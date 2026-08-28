"""The harness sidecar FastAPI application.

State model
-----------
- One ``Hive`` instance per conversation_id (fresh store + comb per
  conversation â€” per-conversation isolation is mandatory, HIVE-HANDOFF Â§6.0 #14).
  Instances are created lazily on the first turn and dropped by /v1/hive/reset.
- Conversations persist to ``state_dir`` (default ./harness_state, one atomic
  JSON per conversation using the same store serialization as the benchmark's
  checkpoint/resume) and reload lazily on first touch after a restart, so the
  hive survives sidecar restarts. /v1/hive/reset deletes memory AND disk.
- One shared ultra-small drone across conversations (a per-conversation encoder
  would multiply VRAM/RAM for nothing); inference is read-only.
- Per-conversation locks serialize turns within a conversation; different
  conversations may proceed in parallel. Generation calls are blocking
  (streaming is a v2 concern) â€” sync endpoints run in FastAPI's threadpool.
- Providers: loaded from providers.local.json at startup (or --providers-file),
  replaceable at runtime via POST /v1/provider/config.

Secrets: api_key values are never echoed back (masked as "***") and are only
written to the local providers file; NDJSON event logs redact separately.
"""

from __future__ import annotations

import hashlib
import html
import json
import os
import re
import subprocess
import sys
import threading
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Callable, List, Optional

import psutil
import requests
from fastapi import FastAPI, File, HTTPException, Query, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, RedirectResponse, StreamingResponse
from pydantic import BaseModel
import queue

from harness.agent import DshAgentService
from harness.commands import ConsoleCommands
from harness.trainer import (
    draft_candidate,
    evaluate_candidate,
    mine_evidence,
    promote,
    summarize_evidence,
)

from backend.cache_manager import KVCacheManager
from backend.engines import (
    EngineProfile,
    EngineRegistry,
    engines_path,
    load_engines,
    save_engines,
)
from backend.openai_compat import OpenAICompatBackend
from backend.providers import (
    MASK,
    Provider,
    ProviderRegistry,
    backend_kwargs,
    load_registry,
    providers_path,
    save_registry,
)
from cortex.config import HiveConfig
from cortex.hive import Hive
from experiments.model_probe import _list_models, probe_model
from harness.models import LlamaServerManager
from harness.reports import (
    render_report_page,
    render_runs_page,
    render_server_page,
    resolve_run_dir,
)
from logs.event_logger import EventLogger
from retention.store import ContextStore


def _list_runs(runs_root: Path) -> list[dict]:
    """Available run bundles under runs_root (newest first)."""
    root = Path(runs_root)
    if not root.is_dir():
        return []
    entries = []
    for child in sorted(root.iterdir(), key=lambda p: p.stat().st_mtime,
                        reverse=True):
        if not child.is_dir():
            continue
        entries.append({
            "name": child.name,
            "has_report": (child / "run_report.json").is_file(),
            "modified": datetime.fromtimestamp(child.stat().st_mtime)
            .strftime("%Y-%m-%d %H:%M:%S"),
        })
    return entries

REPO_ROOT = Path(__file__).resolve().parents[2]
# Hive mode (AFK) canonical state - workspace-level so all projects share one source.
MODE_FILE = Path(os.environ.get("HIVE_MODE_FILE", str(Path(REPO_ROOT).parent / "HIVE-MODE.json")))
RESEARCH_QUEUE = Path(os.environ.get(
    "HIVE_RESEARCH_QUEUE", str(Path(REPO_ROOT).parent / "RESEARCH-QUEUE.md")))
DEFAULT_RUNS_ROOT = REPO_ROOT / "runs"
DEFAULT_PORT = 8765

# generate_data flags the sidecar may forward (whitelist: no arbitrary CLI).
PROTOCOL_FLAGS_INT = {"max_convs": "--max-convs", "max_turns": "--max-turns",
                      "checkpoint_every": "--checkpoint-every"}
PROTOCOL_FLAGS_STR = {"model": "--model", "base_url": "--base-url",
                      "provider": "--provider", "conversations": "--conversations"}
PROTOCOL_FLAGS_BOOL = {"protocol": "--protocol", "baselines": "--baselines",
                       "no_thinking": "--no-thinking"}

_popen = subprocess.Popen  # module-level so tests can intercept

# HTML consoles change with the code; never let a browser cache them.
_NO_STORE = {"Cache-Control": "no-store"}

def _hardware_summary() -> dict:
    """Host VRAM/RAM summary for fit estimates (nvidia-smi + psutil, RAM fallback)."""
    try:
        vm = psutil.virtual_memory()
        total_ram_gb = round(vm.total / (1024 ** 3), 2)
        available_ram_gb = round(vm.available / (1024 ** 3), 2)
    except Exception:
        total_ram_gb = 8.0
        available_ram_gb = 8.0
    vram_gb: Optional[float] = None
    devices: list[dict] = []
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
            timeout=2, text=True, stderr=subprocess.DEVNULL,
        )
        for line in out.strip().splitlines():
            line = line.strip()
            if not line:
                continue
            last = line.rfind(",")
            if last < 0:
                continue
            name = line[:last].strip().strip('"')
            mem_str = line[last + 1:].strip()
            try:
                mib = int(mem_str)
            except ValueError:
                continue
            gb = round(mib / 1024, 2)
            vram_gb = max(vram_gb or 0, gb)
            devices.append({"backend": "cuda", "name": name, "memory_gb": gb})
    except Exception:
        pass
    available_gb = round(vram_gb, 2) if vram_gb is not None else total_ram_gb
    return {
        "total_ram_gb": total_ram_gb,
        "available_ram_gb": available_ram_gb,
        "vram_gb": vram_gb,
        "available_gb": available_gb,
        "devices": devices,
        "vram_source": "nvidia-smi" if vram_gb is not None else "ram",
    }

def _read_gguf_metadata(path: Path) -> dict:
    """Best-effort GGUF header parse for auto-preset (block_count, context, etc).

    Reads the GGUF magic + version + metadata KV section and extracts a handful
    of known keys (general.architecture, *block_count, *context_length,
    *embedding_length) without requiring a full gguf library. Missing or
    unreadable files return {}. This is the gguf-metadata source the Auto
    button combines with GET /v1/models/local size_gb and GET /v1/server/status
    hardware.
    """
    import struct

    try:
        with open(path, "rb") as fh:
            magic = fh.read(4)
            if magic != b"GGUF":
                return {}
            ver = struct.unpack("<I", fh.read(4))[0]
            # v3 uses 64-bit counts, v1/v2 32-bit; try 64 then fallback
            pos = fh.tell()
            try:
                tc = struct.unpack("<Q", fh.read(8))[0]
                mc = struct.unpack("<Q", fh.read(8))[0]
                # sanity: metadata count should be reasonable (< 10000)
                if mc > 10000:
                    raise ValueError("implausible")
            except Exception:
                fh.seek(pos)
                tc = struct.unpack("<I", fh.read(4))[0]
                mc = struct.unpack("<I", fh.read(4))[0]
            out: dict = {}
            for _ in range(int(mc)):
                try:
                    klen = struct.unpack("<Q" if ver >= 3 else "<I", fh.read(8 if ver >= 3 else 4))[0]
                except Exception:
                    break
                if klen > 500:
                    break
                key = fh.read(int(klen)).decode("utf-8", errors="ignore")
                try:
                    ktype = struct.unpack("<I", fh.read(4))[0]
                except Exception:
                    break
                # we care about a few string/uint32/uint64 keys
                try:
                    if ktype == 0:  # uint8
                        val = struct.unpack("<B", fh.read(1))[0]
                    elif ktype == 1:  # int8
                        val = struct.unpack("<b", fh.read(1))[0]
                    elif ktype == 2:  # uint16
                        val = struct.unpack("<H", fh.read(2))[0]
                    elif ktype == 3:  # int16
                        val = struct.unpack("<h", fh.read(2))[0]
                    elif ktype == 4:  # uint32
                        val = struct.unpack("<I", fh.read(4))[0]
                    elif ktype == 5:  # int32
                        val = struct.unpack("<i", fh.read(4))[0]
                    elif ktype == 6:  # float32
                        val = struct.unpack("<f", fh.read(4))[0]
                    elif ktype == 7:  # bool
                        val = bool(struct.unpack("<B", fh.read(1))[0])
                    elif ktype == 8:  # string
                        slen = struct.unpack("<Q" if ver >= 3 else "<I", fh.read(8 if ver >= 3 else 4))[0]
                        val = fh.read(int(slen)).decode("utf-8", errors="ignore")
                    elif ktype == 9:  # array
                        atype = struct.unpack("<I", fh.read(4))[0]
                        alen = struct.unpack("<Q" if ver >= 3 else "<I", fh.read(8 if ver >= 3 else 4))[0]
                        # skip arrays — not needed for presets
                        if atype == 8:  # string array
                            for __ in range(int(alen)):
                                sl = struct.unpack("<Q" if ver >= 3 else "<I", fh.read(8 if ver >= 3 else 4))[0]
                                fh.read(int(sl))
                            val = f"<array:{alen}>"
                        else:
                            size = {0:1,1:1,2:2,3:2,4:4,5:4,6:4,7:1,10:8,11:8,12:8}.get(atype, 1)
                            fh.read(int(alen)*size)
                            val = f"<array:{alen}>"
                    elif ktype == 10:  # uint64
                        val = struct.unpack("<Q", fh.read(8))[0]
                    elif ktype == 11:  # int64
                        val = struct.unpack("<q", fh.read(8))[0]
                    elif ktype == 12:  # float64
                        val = struct.unpack("<d", fh.read(8))[0]
                    else:
                        break
                except Exception:
                    break
                # keep only interesting keys
                if any(key.endswith(s) for s in (".block_count", ".context_length", ".embedding_length", ".feed_forward_length")) \
                   or key in ("general.architecture", "general.name", "general.parameter_count", "general.quantization_version"):
                    out[key] = val
                # early exit after we have architecture + block_count
                if "general.architecture" in out and any(k.endswith(".block_count") for k in out):
                    # keep reading a few more but not the whole file
                    if len(out) > 12:
                        break
            return out
    except Exception:
        return {}


def _auto_preset_load_options(size_gb: float, hardware_gb: float, file_name: str, gguf_meta: dict) -> dict:
    """Compute Auto preset load_options from hardware + model size + gguf-metadata.

    Matches the UI contract: qwen3-4b on 8GB → gpu_layers 28 + 8k ctx,
    qwen3-32b on 8GB → 12 layers + 4k, larger VRAM → larger offload/ctx.
    Other load options (threads, flash_attn, kv quant) are tuned with context.
    """
    name = (file_name or "").lower()
    meta = gguf_meta or {}
    # parameter count from gguf if available
    params_b = None
    for k in ("general.parameter_count", "general.parameter_count", "parameter_count"):
        if k in meta:
            try:
                params_b = float(meta[k]) / 1e9
                break
            except Exception:
                pass
    # block_count → total layers
    block_count = None
    for k, v in meta.items():
        if k.endswith(".block_count"):
            try:
                block_count = int(v)
                break
            except Exception:
                pass
    est_layers = block_count
    if not est_layers:
        if "32b" in name or "30b" in name or (params_b and params_b >= 30) or size_gb > 15:
            est_layers = 62
        elif "14b" in name or "13b" in name or (params_b and params_b >= 13):
            est_layers = 40
        elif "7b" in name or "8b" in name or (params_b and params_b >= 7):
            est_layers = 32
        elif "4b" in name or "3b" in name or "qwen3-4b" in name or (params_b and params_b >= 3):
            est_layers = 36
        else:
            est_layers = 32
    is_32 = "32b" in name or "30b" in name or (params_b and params_b >= 30) or size_gb > 15
    is_4 = "4b" in name or "3b" in name or "qwen3-4b" in name or (params_b is not None and 3 <= params_b < 6) or (2 <= size_gb < 6)
    avail = float(hardware_gb) if hardware_gb else 8.0
    if is_32:
        if avail <= 9:
            gpu_layers, ctx = 12, 4096
        elif avail <= 16:
            gpu_layers, ctx = 20, 8192
        elif avail <= 24:
            gpu_layers, ctx = min(40, est_layers), 8192
        else:
            gpu_layers, ctx = 999, 16384
    elif is_4:
        if avail <= 9:
            gpu_layers, ctx = 28, 8192
        elif avail <= 16:
            gpu_layers, ctx = 999, 16384
        else:
            gpu_layers, ctx = 999, 32768
    else:
        if size_gb + 2.0 <= avail * 0.9:
            gpu_layers, ctx = 999, 8192
        elif size_gb + 1.0 <= avail * 1.1:
            gpu_layers, ctx = min(28, est_layers), 8192
        else:
            gpu_layers, ctx = min(12, est_layers), 4096
    if gpu_layers != 999:
        gpu_layers = min(gpu_layers, est_layers)
    out = {"gpu_layers": gpu_layers, "context": ctx, "ctx_size": ctx}
    # advisory extras
    out["flash_attn"] = ctx >= 8192
    if ctx > 16384:
        out["cache_type_k"] = "q8_0"
        out["cache_type_v"] = "q8_0"
    return out


# Streaming upstream (llama-server / any OpenAI-compatible provider); module
# level so tests can inject a fake SSE transport.
_upstream_stream = requests.post


class _State:
    """Mutable app state: hives, locks, providers, engines, factories."""

    def __init__(
        self,
        ultra_factory: Callable[[], object],
        backend_factory: Callable[[Optional[str]], object],
        runs_root: Path,
        providers_file: Optional[Path],
        log_dir: str,
        state_dir: Optional[Path] = None,
        engines_file: Optional[Path] = None,
    ) -> None:
        self.ultra_factory = ultra_factory
        self.backend_factory = backend_factory
        self.runs_root = runs_root
        self.providers_file = providers_file
        self.engines_file = engines_file
        self.log_dir = log_dir
        # Conversations persist here across restarts; None/empty disables.
        self.state_dir = Path(state_dir) if state_dir else None
        if self.state_dir is not None:
            self.state_dir.mkdir(parents=True, exist_ok=True)
        self.registry = ProviderRegistry()
        self.engines = EngineRegistry()
        self._ultra = None
        self.hives: dict[str, Hive] = {}
        self.locks: dict[str, threading.Lock] = {}
        self.global_lock = threading.Lock()
        # Conversation lifecycle: LRU-bounded so a long-running sidecar cannot
        # accumulate hives/loggers from every browser session that ever opened.
        self.max_conversations = int(os.environ.get("HARNESS_MAX_CONVERSATIONS", "50"))
        self._last_access: dict[str, float] = {}
        self._inflight: set[str] = set()
        self._loggers: dict[str, EventLogger] = {}
        self._conv_provider: dict[str, str] = {}  # per-conversation override

    def ultra(self):
        if self._ultra is None:
            self._ultra = self.ultra_factory()
        return self._ultra

    def _conv_path(self, conversation_id: str) -> Optional[Path]:
        """Per-conversation state file. Content-hashed name: arbitrary ids
        (session UUIDs, workspace keys, user input) stay safe on disk."""
        if self.state_dir is None:
            return None
        digest = hashlib.md5(conversation_id.encode("utf-8")).hexdigest()[:16]
        return self.state_dir / f"conv-{digest}.json"

    def save_conversation(self, conversation_id: str, hive: Hive) -> None:
        """Persist one conversation atomically (tmp file + os.replace)."""
        path = self._conv_path(conversation_id)
        if path is None:
            return
        payload = {
            "conversation_id": conversation_id,
            "turn": hive.turn,
            "with_backend": hive.backend is not None,
            "config": hive.config.to_dict(),
            "store": hive.store.to_dict(),
        }
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(payload), encoding="utf-8")
        os.replace(tmp, path)

    def drop_conversation(self, conversation_id: str) -> None:
        path = self._conv_path(conversation_id)
        if path is not None and path.exists():
            path.unlink()

    def hive_for(
        self, conversation_id: str, config_overrides: dict | None,
        with_backend: bool = True, engine: Optional[str] = None,
    ) -> Hive:
        """Get or lazily create the conversation's hive.

        A conversation not in memory but present in ``state_dir`` restores
        from disk (same serialization as the benchmark's checkpoint/resume),
        so the hive survives sidecar restarts. In-memory hives are LRU-bounded
        (``HARNESS_MAX_CONVERSATIONS``); evicted conversations are persisted
        first and transparently restore on their next touch.

        ``with_backend=False`` (the curate/observe flow, where the caller's
        own shell generates) creates the hive without an LLM backend; a
        conversation is driven either fully (/v1/hive/turn) or externally
        (curate + observe), whichever touches it first wins.
        """
        with self.global_lock:
            hive = self.hives.get(conversation_id)
            if hive is not None:
                self._last_access[conversation_id] = time.monotonic()
                return hive

            def build(cfg: HiveConfig, backend: object | None) -> Hive:
                logger = self._loggers.get(conversation_id)
                if logger is None:
                    logger = EventLogger(log_dir=self.log_dir)
                    self._loggers[conversation_id] = logger
                h = Hive(
                    config=cfg,
                    ultra=self.ultra(),
                    backend=backend,
                    logger=logger,
                )
                self.hives[conversation_id] = h
                self.locks.setdefault(conversation_id, threading.Lock())
                return h

            path = self._conv_path(conversation_id)
            if path is not None and path.exists():
                try:
                    data = json.loads(path.read_text(encoding="utf-8"))
                    hive = build(HiveConfig.from_dict(data["config"]),
                                 self.backend_factory(None)
                                 if data.get("with_backend") else None)
                    hive.store = ContextStore.from_dict(
                        data["store"], embed_fn=hive.ultra.embed
                    )
                    hive.turn = int(data["turn"])
                    self._last_access[conversation_id] = time.monotonic()
                    self._evict_locked(exclude=conversation_id)
                    return hive
                except (ValueError, KeyError, TypeError, OSError) as exc:
                    print(f"harness: restoring {conversation_id} failed ({exc}); "
                          "starting fresh", file=sys.stderr)

            config = HiveConfig(confidence_mode="off")
            if config_overrides:
                merged = {**config.to_dict(), **config_overrides}
                config = HiveConfig.from_dict(merged)
            if not config.sampling and self.engines.engines:
                # Engine sampling defaults apply when the caller did not
                # specify sampling (per-call / per-config overrides win).
                try:
                    profile = self.engines.resolve(engine)
                except LookupError:
                    profile = None
                if profile is not None and profile.sampling:
                    config.sampling = profile.sampling
            hive = build(config, self.backend_factory(None) if with_backend else None)
            self._last_access[conversation_id] = time.monotonic()
            self._evict_locked(exclude=conversation_id)
            return hive

    def _evict_locked(self, exclude: str) -> int:
        """LRU-evict idle conversations beyond the cap. Caller holds the
        global lock; in-flight conversations are never evicted, and evicted
        state is persisted first (restore-on-touch keeps it reachable)."""
        evicted = 0
        while len(self.hives) > self.max_conversations:
            candidates = [cid for cid in self.hives
                          if cid != exclude and cid not in self._inflight]
            if not candidates:
                break
            oldest = min(candidates, key=lambda c: self._last_access.get(c, 0.0))
            self.save_conversation(oldest, self.hives[oldest])
            logger = self._loggers.pop(oldest, None)
            if logger is not None:
                try:
                    logger.close()
                except Exception:  # noqa: BLE001 - eviction must not fail
                    pass
            self.hives.pop(oldest, None)
            self.locks.pop(oldest, None)
            self._last_access.pop(oldest, None)
            evicted += 1
        return evicted

    def begin(self, conversation_id: str) -> None:
        self._inflight.add(conversation_id)
        self._last_access[conversation_id] = time.monotonic()

    def end(self, conversation_id: str) -> None:
        self._inflight.discard(conversation_id)

    def drop(self, conversation_id: str) -> None:
        with self.global_lock:
            self.hives.pop(conversation_id, None)
            self.locks.pop(conversation_id, None)
            self._last_access.pop(conversation_id, None)
            logger = self._loggers.pop(conversation_id, None)
        if logger is not None:
            try:
                logger.close()
            except Exception:  # noqa: BLE001
                pass
        self.drop_conversation(conversation_id)

    def lock_for(self, conversation_id: str) -> threading.Lock:
        with self.global_lock:
            return self.locks.setdefault(conversation_id, threading.Lock())


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------
class TurnRequest(BaseModel):
    query: str
    conversation_id: str = "default"
    model: Optional[str] = None  # override the provider's model for this turn's hive
    provider: Optional[str] = None  # per-conversation inference target (multi-model)
    engine: Optional[str] = None  # engine profile name (sampling defaults apply)
    config: Optional[dict] = None  # HiveConfig overrides (applied on creation)


class ResetRequest(BaseModel):
    conversation_id: str


class CurateRequest(BaseModel):
    query: str
    conversation_id: str = "default"
    engine: Optional[str] = None
    config: Optional[dict] = None


class ObserveRequest(BaseModel):
    conversation_id: str
    reply: str


class ProtocolRunRequest(BaseModel):
    mode: str = "mock"  # live | mock
    args: dict = {}


class ProviderEntry(BaseModel):
    name: str
    base_url: str
    api_key: str = ""
    model: str = ""
    headers: dict = {}


class ProviderConfigRequest(BaseModel):
    providers: list[ProviderEntry]
    default: str = ""
    persist: bool = False


class ServerStartRequest(BaseModel):
    model: Optional[str] = None  # local library name or path
    hf_repo: Optional[str] = None  # passthrough to llama-server --hf-repo
    hf_file: Optional[str] = None
    key: Optional[str] = None  # instance key (defaults to the model stem)
    port: Optional[int] = None
    ctx_size: int = 8192
    ngl: int = 999  # GPU layers (Vulkan build: all layers on the RX 7900 XT)
    register_provider: bool = True
    claim_default: bool = True  # first load claims the default provider slot
    # extra llama-server launch flags (wired to the UI settings panel)
    threads: Optional[int] = None
    flash_attn: bool = False
    parallel_slots: Optional[int] = None
    cache_type_k: Optional[str] = None  # f16 | q8_0 | q4_0 ...
    cache_type_v: Optional[str] = None
    batch_size: Optional[int] = None
    ubatch_size: Optional[int] = None
    alias: Optional[str] = None
    mlock: bool = False
    no_mmap: bool = False
    api_key: Optional[str] = None  # protect llama-server (--api-key)
    backend: Optional[str] = None  # vulkan | rocm | cuda | cpu | sycl (binary under tools/backends/<backend>/)

    def extra_args(self) -> list[str]:
        args: list[str] = []
        if self.threads:
            args += ["-t", str(self.threads)]
        if self.flash_attn:
            # current llama.cpp: -fa takes on|off|auto (a bare -fa would eat
            # the next flag as its value)
            args += ["-fa", "on"]
        if self.parallel_slots:
            args += ["-np", str(self.parallel_slots)]
        if self.cache_type_k:
            args += ["--cache-type-k", self.cache_type_k]
        if self.cache_type_v:
            args += ["--cache-type-v", self.cache_type_v]
        if self.batch_size:
            args += ["-b", str(self.batch_size)]
        if self.ubatch_size:
            args += ["-ub", str(self.ubatch_size)]
        if self.alias:
            args += ["--alias", self.alias]
        if self.mlock:
            args += ["--mlock"]
        if self.no_mmap:
            args += ["--no-mmap"]
        if self.api_key:
            args += ["--api-key", self.api_key]
        return args

    def load_options(self) -> dict:
        """Advisory engine record of the launch configuration actually used."""
        out = {"context": self.ctx_size, "gpu_layers": self.ngl}
        for key, value in (("threads", self.threads),
                           ("flash_attn", self.flash_attn or None),
                           ("parallel_slots", self.parallel_slots),
                           ("cache_type_k", self.cache_type_k),
                           ("cache_type_v", self.cache_type_v),
                           ("batch_size", self.batch_size),
                           ("ubatch_size", self.ubatch_size),
                           ("alias", self.alias),
                           ("mlock", self.mlock or None),
                           ("no_mmap", self.no_mmap or None)):
            if value is not None:
                out[key] = value
        return out


class HubDownloadRequest(BaseModel):
    repo: str
    file: str


class ServerUnloadRequest(BaseModel):
    key: str


class StreamTurnRequest(BaseModel):
    query: str
    conversation_id: str = "default"
    engine: Optional[str] = None
    config: Optional[dict] = None


class AgentMessageRequest(BaseModel):
    message: str
    conversation_id: str = "default"


class CommandRunRequest(BaseModel):
    line: str
    conversation_id: str = "default"


class EngineEntry(BaseModel):
    name: str
    kind: str = "lmstudio"
    base_url: str = ""
    load_options: dict = {}
    capabilities: list[str] = []
    sampling: dict = {}


class EngineConfigRequest(BaseModel):
    engines: list[EngineEntry]
    default: str = ""
    persist: bool = False


def _cors_origins() -> list[str]:
    """Console origins. Default: localhost dev origins only â€” agent mode can
    execute code, so blanket CORS (*) is opt-in via HARNESS_CORS_ORIGINS=*."""
    raw = os.environ.get("HARNESS_CORS_ORIGINS", "").strip()
    if raw == "*":
        return ["*"]
    if raw:
        return [o.strip() for o in raw.split(",") if o.strip()]
    return ["http://localhost:5173", "http://127.0.0.1:5173",
            "http://localhost:3000", "http://127.0.0.1:3000",
            "http://localhost:8765", "http://127.0.0.1:8765"]


def _required_token() -> str:
    """When HARNESS_TOKEN is set, /v1/* mutations require this bearer token."""
    return os.environ.get("HARNESS_TOKEN", "").strip()


def create_app(
    ultra_factory: Optional[Callable[[], object]] = None,
    backend_factory: Optional[Callable[[Optional[str]], object]] = None,
    runs_root: Optional[Path] = None,
    providers_file: Optional[Path] = None,
    log_dir: str = "logs",
    state_dir: Optional[Path] = None,
    engines_file: Optional[Path] = None,
    models_manager: Optional[LlamaServerManager] = None,
    llama_port: int = 1234,
) -> FastAPI:
    """Build the sidecar app.

    ``ultra_factory`` / ``backend_factory`` are injectable for offline tests;
    defaults build the real L3-v2 drone and a provider-driven OpenAI-compat
    backend (LM Studio on localhost:1234 when no providers are configured).
    ``state_dir=None`` defaults to ./harness_state (conversations survive
    restarts); passing an empty string disables persistence.
    """
    from sieve.ultra_small import UltraSmallDrone

    embedding_backend = os.environ.get("HARNESS_EMBEDDING_BACKEND", "local")
    embedding_url = os.environ.get("HARNESS_EMBEDDING_URL", "")
    embedding_model = os.environ.get("HARNESS_EMBEDDING_MODEL", "default")

    def _default_ultra():
        if embedding_backend == "served" and embedding_url:
            from sieve.served import ServedEmbeddingDrone

            return ServedEmbeddingDrone(base_url=embedding_url,
                                        model=embedding_model)
        return UltraSmallDrone(confidence_mode="off")

    def _default_backend(model: Optional[str], provider: Optional[str] = None):
        kw = backend_kwargs(st.registry.resolve(provider))
        if model:
            kw["model"] = model
        return OpenAICompatBackend(**kw)

    app = FastAPI(title="Hive Studio", version="0.1.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=_cors_origins(),
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.middleware("http")
    async def token_guard(request: Request, call_next):
        required = _required_token()
        if required and request.url.path.startswith("/v1/"):
            supplied = request.headers.get("x-hive-token", "")
            if supplied != required:
                from fastapi.responses import JSONResponse

                return JSONResponse({"detail": "invalid or missing token"},
                                    status_code=401)
        return await call_next(request)

    st = _State(
        ultra_factory=ultra_factory or _default_ultra,
        backend_factory=backend_factory or _default_backend,
        runs_root=Path(runs_root) if runs_root else DEFAULT_RUNS_ROOT,
        providers_file=Path(providers_file) if providers_file else None,
        log_dir=log_dir,
        state_dir=state_dir if state_dir is not None else Path("harness_state"),
        engines_file=Path(engines_file) if engines_file else None,
    )
    try:
        st.registry = load_registry(providers_file)
    except (ValueError, OSError) as exc:
        print(f"harness: ignoring unreadable providers config ({exc})", file=sys.stderr)
    try:
        st.engines = load_engines(engines_file)
    except (ValueError, OSError) as exc:
        print(f"harness: ignoring unreadable engines config ({exc})", file=sys.stderr)
    app.state.harness = st

    @app.get("/health")
    def health():
        return {"ok": True, "conversations": len(st.hives)}

    # ------------------------------------------------------------------
    @app.post("/v1/hive/turn")
    def hive_turn(req: TurnRequest):
        query = (req.query or "").strip()
        if not query:
            raise HTTPException(422, "query must not be empty")
        hive = st.hive_for(req.conversation_id, req.config, engine=req.engine)
        # Per-conversation inference target: provider and/or model override
        # swaps the conversation's backend (multi-model: pick any loaded one).
        current_provider = st._conv_provider.get(req.conversation_id)
        wants_backend = (req.provider and req.provider != current_provider) \
            or (req.model and isinstance(hive.backend, OpenAICompatBackend)
                and req.model != hive.backend.model)
        if wants_backend and isinstance(hive.backend, OpenAICompatBackend):
            new_backend = st.backend_factory(req.model, provider=req.provider)
            hive.backend = new_backend
            hive.cache = KVCacheManager(new_backend)
            st._conv_provider[req.conversation_id] = req.provider \
                or st.registry.default
        st.begin(req.conversation_id)
        with st.lock_for(req.conversation_id):
            result = hive.process_turn(req.query, conversation_id=req.conversation_id)
            st.save_conversation(req.conversation_id, hive)
        st.end(req.conversation_id)
        assembled = result.assembled
        return {
            "conversation_id": req.conversation_id,
            "turn": result.turn,
            "reply": result.reply,
            "assembled_content": assembled.content if assembled is not None else "",
            "token_count": result.token_count,
            "budget": result.budget,
            "mode": result.mode,
            "error": result.error,
            "timings": result.timings,
            "pes": result.pes,
            "degradation_level": result.degradation_level,
            "inspection": hive.inspect_turn(result),
        }

    @app.get("/v1/hive/inspect/{conversation_id}")
    def hive_inspect(conversation_id: str):
        """Last turn's full curation detail for the prompt inspector."""
        with st.global_lock:
            hive = st.hives.get(conversation_id)
        if hive is None:
            raise HTTPException(404, f"no such conversation: {conversation_id}")
        if not hasattr(hive, "_last_turn_result") or hive._last_turn_result is None:
            raise HTTPException(404, "no turn has been processed yet")
        return hive.inspect_turn(hive._last_turn_result)

    @app.post("/v1/hive/reset")
    def hive_reset(req: ResetRequest):
        st.drop(req.conversation_id)
        return {"ok": True}

    # ------------------------------------------------------------------
    # Curate / observe (Seam A, dsh-hive flow): the caller's own shell
    # generates â€” the sidecar only assembles context and ingests replies.
    @app.post("/v1/hive/curate")
    def hive_curate(req: CurateRequest):
        query = (req.query or "").strip()
        if not query:
            raise HTTPException(422, "query must not be empty")
        hive = st.hive_for(req.conversation_id, req.config, with_backend=False,
                           engine=req.engine)
        with st.lock_for(req.conversation_id):
            result = hive.process_turn(query, conversation_id=req.conversation_id)
            st.save_conversation(req.conversation_id, hive)
        assembled = result.assembled
        return {
            "conversation_id": req.conversation_id,
            "turn": result.turn,
            "assembled_content": assembled.content if assembled is not None else "",
            "token_count": result.token_count,
            "budget": result.budget,
            "mode": result.mode,
            "error": result.error,
            "timings": result.timings,
            "pes": result.pes,
            "degradation_level": result.degradation_level,
        }

    @app.post("/v1/hive/observe")
    def hive_observe(req: ObserveRequest):
        # lazily create: external integrators may observe before ever calling
        # curate (e.g. feeding back a reply for a session the studio has
        # never seen); the conversation materializes here.
        hive = st.hive_for(req.conversation_id, None, with_backend=False)
        reply = (req.reply or "").strip()
        stored = False
        if reply and not (
            hive.config.filter_hedge_replies and Hive._is_hedge_reply(reply)
        ):
            st.begin(req.conversation_id)
            with st.lock_for(req.conversation_id):
                hive.store.add_chunk(hive.turn, reply)
                st.save_conversation(req.conversation_id, hive)
            st.end(req.conversation_id)
            stored = True
        return {"ok": True, "stored": stored, "turn": hive.turn}

    # ------------------------------------------------------------------
    # Streaming chat (LM-Studio-style token stream) THROUGH the hive:
    # curate -> stream the provider's SSE -> observe the reply back into
    # the store. Events: {type: meta|delta|done|error}.
    @app.post("/v1/hive/stream")
    async def hive_stream(req: StreamTurnRequest):
        query = (req.query or "").strip()
        if not query:
            raise HTTPException(422, "query must not be empty")
        try:
            provider = st.registry.resolve(None)
        except LookupError:
            raise HTTPException(502, "no provider configured; start a local "
                                     "server or configure one")
        base_url = provider.base_url.rstrip("/")
        headers = {"Authorization": f"Bearer {provider.api_key or 'lm-studio'}",
                   **provider.extra_headers}
        hive = st.hive_for(req.conversation_id, req.config, with_backend=False)
        st.begin(req.conversation_id)
        with st.lock_for(req.conversation_id):
            result = hive.process_turn(query, conversation_id=req.conversation_id)
            st.save_conversation(req.conversation_id, hive)
        st.end(req.conversation_id)
        assembled = result.assembled
        curated = assembled.content if assembled is not None else ""
        payload = {
            "model": provider.model or "local",
            "messages": [
                {"role": "system", "content": curated or "You are a helpful assistant."},
                {"role": "user", "content": query},
            ],
            "stream": True,
            "stream_options": {"include_usage": True},
            **(hive.config.sampling or {}),
        }
        if hive.config.max_tokens:
            payload["max_tokens"] = hive.config.max_tokens

        def sse():
            yield "data: " + json.dumps({
                "type": "meta", "turn": result.turn,
                "token_count": result.token_count, "budget": result.budget,
                "curated_chars": len(curated), "mode": result.mode,
            }) + "\n\n"

            started = time.time()
            parts: list[str] = []
            usage: dict = {}
            try:
                resp = _upstream_stream(
                    f"{base_url}/v1/chat/completions", json=payload,
                    headers=headers, stream=True, timeout=600,
                )
                resp.raise_for_status()
                for raw in resp.iter_lines(decode_unicode=True):
                    if not raw:
                        continue
                    line = raw[6:].strip() if raw.startswith("data:") else raw.strip()
                    if not line or line == "[DONE]":
                        continue
                    try:
                        chunk = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    usage = chunk.get("usage") or usage
                    for choice in chunk.get("choices") or []:
                        delta = choice.get("delta") or {}
                        text = delta.get("content")
                        if text:
                            parts.append(text)
                            yield "data: " + json.dumps({
                                "type": "delta", "text": text}) + "\n\n"
            except Exception as exc:  # noqa: BLE001 - surfaced as an event
                yield "data: " + json.dumps({
                    "type": "error", "error": str(exc)}) + "\n\n"

            reply = "".join(parts)
            stored = False
            if reply.strip() and not (
                hive.config.filter_hedge_replies
                and Hive._is_hedge_reply(reply)
            ):
                hive.store.add_chunk(hive.turn, reply)
                st.save_conversation(req.conversation_id, hive)
                stored = True
            elapsed = max(time.time() - started, 1e-6)
            completion_tokens = (usage or {}).get("completion_tokens") or 0
            yield "data: " + json.dumps({
                "type": "done", "stored": stored,
                "tokens": completion_tokens,
                "seconds": round(elapsed, 2),
                "tokens_per_sec": round(completion_tokens / elapsed, 1)
                if completion_tokens else None,
            }) + "\n\n"

        return StreamingResponse(sse(), media_type="text/event-stream")

    @app.get("/v1/hive/defaults")
    def hive_defaults():
        """HiveConfig defaults â€” the source for the UI tuning form. Overrides
        ride each turn request's `config` and apply when a conversation is
        created (reset to re-tune)."""
        return HiveConfig().to_dict()

    @app.get("/v1/hive/state")
    def hive_state(conversation_id: Optional[str] = Query(default=None)):
        def snapshot(h: Hive) -> dict:
            return {
                "turn": h.turn,
                "store_chunks": len(h.store.all_chunks()),
                "comb_stats": dict(h.comb_stats),
            }

        if conversation_id:
            with st.global_lock:
                hive = st.hives.get(conversation_id)
            if hive is None and st.state_dir is not None \
                    and st._conv_path(conversation_id).exists():
                # lazy-restore a persisted conversation so state survives restarts
                hive = st.hive_for(conversation_id, None)
            if hive is None:
                raise HTTPException(404, f"no such conversation: {conversation_id}")
            return {**snapshot(hive), "conversation_id": conversation_id}
        with st.global_lock:
            items = {cid: snapshot(h) for cid, h in st.hives.items()}
        return {"count": len(items), "conversations": items}

    # ------------------------------------------------------------------
    @app.get("/v1/models")
    def models(
        probe: bool = Query(default=False),
        provider: Optional[str] = Query(default=None),
        base_url: Optional[str] = Query(default=None),
    ):
        target = base_url
        if not target:
            try:
                target = st.registry.resolve(provider).base_url
            except LookupError:
                target = "http://localhost:1234"
        try:
            ids = _list_models(target)
        except Exception as exc:  # noqa: BLE001 - surfaced as 502 to the caller
            raise HTTPException(502, f"cannot list models from {target}: {exc}")
        out = {"base_url": target, "models": ids, "probe": None}
        if probe:
            results = [probe_model(target, m).__dict__ for m in ids]
            out["probe"] = results
        return out

    # ------------------------------------------------------------------
    # Built-in mock OpenAI-compatible chat completions: pairs with
    # `python -m harness --mock` so a dsh shell (pi-ai openai-completions
    # route) can run end-to-end offline. The reply deterministically echoes
    # what the request actually contained â€” context size and whether hive
    # content reached the model â€” which makes it a live probe of Seam A.
    # When the conversation asks for the benchmark, it emits a proper
    # hive_bench_run tool call and then acknowledges the tool result, so the
    # full agent loop (request -> tool_call -> tool/result -> answer) is
    # exercised offline.
    def _mock_reply(payload: dict) -> str:
        messages = payload.get("messages") or []
        system_txt = ""
        user_txt = ""
        for m in messages:
            if m.get("role") == "system" and not system_txt:
                system_txt = str(m.get("content") or "")
            elif m.get("role") == "user":
                user_txt = str(m.get("content") or "")
        # the exact marker dsh-hive appends as a snapshot user message
        curated = any(
            "hive-curated-context" in str(m.get("content") or "")
            for m in messages
        )
        head = " ".join(system_txt.split())[:160]
        return (
            f"[hive-mock] model={payload.get('model', '?')} "
            f"system={len(system_txt)}ch user={len(user_txt)}ch "
            f"hive_context={'yes' if curated else 'no'} "
            f"context_head={head!r}"
        )

    def _message_text(message: object) -> str:
        if isinstance(message, str):
            return message
        if isinstance(message, list):
            parts = []
            for block in message:
                if isinstance(block, dict):
                    parts.append(str(block.get("text") or ""))
                else:
                    parts.append(str(block))
            return "".join(parts)
        return str(message or "")

    def _mock_chat_decision(payload: dict) -> dict:
        """Return {'reply': str} or {'tool_call': (name, arguments_json)}."""
        messages = payload.get("messages") or []
        tool_results = [m for m in messages if m.get("role") == "tool"]
        if tool_results:
            last = tool_results[-1]
            text = _message_text(last.get("content"))
            lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
            verdicts = next(
                (ln for ln in lines if "|" in ln), ""
            )
            pes = next((ln for ln in lines if "PES" in ln.upper()), "")
            reply = "The HiveBench run completed."
            if pes:
                reply += f" {pes}."
            if verdicts:
                reply += f" Verdicts: {verdicts}"
            return {"reply": reply}
        user_text = " ".join(
            str(m.get("content") or "") for m in messages if m.get("role") == "user"
        ).lower()
        if "hive_bench" in user_text or "benchmark" in user_text \
                or "p1-p11" in user_text or "p1â€“p11" in user_text:
            match = re.search(r"(\d+)\s+conv", user_text)
            max_convs = int(match.group(1)) if match else 2
            return {"tool_call": (
                "hive_bench_run",
                json.dumps({"mode": "mock", "max_convs": max_convs,
                            "protocol": True}),
            )}
        return {"reply": _mock_reply(payload)}

    def _mock_completion_payload(payload: dict, decision: dict,
                                 cid: str, created: int) -> tuple[dict, dict]:
        usage = {
            "prompt_tokens": sum(len(str(m.get("content") or "").split())
                                 for m in (payload.get("messages") or [])),
            "completion_tokens": 40, "total_tokens": 0,
        }
        model = payload.get("model", "mock")
        if "tool_call" in decision:
            name, arguments = decision["tool_call"]
            message = {
                "role": "assistant", "content": None,
                "tool_calls": [{
                    "id": f"call_{uuid.uuid4().hex[:16]}",
                    "type": "function",
                    "function": {"name": name, "arguments": arguments},
                }],
            }
            finish = "tool_calls"
            completion_tokens = len(arguments.split()) + 6
        else:
            message = {"role": "assistant", "content": decision["reply"]}
            finish = "stop"
            completion_tokens = len(decision["reply"].split())
        usage["completion_tokens"] = completion_tokens
        usage["total_tokens"] = usage["prompt_tokens"] + completion_tokens
        return message, {"model": model, "finish": finish, "usage": usage}

    @app.post("/v1/chat/completions")
    async def mock_chat_completions(request: Request):
        payload = await request.json()
        debug_dir = os.environ.get("HARNESS_DEBUG_CHAT")
        if debug_dir:
            Path(debug_dir).mkdir(parents=True, exist_ok=True)
            with open(Path(debug_dir) / f"chat_{int(time.time() * 1000)}.json",
                      "w", encoding="utf-8") as fh:
                json.dump(payload, fh, indent=1, default=str)
        decision = _mock_chat_decision(payload)
        cid = f"chatcmpl-mock-{uuid.uuid4().hex[:12]}"
        created = int(time.time())
        message, meta = _mock_completion_payload(payload, decision, cid, created)

        if not payload.get("stream"):
            return {
                "id": cid, "object": "chat.completion", "created": created,
                "model": meta["model"],
                "choices": [{
                    "index": 0, "message": message,
                    "finish_reason": meta["finish"],
                }],
                "usage": meta["usage"],
            }

        if "tool_call" in decision:
            tc = message["tool_calls"][0]
            chunks = [
                {"delta": {"role": "assistant", "tool_calls": [{
                    "index": 0, "id": tc["id"], "type": "function",
                    "function": {"name": tc["function"]["name"],
                                 "arguments": ""},
                }]}},
                {"delta": {"tool_calls": [{"index": 0, "function": {
                    "arguments": tc["function"]["arguments"]}}]}},
                {"delta": {}, "finish_reason": meta["finish"]},
            ]
        else:
            content = message["content"]
            pieces = [content[i:i + 24] for i in range(0, len(content), 24)] or [""]
            chunks = [{"delta": {"role": "assistant", "content": p}} for p in pieces]
            chunks.append({"delta": {}, "finish_reason": "stop"})

        chunks[-1]["finish_reason"] = meta["finish"]

        def sse():
            for part in chunks:
                choice = {"index": 0,
                          "delta": part.get("delta", {}),
                          "finish_reason": part.get("finish_reason")}
                body = {
                    "id": cid, "object": "chat.completion.chunk",
                    "created": created, "model": meta["model"],
                    "choices": [choice],
                }
                if part is chunks[-1]:
                    body["usage"] = meta["usage"]
                yield "data: " + json.dumps(body) + "\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(sse(), media_type="text/event-stream")

    # ------------------------------------------------------------------
    # Real OpenAI-compatible passthrough (curated) â€” Mode A integration
    # (OpenCode, dsh, any OpenAI client): standard /chat/completions wire
    # shape, curated system context, the reply observed back into the
    # store. Conversation key: X-Hive-Conversation header > payload "user"
    # > "default".
    @app.post("/v1/openai/chat/completions")
    async def openai_chat_completions(request: Request):
        payload = await request.json()
        messages = payload.get("messages") or []
        if not messages:
            raise HTTPException(422, "messages must not be empty")
        query = ""
        for m in reversed(messages):
            content = m.get("content") if m.get("role") == "user" else None
            if isinstance(content, str) and content.strip():
                query = content
                break
        if not query.strip():
            raise HTTPException(422, "no user message with text content")
        cid = (request.headers.get("X-Hive-Conversation")
               or (payload.get("user") or "") or "default")
        try:
            provider = st.registry.resolve(None)
        except LookupError:
            raise HTTPException(
                502, "no provider configured; configure one via /v1/provider/config "
                     "or providers.local.json")
        base_url = provider.base_url.rstrip("/")
        headers = {"Authorization": f"Bearer {provider.api_key or 'lm-studio'}",
                   **provider.extra_headers}
        hive = st.hive_for(cid, payload.get("config"), with_backend=False)
        with st.lock_for(cid):
            result = hive.process_turn(query, conversation_id=cid)
            st.save_conversation(cid, hive)
        curated = result.assembled.content if result.assembled is not None else ""
        merged_sys = curated or "You are a helpful assistant."
        if messages and messages[0].get("role") == "system" \
                and messages[0].get("content"):
            merged_sys = merged_sys + "\n\n" + messages[0]["content"]
        stream = bool(payload.get("stream"))
        upstream = {
            **payload,
            "model": provider.model or payload.get("model") or "local",
            "stream": stream,
            "messages": [{"role": "system", "content": merged_sys}] + messages[1:],
        }
        upstream.setdefault("stream_options", {"include_usage": True})

        def observe(reply: str) -> bool:
            stored = False
            if reply.strip() and not (
                hive.config.filter_hedge_replies
                and Hive._is_hedge_reply(reply)
            ):
                hive.store.add_chunk(hive.turn, reply)
                st.save_conversation(cid, hive)
                stored = True
            return stored

        if not stream:
            resp = _upstream_stream(
                f"{base_url}/v1/chat/completions", json=upstream,
                headers=headers, timeout=600,
            )
            resp.raise_for_status()
            data = resp.json()
            try:
                observe(data["choices"][0]["message"]["content"] or "")
            except (KeyError, IndexError):
                pass
            return data

        def sse():
            parts: list[str] = []
            try:
                resp = _upstream_stream(
                    f"{base_url}/v1/chat/completions", json=upstream,
                    headers=headers, stream=True, timeout=600,
                )
                resp.raise_for_status()
                for raw in resp.iter_lines(decode_unicode=True):
                    if not raw:
                        continue
                    line = raw[6:].strip() if raw.startswith("data:") else raw.strip()
                    if not line:
                        continue
                    if line == "[DONE]":
                        yield "data: [DONE]\n\n"
                        break
                    try:
                        chunk = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    for choice in chunk.get("choices") or []:
                        delta = choice.get("delta") or {}
                        if delta.get("content"):
                            parts.append(delta["content"])
                    yield "data: " + json.dumps(chunk) + "\n\n"
            except Exception as exc:  # noqa: BLE001 - surfaced as an SSE error event
                yield "data: " + json.dumps({
                    "error": {"message": str(exc), "type": "hive_upstream_error"},
                }) + "\n\n"
            observe("".join(parts))

        return StreamingResponse(sse(), media_type="text/event-stream")
        reg = ProviderRegistry(default=req.default)
        for entry in req.providers:
            data = entry.model_dump()
            if data.get("api_key") == MASK:
                # the UI echoes the mask back for untouched keys â€” keep the
                # stored secret instead of overwriting it with "***"
                previous = [p for p in st.registry.providers
                            if p.name.lower() == str(data.get("name", "")).lower()]
                data["api_key"] = previous[0].api_key if previous else ""
            try:
                reg.providers.append(Provider.from_dict(data))
            except ValueError as exc:
                raise HTTPException(422, str(exc))
        st.registry = reg
        persisted = None
        if req.persist:
            path = save_registry(reg, st.providers_file)
            persisted = str(path)
        return {"ok": True, "default": reg.default,
                "providers": reg.redacted(), "persisted_to": persisted}

    @app.post("/v1/provider/config")
    def set_providers(req: ProviderConfigRequest):
        reg = ProviderRegistry(default=req.default)
        for entry in req.providers:
            data = entry.model_dump()
            if data.get("api_key") == MASK:
                # the UI echoes the mask back for untouched keys â€” keep the
                # stored secret instead of overwriting it with "***"
                previous = [p for p in st.registry.providers
                            if p.name.lower() == str(data.get("name", "")).lower()]
                data["api_key"] = previous[0].api_key if previous else ""
            try:
                reg.providers.append(Provider.from_dict(data))
            except ValueError as exc:
                raise HTTPException(422, str(exc))
        st.registry = reg
        persisted = None
        if req.persist:
            path = save_registry(reg, st.providers_file)
            persisted = str(path)
        return {"ok": True, "default": reg.default,
                "providers": reg.redacted(), "persisted_to": persisted}

    @app.get("/v1/provider/config")
    def get_providers():
        return {
            "default": st.registry.default,
            "providers": st.registry.redacted(),
            "file": str(providers_path(st.providers_file)),
        }

    # ------------------------------------------------------------------
    @app.post("/v1/engines")
    def set_engines(req: EngineConfigRequest):
        reg = EngineRegistry(default=req.default)
        for entry in req.engines:
            try:
                reg.engines.append(EngineProfile.from_dict(entry.model_dump()))
            except ValueError as exc:
                raise HTTPException(422, str(exc))
        st.engines = reg
        persisted = None
        if req.persist:
            path = save_engines(reg, st.engines_file)
            persisted = str(path)
        return {"ok": True, "default": reg.default,
                "engines": [e.to_dict() for e in reg.engines],
                "persisted_to": persisted}

    @app.get("/v1/engines")
    def get_engines():
        return {
            "default": st.engines.default,
            "engines": [e.to_dict() for e in st.engines.engines],
            "file": str(engines_path(st.engines_file)),
        }

    @app.get("/v1/engines/preset")
    def engines_preset(file: str = Query(...)):
        """Auto preset for the Engine profiles section.

        Combines GET /v1/server/status hardware + GET /v1/models/local
        size_gb + gguf-metadata (parsed from the GGUF header) into a
        ready-to-apply load_options preset. The Studio's Auto button calls
        this and then saves the profile. Examples: qwen3-4b on 8GB →
        gpu_layers 28 + 8k ctx, qwen3-32b → 12 + 4k.
        """
        # hardware: available_gb prefers VRAM when present, else RAM
        hw = _hardware_summary()
        hardware_gb = float(hw.get("available_gb") or hw.get("vram_gb") or 8.0)
        # model size + gguf metadata
        size_gb = 0.0
        gguf_meta: dict = {}
        try:
            models = models_manager.list_local()
            entry = next((m for m in models if m.get("file") == file), None)
            if entry is not None:
                size_gb = float(entry.get("size_gb") or 0)
            # try GGUF header even if not in list (e.g. absolute path)
            cand = models_manager.resolve_model(file) if file else None
            if cand and cand.is_file():
                gguf_meta = _read_gguf_metadata(cand)
        except Exception:
            pass
        # fallback: infer from file name when model not yet local
        if size_gb == 0:
            low = file.lower()
            if "32b" in low:
                size_gb = 18.0
            elif "14b" in low or "13b" in low:
                size_gb = 8.0
            elif "7b" in low or "8b" in low:
                size_gb = 4.5
            elif "4b" in low:
                size_gb = 2.5
        preset = _auto_preset_load_options(size_gb, hardware_gb, file, gguf_meta)
        return {
            "file": file,
            "hardware": hw,
            "model": {"file": file, "size_gb": size_gb, "gguf_metadata": gguf_meta},
            "preset": preset,
            "load_options": preset,
        }

    # ------------------------------------------------------------------
    # Model management (M4): own llama.cpp server + live Hugging Face hub.
    if models_manager is None:
        models_manager = LlamaServerManager(log_dir=Path(log_dir),
                                            port=llama_port)
    app.state.models = models_manager

    def register_local_remove(key: str) -> None:
        """Retire a local instance's provider + engine profile."""
        prov_name = f"local-{key}"
        st.registry.providers = [
            p for p in st.registry.providers
            if p.name.lower() != prov_name.lower()
        ]
        st.engines.engines = [
            e for e in st.engines.engines if e.name.lower() != prov_name.lower()
        ]

    def register_local(info: dict, load_options: Optional[dict] = None,
                       api_key: Optional[str] = None,
                       key: Optional[str] = None,
                       claim_default: bool = True) -> str:
        """Register one manager-loaded llama-server as provider
        ``local-<key>`` + engine profile with the real launch options. Shared
        by the start endpoint and CLI auto-start. Returns the provider name.
        Multi-model: every loaded instance gets its own provider; the first
        (or an explicitly re-loaded default) claims the default slot."""
        if not info.get("running"):
            return ""
        inst_key = key or "local"
        prov_name = f"local-{inst_key}"
        base_url = f"http://{models_manager.host}:{info['port']}"
        prov = Provider(name=prov_name, base_url=base_url,
                        api_key=api_key or "lm-studio",
                        model=str(info.get("model") or ""))
        st.registry.providers = [
            p for p in st.registry.providers if p.name.lower() != prov_name.lower()
        ] + [prov]
        if claim_default:
            # a managed launch IS the studio's brain: claim the default
            # (the user explicitly loaded this model; /provider switches back)
            st.registry.default = prov_name
        try:
            save_registry(st.registry, st.providers_file)
        except OSError as exc:
            print(f"harness: could not persist providers config ({exc})",
                  file=sys.stderr)
        profile = EngineProfile(
            name=prov_name, kind="llama_cpp", base_url=base_url,
            load_options=load_options
            or {"context": 8192, "gpu_layers": 999},
            capabilities=["streaming", "prefix_caching"],
        )
        st.engines.engines = [
            e for e in st.engines.engines if e.name.lower() != prov_name.lower()
        ] + [profile]
        if not st.engines.default or st.engines.default.startswith("local"):
            st.engines.default = prov_name
        # persist so CLI auto-start (and /model) reuse these launch settings
        try:
            save_engines(st.engines, st.engines_file)
        except OSError as exc:
            print(f"harness: could not persist engines config ({exc})",
                  file=sys.stderr)
        return prov_name

    app.state.register_local = register_local

    # ------------------------------------------------------------------
    # A/B compare — bench two Engine profiles side-by-side:
    #   start A on basePort, B on basePort+1, then POST 3 prompts to
    #   each POST /v1/chat/completions and compare tok/s to determine winner.
    #   Frontend grid cell uses this via the Bench button and winner badge.
    # ------------------------------------------------------------------
    _AB_DEFAULT_PROMPTS = [
        "Hello, how are you?",
        "Write a short story about a cat.",
        "Explain quantum physics briefly.",
    ]

    def _ab_ensure_started(profile: EngineProfile, port: int) -> str:
        """Ensure the profile's model is serving on ``port``; return base_url.

        Best-effort: if a server is already on that port, reuse it; if
        ``load()`` fails because the model is not found or already loaded,
        still return the constructed URL so the bench can proceed (tests mock
        the upstream POST). The helper prefers a local GGUF matching the
        profile name when available.
        """
        base_url = f"http://{getattr(models_manager, 'host', '127.0.0.1')}:{port}"
        try:
            # already serving on this port?
            try:
                stt = models_manager.status()
                for inst in stt.get("instances", []) or []:
                    if int(inst.get("port", -1)) == int(port):
                        return base_url
            except Exception:
                pass
            # pick a model file for the load
            model_file = None
            try:
                local = models_manager.list_local()
                if local:
                    for m in local:
                        f = str(m.get("file") or "")
                        n = str(m.get("name") or "")
                        if profile.name.lower() in f.lower() or profile.name.lower() in n.lower():
                            model_file = f
                            break
                    if not model_file:
                        model_file = local[0].get("file")
            except Exception:
                pass
            opts = profile.load_options if isinstance(profile.load_options, dict) else {}
            try:
                ctx_sz = int(opts.get("context") or opts.get("ctx_size") or opts.get("contextLength") or 8192)
            except Exception:
                ctx_sz = 8192
            try:
                ngl = int(opts.get("gpu_layers") or opts.get("ngl") or 999)
            except Exception:
                ngl = 999
            # attempt load only if not already serving
            if not any(int(inst.get("port", -1)) == int(port) for inst in (models_manager.status().get("instances", []) or []) if isinstance(inst, dict)):
                try:
                    models_manager.load(model=model_file, port=port, ctx_size=ctx_sz, ngl=ngl, extra_args=[])
                except RuntimeError as exc:
                    msg = str(exc).lower()
                    if "already loaded" in msg or "already serving" in msg or "port" in msg or "neither a local" in msg or "not found" in msg:
                        pass
                    else:
                        raise
        except Exception:
            pass
        return base_url

    def _ab_bench_one(base_url: str, prompts: list[str]) -> dict:
        """POST 3 prompts to ``base_url/v1/chat/completions`` and measure tok/s."""
        total_tokens = 0
        total_time = 0.0
        results: list[dict] = []
        for prompt in prompts[:3]:
            payload = {
                "model": "bench",
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 64,
            }
            start = time.time()
            resp = requests.post(f"{base_url.rstrip('/')}/v1/chat/completions", json=payload, timeout=30)
            resp.raise_for_status()
            data = resp.json() if hasattr(resp, "json") else json.loads(resp.text)
            usage = data.get("usage") if isinstance(data, dict) else {}
            comp = None
            if isinstance(usage, dict):
                comp = usage.get("completion_tokens")
                if comp is None:
                    comp = usage.get("completionTokens")
            if comp is None:
                try:
                    content = data["choices"][0]["message"]["content"] or ""
                    comp = len(str(content).split())
                except Exception:
                    comp = 0
            try:
                comp = int(comp)
            except Exception:
                comp = 0
            elapsed = max(time.time() - start, 1e-6)
            tps = comp / elapsed if elapsed else 0
            total_tokens += comp
            total_time += elapsed
            results.append({"prompt": prompt, "tokens": comp, "seconds": round(elapsed, 4), "tok_per_sec": round(tps, 2)})
        avg = (total_tokens / total_time) if total_time else 0
        return {"tok_per_sec": avg, "total_tokens": total_tokens, "total_seconds": round(total_time, 4), "results": results}

    async def _ab_handle(request: Request):
        body = {}
        try:
            body = await request.json()
        except Exception:
            body = {}
        # flexible profile keys
        profile_a = body.get("profile_a") or body.get("profileA") or body.get("a") or body.get("A") or body.get("profile_a_name")
        profile_b = body.get("profile_b") or body.get("profileB") or body.get("b") or body.get("B") or body.get("profile_b_name")
        base_port_raw = body.get("basePort") if body.get("basePort") is not None else body.get("base_port") if body.get("base_port") is not None else body.get("baseport")
        prompts = body.get("prompts") or body.get("prompt_list") or _AB_DEFAULT_PROMPTS
        if isinstance(prompts, str):
            prompts = [prompts]
        prompts = [str(p) for p in (prompts or _AB_DEFAULT_PROMPTS)][:3]
        while len(prompts) < 3:
            prompts += _AB_DEFAULT_PROMPTS[len(prompts):3]
        if not profile_a or not profile_b:
            raise HTTPException(422, "profile_a and profile_b are required")
        try:
            eng_a = st.engines.resolve(profile_a)
        except LookupError as exc:
            raise HTTPException(404, str(exc))
        try:
            eng_b = st.engines.resolve(profile_b)
        except LookupError as exc:
            raise HTTPException(404, str(exc))
        try:
            bp = int(base_port_raw) if base_port_raw is not None else int(getattr(models_manager, "port", 1234))
        except Exception:
            bp = 1234
        if bp < 1024 or bp > 65534:
            raise HTTPException(422, "basePort out of range (1024-65534)")
        port_a = bp
        port_b = bp + 1
        # start both on basePort and basePort+1
        url_a = _ab_ensure_started(eng_a, port_a)
        url_b = _ab_ensure_started(eng_b, port_b)
        # bench each — 3 POST /v1/chat/completions, measure tok/s
        try:
            res_a = _ab_bench_one(url_a, prompts)
        except Exception as exc:
            raise HTTPException(502, f"A bench failed on {url_a}: {exc}")
        try:
            res_b = _ab_bench_one(url_b, prompts)
        except Exception as exc:
            raise HTTPException(502, f"B bench failed on {url_b}: {exc}")
        a_tps = float(res_a.get("tok_per_sec") or 0)
        b_tps = float(res_b.get("tok_per_sec") or 0)
        if a_tps > b_tps:
            winner = "A"
        elif b_tps > a_tps:
            winner = "B"
        else:
            winner = "tie"
        # small difference within 2% counts as tie (noise)
        if winner != "tie":
            mx = max(a_tps, b_tps)
            if mx and abs(a_tps - b_tps) / mx < 0.02:
                winner = "tie"
        return {
            "winner": winner,
            "a_tok_per_sec": round(a_tps, 2),
            "b_tok_per_sec": round(b_tps, 2),
            "a": {"profile": eng_a.name, "port": port_a, "base_url": url_a, "tok_per_sec": round(a_tps, 2), **res_a},
            "b": {"profile": eng_b.name, "port": port_b, "base_url": url_b, "tok_per_sec": round(b_tps, 2), **res_b},
            "basePort": port_a,
            "base_port": port_a,
            "port_a": port_a,
            "port_b": port_b,
            "prompts": prompts,
            # aliases for frontend fallbacks
            "tokens_per_sec_a": round(a_tps, 2),
            "tokens_per_sec_b": round(b_tps, 2),
            "profile_a": eng_a.name,
            "profile_b": eng_b.name,
        }

    @app.post("/v1/engines/ab/bench")
    async def engines_ab_bench(request: Request):
        return await _ab_handle(request)

    @app.post("/v1/engines/bench")
    async def engines_bench_alias(request: Request):
        return await _ab_handle(request)

    @app.post("/v1/ab/bench")
    async def ab_bench_alias(request: Request):
        return await _ab_handle(request)

    agent_service = DshAgentService(
        default_cwd=REPO_ROOT,
        session_root=REPO_ROOT / "harness_state" / "dsh_sessions",
    )
    app.state.agent = agent_service

    commands = ConsoleCommands(
        st=st,
        models=models_manager,
        agent=agent_service,
        transcripts_dir=REPO_ROOT / "transcripts",
    )
    commands._app = app  # register_local lives on the app instance
    app.state.commands = commands

    @app.get("/v1/commands")
    def list_commands():
        return {"commands": commands.descriptors()}

    @app.post("/v1/commands/run")
    def run_command(req: CommandRunRequest):
        result = commands.run(req.line, req.conversation_id)
        return result.to_dict()

    @app.post("/v1/agent/stream")
    async def agent_stream(req: AgentMessageRequest):
        message = (req.message or "").strip()
        if not message:
            raise HTTPException(422, "message must not be empty")
        try:
            provider = st.registry.resolve(None)
        except LookupError as exc:
            raise HTTPException(502, str(exc))
        base_url = provider.base_url.rstrip("/") + "/v1"
        q: "queue.Queue[dict]" = queue.Queue()

        def worker():
            try:
                out = agent_service.run_turn(
                    req.conversation_id, message,
                    base_url=base_url,
                    api_key=provider.api_key or "lm-studio",
                    model=provider.model or "local",
                    on_event=q.put,
                )
                q.put({"type": "done", **out})
            except Exception as exc:  # noqa: BLE001 - surfaced as an event
                q.put({"type": "error", "error": str(exc)})

        threading.Thread(target=worker, daemon=True).start()

        def sse():
            while True:
                event = q.get()
                yield "data: " + json.dumps(event, default=str) + "\n\n"
                if event.get("type") in ("done", "error"):
                    return

        return StreamingResponse(sse(), media_type="text/event-stream")

    @app.get("/v1/agent/status")
    def agent_status():
        return {"runtime_running": agent_service.runtime_running,
                "permission_policy": agent_service.permission_policy,
                "busy": agent_service._inflight}

    @app.post("/v1/agent/cancel")
    def agent_cancel():
        return agent_service.cancel()

    # ------------------------------------------------------------------
    # Preset trainer (X15-X18)
    @app.get("/v1/trainer/evidence")
    def trainer_evidence():
        """Mine session logs for tool-use patterns and outcomes."""
        session_root = REPO_ROOT / "harness_state" / "dsh_sessions"
        all_tools = ["bash", "str_replace_editor", "fs_search", "web",
                     "subagent", "todo", "code_runtime", "skill"]
        evidences = mine_evidence(session_root, all_tools)
        return {"evidence": [e.to_dict() for e in evidences],
                "summary": summarize_evidence(evidences)}

    @app.post("/v1/trainer/draft")
    def trainer_draft(body: dict):
        """Draft a candidate preset from the evidence."""
        name = body.get("name") or f"candidate-{int(time.time())}"
        session_root = REPO_ROOT / "harness_state" / "dsh_sessions"
        all_tools = body.get("all_tools") or [
            "bash", "str_replace_editor", "fs_search", "web",
            "subagent", "todo", "code_runtime", "skill"]
        evidences = mine_evidence(session_root, all_tools)
        summary = summarize_evidence(evidences)
        baseline = Path(body.get("baseline") or "")
        if not baseline.is_file():
            raise HTTPException(422, f"baseline preset not found: {baseline}")
        output = REPO_ROOT / "harness_state" / "trainer_candidates"
        candidate = draft_candidate(summary, baseline, output, name)
        return candidate.to_dict()

    @app.post("/v1/trainer/evaluate")
    def trainer_evaluate(body: dict):
        """Run candidate vs baseline on bench tasks, score, compare."""
        candidate_path = Path(body.get("candidate") or "")
        meta_path = candidate_path.with_suffix("").with_suffix(".trainer-meta.json") \
            if candidate_path.suffix == ".yml" else \
            candidate_path.parent / (candidate_path.stem + ".trainer-meta.json")
        if not meta_path.is_file():
            raise HTTPException(404, f"trainer meta not found: {meta_path}")
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        from deepseek_harness import DeepSeekHarness

        def factory(config_path):
            return DeepSeekHarness(
                provider="deepseek-official",
                base_url=os.environ.get("HARNESS_EMBEDDING_URL",
                                        "http://127.0.0.1:1235/v1"),
                api_key="lm-studio",
                model=os.environ.get("HARNESS_AGENT_MODEL", "default"),
                cwd=str(REPO_ROOT),
                session_root=str(REPO_ROOT / "harness_state" / "dsh_sessions"),
                cordis=str(config_path),
            )

        tasks = body.get("tasks") or [
            "List the files in the current directory.",
            "Create a file named trainer-test.txt containing 'hello'.",
            "Search for the word 'hive' in .py files and report matches.",
        ]
        candidate = draft_candidate.__wrapped__ if hasattr(
            draft_candidate, "__wrapped__") else None
        from harness.trainer import CandidatePreset

        cand = CandidatePreset(
            name=candidate_path.stem, baseline=meta.get("baseline", ""),
            changes=meta.get("changes", {}),
            evidence_summary=meta.get("evidence_summary", {}),
            path=str(candidate_path),
        )
        results = evaluate_candidate(cand, factory, tasks)
        meta["eval_result"] = results
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        return results

    @app.get("/v1/server/status")
    def server_status():
        data = models_manager.status()
        try:
            data["hardware"] = _hardware_summary()
        except Exception:
            data["hardware"] = {
                "available_gb": 8.0, "total_ram_gb": 8.0,
                "vram_gb": None, "available_ram_gb": 8.0,
                "devices": [], "vram_source": "ram",
            }
        return data

    @app.get("/v1/server/log")
    def server_log(tail: int = 120):
        return {"lines": models_manager.server_log(tail)}

    @app.get("/v1/server/memory")
    def server_memory():
        """Sidecar RSS + conversation accounting — the leak-detection probe."""
        import psutil

        proc = psutil.Process()
        rss_mb = proc.memory_info().rss / (1024 * 1024)
        return {
            "rss_mb": round(rss_mb, 1),
            "conversations_in_memory": len(st.hives),
            "max_conversations": st.max_conversations,
            "loggers": len(st._loggers),
            "downloads": len(models_manager._downloads),
            "threads": threading.active_count(),
        }

    @app.get("/v1/server/metrics")
    def server_metrics():
        return models_manager.server_metrics()

    @app.delete("/v1/models/local")
    def delete_local_model(file: str = Query(...)):
        try:
            return models_manager.delete_local(file)
        except (ValueError, FileNotFoundError) as exc:
            raise HTTPException(400, str(exc))


    @app.get("/v1/research/queue")
    def research_queue_get():
        """Pending deep-research questions. Execution is QUEEN-only: entries are
        picked up when the primary session next wakes."""
        if not RESEARCH_QUEUE.exists():
            return {"items": []}
        items = []
        for line in RESEARCH_QUEUE.read_text(encoding="utf-8-sig").splitlines():
            s = line.strip()
            if s.startswith("- [ ] "):
                items.append(s[6:])
        return {"items": items}

    @app.post("/v1/research/queue")
    async def research_queue_add(req: Request):
        body = await req.json()
        q = str(body.get("question", "")).strip()
        if not q:
            raise HTTPException(422, "question is required")
        q = q[:500]
        with RESEARCH_QUEUE.open("a", encoding="utf-8") as fh:
            fh.write(f"- [ ] {q}\n")
        return {"queued": True, "question": q}


    @app.get("/v1/hive/mode")
    def hive_mode_get():
        if MODE_FILE.exists():
            try:
                data = json.loads(MODE_FILE.read_text(encoding="utf-8-sig"))
                return {"afk": True, **data, "_file": str(MODE_FILE)}
            except Exception as exc:
                return {"afk": False, "error": f"unreadable mode file: {exc}",
                        "_file": str(MODE_FILE)}
        return {"afk": False, "_file": str(MODE_FILE)}

    @app.post("/v1/hive/mode")
    async def hive_mode_set(req: Request):
        body = await req.json()
        afk = bool(body.get("afk"))
        note = str(body.get("note", ""))[:200]
        if afk:
            payload = {"mode": "AFK",
                       "since": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                       "operator": "away", "note": note,
                       "preapproved": ["GREEN/YELLOW fixes",
                                       "catalog+doc regeneration",
                                       "executing HIVE-PLAN orders",
                                       "approved-proposal implementation",
                                       "gate bug fixes"],
                       "queue_for_return": ["pushes to public masters",
                                            "PR merges",
                                            "policy/protocol changes",
                                            "RED defects beyond containment"]}
            MODE_FILE.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        elif MODE_FILE.exists():
            MODE_FILE.unlink()
        return {"afk": afk, "note": note}

    @app.post("/v1/server/stop")
    def server_stop(key: Optional[str] = Query(default=None)):
        """Unload one instance (key) or the whole local fleet (no key)."""
        if key:
            try:
                return models_manager.unload(key)
            except RuntimeError as exc:
                raise HTTPException(404, str(exc))
        for inst in list(models_manager.status()["instances"]):
            register_local_remove(inst["key"])
        return models_manager.stop_all()

    @app.post("/v1/server/start")
    def server_start(req: ServerStartRequest):
        try:
            info = models_manager.load(
                model=req.model, hf_repo=req.hf_repo, hf_file=req.hf_file,
                key=req.key, port=req.port, ctx_size=req.ctx_size,
                ngl=req.ngl, extra_args=req.extra_args(),
                backend=req.backend,
            )
        except RuntimeError as exc:
            message = str(exc)
            code = 502
            if "already loaded" in message or "already serving" in message:
                code = 409
            elif "not found at" in message or "neither a local file" in message:
                code = 400
            raise HTTPException(code, message)
        prov_name = ""
        if req.register_provider:
            prov_name = register_local(
                info, load_options=req.load_options(), api_key=req.api_key,
                key=info["key"], claim_default=bool(req.claim_default),
            )
            save_registry(st.registry, st.providers_file)
        return {**info, "provider": prov_name,
                "provider_registered": bool(req.register_provider)}

    @app.post("/v1/server/unload")
    def server_unload(req: ServerUnloadRequest):
        """Unload one instance and retire its provider + engine profile."""
        try:
            result = models_manager.unload(req.key)
        except RuntimeError as exc:
            raise HTTPException(404, str(exc))
        prov_name = f"local-{req.key}"
        st.registry.providers = [
            p for p in st.registry.providers
            if p.name.lower() != prov_name.lower()
        ]
        if st.registry.default == prov_name:
            st.registry.default = next(
                (p.name for p in st.registry.providers), "")
        st.engines.engines = [
            e for e in st.engines.engines if e.name.lower() != prov_name.lower()
        ]
        if st.engines.default == prov_name:
            st.engines.default = next(
                (e.name for e in st.engines.engines), "")
        try:
            save_registry(st.registry, st.providers_file)
            save_engines(st.engines, st.engines_file)
        except OSError:
            pass
        return {**result, "provider": prov_name}

    @app.get("/v1/models/local")
    def local_models():
        return {"models_dir": str(models_manager.models_dir),
                "models": models_manager.list_local()}

    @app.post("/v1/models/local/import")
    async def import_local_models(files: List[UploadFile] = File(...)):
        """Import GGUF files from a local folder (webkitdirectory upload).

        Accepts multiple .gguf files via multipart/form-data; preserves
        webkitRelativePath subfolders when present, path-traversal safe.
        """
        imported: List[str] = []
        errors: List[str] = []
        root = models_manager.models_dir.resolve()
        for f in files:
            filename = (f.filename or "").strip()
            if not filename:
                continue
            if not filename.lower().endswith(".gguf"):
                errors.append(f"{filename}: not a .gguf file")
                continue
            # webkitRelativePath may be like "my-models/sub/model.gguf"
            p = Path(filename)
            if p.is_absolute() or ".." in p.parts:
                errors.append(f"{filename}: invalid path")
                continue
            # Resolve destination: try preserving relative path, fallback to basename
            dest = (models_manager.models_dir / p).resolve()
            if root not in dest.parents and dest != root:
                dest = (models_manager.models_dir / p.name).resolve()
                if root not in dest.parents and dest != root:
                    errors.append(f"{filename}: path traversal detected")
                    continue
            try:
                dest.parent.mkdir(parents=True, exist_ok=True)
                with dest.open("wb") as out:
                    while True:
                        chunk = await f.read(1024 * 1024)
                        if not chunk:
                            break
                        out.write(chunk)
                try:
                    rel = str(dest.relative_to(root))
                except ValueError:
                    rel = dest.name
                imported.append(rel)
            except Exception as exc:  # noqa: BLE001
                errors.append(f"{filename}: {exc}")
        return {"imported": imported, "errors": errors, "models_dir": str(models_manager.models_dir)}

    @app.post("/v1/models/local/import-path")
    async def import_local_path(request: Request):
        """Link an external folder without copying (instant)."""
        try:
            body = await request.json()
        except Exception:
            raise HTTPException(422, "invalid JSON")
        folder = str(body.get("folder") or "").strip()
        if not folder:
            raise HTTPException(422, "folder is required")
        try:
            return models_manager.import_local_path(folder)
        except FileNotFoundError as exc:
            raise HTTPException(404, str(exc))
        except (ValueError, RuntimeError) as exc:
            raise HTTPException(400, str(exc))

    @app.post("/v1/models/local/path")
    async def set_local_path(request: Request):
        """Point the library at a chosen folder (no copy, instant).

        Default is LM Studio's ``.lmstudio/models`` when it exists, else
        ``models/gguf`` (created for Hugging Face downloads).
        """
        try:
            body = await request.json()
        except Exception:
            raise HTTPException(422, "invalid JSON")
        folder = str(body.get("folder") or body.get("path") or "").strip()
        if not folder:
            raise HTTPException(422, "folder is required")
        try:
            return models_manager.set_models_dir(folder)
        except FileNotFoundError as exc:
            raise HTTPException(404, str(exc))
        except (NotADirectoryError, PermissionError, ValueError, RuntimeError, OSError) as exc:
            raise HTTPException(400, str(exc))

    @app.get("/v1/models/hub")
    def hub_search(q: str = "", limit: int = 12):
        try:
            return {"results": models_manager.hub_search(q, limit)}
        except Exception as exc:  # noqa: BLE001 - network errors surface as 502
            raise HTTPException(502, f"hugging face search failed: {exc}")

    @app.get("/v1/models/hub/files/{repo:path}")
    def hub_files(repo: str):
        if not repo.strip():
            raise HTTPException(422, "repo must not be empty")
        try:
            return {"repo": repo, "files": models_manager.hub_files(repo)}
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(502, f"cannot list '{repo}': {exc}")

    @app.post("/v1/models/hub/download")
    def hub_download(req: HubDownloadRequest):
        return models_manager.download(req.repo, req.file)

    @app.get("/v1/models/hub/downloads")
    def hub_downloads():
        return {"downloads": models_manager.downloads_status()}

    @app.get("/server", response_class=HTMLResponse)
    def server_page():
        return HTMLResponse(render_server_page(), headers=_NO_STORE)

    # ------------------------------------------------------------------
    @app.post("/v1/protocol/run")
    def protocol_run(req: ProtocolRunRequest):
        mode = req.mode if req.mode in ("live", "mock") else ""
        if not mode:
            raise HTTPException(422, "mode must be 'live' or 'mock'")
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = st.runs_root / f"protocol_{stamp}"
        cmd = [sys.executable, "-m", "experiments.generate_data",
               f"--{mode}", "--output", str(run_dir)]
        a = req.args or {}
        for key, flag in PROTOCOL_FLAGS_INT.items():
            if key in a:
                cmd += [flag, str(int(a[key]))]
        for key, flag in PROTOCOL_FLAGS_STR.items():
            if key in a and a[key]:
                cmd += [flag, str(a[key])]
        for key, flag in PROTOCOL_FLAGS_BOOL.items():
            if a.get(key):
                cmd.append(flag)
        run_dir.mkdir(parents=True, exist_ok=True)
        proc = _popen(
            cmd, cwd=str(REPO_ROOT),
            stdout=open(run_dir / "run_stdout.log", "ab"),
            stderr=subprocess.STDOUT,
        )
        return {"run_dir": str(run_dir), "pid": proc.pid}

    @app.get("/v1/report/{run_dir:path}")
    def report(run_dir: str):
        target = resolve_run_dir(st.runs_root, run_dir)
        path = target / "run_report.json"
        if not path.is_file():
            raise HTTPException(404, f"no run_report.json under {target}")
        return json.loads(path.read_text(encoding="utf-8"))

    @app.get("/v1/runs")
    def runs_index():
        return {"runs": _list_runs(st.runs_root)}

    # ------------------------------------------------------------------
    # Report views (Seam B): server-rendered HTML over run bundles.
    @app.get("/view/{run_dir:path}", response_class=HTMLResponse)
    def view_report(run_dir: str):
        target = resolve_run_dir(st.runs_root, run_dir)
        path = target / "run_report.json"
        if not path.is_file():
            raise HTTPException(404, f"no run_report.json under {target}")
        report = json.loads(path.read_text(encoding="utf-8"))
        return HTMLResponse(render_report_page(report, target.name), headers=_NO_STORE)

    @app.get("/runs", response_class=HTMLResponse)
    def view_runs():
        return HTMLResponse(render_runs_page(_list_runs(st.runs_root)), headers=_NO_STORE)

    @app.get("/", response_class=HTMLResponse)
    def index():
        return RedirectResponse("/runs")

    return app
