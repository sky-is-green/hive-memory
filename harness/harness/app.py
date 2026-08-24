"""The harness sidecar FastAPI application.

State model
-----------
- One ``Hive`` instance per conversation_id (fresh store + comb per
  conversation — per-conversation isolation is mandatory, HIVE-HANDOFF §6.0 #14).
  Instances are created lazily on the first turn and dropped by /v1/hive/reset.
- Conversations persist to ``state_dir`` (default ./harness_state, one atomic
  JSON per conversation using the same store serialization as the benchmark's
  checkpoint/resume) and reload lazily on first touch after a restart, so the
  hive survives sidecar restarts. /v1/hive/reset deletes memory AND disk.
- One shared ultra-small drone across conversations (a per-conversation encoder
  would multiply VRAM/RAM for nothing); inference is read-only.
- Per-conversation locks serialize turns within a conversation; different
  conversations may proceed in parallel. Generation calls are blocking
  (streaming is a v2 concern) — sync endpoints run in FastAPI's threadpool.
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
from typing import Callable, Optional

import requests
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, RedirectResponse, StreamingResponse
from pydantic import BaseModel
import queue

from harness.agent import DshAgentService
from harness.commands import ConsoleCommands

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
        so the hive survives sidecar restarts.

        ``with_backend=False`` (the curate/observe flow, where the caller's
        own shell generates) creates the hive without an LLM backend; a
        conversation is driven either fully (/v1/hive/turn) or externally
        (curate + observe), whichever touches it first wins.
        """
        with self.global_lock:
            hive = self.hives.get(conversation_id)
            if hive is not None:
                return hive

            def build(cfg: HiveConfig, backend: object | None) -> Hive:
                h = Hive(
                    config=cfg,
                    ultra=self.ultra(),
                    backend=backend,
                    logger=EventLogger(log_dir=self.log_dir),
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
            return build(config, self.backend_factory(None) if with_backend else None)

    def drop(self, conversation_id: str) -> None:
        with self.global_lock:
            self.hives.pop(conversation_id, None)
            self.locks.pop(conversation_id, None)
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
    port: Optional[int] = None
    ctx_size: int = 8192
    ngl: int = 999  # GPU layers (Vulkan build: all layers on the RX 7900 XT)
    register_provider: bool = True
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
    """Console origins. Default: localhost dev origins only — agent mode can
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

    def _default_ultra():
        return UltraSmallDrone(confidence_mode="off")

    def _default_backend(model: Optional[str]):
        kw = backend_kwargs(st.registry.resolve(None))
        if model:
            kw["model"] = model
        return OpenAICompatBackend(**kw)

    app = FastAPI(title="HiveBench Studio harness", version="0.1.0")
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
        if req.model and isinstance(hive.backend, OpenAICompatBackend) \
                and req.model != hive.backend.model:
            new_backend = st.backend_factory(req.model)
            hive.backend = new_backend
            hive.cache = KVCacheManager(new_backend)
        with st.lock_for(req.conversation_id):
            result = hive.process_turn(req.query, conversation_id=req.conversation_id)
            st.save_conversation(req.conversation_id, hive)
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
        }

    @app.post("/v1/hive/reset")
    def hive_reset(req: ResetRequest):
        st.drop(req.conversation_id)
        return {"ok": True}

    # ------------------------------------------------------------------
    # Curate / observe (Seam A, dsh-hive flow): the caller's own shell
    # generates — the sidecar only assembles context and ingests replies.
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
        with st.global_lock:
            hive = st.hives.get(req.conversation_id)
        if hive is None:
            raise HTTPException(404, f"no such conversation: {req.conversation_id}")
        reply = (req.reply or "").strip()
        stored = False
        if reply and not (
            hive.config.filter_hedge_replies and Hive._is_hedge_reply(reply)
        ):
            with st.lock_for(req.conversation_id):
                hive.store.add_chunk(hive.turn, reply)
                st.save_conversation(req.conversation_id, hive)
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
        with st.lock_for(req.conversation_id):
            result = hive.process_turn(query, conversation_id=req.conversation_id)
            st.save_conversation(req.conversation_id, hive)
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
        """HiveConfig defaults — the source for the UI tuning form. Overrides
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
    # what the request actually contained — context size and whether hive
    # content reached the model — which makes it a live probe of Seam A.
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
                or "p1-p11" in user_text or "p1–p11" in user_text:
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
    @app.post("/v1/provider/config")
    def set_providers(req: ProviderConfigRequest):
        reg = ProviderRegistry(default=req.default)
        for entry in req.providers:
            data = entry.model_dump()
            if data.get("api_key") == MASK:
                # the UI echoes the mask back for untouched keys — keep the
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

    # ------------------------------------------------------------------
    # Model management (M4): own llama.cpp server + live Hugging Face hub.
    if models_manager is None:
        models_manager = LlamaServerManager(log_dir=Path(log_dir),
                                            port=llama_port)
    app.state.models = models_manager

    def register_local(info: dict, load_options: Optional[dict] = None,
                       api_key: Optional[str] = None) -> None:
        """Point the app at a manager-started llama-server: provider `local`
        + engine profile with the real launch options. Shared by the start
        endpoint and CLI auto-start so both paths route identically."""
        if not info.get("running"):
            return
        base_url = f"http://{models_manager.host}:{info['port']}"
        prov = Provider(name="local", base_url=base_url,
                        api_key=api_key or "lm-studio",
                        model=str(info.get("model") or ""))
        st.registry.providers = [
            p for p in st.registry.providers if p.name.lower() != "local"
        ] + [prov]
        if not st.registry.default:
            st.registry.default = "local"
        profile = EngineProfile(
            name="local", kind="llama_cpp", base_url=base_url,
            load_options=load_options
            or {"context": 8192, "gpu_layers": 999},
            capabilities=["streaming", "prefix_caching"],
        )
        st.engines.engines = [
            e for e in st.engines.engines if e.name.lower() != "local"
        ] + [profile]
        if not st.engines.default:
            st.engines.default = "local"
        # persist so CLI auto-start (and /model) reuse these launch settings
        try:
            save_engines(st.engines, st.engines_file)
        except OSError as exc:
            print(f"harness: could not persist engines config ({exc})",
                  file=sys.stderr)

    app.state.register_local = register_local

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

    @app.get("/v1/server/status")
    def server_status():
        return models_manager.status()

    @app.get("/v1/server/log")
    def server_log(tail: int = 120):
        return {"lines": models_manager.server_log(tail)}

    @app.get("/v1/server/metrics")
    def server_metrics():
        return models_manager.server_metrics()

    @app.delete("/v1/models/local")
    def delete_local_model(file: str = Query(...)):
        try:
            return models_manager.delete_local(file)
        except (ValueError, FileNotFoundError) as exc:
            raise HTTPException(400, str(exc))

    @app.post("/v1/server/stop")
    def server_stop():
        return models_manager.stop()

    @app.post("/v1/server/start")
    def server_start(req: ServerStartRequest):
        try:
            info = models_manager.start(
                model=req.model, hf_repo=req.hf_repo, hf_file=req.hf_file,
                port=req.port, ctx_size=req.ctx_size, ngl=req.ngl,
                extra_args=req.extra_args(),
            )
        except RuntimeError as exc:
            message = str(exc)
            code = 502
            if "already running" in message:
                code = 409
            elif "not found at" in message or "neither a local file" in message:
                code = 400
            raise HTTPException(code, message)
        if req.register_provider:
            register_local(info, load_options=req.load_options(),
                           api_key=req.api_key)
        return {**info, "provider_registered": bool(req.register_provider)}

    @app.get("/v1/models/local")
    def local_models():
        return {"models_dir": str(models_manager.models_dir),
                "models": models_manager.list_local()}

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
