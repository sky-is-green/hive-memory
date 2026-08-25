"""Local llama.cpp server management + Hugging Face model acquisition.

Owns the "load models in our own app" layer (HARNESS-SPEC M4), LM-Studio
style: **multiple models can be loaded at once**, each as its own
``llama-server`` process on its own port, independently unloadable, every
one registered as a provider so chat/agent/benchmarks can target it by name.

- Discovery is **live**: hub search and repo file listings hit the public
  Hugging Face API per request, so newly released models appear immediately —
  nothing model-specific is hardcoded here. Managed downloads run in
  background threads via ``huggingface_hub`` into ``models_dir``.
- ``llama-server --hf-repo/--hf-file`` is passed through verbatim when a
  requested repo/file is not yet local, so day-one releases work even before
  a managed download exists.

Injectable seams (``spawner``, ``prober``, module-level ``_hf_download``)
keep the whole manager unit-testable offline.
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import requests

REPO_ROOT = Path(__file__).resolve().parents[2]
HF_API = "https://huggingface.co/api"


def _default_binary() -> Path:
    """HARNESS_LLAMA_SERVER override > tools/llama.cpp/llama-server(.exe)."""
    env = os.environ.get("HARNESS_LLAMA_SERVER")
    if env:
        return Path(env)
    exe = "llama-server.exe" if os.name == "nt" else "llama-server"
    return REPO_ROOT / "tools" / "llama.cpp" / exe


def _probe(base_url: str, timeout: float = 5.0):
    """Return the first model id advertised by the server, True if reachable
    but silent about ids, False when unreachable."""
    try:
        resp = requests.get(f"{base_url}/v1/models", timeout=timeout)
    except requests.RequestException:
        return False
    if not resp.ok:
        return False
    try:
        ids = [m.get("id", "") for m in resp.json().get("data", [])]
    except ValueError:
        return True
    ids = [i for i in ids if i]
    return ids[0] if ids else True


def _hf_download(repo_id: str, filename: str, dest_dir: Path,
                 token: Optional[str] = None) -> Path:
    """Blocking single-file HF download (runs inside a worker thread)."""
    from huggingface_hub import hf_hub_download

    return Path(hf_hub_download(
        repo_id=repo_id, filename=filename, local_dir=str(dest_dir),
        token=token or os.environ.get("HF_TOKEN") or None,
    ))


def _port_in_use(host: str, port: int, timeout: float = 1.0) -> bool:
    """True when something already accepts TCP connections on host:port."""
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def _pid_listening_on(port: int) -> Optional[int]:
    """Pid of whatever process LISTENs on this port (best effort)."""
    try:
        import psutil

        for conn in psutil.net_connections(kind="inet"):
            if (conn.status == psutil.CONN_LISTEN and conn.laddr
                    and conn.laddr.port == port and conn.pid):
                return conn.pid
    except Exception:  # noqa: BLE001 - psutil limits are environment-specific
        return None
    return None


def _process_name(pid: Optional[int]) -> str:
    if not pid:
        return ""
    try:
        import psutil

        return psutil.Process(pid).name() or ""
    except Exception:  # noqa: BLE001
        return ""


def _terminate_pid(pid: int) -> None:
    try:
        import psutil

        proc = psutil.Process(pid)
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except Exception:  # noqa: BLE001
            proc.kill()
    except Exception:  # noqa: BLE001 - already gone
        pass


def launch_extra_args(load_options: dict) -> list[str]:
    """llama-server argv extras recorded in an engine profile's load_options
    (shared by the start endpoint, /model command, and CLI auto-start)."""
    args: list[str] = []
    if load_options.get("threads"):
        args += ["-t", str(load_options["threads"])]
    if load_options.get("flash_attn"):
        args += ["-fa", "on"]
    if load_options.get("parallel_slots"):
        args += ["-np", str(load_options["parallel_slots"])]
    if load_options.get("cache_type_k"):
        args += ["--cache-type-k", load_options["cache_type_k"]]
    if load_options.get("cache_type_v"):
        args += ["--cache-type-v", load_options["cache_type_v"]]
    if load_options.get("batch_size"):
        args += ["-b", str(load_options["batch_size"])]
    if load_options.get("ubatch_size"):
        args += ["-ub", str(load_options["ubatch_size"])]
    if load_options.get("alias"):
        args += ["--alias", load_options["alias"]]
    if load_options.get("mlock"):
        args += ["--mlock"]
    if load_options.get("no_mmap"):
        args += ["--no-mmap"]
    return args


@dataclass
class ServerInstance:
    """One loaded model: its llama-server process, port, and identity."""

    key: str
    port: int
    model: str
    pid: Optional[int] = None
    adopted: bool = False
    started_at: str = ""
    proc: object = field(default=None, repr=False, compare=False)

    def to_dict(self) -> dict:
        return {
            "key": self.key,
            "running": True,
            "healthy": True,
            "port": self.port,
            "base_url": f"http://127.0.0.1:{self.port}",
            "model": self.model,
            "pid": self.pid,
            "adopted": self.adopted,
            "started_at": self.started_at,
        }


class DownloadJob:
    """Status record for one managed HF file download."""

    def __init__(self, key: str, repo: str, filename: str) -> None:
        self.key = key
        self.repo = repo
        self.filename = filename
        self.state = "queued"  # queued | downloading | done | error
        self.error = ""
        self.path: Optional[str] = None
        self.started_at = time.time()
        self.finished_at: Optional[float] = None

    def to_dict(self) -> dict:
        return {
            "key": self.key, "repo": self.repo, "filename": self.filename,
            "state": self.state, "error": self.error, "path": self.path,
            "elapsed_s": round((self.finished_at or time.time())
                               - self.started_at, 1),
        }


class LlamaServerManager:
    """Multi-model llama-server fleet, downloads, and the local library.

    Each ``load()`` spawns (or adopts) one llama-server on its own port and
    registers it under a unique key; ``unload(key)`` stops exactly that one.
    """

    def __init__(
        self,
        binary: Optional[Path] = None,
        models_dir: Optional[Path] = None,
        host: str = "127.0.0.1",
        port: int = 1234,
        log_dir: Optional[Path] = None,
        spawner: Callable[..., object] = subprocess.Popen,
        prober: Optional[Callable[[str], object]] = None,
        startup_timeout: float = 300.0,
    ) -> None:
        self.binary = Path(binary) if binary else _default_binary()
        self.models_dir = Path(models_dir) if models_dir \
            else REPO_ROOT / "models" / "gguf"
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.host = host
        self.port = port
        self.log_dir = Path(log_dir) if log_dir else REPO_ROOT / "logs"
        self.spawner = spawner
        self.prober = prober or _probe
        self.startup_timeout = startup_timeout
        self._lock = threading.RLock()
        self._instances: dict[str, ServerInstance] = {}
        self._downloads: dict[str, DownloadJob] = {}

    # ------------------------------------------------------------------
    # local library
    # ------------------------------------------------------------------
    def server_log(self, tail: int = 120) -> list[str]:
        """Last lines of the llama-server output (startup errors land here).

        Logs are per-port (``llama_server_<port>.log``); the most recently
        written file wins."""
        candidates = sorted(self.log_dir.glob("llama_server*.log"),
                            key=lambda p: p.stat().st_mtime, reverse=True)
        if not candidates:
            return []
        text = candidates[0].read_text(encoding="utf-8", errors="replace")
        return text.splitlines()[-max(1, tail):]

    def list_local(self) -> list[dict]:
        entries = []
        for p in sorted(self.models_dir.rglob("*.gguf")):
            stat = p.stat()
            entries.append({
                "name": p.stem,
                "file": str(p.relative_to(self.models_dir)),
                "size_gb": round(stat.st_size / (1024 ** 3), 2),
                "modified": time.strftime("%Y-%m-%d %H:%M",
                                          time.localtime(stat.st_mtime)),
            })
        return sorted(entries, key=lambda e: e["modified"], reverse=True)

    def resolve_model(self, name: str) -> Optional[Path]:
        """A model arg is either a local library name/file or a real path."""
        candidate = Path(name)
        if candidate.is_file():
            return candidate.resolve()
        direct = self.models_dir / name
        if direct.is_file():
            return direct.resolve()
        matches = [p for p in self.models_dir.rglob("*.gguf")
                   if name in p.stem or name == p.name]
        return matches[0].resolve() if matches else None

    # ------------------------------------------------------------------
    # lifecycle
    # ------------------------------------------------------------------
    def _instance_alive(self, inst: ServerInstance) -> bool:
        if inst.adopted:
            return True
        return inst.proc is not None and inst.proc.poll() is None

    def _probe_instance(self, inst: ServerInstance) -> object:
        try:
            return self.prober(f"http://{self.host}:{inst.port}")
        except Exception:  # noqa: BLE001 - a dead instance probes False
            return False

    def _prune_dead(self) -> None:
        for key in list(self._instances):
            inst = self._instances[key]
            if not self._instance_alive(inst):
                del self._instances[key]

    def _free_port(self, preferred: Optional[int] = None) -> int:
        """First free port at/below-or-after ``preferred`` (our own loaded
        instances' ports are busy by definition, so they are skipped)."""
        port = preferred if preferred is not None else self.port
        for _ in range(64):
            if not _port_in_use(self.host, port):
                return port
            port += 1
        raise RuntimeError("no free port found in the llama-server range")

    def _unique_key(self, base: str) -> str:
        key = base
        n = 2
        while key in self._instances:
            key = f"{base}-{n}"
            n += 1
        return key

    def status(self) -> dict:
        with self._lock:
            self._prune_dead()
            instances = [i.to_dict() for i in
                         sorted(self._instances.values(), key=lambda i: i.key)]
        for inst_dict in instances:
            inst = self._instances[inst_dict["key"]]
            probe = self._probe_instance(inst)
            inst_dict["healthy"] = bool(probe)
            if probe and probe is not True:
                inst_dict["model"] = probe
        primary = instances[0] if instances else None
        return {
            "running": bool(instances),
            "healthy": bool(instances) and all(i["healthy"] for i in instances),
            "binary": str(self.binary),
            "binary_found": self.binary.is_file(),
            "port": primary["port"] if primary else self.port,
            "base_url": primary["base_url"] if primary
            else f"http://{self.host}:{self.port}",
            "model": primary["model"] if primary else None,
            "pid": primary["pid"] if primary else None,
            "started_at": primary["started_at"] if primary else None,
            "instances": instances,
            "local_models": self.list_local(),
            "downloads": self.downloads_status(),
        }

    def load(
        self,
        model: Optional[str] = None,
        hf_repo: Optional[str] = None,
        hf_file: Optional[str] = None,
        key: Optional[str] = None,
        port: Optional[int] = None,
        ctx_size: int = 8192,
        ngl: int = 999,
        extra_args: Optional[list[str]] = None,
    ) -> dict:
        """Load one model as its own llama-server instance. Priority: local
        ``model`` > --hf-repo/--hf-file passthrough. Loading the same model
        twice is rejected; different models coexist on different ports."""
        if not self.binary.is_file():
            raise RuntimeError(
                f"llama-server not found at {self.binary}. Set HARNESS_LLAMA_SERVER "
                "or extract a llama.cpp Vulkan release into tools/llama.cpp/."
            )
        resolved = self.resolve_model(model) if model else None
        if model and resolved is None and not (hf_repo and hf_file):
            raise RuntimeError(
                f"model '{model}' is neither a local file nor paired with "
                "--hf-repo/--hf-file; see GET /v1/models/local"
            )
        with self._lock:
            self._prune_dead()
            if resolved is not None:
                for inst in self._instances.values():
                    if inst.model in (resolved.name, resolved.stem, str(resolved)):
                        raise RuntimeError(
                            f"'{resolved.name}' is already loaded "
                            f"(instance '{inst.key}' on port {inst.port})"
                        )

        model_name = (resolved.name if resolved else (hf_file or "model"))
        base_key = key or Path(model_name).stem or "model"
        use_port = int(port) if port else self._free_port(self.port)
        # Pre-flight: something is already on this port. If it is a healthy
        # llama-server, ADOPT it (sidecar restarts must not orphan or fight
        # their own servers); anything else is refused loudly.
        if _port_in_use(self.host, use_port):
            base_url = f"http://{self.host}:{use_port}"
            listener_pid = _pid_listening_on(use_port)
            listener_name = _process_name(listener_pid)
            try:
                probe = self.prober(base_url)
            except Exception:  # noqa: BLE001 - a broken squatter is not ours
                probe = False
            if probe and "llama" in listener_name.lower():
                inst = ServerInstance(
                    key=self._unique_key(base_key),
                    port=use_port,
                    model=(probe if probe is not True else model_name),
                    pid=listener_pid,
                    adopted=True,
                    started_at=time.strftime("%Y-%m-%d %H:%M:%S"),
                )
                with self._lock:
                    self._instances[inst.key] = inst
                return inst.to_dict()
            raise RuntimeError(
                f"port {use_port} is already serving"
                + (f" (pid {listener_pid}: {listener_name})" if listener_pid else "")
                + "; stop that server or pick another port."
            )

        cmd = [str(self.binary)]
        if resolved is not None:
            cmd += ["-m", str(resolved)]
        elif hf_repo and hf_file:
            cmd += ["--hf-repo", hf_repo, "--hf-file", hf_file]
        cmd += [
            "--host", self.host,
            "--port", str(use_port),
            "-ngl", str(ngl),
            "-c", str(ctx_size),
        ]
        cmd += [str(a) for a in (extra_args or [])]

        self.log_dir.mkdir(parents=True, exist_ok=True)
        log_path = self.log_dir / f"llama_server_{use_port}.log"
        log_handle = open(log_path, "ab")
        try:
            proc = self.spawner(cmd, stdout=log_handle, stderr=subprocess.STDOUT)
        finally:
            log_handle.close()

        base_url = f"http://{self.host}:{use_port}"
        deadline = time.time() + self.startup_timeout
        model_id = None
        while time.time() < deadline:
            if proc.poll() is not None:
                raise RuntimeError(
                    f"llama-server exited during startup (code {proc.poll()}); "
                    f"see {log_path}"
                )
            probe = self.prober(base_url)
            if probe:
                # The probe must answer for OUR process: re-check liveness so a
                # foreign server that raced the bind is not misattributed.
                time.sleep(0.5)
                if proc.poll() is not None:
                    raise RuntimeError(
                        f"llama-server exited during startup (code {proc.poll()}); "
                        f"see {log_path}"
                    )
                model_id = probe if probe is not True else None
                break
            time.sleep(0.5)
        else:
            try:
                proc.terminate()
            except Exception:  # noqa: BLE001 - best-effort cleanup
                pass
            raise RuntimeError(
                f"llama-server did not become healthy within "
                f"{self.startup_timeout:.0f}s; see {log_path}"
            )

        inst = ServerInstance(
            key=self._unique_key(base_key),
            port=use_port,
            model=model_id or model_name,
            pid=getattr(proc, "pid", None),
            started_at=time.strftime("%Y-%m-%d %H:%M:%S"),
            proc=proc,
        )
        with self._lock:
            self._instances[inst.key] = inst
        return inst.to_dict()

    def unload(self, key: str) -> dict:
        with self._lock:
            inst = self._instances.pop(key, None)
        if inst is None:
            raise RuntimeError(f"no such loaded instance: {key}")
        if inst.proc is not None:
            try:
                inst.proc.terminate()
                try:
                    inst.proc.wait(timeout=10)
                except Exception:  # noqa: BLE001
                    inst.proc.kill()
            except Exception:  # noqa: BLE001 - already gone
                pass
        elif inst.pid:
            _terminate_pid(inst.pid)
        return {"ok": True, "unloaded": key}

    def stop_all(self) -> dict:
        with self._lock:
            keys = list(self._instances)
        for key in keys:
            try:
                self.unload(key)
            except RuntimeError:
                pass
        return {"ok": True, "stopped": len(keys)}

    # Backward-compatible single-instance vocabulary -----------------------
    def start(self, **kwargs) -> dict:
        return self.load(**kwargs)

    def stop(self) -> dict:
        return self.stop_all()

    # ------------------------------------------------------------------
    # Hugging Face (live hub API — nothing cached/hardcoded)
    # ------------------------------------------------------------------
    def hub_search(self, query: str = "", limit: int = 12) -> list[dict]:
        params = {"filter": "gguf", "sort": "downloads", "limit": limit}
        if query:
            params["search"] = query
        resp = requests.get(f"{HF_API}/models", params=params, timeout=15)
        resp.raise_for_status()
        return [
            {
                "repo": m.get("id", ""),
                "downloads": m.get("downloads", 0),
                "likes": m.get("likes", 0),
                "last_modified": (m.get("lastModified") or "")[:10],
            }
            for m in resp.json()
        ]

    def hub_files(self, repo: str) -> list[dict]:
        resp = requests.get(f"{HF_API}/models/{repo}/tree/main", timeout=20)
        resp.raise_for_status()
        return [
            {"file": e["path"], "size_gb": round(e.get("size", 0) / (1024 ** 3), 2)}
            for e in resp.json()
            if e.get("path", "").endswith(".gguf")
            and not e["path"].startswith("mmproj")
        ]

    def delete_local(self, file: str) -> dict:
        """Remove one GGUF from the local library (path-traversal safe)."""
        root = self.models_dir.resolve()
        target = (root / file).resolve()
        if root not in target.parents and target != root:
            raise ValueError("file must stay inside the models directory")
        if not target.is_file():
            raise FileNotFoundError(f"no such model file: {file}")
        target.unlink()
        return {"ok": True, "deleted": str(target)}

    def server_log(self, tail: int = 120) -> list[str]:
        """Last lines of the llama-server logs (startup errors live here)."""
        lines: list[str] = []
        for path in sorted(self.log_dir.glob("llama_server*.log")):
            lines += path.read_text(encoding="utf-8",
                                    errors="replace").splitlines()[-max(1, tail):]
        return lines

    def server_metrics(self, key: Optional[str] = None) -> dict:
        """Proxy llama-server's prometheus /metrics (kv usage, throughput)."""
        with self._lock:
            inst = self._instances.get(key) if key else None
            port = inst.port if inst else (
                next(iter(self._instances.values())).port
                if self._instances else self.port)
        base = f"http://{self.host}:{port}"
        try:
            resp = requests.get(f"{base}/metrics", timeout=3)
        except requests.RequestException:
            return {"available": False}
        if not resp.ok:
            return {"available": False}
        gauges = {}
        for line in resp.text.splitlines():
            if line.startswith("#") or " " not in line:
                continue
            k, _, v = line.partition(" ")
            try:
                gauges[k] = float(v)
            except ValueError:
                continue
        return {"available": True, "gauges": gauges}

    def downloads_status(self) -> list[dict]:
        self._prune_downloads()
        return [job.to_dict() for job in self._downloads.values()]

    def _prune_downloads(self, keep: int = 20) -> None:
        """Drop finished/failed jobs beyond the newest ``keep`` — the registry
        must not grow with every hub download a browser ever started."""
        with self._lock:
            finished = sorted(
                (j for j in self._downloads.values()
                 if j.state in ("done", "error")),
                key=lambda j: j.started_at,
            )
            excess = max(0, len(finished) - keep)
            for job in finished[:excess]:
                self._downloads.pop(job.key, None)

    def download(self, repo: str, filename: str) -> dict:
        key = f"{repo}:{filename}"
        with self._lock:
            existing = self._downloads.get(key)
            if existing and existing.state in ("queued", "downloading"):
                return existing.to_dict()
            job = DownloadJob(key, repo, filename)
            self._downloads[key] = job

        def run(job: DownloadJob) -> None:
            job.state = "downloading"
            try:
                path = _hf_download(repo, filename, self.models_dir)
                job.state = "done"
                job.path = str(path)
            except Exception as exc:  # noqa: BLE001 - surfaced via status
                job.state = "error"
                job.error = str(exc)
            finally:
                job.finished_at = time.time()

        threading.Thread(target=run, args=(job,), daemon=True).start()
        return job.to_dict()
