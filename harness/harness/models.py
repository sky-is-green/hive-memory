"""Local llama.cpp server management + Hugging Face model acquisition.

Owns the "load models in our own app" layer (HARNESS-SPEC M4):

- ``LlamaServerManager`` spawns/stops one ``llama-server`` subprocess
  (OpenAI-compatible, Vulkan build for AMD GPUs) and health-checks it.
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


class LlamaServerManager:
    """One llama-server process, its downloads, and the local model library."""

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
        self._lock = threading.Lock()
        self._proc = None
        self._meta: dict = {}
        self._downloads: dict[str, DownloadJob] = {}

    # ------------------------------------------------------------------
    # local library
    # ------------------------------------------------------------------
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
    def status(self) -> dict:
        with self._lock:
            meta = dict(self._meta)
            proc_alive = self._proc is not None and self._proc.poll() is None
        running = bool(meta.get("running")) and (proc_alive or meta.get("adopted"))
        info = {
            "running": running,
            "adopted": bool(meta.get("adopted")),
            "binary": str(self.binary),
            "binary_found": self.binary.is_file(),
            "port": meta.get("port", self.port),
            "base_url": f"http://{self.host}:{meta.get('port', self.port)}",
            "model": meta.get("model"),
            "pid": meta.get("pid"),
            "started_at": meta.get("started_at"),
            "local_models": self.list_local(),
            "downloads": [j.to_dict() for j in self._downloads.values()],
        }
        if running:
            probe = self.prober(info["base_url"])
            info["healthy"] = bool(probe)
            if probe and probe is not True:
                info["model"] = probe
            if not info["healthy"] and meta.get("adopted"):
                # an adopted server that stopped answering is gone
                info["running"] = False
        else:
            info["healthy"] = False
        return info

    def start(
        self,
        model: Optional[str] = None,
        hf_repo: Optional[str] = None,
        hf_file: Optional[str] = None,
        port: Optional[int] = None,
        ctx_size: int = 8192,
        ngl: int = 999,
        extra_args: Optional[list[str]] = None,
    ) -> dict:
        """Start llama-server. Priority: local ``model`` > --hf-repo/--hf-file
        passthrough (llama-server downloads into the shared HF cache)."""
        if not self.binary.is_file():
            raise RuntimeError(
                f"llama-server not found at {self.binary}. Set HARNESS_LLAMA_SERVER "
                "or extract a llama.cpp Vulkan release into tools/llama.cpp/."
            )
        with self._lock:
            if self._proc is not None and self._proc.poll() is None:
                raise RuntimeError(
                    f"already running on port {self._meta.get('port', self.port)}; "
                    "stop it first"
                )
        resolved = self.resolve_model(model) if model else None
        if model and resolved is None and not (hf_repo and hf_file):
            raise RuntimeError(
                f"model '{model}' is neither a local file nor paired with "
                "--hf-repo/--hf-file; see GET /v1/models/local"
            )
        use_port = int(port) if port else self.port

        # Pre-flight: something is already on this port. If it is a healthy
        # llama-server, ADOPT it (sidecar restarts must not orphan or fight
        # their own server); anything else is refused loudly.
        if _port_in_use(self.host, use_port):
            base_url = f"http://{self.host}:{use_port}"
            listener_pid = _pid_listening_on(use_port)
            listener_name = _process_name(listener_pid)
            try:
                probe = self.prober(base_url)
            except Exception:  # noqa: BLE001 - a broken squatter is not ours
                probe = False
            if probe and "llama" in listener_name.lower():
                with self._lock:
                    self._proc = None
                    self._meta = {
                        "running": True,
                        "adopted": True,
                        "pid": listener_pid,
                        "port": use_port,
                        "model": (probe if probe is not True
                                  else (resolved.name if resolved
                                        else (hf_file or ""))),
                        "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                    }
                return self.status()
            raise RuntimeError(
                f"port {use_port} is already serving"
                + (f" (pid {listener_pid}: {listener_name})" if listener_pid else "")
                + "; stop it or pass port=<free port>."
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
        log_path = self.log_dir / "llama_server.log"
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

        with self._lock:
            self._proc = proc
            self._meta = {
                "running": True,
                "pid": getattr(proc, "pid", None),
                "port": use_port,
                "model": model_id or (resolved.name if resolved
                                      else (hf_file or "")),
                "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            }
        return self.status()

    def stop(self) -> dict:
        with self._lock:
            proc, self._proc = self._proc, None
            meta, self._meta = self._meta, {}
        adopted_pid = meta.get("pid") if meta.get("adopted") else None
        if proc is not None:
            try:
                proc.terminate()
                try:
                    proc.wait(timeout=10)
                except Exception:  # noqa: BLE001 - force after grace period
                    proc.kill()
            except Exception:  # noqa: BLE001 - already gone
                pass
        elif adopted_pid:
            _terminate_pid(int(adopted_pid))
        return {"ok": True}

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
        """Last lines of the llama-server log (startup errors live here)."""
        path = self.log_dir / "llama_server.log"
        if not path.is_file():
            return []
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        return lines[-max(1, tail):]

    def server_metrics(self) -> dict:
        """Proxy llama-server's prometheus /metrics (kv usage, throughput)."""
        base = f"http://{self.host}:{self._meta.get('port', self.port)}"
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
            key, _, value = line.partition(" ")
            try:
                gauges[key] = float(value)
            except ValueError:
                continue
        return {"available": True, "gauges": gauges}

    def downloads_status(self) -> list[dict]:
        return [job.to_dict() for job in self._downloads.values()]

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
