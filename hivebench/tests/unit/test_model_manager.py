"""Unit tests for the model-management layer (HARNESS-SPEC M4).

All offline: the llama-server process is a fake spawned-command recorder, the
health prober is injectable, and the HF download implementation is monkey-
patched — no binary, GPU, or network needed.
"""

import json
import socket
import threading
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import harness.models as models_module
from backend.openai_compat import OpenAICompatBackend
from cortex.e2e import FakeUltraSmall, MockTransport
from harness.app import create_app
from harness.models import LlamaServerManager


class FakeProc:
    def __init__(self):
        self.pid = 4321
        self.terminated = False
        self.returncode = None

    def poll(self):
        return self.returncode

    def terminate(self):
        self.terminated = True
        self.returncode = 0

    def kill(self):
        self.terminate()


@pytest.fixture()
def manager(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    spawned = {}

    def fake_spawner(cmd, stdout=None, stderr=None):
        spawned["cmd"] = cmd
        return FakeProc()

    # healthy immediately, advertising the model id from the -m argument
    def fake_prober(base_url):
        cmd = spawned.get("cmd", [])
        gguf = next((a for a in cmd if a.endswith(".gguf")), "fake-model")
        return Path(gguf).stem

    blocker = socket.socket()
    blocker.bind(("127.0.0.1", 0))
    free_port = blocker.getsockname()[1]
    blocker.close()

    mgr = LlamaServerManager(
        binary=tmp_path / "llama-server.exe",
        models_dir=tmp_path / "models" / "gguf",
        log_dir=tmp_path / "logs",
        port=free_port,
        spawner=fake_spawner,
        prober=fake_prober,
        startup_timeout=5,
    )
    mgr._spawned = spawned  # test access
    mgr.binary.write_bytes(b"")  # binary exists
    return mgr


# ---------------------------------------------------------------------------
# lifecycle
# ---------------------------------------------------------------------------
def test_start_builds_expected_command_and_registers_health(manager):
    (manager.models_dir / "qwen3.8-8b-q4_k_m.gguf").write_bytes(b"x")
    info = manager.start(model="qwen3.8-8b", ctx_size=16384, ngl=999)
    assert info["running"] is True and info["healthy"] is True
    assert info["model"] == "qwen3.8-8b-q4_k_m"
    cmd = manager._spawned["cmd"]
    assert "-m" in cmd and str(manager.models_dir / "qwen3.8-8b-q4_k_m.gguf") in cmd
    i = cmd.index("--port")
    assert cmd[i + 1] == str(manager.port)
    j = cmd.index("-ngl")
    assert cmd[j + 1] == "999"
    k = cmd.index("-c")
    assert cmd[k + 1] == "16384"


def test_start_refuses_missing_binary(tmp_path):
    mgr = LlamaServerManager(binary=tmp_path / "nope.exe",
                             models_dir=tmp_path / "m",
                             log_dir=tmp_path / "logs")
    with pytest.raises(RuntimeError) as exc:
        mgr.start(model="x.gguf")
    assert "HARNESS_LLAMA_SERVER" in str(exc.value)


def test_start_refuses_unknown_model_without_hf_fallback(manager):
    with pytest.raises(RuntimeError) as exc:
        manager.start(model="not-downloaded")
    assert "neither a local file" in str(exc.value)


def test_hf_passthrough_when_model_not_local(manager):
    manager.start(hf_repo="unsloth/Qwen3.8-9B-GGUF",
                  hf_file="Qwen3.8-9B-UD-Q4_K_M.gguf")
    cmd = manager._spawned["cmd"]
    assert "--hf-repo" in cmd and "unsloth/Qwen3.8-9B-GGUF" in cmd
    assert "--hf-file" in cmd and "Qwen3.8-9B-UD-Q4_K_M.gguf" in cmd


def test_launch_settings_reach_the_command_line(client, tmp_path):
    """The settings-panel launch flags must land as real llama-server args."""
    c, app = client
    (app.state.models.models_dir / "demo.gguf").write_bytes(b"x")
    c.post("/v1/server/stop")  # in case another test left it running
    r = c.post("/v1/server/start", json={
        "model": "demo.gguf", "threads": 6, "flash_attn": True,
        "parallel_slots": 2, "cache_type_k": "q8_0", "cache_type_v": "q8_0",
        "batch_size": 1024, "ubatch_size": 512, "alias": "demo-model",
        "mlock": True,
    }).json()
    assert r["running"] is True
    cmd = app.state.models._spawned["cmd"]
    for expected in ("-t", "6", "-fa", "-np", "2", "--cache-type-k", "q8_0",
                     "--cache-type-v", "q8_0", "-b", "1024", "-ub", "512",
                     "--alias", "demo-model", "--mlock"):
        assert expected in cmd, expected
    # engine load_options record what was actually launched
    engines = {e["name"]: e for e in c.get("/v1/engines").json()["engines"]}
    opts = engines["local"]["load_options"]
    assert opts["threads"] == 6 and opts["flash_attn"] is True
    assert opts["cache_type_k"] == "q8_0" and opts["parallel_slots"] == 2
    c.post("/v1/server/stop")


def test_hive_defaults_endpoint(client):
    c, _app = client
    body = c.get("/v1/hive/defaults").json()
    assert body["max_context"] > 0
    assert "comb_gate_threshold" in body
    assert "sampling" in body  # co-added field stays exposed for the UI


def test_stop_terminates_and_status_reports_stopped(manager):
    (manager.models_dir / "m.gguf").write_bytes(b"x")
    manager.start(model="m.gguf")
    result = manager.stop()
    assert result["ok"]
    st = manager.status()
    assert st["running"] is False and st["healthy"] is False


def test_double_start_rejected_while_running(manager):
    (manager.models_dir / "m.gguf").write_bytes(b"x")
    manager.start(model="m.gguf")
    with pytest.raises(RuntimeError) as exc:
        manager.start(model="m.gguf")
    assert "already running" in str(exc.value)


def test_start_refuses_port_already_in_use(manager):
    (manager.models_dir / "m.gguf").write_bytes(b"x")
    blocker = socket.socket()
    blocker.bind(("127.0.0.1", 0))
    blocker.listen(1)
    port = blocker.getsockname()[1]
    try:
        with pytest.raises(RuntimeError) as exc:
            manager.start(model="m.gguf", port=port)
        assert "already serving" in str(exc.value)
    finally:
        blocker.close()
    # nothing was spawned for a refused start
    assert "cmd" not in manager._spawned


def test_unhealthy_startup_is_rolled_back(manager, monkeypatch):
    monkeypatch.setattr(manager, "prober", lambda url: False)
    (manager.models_dir / "bad.gguf").write_bytes(b"x")
    with pytest.raises(RuntimeError) as exc:
        manager.start(model="bad.gguf", )
    assert "did not become healthy" in str(exc.value)
    assert manager.status()["running"] is False


# ---------------------------------------------------------------------------
# hub integration (network seams patched)
# ---------------------------------------------------------------------------
class FakeResp:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self._payload


def test_hub_search_shapes_results(manager, monkeypatch):
    monkeypatch.setattr(models_module.requests, "get",
                        lambda url, params=None, timeout=None: FakeResp([
                            {"id": "unsloth/Qwen3.8-9B-GGUF", "downloads": 99,
                             "likes": 10, "lastModified": "2026-08-20T00:00:00"},
                        ]))
    out = manager.hub_search("qwen3.8")
    assert out[0]["repo"] == "unsloth/Qwen3.8-9B-GGUF"
    assert out[0]["last_modified"] == "2026-08-20"


def test_hub_files_filters_mmproj_and_non_gguf(manager, monkeypatch):
    monkeypatch.setattr(models_module.requests, "get",
                        lambda url, params=None, timeout=None: FakeResp([
                            {"path": "model-UD-Q4_K_M.gguf", "size": 5 * 1024 ** 3},
                            {"path": "mmproj-model.gguf", "size": 1},
                            {"path": "README.md", "size": 1},
                        ]))
    files = manager.hub_files("some/repo")
    assert [f["file"] for f in files] == ["model-UD-Q4_K_M.gguf"]
    assert files[0]["size_gb"] == 5.0


def test_download_lifecycle_and_local_listing(manager, monkeypatch):
    started = threading.Event()

    def fake_download(repo, filename, dest_dir, token=None):
        path = Path(dest_dir) / filename
        path.write_bytes(b"weights" * 1024)  # >0.01 GB rounding is not needed; just non-empty
        started.set()
        return path

    monkeypatch.setattr(models_module, "_hf_download", fake_download)
    job = manager.download("some/repo", "model-q4.gguf")
    assert job["state"] in ("queued", "downloading", "done")
    assert started.wait(timeout=5)
    deadline = time.time() + 5
    statuses = {j.key: j for j in manager._downloads.values()}
    done = statuses["some/repo:model-q4.gguf"]
    while done.state != "done" and time.time() < deadline:
        time.sleep(0.05)
    assert done.state == "done"
    local = manager.list_local()
    assert any(m["file"] == "model-q4.gguf" for m in local)


def test_duplicate_download_returns_same_job(manager, monkeypatch):
    gate = threading.Event()

    def slow_download(repo, filename, dest_dir, token=None):
        gate.wait(timeout=5)
        return Path(dest_dir) / filename

    monkeypatch.setattr(models_module, "_hf_download", slow_download)
    first = manager.download("r", "f.gguf")
    second = manager.download("r", "f.gguf")
    assert first["key"] == second["key"]
    gate.set()


def test_download_error_is_captured_not_raised(manager, monkeypatch):
    def failing(repo, filename, dest_dir, token=None):
        raise RuntimeError("hub 404")

    monkeypatch.setattr(models_module, "_hf_download", failing)
    manager.download("bad/repo", "x.gguf")
    import time

    deadline = time.time() + 5
    job = list(manager._downloads.values())[0]
    while job.state not in ("error", "done") and time.time() < deadline:
        time.sleep(0.05)
    assert job.state == "error"
    assert "404" in job.error


# ---------------------------------------------------------------------------
# sidecar endpoints
# ---------------------------------------------------------------------------
@pytest.fixture()
def client(tmp_path, monkeypatch, manager):
    monkeypatch.chdir(tmp_path)

    def backend_factory(model=None):
        return OpenAICompatBackend(
            base_url="http://mock", model=model or "mock-model",
            transport=MockTransport(latency_ms=0),
        )

    app = create_app(
        ultra_factory=FakeUltraSmall,
        backend_factory=backend_factory,
        runs_root=tmp_path / "runs",
        providers_file=tmp_path / "providers.local.json",
        log_dir=str(tmp_path / "logs"),
        state_dir=tmp_path / "harness_state",
        models_manager=manager,
    )
    with TestClient(app) as c:
        yield c, app


def _clean(obj):
    return json.loads(json.dumps(obj))


def test_server_endpoints_roundtrip(client, tmp_path):
    c, app = client
    body = c.get("/v1/server/status").json()
    assert body["running"] is False
    assert body["binary_found"] is True

    (app.state.models.models_dir / "demo.gguf").write_bytes(b"x")
    r = c.post("/v1/server/start", json={"model": "demo.gguf"}).json()
    assert r["running"] is True and r["provider_registered"] is True

    # start registered provider 'local' + engine profile 'local'
    provs = {p["name"]: p for p in c.get("/v1/provider/config").json()["providers"]}
    assert provs["local"]["base_url"].endswith(f":{app.state.models.port}")
    engines = c.get("/v1/engines").json()
    by_name = {e["name"]: e for e in engines["engines"]}
    assert engines["default"] == "local"
    assert by_name["local"]["kind"] == "llama_cpp"
    assert by_name["local"]["load_options"]["gpu_layers"] == 999

    assert c.post("/v1/server/stop").json()["ok"]

    # conflict while running
    c.post("/v1/server/start", json={"model": "demo.gguf"})
    conflict = c.post("/v1/server/start", json={"model": "demo.gguf"})
    assert conflict.status_code == 409


def test_models_local_and_hub_routes(client, monkeypatch):
    c, app = client
    (app.state.models.models_dir / "a.gguf").write_bytes(b"x" * 2048)
    body = c.get("/v1/models/local").json()
    assert body["models"][0]["file"] == "a.gguf"

    monkeypatch.setattr(models_module.requests, "get",
                        lambda url, params=None, timeout=None: FakeResp([]))
    assert c.get("/v1/models/hub", params={"q": "gemma"}).json()["results"] == []
    files = c.get("/v1/models/hub/files/some/repo")
    assert files.status_code == 200

    dl = c.post("/v1/models/hub/download",
                json={"repo": "r", "file": "f.gguf"})
    assert dl.status_code == 200
    downloads = c.get("/v1/models/hub/downloads").json()["downloads"]
    assert any(d["key"] == "r:f.gguf" for d in downloads)


def test_server_page_serves(client):
    c, _app = client
    page = c.get("/server")
    assert page.status_code == 200
    assert "HiveBench Studio console" in page.text
    assert "/v1/server/status" in page.text
    # the chat pane talks to the loaded model through the hive
    assert "/v1/hive/turn" in page.text
    assert "chatlog" in page.text
    assert page.headers.get("cache-control") == "no-store"
