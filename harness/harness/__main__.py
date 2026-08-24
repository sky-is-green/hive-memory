"""Run the harness sidecar: ``python -m harness [--host H] [--port P]``."""

from __future__ import annotations

import argparse
from pathlib import Path

import uvicorn

from backend.providers import providers_path
from cortex.e2e import FakeUltraSmall, MockTransport
from harness.app import DEFAULT_PORT, OpenAICompatBackend, create_app
from harness.models import LlamaServerManager


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="HiveBench Studio harness sidecar (FastAPI, local-only)",
    )
    parser.add_argument("--host", default="127.0.0.1",
                        help="bind address (default 127.0.0.1; keep it local)")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--providers-file", default="",
                        help="path to the providers JSON config "
                             "(default providers.local.json)")
    parser.add_argument("--log-dir", default="logs",
                        help="NDJSON event-log directory for hive turns")
    parser.add_argument(
        "--state-dir", default="harness_state",
        help="directory where conversations persist across restarts "
             "(default ./harness_state; pass an empty string to disable)",
    )
    parser.add_argument("--runs-root", default="",
                        help="root directory holding run bundles (default ./runs)")
    parser.add_argument(
        "--mock", action="store_true",
        help="offline mode: fake drone + mock backend (no LM Studio needed)",
    )
    parser.add_argument("--llama-server", default="",
                        help="path to llama-server binary (default "
                             "HARNESS_LLAMA_SERVER or tools/llama.cpp/)")
    parser.add_argument("--llama-port", type=int, default=1234,
                        help="port for the managed llama-server (default 1234, "
                             "LM Studio-compatible)")
    parser.add_argument("--models-dir", default="",
                        help="local GGUF library directory (default models/gguf)")
    parser.add_argument(
        "--no-auto-start", action="store_true",
        help="do not auto-launch llama-server even when a local model exists",
    )
    args = parser.parse_args(argv)

    kwargs = {
        "providers_file": providers_path(args.providers_file or None),
        "log_dir": args.log_dir,
        "state_dir": args.state_dir,  # empty string disables persistence
    }
    if args.runs_root:
        kwargs["runs_root"] = args.runs_root
    if args.mock:
        def mock_backend(model):
            return OpenAICompatBackend(transport=MockTransport(), model=model or "mock")

        kwargs["ultra_factory"] = FakeUltraSmall
        kwargs["backend_factory"] = mock_backend
        kwargs["runs_root"] = args.runs_root or Path("runs_mock")

    kwargs["models_manager"] = LlamaServerManager(
        binary=Path(args.llama_server) if args.llama_server else None,
        models_dir=Path(args.models_dir) if args.models_dir else None,
        log_dir=Path(args.log_dir),
        port=args.llama_port,
    )
    app = create_app(**kwargs)

    # Auto-start the local llama.cpp server when a model is available
    # (zero-step live usage; --no-auto-start opts out). Registration mirrors
    # POST /v1/server/start so hive turns route to it immediately.
    if not args.no_auto_start and not args.mock:
        manager = app.state.models
        try:
            local = manager.list_local()
            if local and not manager.status()["running"]:
                newest = local[0]["file"]
                print(f"auto-start: launching llama-server with {newest}")
                info = manager.start(model=newest, port=args.llama_port)
                app.state.register_local(info)
                print(f"auto-start: serving http://{manager.host}:{info['port']} "
                      f"(model: {info.get('model')})")
        except Exception as exc:  # noqa: BLE001 - never block boot on this
            print(f"auto-start: skipped ({exc})")

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
