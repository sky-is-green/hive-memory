"""Run the harness sidecar: ``python -m harness [--host H] [--port P]``."""

from __future__ import annotations

import argparse
from pathlib import Path

import uvicorn

from backend.providers import providers_path
from cortex.e2e import FakeUltraSmall, MockTransport
from harness.app import DEFAULT_PORT, OpenAICompatBackend, create_app
from harness.models import launch_extra_args, LlamaServerManager


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
    parser.add_argument(
        "--setup", action="store_true",
        help="guided first-run check: create providers.local.json if missing, "
             "probe for a reachable backend, and print the next command",
    )
    args = parser.parse_args(argv)

    if args.setup:
        return _setup(providers_path(args.providers_file or None))

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

    # Auto-start (or adopt) the local llama.cpp server when a model is
    # available, reusing the saved launch settings from the engine profile
    # (--no-auto-start opts out). Mirrors POST /v1/server/start.
    if not args.no_auto_start and not args.mock:
        manager = app.state.models
        try:
            local = manager.list_local()
            port_busy = manager.status()["running"]
            if port_busy:
                info = manager.start(model=None, port=args.llama_port)
                app.state.register_local(info)
                print(f"auto-start: adopted llama-server on port {info['port']}")
            elif local:
                newest = local[0]["file"]
                load_options = {}
                try:
                    engine = app.state.harness.engines.resolve("local")
                    load_options = dict(engine.load_options or {})
                except LookupError:
                    pass
                print(f"auto-start: launching llama-server with {newest}")
                info = manager.start(
                    model=newest, port=args.llama_port,
                    ctx_size=int(load_options.get("context", 8192)),
                    ngl=int(load_options.get("gpu_layers", 999)),
                    extra_args=launch_extra_args(load_options),
                )
                app.state.register_local(info, load_options=load_options)
                print(f"auto-start: serving http://{manager.host}:{info['port']} "
                      f"(model: {info.get('model')})")
        except Exception as exc:  # noqa: BLE001 - never block boot on this
            print(f"auto-start: skipped ({exc})")

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    return 0


def _setup(providers_file: Path) -> int:
    """Guided first run: config, backend probe, next command."""
    import shutil

    from backend.providers import DEFAULT_PROVIDERS_FILE

    print("HiveBench Studio — first-run setup")
    print("-" * 46)

    # 1. providers config
    if not providers_file.exists():
        example = Path("providers.example.json")
        if example.exists():
            providers_file.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(example, providers_file)
            print(f"  [ok] created {providers_file} from providers.example.json")
            print("       (default provider is 'lmstudio' on localhost:1234)")
        else:
            print(f"  [..] no providers.example.json found; using defaults "
                  f"({DEFAULT_PROVIDERS_FILE} not required)")
    else:
        print(f"  [ok] providers config present: {providers_file}")

    # 2. backend probe: LM Studio on :1234 (or a provider's base_url)
    from backend.openai_compat import OpenAICompatBackend

    found_backend = False
    for label, base in (("LM Studio (localhost:1234)", "http://localhost:1234"),):
        try:
            ok = OpenAICompatBackend(base_url=base).health()
        except Exception:  # noqa: BLE001
            ok = False
        print(f"  [{'ok' if ok else '..'}] {label}: "
              f"{'reachable' if ok else 'not running'}")
        found_backend = found_backend or ok

    # 3. local llama-server binary (managed mode)
    llama_bin = Path("tools/llama.cpp/llama-server.exe")
    if llama_bin.exists():
        print(f"  [ok] local llama-server found ({llama_bin}); "
              "auto-start will serve models/gguf on :1234")
    else:
        print("  [..] no local llama-server (auto-start needs models/gguf)")

    # 4. next step
    print("-" * 46)
    if found_backend:
        print("Ready. Start the studio with:")
        print("    python -m harness            # or: hivebench-harness")
        print("Then open http://127.0.0.1:8765")
    else:
        print("No reachable backend yet. Either:")
        print("  1. start LM Studio with a model loaded (localhost:1234), or")
        print("  2. drop GGUF files into models/gguf and run:")
        print("     python -m harness")
        print("     (auto-start will launch llama-server for you)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
