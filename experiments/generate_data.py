"""Unified data-generation workflow.

Drives the full Hive pipeline over a set of conversations and writes a
self-contained run directory with:

  - NDJSON event logs (correlation-tagged, redacted, rotated)
  - a ground-truth SQLite DB (routing decisions + oracle labels)
  - a per-conversation E2E report
  - optional P1-P10 protocol report (--protocol)
  - optional LM Studio / FIFO baselines (--baselines)

Run directory layout::

    runs/<timestamp>/
        run_report.json
        ground_truth.sqlite
        logs/events-*.ndjson
        labels/...

Usage::

    # Live against LM Studio (recommended for real data):
    python -m experiments.generate_data --live --max-convs 20

    # Offline to validate the workflow / generate synthetic data:
    python -m experiments.generate_data --mock --max-convs 5

    # Full: live + protocol + baselines:
    python -m experiments.generate_data --live --protocol --baselines
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
import uuid
from collections import deque
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Optional

from backend.cache_manager import KVCacheManager
from backend.lmstudio import LMStudioBackend
from cortex.baselines.runner import load_conversations
from cortex.config import HiveConfig
from cortex.e2e import FakeUltraSmall, MockTransport
from cortex.hive import Hive
from cortex.routing import DroneRouter
from experiments.dashboard import KeepAwake, TermDashboard
from logs.event_logger import EventLogger
from oracle.async_oracle import AsyncOracle, TurnRecord
from oracle.ground_truth import GroundTruthDB
from oracle.labeling import generate_all
from retention.store import ContextStore
from sieve.medium import MediumDrone
from sieve.ultra_small import UltraSmallDrone

DEFAULT_PINNED_PREFIX = (
    "You are an assistant operating in the Hive Memory system. "
    "Answer using only the provided context and conversation history."
)


# ---------------------------------------------------------------------------
# Progress reporting
# ---------------------------------------------------------------------------
def _fmt_seconds(seconds: float) -> str:
    seconds = max(0, int(seconds))
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    return f"{h:d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


def _total_user_turns(conversations, max_turns) -> int:
    total = 0
    for conv in conversations:
        n = sum(1 for t in conv.get("turns", []) if t.get("role") == "user")
        total += min(n, max_turns) if max_turns else n
    return total


def _print_progress(done: int, total: int, start: float, **ctx) -> None:
    """Persistent, newline-terminated per-turn status line.

    Used when the terminal dashboard is off. A plain ``\\n`` line (not a
    carriage-return overwrite) so it scrolls as a readable log and is visible
    even when stdout is redirected/captured (launcher app, pipes, GUI) rather
    than a TTY — this keeps the console updating and errors greppable.
    """
    if total <= 0:
        return
    elapsed = time.time() - start
    pct = done / total * 100.0
    eta = ctx.get("eta") or ((elapsed / done) * (total - done) if done else 0.0)
    sys.stdout.write(
        f"  turn {ctx.get('turn', '?')}/{ctx.get('cap', '?')}  "
        f"conv {ctx.get('conv', '?')}/{ctx.get('conv_total', '?')}  "
        f"{done}/{total} ({pct:5.1f}%)  "
        f"elapsed {_fmt_seconds(elapsed)}  ETA {_fmt_seconds(eta)}\n"
    )
    sys.stdout.flush()


def _phase(msg: str) -> None:
    if _QUIET_STDOUT:
        return
    print(f"\n--- {msg} ---", flush=True)


# Module-level flag: when a terminal dashboard owns the screen, suppress the
# CLI banner/progress writes so the two never fight over stdout.
_QUIET_STDOUT = False



def _resolve_backend(args):
    """Return (backend, ultra, live) or raise a clear error."""
    if args.mock:
        return (
            LMStudioBackend(base_url=args.base_url, model=args.model,
                            transport=MockTransport(),
                            disable_thinking=args.no_thinking),
            FakeUltraSmall(),
            False,
        )
    ultra = UltraSmallDrone(confidence_mode=args.confidence)
    ultra._ensure_loaded()
    backend = LMStudioBackend(base_url=args.base_url, model=args.model,
                              disable_thinking=args.no_thinking)
    if not backend.health():
        raise RuntimeError(
            f"LM Studio not reachable at {args.base_url}. Start it with a model "
            "loaded, or pass --mock to run offline."
        )
    return backend, ultra, True


def _pid_alive(pid: int) -> bool:
    try:
        import psutil

        return psutil.pid_exists(pid)
    except Exception:  # pragma: no cover
        return True  # conservative: assume alive if we cannot tell


def _acquire_run_lock(run_dir) -> bool:
    """Refuse to start if another live process already owns this run dir.

    Self-healing: a lock left by a crashed/killed process (dead PID) is
    overwritten. A lock held by this same PID (resume in-process) is allowed.
    """
    lock_path = run_dir / "run.lock"
    if lock_path.exists():
        try:
            old = int(lock_path.read_text(encoding="utf-8").strip())
        except (ValueError, OSError):
            old = None
        if old is not None and old != os.getpid() and _pid_alive(old):
            return False
    lock_path.write_text(str(os.getpid()), encoding="utf-8")
    return True


def _run_conversations(hive, conversations, max_turns, conversation_id=None,
                       show_progress=True, resume=None, checkpoint_path=None,
                       checkpoint_every=10, run_args=None, dashboards=None):
    """Run conversations through the hive, optionally resuming a prior run.

    ``resume`` (dict) carries ``conv_index`` (0-based index of the conversation
    being processed), ``turn_index`` (user turns already done inside it),
    ``records`` (completed conversation records), ``current_record`` (the partial
    record of the conversation in progress, if any), and ``done`` (total turns).
    A checkpoint is written every ``checkpoint_every`` turns and at each
    conversation boundary when ``checkpoint_path`` is set, so a crashed run can
    be restarted in place without losing the partial conversation.
    """
    start_conv = resume.get("conv_index", 0) if resume else 0
    start_turn = resume.get("turn_index", 0) if resume else 0
    records = list(resume.get("records", [])) if resume else []
    current_record = resume.get("current_record") if resume else None
    done = resume.get("done", 0) if resume else 0

    total = _total_user_turns(conversations, max_turns)
    start = time.time()
    conv_total = len(conversations)
    turn_times = deque(maxlen=20)  # rolling window of recent turn durations (ms)

    def save_checkpoint(conv_index, turn_index, rec):
        if checkpoint_path is None:
            return
        checkpoint_path.write_text(json.dumps({
            "version": 1,
            "run_id": hive.run_id,
            "config": hive.config.to_dict(),
            "store": hive.store.to_dict(),
            "hive_turn": hive.turn,
            "progress": {
                "conv_index": conv_index,
                "turn_index": turn_index,
                "records": records,
                "current_record": rec,
                "done": done,
            },
            "run_args": run_args or {},
        }, indent=2, default=str), encoding="utf-8")

    for ci in range(start_conv, conv_total):
        conv = conversations[ci]
        if ci == start_conv and current_record is not None:
            conv_record = current_record
        else:
            conv_record = {
                "conversation_id": conv.get("conversation_id", "unknown"),
                "profile": conv.get("profile", "unknown"),
                "turns": [],
            }
        n_user = sum(1 for t in conv.get("turns", []) if t.get("role") == "user")
        cap = min(n_user, max_turns) if max_turns else n_user
        turn_count = 0
        for td in conv.get("turns", []):
            if td.get("role") != "user":
                continue
            turn_count += 1
            if max_turns and turn_count > max_turns:
                break
            if ci == start_conv and turn_count <= start_turn:
                continue  # already processed before the checkpoint
            res = hive.process_turn(td["content"], conversation_id=conversation_id)
            conv_record["turns"].append({
                "turn": res.turn,
                "query": res.query,
                "reply": res.reply,
                "assembled_content": res.assembled.content if res.assembled else "",
                "mode": res.mode,
                "pes": res.pes,
                "degradation_level": res.degradation_level,
                "timings": res.timings,
                "token_count": res.assembled.token_count if res.assembled else 0,
                "budget": res.assembled.budget if res.assembled else 0,
                "completion_tokens": (
                    (getattr(hive.backend, "last_usage", {}) or {}).get("completion_tokens")
                ),
            })
            done += 1
            # ETA from the median of recent turn durations (robust to outlier
            # slow/sleep-contaminated turns and responsive to current speed),
            # rather than the run-wide average which stays pinned high by the
            # early turns and barely moves.
            turn_times.append(res.timings.get("total_ms", 0.0))
            eta_s = statistics.median(turn_times) / 1000.0 * (total - done)
            for d in (dashboards or []):
                d.update_progress(
                    done, total, time.time() - start, eta_s,
                    ci + 1, conv_total, turn_count, cap,
                )
                d.add_turn(
                    td["content"], res.reply, res.pes,
                    res.timings.get("generation_ms", 0.0),
                    res.timings.get("total_ms", 0.0),
                )
            if checkpoint_path is not None and done % checkpoint_every == 0:
                save_checkpoint(ci, turn_count, conv_record)
            if show_progress:
                _print_progress(done, total, start, eta=eta_s,
                                conv=ci + 1, conv_total=conv_total, turn=turn_count, cap=cap)
                if res.error:
                    print(f"  ! turn {res.turn} ERROR: {res.error}", file=sys.stderr)
                elif not (res.reply or "").strip():
                    print(f"  ! turn {res.turn} WARNING: empty reply", file=sys.stderr)
                elif res.mode == "fifo_fallback":
                    print(
                        f"  ! turn {res.turn} WARNING: fifo fallback "
                        f"(degradation level {res.degradation_level})",
                        file=sys.stderr,
                    )
        records.append(conv_record)
        # boundary checkpoint: this conversation is now complete in the record
        save_checkpoint(ci + 1, 0, None)
        current_record = None
    if show_progress:
        print()
    return records


def _aggregate(records):
    turns = [t for c in records for t in c["turns"]]
    if not turns:
        return {}
    peps = [t["pes"] for t in turns]
    lat = [t["timings"].get("total_ms", 0.0) for t in turns]
    return {
        "user_turns": len(turns),
        "conversations": len(records),
        "avg_pes": round(sum(peps) / len(peps), 2),
        "min_pes": round(min(peps), 2),
        "avg_total_ms": round(sum(lat) / len(lat), 2),
        "fifo_fallbacks": sum(1 for t in turns if t["mode"] == "fifo_fallback"),
        "drift_events": sum(1 for t in turns if t["degradation_level"] >= 1),
    }


def _populate_ground_truth(db, records, oracle, logger=None, push=None, workers=1):
    """Record routing decisions and oracle labels from a run.

    The LLM-as-judge label calls are independent and dominate this phase's wall
    time, so they run concurrently on a thread pool (``workers`` > 1). Results
    are collected back on the calling thread and written to the DB there, since
    the SQLite connection is single-threaded. Concurrency only helps if the
    backend serves parallel slots (llama.cpp ``-np`` / LM Studio parallel
    requests); with a single slot the requests queue (harmless).
    """
    for conv in records:
        for t in conv["turns"]:
            query = t["query"]
            route = DroneRouter().route(query)
            db.record_routing_decision(t["turn"], _router_score(route), route.route_to)
    # oracle labels on a sample of assembled turns
    sampled = [t for c in records for t in c["turns"]][::10]
    if push is not None:
        push("set_phase", f"ground truth labeling ({len(sampled)} sampled)")
    if not sampled:
        return

    def eval_one(t):
        try:
            label = oracle.evaluate_turn(
                TurnRecord(
                    turn=t["turn"],
                    assembled_context=t.get("assembled_content") or "",
                    user_query=t["query"], llm_response=t["reply"], chunk_ids=[],
                )
            )
            return label, None
        except Exception as exc:  # noqa: BLE001 — one bad oracle call must not kill the run
            return None, str(exc)

    if workers > 1 and len(sampled) > 1:
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=workers) as pool:
            results = list(pool.map(eval_one, sampled))
    else:
        results = [eval_one(t) for t in sampled]

    for i, (t, (label, err)) in enumerate(zip(sampled, results), 1):
        if err is not None:
            if logger is not None:
                logger.log("oracle", "label_failed",
                           {"turn": t["turn"], "error": err[:200]})
            if push is not None:
                push("add_line", f"oracle {i}/{len(sampled)} turn {t['turn']}: FAILED ({err})")
            # Always surface on stderr so a label failure is visible even with no
            # terminal dashboard / captured stdout.
            print(f"  oracle {i}/{len(sampled)} turn {t['turn']}: label FAILED ({err})",
                  file=sys.stderr)
            continue
        # if no chunk ids, record one aggregate label per sampled turn
        db.record_oracle_label(t["turn"], "aggregate", True,
                               label.context_sufficient, label.sufficiency_score)
        if push is not None:
            push("add_line", f"oracle {i}/{len(sampled)} turn {t['turn']}: "
                             f"sufficient={label.context_sufficient} score={label.sufficiency_score}")
        print(f"  oracle {i}/{len(sampled)} turn {t['turn']}: "
              f"sufficient={label.context_sufficient} score={label.sufficiency_score}",
              file=sys.stderr)


def _router_score(decision) -> float:
    try:
        return len(decision.reason.split(",")) if decision.reason else 0
    except Exception:  # noqa: BLE001
        return 0


def _compute_post_run_pes(records, db):
    """Post-run Pipeline Efficiency Score from ground-truth + measured metrics.

    The per-turn in-process PES (``Hive.process_turn``) only sees latency and
    context utilization, so in live runs it floors near zero (the paper's
    LatencyHealth is ms-calibrated and live generation is seconds). This computes
    the paper's real PES after the run, when oracle retrieval/routing metrics
    exist. Components that are genuinely unavailable are dropped (the scorer
    renormalizes). Returns a dict, or None with no turns.
    """
    from cortex.efficiency import (
        EfficiencyScorer,
        context_utilization,
        latency_health,
        throughput_health,
    )

    turns = [t for c in records for t in c["turns"]]
    if not turns:
        return None
    n = len(turns)
    avg_total_ms = sum(t["timings"].get("total_ms", 0.0) for t in turns) / n
    avg_util = sum(t["token_count"] / max(t["budget"], 1) for t in turns) / n
    completions = [t.get("completion_tokens") for t in turns if t.get("completion_tokens")]
    total_gen_s = sum(t["timings"].get("generation_ms", 0.0) for t in turns) / 1000.0
    actual_tps = sum(completions) / total_gen_s if completions and total_gen_s > 0 else None
    baseline_tps = 30.0

    has_labels = db.label_count() > 0
    precision = db.retrieval_precision() if has_labels else None
    routing = db.routing_accuracy()

    result = EfficiencyScorer().compute(
        retrieval_precision=precision,
        routing_accuracy=routing,
        avg_latency_ms=avg_total_ms,
        actual_tps=actual_tps,
        baseline_tps=baseline_tps,
        budget_used=avg_util,
        budget_total=1.0,
    )
    notes = []
    if latency_health(avg_total_ms) == 0 and avg_total_ms > 200:
        notes.append(
            f"LatencyHealth floors at 0 because the paper's formula is ms-calibrated "
            f"(50ms=100, 200ms=0) and live turns average ~{avg_total_ms / 1000:.0f}s; "
            "the PES below is therefore driven by retrieval/routing/throughput/utilization."
        )
    return {
        "pes": result.composite,
        "band": result.band,
        "breakdown": result.breakdown,
        "active_components": result.active_components,
        "components": {
            "retrieval_precision": round(precision, 1) if precision is not None else None,
            "routing_accuracy": round(routing, 1),
            "latency_health": round(latency_health(avg_total_ms), 1),
            "throughput_health": round(throughput_health(actual_tps, baseline_tps), 1) if actual_tps else None,
            "context_utilization": round(context_utilization(avg_util, 1.0), 1),
        },
        "measurements": {
            "turns": n,
            "avg_total_ms": round(avg_total_ms, 1),
            "avg_context_utilization_pct": round(avg_util * 100.0, 1),
            "actual_tps": round(actual_tps, 1) if actual_tps else None,
            "baseline_tps": baseline_tps,
            "oracle_labels": db.label_count(),
        },
        "notes": notes,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Hive data-generation workflow")
    parser.add_argument("--live", action="store_true", help="use real LM Studio + all-MiniLM")
    parser.add_argument("--mock", action="store_true", help="offline (fake drone + mock backend)")
    parser.add_argument("--conversations", default="tests/fixtures/generated")
    parser.add_argument("--max-convs", type=int, default=50)
    parser.add_argument("--max-turns", type=int, default=None)
    parser.add_argument("--base-url", default="http://localhost:1234")
    parser.add_argument("--model", default="")
    parser.add_argument("--pinned-prefix", default=DEFAULT_PINNED_PREFIX)
    parser.add_argument("--output", default="")
    parser.add_argument("--protocol", action="store_true", help="also run P1-P10")
    parser.add_argument("--baselines", action="store_true", help="also run baseline harnesses")
    parser.add_argument(
        "--max-tokens", type=int, default=None,
        help="cap E2E reply length (default: uncapped, 4096 ceiling). On reasoning "
             "models a small cap yields empty replies unless --no-thinking is used",
    )
    parser.add_argument(
        "--baseline-max-tokens", type=int, default=None,
        help="cap baseline reply length (baseline tps is a decode measurement)",
    )
    parser.add_argument(
        "--checkpoint-every", type=int, default=10,
        help="write a resume checkpoint every N turns",
    )
    parser.add_argument(
        "--resume", default="",
        help="resume a run directory (reads its checkpoint.json)",
    )
    parser.add_argument(
        "--term", action="store_true",
        help="render a live dashboard inside the terminal (ANSI, no deps)",
    )
    parser.add_argument(
        "--confidence", choices=("mcdropout", "single", "off"), default="mcdropout",
        help="drone confidence mode; 'off' skips the MC-dropout passes (the stock "
             "embedding model yields confidence ~1.0 regardless)",
    )
    parser.add_argument(
        "--oracle-workers", type=int, default=4,
        help="parallel workers for the ground-truth oracle-labeling phase. "
             "Only speeds up when the backend serves parallel slots (llama.cpp "
             "-np / LM Studio parallel requests); single-slot servers queue "
             "the requests (harmless, no speedup)",
    )
    parser.add_argument(
        "--no-thinking", action="store_true",
        help="send enable_thinking=false with every request, so reasoning models "
             "skip chain-of-thought (faster, and reply caps stop yielding empty "
             "output). Combine with LM Studio's 'thinking' toggle for models that "
             "only honor the GUI setting.",
    )
    args = parser.parse_args(argv)

    # Which options did the user pass explicitly? Used by --resume so it only
    # restores checkpoint parameters the user did NOT override on the command
    # line (e.g. a "--resume X --max-convs 3" run must stay at 3, not silently
    # resume the original run's max_convs).
    _defaults = vars(parser.parse_args([]))
    explicit = {k for k, v in vars(args).items() if v != _defaults.get(k)}

    if not args.live and not args.mock:
        args.mock = True  # default to offline for safety
    if args.live and args.mock:
        print("pass either --live or --mock, not both")
        return 2
    if args.resume and args.output:
        print("--output is ignored when --resume is given")
        args.output = ""

    # --- resume: restore run parameters from the checkpoint ---
    resume_ckpt = None
    if args.resume:
        rdir = Path(args.resume)
        ckpt_path = rdir / "checkpoint.json"
        if not ckpt_path.exists():
            print(f"no checkpoint found at {ckpt_path}")
            return 3
        resume_ckpt = json.loads(ckpt_path.read_text(encoding="utf-8"))
        ra = resume_ckpt.get("run_args", {})
        for key in ("conversations", "max_convs", "max_turns", "base_url",
                    "model", "pinned_prefix", "confidence", "no_thinking"):
            if ra.get(key) is not None and key not in explicit:
                setattr(args, key, ra[key])
        if "mode" in ra and "live" not in explicit and "mock" not in explicit:
            if ra["mode"] == "mock":
                args.mock, args.live = True, False
            elif ra["mode"] == "live":
                args.mock, args.live = False, True
        print(f"resuming run {resume_ckpt.get('run_id')} from {args.resume}")
        print(f"  conversation {resume_ckpt['progress'].get('conv_index', 0) + 1}, "
              f"turn {resume_ckpt['progress'].get('turn_index', 0)} "
              f"({resume_ckpt['progress'].get('done', 0)} turns done)")
        overridden = [k for k in ("conversations", "max_convs", "max_turns",
                                  "base_url", "model", "confidence", "no_thinking")
                      if k not in explicit and ra.get(k) is not None]
        if overridden:
            print("  restored from checkpoint:", ", ".join(sorted(overridden)))

    if args.protocol and args.max_tokens:
        print("note: --max-tokens caps only the E2E phase; P1-P10 generations are uncapped")

    # --- run directory ---
    run_dir = (Path(args.resume) if args.resume else
               (Path(args.output) if args.output else
                Path("runs") / datetime.now().strftime("%Y%m%d_%H%M%S")))
    log_dir = run_dir / "logs"
    (run_dir / "labels").mkdir(parents=True, exist_ok=True)
    logger = EventLogger(log_dir=log_dir)

    # --- backend + hive ---
    try:
        backend, ultra, live = _resolve_backend(args)
    except RuntimeError as exc:
        logger.close()
        print(str(exc))
        return 3

    if not _acquire_run_lock(run_dir):
        logger.close()
        print(f"run directory {run_dir} is already in use by a live process (run.lock)")
        return 3

    config = (HiveConfig.from_dict(resume_ckpt["config"]) if resume_ckpt
              else HiveConfig(max_tokens=args.max_tokens))
    hive = Hive(
        config=config,
        ultra=ultra,
        medium=MediumDrone(score_pair_fn=lambda q, c: 0.5),
        backend=backend,
        logger=logger,
        pinned_prefix=args.pinned_prefix,
    )
    if resume_ckpt:
        hive.run_id = resume_ckpt["run_id"]
        hive.store = ContextStore.from_dict(
            resume_ckpt["store"], embed_fn=hive.ultra.embed
        )
        hive.turn = resume_ckpt["hive_turn"]

    # Pinned prefix must reach the backend as a byte-stable leading system
    # message for llama.cpp automatic prefix caching (see KVCacheManager).
    if args.pinned_prefix and getattr(backend, "pinned_prefix", None) is None:
        print("warning: backend does not expose pinned_prefix; prefix caching disabled")

    # --- live terminal dashboard + keep-awake (optional) ---
    global _QUIET_STDOUT
    term = TermDashboard(enabled=args.term)
    if term.enabled:
        print("dashboard: terminal dashboard enabled (ANSI)")
    _QUIET_STDOUT = term.enabled  # terminal dashboard owns stdout banners
    dashboards = [d for d in (term,) if d.enabled]

    def push(method: str, *a, **kw) -> None:
        for d in dashboards:
            getattr(d, method)(*a, **kw)

    keep_awake = KeepAwake() if live else None
    if keep_awake is not None and keep_awake.active:
        print("keep-awake: system sleep disabled for this run (ES_SYSTEM_REQUIRED)")

    # --- run ---
    conversations = load_conversations(args.conversations)[: args.max_convs]
    run_args = {
        "mode": "live" if live else "mock",
        "conversations": args.conversations,
        "max_convs": args.max_convs,
        "max_turns": args.max_turns,
        "base_url": args.base_url,
        "model": args.model,
        "pinned_prefix": args.pinned_prefix,
        "confidence": args.confidence,
        "no_thinking": args.no_thinking,
        "conversation_id": run_dir.name,
    }
    _phase(f"1/3 E2E conversations ({len(conversations)} convs, ~{_total_user_turns(conversations, args.max_turns)} turns)")
    push("set_phase", f"1/3 E2E ({len(conversations)} convs)")
    records = _run_conversations(
        hive, conversations, args.max_turns, conversation_id=run_dir.name,
        resume=resume_ckpt.get("progress") if resume_ckpt else None,
        checkpoint_path=run_dir / "checkpoint.json",
        checkpoint_every=args.checkpoint_every,
        run_args=run_args,
        dashboards=dashboards,
        show_progress=not term.enabled,
    )

    # --- ground truth ---
    db_path = run_dir / "ground_truth.sqlite"
    db = GroundTruthDB(db_path)
    oracle = AsyncOracle(generate_fn=_mock_oracle if args.mock else _live_oracle(backend))
    _populate_ground_truth(db, records, oracle, logger=logger, push=push,
                           workers=args.oracle_workers)
    post_run_pes = _compute_post_run_pes(records, db)
    has_labels = db.label_count() > 0
    ground_truth_metrics = {
        "oracle_labels": db.label_count(),
        "retrieval_precision": round(db.retrieval_precision(), 1) if has_labels else None,
        "retrieval_recall": round(db.retrieval_recall(), 1) if has_labels else None,
        "false_eviction_rate": round(db.false_eviction_rate(), 1) if has_labels else None,
        "routing_accuracy": round(db.routing_accuracy(), 1),
    }
    db.close()

    # --- protocol + baselines (optional) ---
    protocol_report = None
    baseline_report = None
    if args.protocol:
        _phase("2/3 P1-P10 protocol")
        push("set_phase", "2/3 P1-P10 protocol")
        from experiments.run_p1_p10 import PredictionSuite, _load_labels

        labels = _load_labels(conversations)
        suite = PredictionSuite(backend, ultra, MediumDrone(score_pair_fn=lambda q, c: 0.5),
                                conversations, labels, oracle, live=live)
        protocol_report = [r.__dict__ for r in suite.run()]
        for r in protocol_report:
            push("add_line", f"P{r['id']} {r['status']}: {r['title']}")
    if args.baselines:
        _phase("3/3 Baselines (LM Studio rolling + FIFO)")
        push("set_phase", "3/3 Baselines")
        baseline_report = _run_baselines(args, run_dir)
        for name, res in (baseline_report or {}).items():
            push("add_line", f"baseline {name}: exit={res.get('exit')} error={res.get('error', '')}")

    logger.flush()
    logger.close()
    for d in dashboards:
        d.close()
    if keep_awake is not None:
        keep_awake.close()

    # --- report ---
    report = {
        "run_id": hive.run_id,
        "mode": "live" if live else "mock",
        "backend": type(backend).__name__,
        "run_dir": str(run_dir.resolve()),
        "generated_at": datetime.now().astimezone().isoformat(),
        "ground_truth_db": str(db_path.resolve()),
        "event_logs": str(log_dir.resolve()),
        "aggregate": _aggregate(records),
        "ground_truth": ground_truth_metrics,
        "post_run_pes": post_run_pes,
        "conversations": records,
        "protocol": protocol_report,
        "baselines": baseline_report,
    }
    (run_dir / "run_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    # --- summary ---
    agg = report["aggregate"]
    print(f"Run {hive.run_id} ({report['mode']}) -> {run_dir.resolve()}")
    print(f"  conversations : {agg.get('conversations', 0)}  turns: {agg.get('user_turns', 0)}")
    print(f"  PES min/avg   : {agg.get('min_pes')} / {agg.get('avg_pes')}")
    if post_run_pes:
        print(f"  post-run PES : {post_run_pes['pes']} ({post_run_pes['band']})")
    print(f"  avg total ms  : {agg.get('avg_total_ms')}")
    print(f"  fifo fallbacks: {agg.get('fifo_fallbacks')}")
    if protocol_report:
        print(f"  protocol      : {sum(1 for r in protocol_report if r['status'] == 'PASS')}/{len(protocol_report)} PASS")
    print(f"  report        : {(run_dir / 'run_report.json').resolve()}")
    (run_dir / "run.lock").unlink(missing_ok=True)
    return 0


def _mock_oracle(prompt: str) -> str:
    return json.dumps({"sufficient": True, "used_pieces": [], "missing": [], "score": 4})


def _live_oracle(backend):
    def fn(prompt: str) -> str:
        # The oracle is a JSON-evaluation task, not a context-answer task: drop
        # the E2E pinned prefix, frame JSON explicitly, and leave headroom for
        # reasoning tokens (reasoning models spend their budget on CoT first).
        backend.pinned_prefix = ""
        return backend.generate(
            "You are an evaluation engine. Respond with ONLY a valid JSON object.",
            prompt, {"temperature": 0.0, "max_tokens": 2048},
        )
    return fn


def _run_baselines(args, run_dir):
    from cortex.baselines import lm_studio_baseline, fifo_baseline

    out = {}
    for module, name in ((lm_studio_baseline, "lm_studio"), (fifo_baseline, "fifo")):
        base = [
            "--conversations", args.conversations,
            "--base-url", args.base_url, "--model", args.model,
            "--output", str(run_dir / f"baseline_{name}.json"),
        ]
        if args.baseline_max_tokens:
            base += ["--max-tokens", str(args.baseline_max_tokens)]
        try:
            code = module.main(base)
            out[name] = {"exit": code, "output": str(run_dir / f"baseline_{name}.json")}
        except Exception as exc:  # noqa: BLE001
            out[name] = {"error": str(exc)}
    return out


if __name__ == "__main__":
    sys.exit(main())
