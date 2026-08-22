"""Unified Hive orchestrator.

Wires all components (store, drones, router, assembler, health monitor, graceful
degradation, KV-cache, logger) into a single ``process_turn()`` entry point that
mirrors the plan's ``Hive`` used by shadow mode / A-B / rollback. It:

  - checks congestion and applies graceful degradation each turn,
  - assembles context (or falls back to FIFO at emergency level),
  - optionally generates via a backend with a stable pinned prefix,
  - records per-turn latency breakdown, PES, and NDJSON events,
  - grows the context store with each exchange.
"""

from __future__ import annotations

import sys
import time
import uuid
from dataclasses import dataclass, field
from typing import Optional

from backend.cache_manager import KVCacheManager
from cortex.baselines.metrics import estimate_tokens
from cortex.baselines.runner import build_fifo_messages
from cortex.config import HiveConfig
from cortex.degradation import GracefulDegradation
from cortex.efficiency import EfficiencyScorer
from cortex.health import PipelineHealthMonitor
from cortex.routing import DroneRouter, EscalationHandler
from focal.assembly import ContextAssembler
from focal.budget import AdaptiveBudget
from membrane.dedup import ContextDeduplicator
from membrane.drift import TopicDriftDetector
from retention.store import ContextStore
from sieve.medium import MediumDrone
from sieve.ultra_small import UltraSmallDrone


@dataclass
class TurnResult:
    turn: int
    query: str
    reply: str
    assembled: object  # AssembledContext or None (FIFO fallback / error)
    timings: dict = field(default_factory=dict)
    congestion: object = None
    degradation_level: int = 0
    mode: str = "hive"  # hive | fifo_fallback | no_backend | error
    pes: float = 0.0
    error: Optional[str] = None


class Hive:
    def __init__(
        self,
        config: Optional[HiveConfig] = None,
        ultra=None,
        medium=None,
        backend=None,
        logger=None,
        store: Optional[ContextStore] = None,
        pinned_prefix: str = "",
        router=None,
    ) -> None:
        self.config = config or HiveConfig()
        self.ultra = ultra or UltraSmallDrone(
            model_name=self.config.ultra_model,
            vocab_boost=self.config.vocab_boost,
            confidence_mode=self.config.confidence_mode,
        )
        if medium is not None:
            self.medium = medium
        elif self.config.enable_medium:
            self.medium = MediumDrone(model_name=self.config.medium_model)
        else:
            # Medium drone is heavy and VRAM-contending; keep a lightweight
            # placeholder until enabled via HiveConfig.enable_medium.
            self.medium = MediumDrone(score_pair_fn=lambda q, c: 0.5)
        self.backend = backend
        self.logger = logger
        self.pinned_prefix = pinned_prefix
        self.store = store or ContextStore(
            embed_fn=self.ultra.embed, max_chunks=self.config.max_chunks
        )
        self.router = router or DroneRouter()
        self.escalation = EscalationHandler()
        self.dedup = ContextDeduplicator(threshold=self.config.dedup_threshold)
        self.drift = TopicDriftDetector(
            embed_fn=self.ultra.embed, threshold=self.config.drift_threshold
        )
        self.budget = AdaptiveBudget()
        self.assembler = ContextAssembler(collect_timings=True)
        self.monitor = PipelineHealthMonitor(logger=logger)
        self.degradation = GracefulDegradation()
        self.cache = KVCacheManager(backend) if backend is not None else None
        self.scorer = EfficiencyScorer()
        self.run_id = uuid.uuid4().hex[:8]
        self.turn = 0
        self._warned_reasoning_starve = False

    # ------------------------------------------------------------------
    def reset_conversation(self) -> None:
        """Start a fresh conversation context.

        Clears the store and resets the turn counter so one conversation's
        chunks never leak into the next. Without this, a benchmark run over
        many conversations shares one store, so late conversations retrieve
        mostly *other* conversations' chunks (cross-conversation
        contamination) and P2 precision collapses.
        """
        self.store = ContextStore(
            embed_fn=self.ultra.embed, max_chunks=self.config.max_chunks
        )
        self.turn = 0

    # ------------------------------------------------------------------
    def process_turn(
        self, query: str, conversation_id: Optional[str] = None, record_exchange: bool = True
    ) -> TurnResult:
        self.turn += 1

        report = self.monitor.check_congestion()
        self.degradation.update(report)
        level = self.degradation.current_level

        timings: dict = {}
        assembled = None
        reply = ""
        error = None
        token_count = 0
        budget = 0
        try:
            if self.degradation.should_fallback_fifo():
                content = self._fifo_context(query)
                reply = self._generate(content, query, timings)
                token_count = estimate_tokens(content)
                budget = 4096
                mode = "fifo_fallback"
            else:
                t0 = time.perf_counter()
                assembled = self.assembler.assemble(
                    query=query, current_turn=self.turn, store=self.store,
                    router=self.router, ultra_small=self.ultra, medium=self.medium,
                    escalation=self.escalation, dedup=self.dedup,
                    drift_detector=self.drift, budget=self.budget,
                    max_context=self.config.max_context,
                    skip_remembrance=self.degradation.should_skip_remembrance(),
                    skip_dedup=self.degradation.should_skip_dedup(),
                )
                timings.update(self.assembler.last_timings)
                timings["assembly_total_ms"] = round((time.perf_counter() - t0) * 1000.0, 3)
                content = assembled.content
                reply = self._generate(content, query, timings)
                token_count = assembled.token_count
                budget = assembled.budget
                mode = "no_backend" if self.backend is None else "hive"
        except Exception as exc:  # noqa: BLE001
            error = str(exc)
            mode = "error"
            if self.logger is not None:
                self.logger.log(
                    "error", "turn_failed",
                    {"query_hash": query[:40], "error": error, "degradation_level": level},
                    run_id=self.run_id, conversation_id=conversation_id, turn_id=self.turn,
                )

        if record_exchange:
            self.store.add_chunk(self.turn, query)
            if reply:
                self.store.add_chunk(self.turn, reply)

        util = token_count / max(budget, 1)
        pes = self.scorer.compute(
            avg_latency_ms=timings.get("total_ms", 0.0),
            budget_used=token_count, budget_total=budget,
        ).composite

        self._log(query, assembled, report, level, timings, conversation_id)
        return TurnResult(
            turn=self.turn, query=query, reply=reply, assembled=assembled,
            timings=timings, congestion=report, degradation_level=level,
            mode=mode, pes=round(pes, 2), error=error,
        )

    # ------------------------------------------------------------------
    def _fifo_context(self, query: str) -> str:
        history = [{"role": "user", "content": c.content} for c in self.store.all_chunks()]
        msgs = build_fifo_messages(history + [{"role": "user", "content": query}])
        return "\n\n".join(m["content"] for m in msgs)

    def _generate(self, content: str, query: str, timings: dict) -> str:
        if self.backend is None:
            timings["generation_ms"] = 0.0
            timings["total_ms"] = round(
                timings.get("assembly_total_ms", 0.0) + timings["generation_ms"], 3
            )
            return ""
        if self.config.sanitize_context:
            from cortex.sanitize import sanitize_context

            content = sanitize_context(content)
        if self.cache is not None:
            self.cache.update_cache(content, persistent_prefix=self.pinned_prefix)
        t1 = time.perf_counter()
        sampling = None
        if self.config.max_tokens:
            sampling = {"max_tokens": self.config.max_tokens}
        reply = self.backend.generate(content, query, sampling)
        if not (reply or "").strip():
            usage = getattr(self.backend, "last_usage", {}) or {}
            reason_toks = usage.get("completion_tokens_details", {}).get("reasoning_tokens")
            if reason_toks and not self._warned_reasoning_starve:
                # A reasoning model burned its whole output budget on chain-of-thought,
                # so the visible reply is empty. Surface this loudly once per run.
                self._warned_reasoning_starve = True
                if self.logger is not None:
                    self.logger.log(
                        "backend", "empty_reply_reasoning_starved",
                        {"max_tokens": self.config.max_tokens, "reasoning_tokens": reason_toks},
                        run_id=self.run_id,
                    )
                print(
                    "\nWARNING: replies are empty — the loaded model spent its whole "
                    f"output budget on reasoning tokens ({reason_toks}). A small "
                    "--max-tokens cap produces no visible answer on a reasoning model. "
                    "Drop --max-tokens (the default 4096 ceiling lets reasoning "
                    "models reason and answer), or load a non-reasoning model.",
                    file=sys.stderr,
                )
        timings["generation_ms"] = round((time.perf_counter() - t1) * 1000.0, 3)
        timings["total_ms"] = round(
            timings.get("assembly_total_ms", 0.0) + timings["generation_ms"], 3
        )
        return reply

    # ------------------------------------------------------------------
    def _log(self, query, assembled, report, level, timings, conversation_id) -> None:
        if self.logger is None:
            return
        routed_to = assembled.routing_decision.route_to if assembled else "fifo"
        self.logger.log(
            "router", "task_classified",
            {"routed_to": routed_to, "latency_ms": 0},
            run_id=self.run_id, conversation_id=conversation_id, turn_id=self.turn,
        )
        self.logger.log(
            "assembly", "context_assembled",
            {
                "total_tokens": assembled.token_count if assembled else 0,
                "chunk_count": assembled.chunks_used if assembled else 0,
                "budget_used": assembled.token_count if assembled else 0,
                "budget_total": assembled.budget if assembled else 0,
                "degradation_level": level,
                "timings": timings,
            },
            latency_ms=timings.get("total_ms", 0.0),
            run_id=self.run_id, conversation_id=conversation_id, turn_id=self.turn,
        )
        if report.severity != "normal":
            self.logger.log(
                "congestion", "congestion_detected", report.__dict__,
                run_id=self.run_id, conversation_id=conversation_id, turn_id=self.turn,
            )
