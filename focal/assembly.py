"""Context assembly (focal layer) — the full hive pipeline per user turn.

Order (Membrane runs before Retention):
  1. Remembrance pass on chunks approaching deletion.
  2. Route the query and score all chunks via the drone fleet.
  3. Deduplicate semantically similar chunks (Membrane) + refresh decay state.
  4. Detect topic drift; if a reset is flagged, build drift penalties (Membrane).
  5. Apply the decay matrix to the surviving chunks (Retention).
  6. Compute the adaptive budget.
  7. Sort by effective score and select within budget.
  8. Return the assembled context string.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from cortex.baselines.metrics import estimate_tokens
from cortex.routing import RoutingDecision
from retention.decay import DecayMatrix
from retention.remembrance import RemembrancePass


@dataclass
class AssembledContext:
    content: str
    token_count: int
    budget: int
    chunks_used: int
    routing_decision: RoutingDecision
    drift_detected: bool
    selected_chunk_ids: list = field(default_factory=list)


class ContextAssembler:
    def __init__(self, collect_timings: bool = False) -> None:
        self.collect_timings = collect_timings
        self.last_timings: dict = {}

    def _tick(self, name: str, t0: float) -> None:
        if self.collect_timings:
            self.last_timings[name] = round((time.perf_counter() - t0) * 1000.0, 3)

    def assemble(
        self,
        query: str,
        current_turn: int,
        store,
        router,
        ultra_small,
        medium,
        escalation,
        dedup,
        drift_detector,
        budget,
        max_context: int = 8192,
        skip_remembrance: bool = False,
        skip_dedup: bool = False,
    ) -> AssembledContext:
        # 1. Remembrance pass
        if not skip_remembrance:
            _t = time.perf_counter()
            deletion_candidates = store.get_deletion_candidates()
            current_topic = query
            remembrance_results = RemembrancePass().process(
                deletion_candidates, current_topic, ultra_small
            )
            self._tick("remembrance_ms", _t)

        # 2. Route + score all chunks
        _t = time.perf_counter()
        routing = router.route(query, store.get_turns())
        all_contents = store.all_contents()
        all_chunks = store.all_chunks()
        if routing.route_to == "escalation":
            scores = escalation.process(query, all_contents, ultra_small, medium)
        elif routing.route_to == "medium":
            scores = medium.score(query, all_contents)
        else:
            scores = ultra_small.score(query, all_contents)
        self._tick("scoring_ms", _t)

        raw_scores = {}
        for i, s in enumerate(scores):
            if i < len(all_chunks):
                raw_scores[all_chunks[i].id] = s.relevance_score

        # 3. Deduplicate (Membrane first) + refresh decay state
        _t = time.perf_counter()
        embeddings = store.all_embeddings()
        if not skip_dedup:
            surviving, refresh_map = dedup.deduplicate(all_chunks, embeddings)
            store.apply_refresh(refresh_map)
        else:
            surviving, refresh_map = all_chunks, {}
        self._tick("dedup_ms", _t)

        # 4. Topic drift -> drift penalties
        _t = time.perf_counter()
        recent = store.get_recent_chunks(3)
        drift = drift_detector.check(recent, all_chunks, ultra_small)
        drift_penalties = {}
        if drift.should_reset:
            drift_penalties = self._apply_drift_reset(surviving, recent)
        self._tick("drift_ms", _t)

        # 5. Decay on surviving chunks (Retention)
        _t = time.perf_counter()
        effective = DecayMatrix().apply(surviving, current_turn, raw_scores, drift_penalties)
        self._tick("decay_ms", _t)

        # 6. Budget
        high_relevance = sum(1 for v in effective.values() if v > 0.6)
        token_budget = budget.compute(routing.route_to, high_relevance, max_context)

        # 7. Sort + select within budget
        _t = time.perf_counter()
        scored = sorted(effective.items(), key=lambda kv: kv[1], reverse=True)
        selected = self._select_within_budget(scored, store, token_budget)
        self._tick("select_ms", _t)

        return AssembledContext(
            content=self._format_context(selected),
            token_count=self._count_tokens(selected),
            budget=token_budget,
            chunks_used=len(selected),
            routing_decision=routing,
            drift_detected=drift.should_reset,
            selected_chunk_ids=[c.id for c in selected],
        )

    def _select_within_budget(self, scored, store, token_budget: int) -> list:
        selected = []
        used = 0
        for cid, _score in scored:
            chunk = store.chunks.get(cid)
            if chunk is None:
                continue
            cost = estimate_tokens(chunk.content)
            if used + cost > token_budget:
                continue
            selected.append(chunk)
            used += cost
        return selected

    @staticmethod
    def _format_context(selected: list) -> str:
        return "\n\n".join(c.content for c in selected)

    @staticmethod
    def _count_tokens(selected: list) -> int:
        return sum(estimate_tokens(c.content) for c in selected)

    @staticmethod
    def _apply_drift_reset(surviving: list, recent: list) -> dict[str, float]:
        recent_ids = {c.id for c in recent}
        return {c.id: (1.0 if c.id in recent_ids else 0.1) for c in surviving}
