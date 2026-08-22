# Hive Memory Architecture — Implementation Plan

Pass this file to any AI (or human) with file access to the repo. It is the executable plan
for building a **Targeted-Masking Drone Hive & Managed Decay Memory Architecture** — an
external, multi-agent context management layer that sits between the user and the primary
LLM inference backend. The system filters, compresses, and curates context before it reaches
the generative model, enabling consistent performance over arbitrarily long conversations
on consumer hardware.

**Relationship to Gatekeeper Studio:** Gatekeeper already operates a coder/auditor drone
pipeline (`coder_auditor_loop.ps1`, `orchestrate_audit.ps1`, confidence-gated audits). The
Hive Memory Architecture is a **complementary infrastructure layer** that manages what the
existing drones *see* — their context window. It does not replace the existing drone system;
it enhances it by ensuring the coder and auditor drones always receive the most relevant,
compressed context regardless of conversation length.

**Core principle:** measure everything, optimize from data. Every component ships with
logging, metrics, and tests. No parameter (decay multiplier, routing threshold, context
budget) is set by intuition — all are tuned against logged evidence.

---

## 1. Architecture Overview

### 1.1 The Pipeline (data flow per user turn)

```
User Input
    │
    ▼
┌─────────────────────────────────────────┐
│  HIVE CONTROLLER (Python orchestration) │
│                                         │
│  1. Classify task complexity            │
│  2. Route to appropriate drone tier     │
│  3. Scan full conversation history      │
│  4. Score relevance per context chunk   │
│  5. Run remembrance pass on evictions   │
│  6. Deduplicate + compress              │
│  7. Detect topic drift                  │
│  8. Apply decay matrix                  │
│  9. Assemble context (adaptive budget)  │
│ 10. Check for congestion                │
└─────────────────────────────────────────┘
    │
    ▼
┌──────────────────────┐    ┌──────────────────────┐
│  ULTRA-SMALL DRONE   │    │    MEDIUM DRONE      │
│  (~80MB, 5ms/query)  │    │   (~400MB, 20-50ms)  │
│  all-MiniLM-L6-v2    │    │   CodeBERT/DeBERTa   │
│  Fast semantic sim   │    │   Domain-aware       │
│  Confidence output   │    │   Validates uncertain │
└──────────────────────┘    └──────────────────────┘
    │                              │
    ▼                              ▼
┌─────────────────────────────────────────┐
│  CONTEXT ASSEMBLY                       │
│  - Confidence-weighted sorting          │
│  - Deduplication (same concept, keep    │
│    densest version)                     │
│  - Adaptive token budget (1k–6k)        │
│  - Topic-drift reset if needed          │
└─────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────┐
│  LLM BACKEND (vLLM / LM Studio)         │
│  - PagedAttention KV-cache             │
│  - Surgical context insertion           │
│  - No full recomputation on edits       │
└─────────────────────────────────────────┘
    │
    ▼
  Response to User
    │
    ▼
┌─────────────────────────────────────────┐
│  ASYNC ORACLE (background, offline)     │
│  - Evaluates: "did the model have       │
│    enough context?"                     │
│  - Labels relevance ground truth        │
│  - Updates decay parameters             │
│  - Zero impact on user-facing latency   │
└─────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────┐
│  LOGGING & METRICS                      │
│  - Every decision logged (NDJSON)       │
│  - Pipeline Efficiency Score (0–100)    │
│  - Congestion detection                 │
│  - Weekly optimization reports          │
└─────────────────────────────────────────┘
```

### 1.2 Component Glossary

| Component | Role | Latency Budget | Size |
|---|---|---|---|
| Ultra-Small Drone | Fast semantic similarity, confidence scoring | 5ms/query, CPU | ~80MB |
| Medium Drone | Domain-aware validation of uncertain chunks | 20–50ms/query, GPU | ~400MB |
| Hive Controller | Orchestration, routing, decay, assembly | <10ms overhead | Script |
| LLM Backend | Generation (Qwen 3.8-27B or similar) | User-facing | 12–24GB VRAM |
| Async Oracle | Ground truth labeling, parameter tuning | Offline, batch | Same as LLM |
| Logging Layer | Decision tracking, metrics, alerts | <1ms/event | Disk I/O |
| Efficiency Scorer | Composite pipeline health score (0–100) | <1ms/calc | Script |
| Congestion Detector | Queue depth + latency spike monitoring | <1ms/poll | Script |

---

## 2. S0 — Foundation: Logging Infrastructure, Test Harness & Baseline

**Goal:** build the measurement infrastructure before any hive component. Every subsequent
section produces data that this foundation captures. Without this, all optimization is
guesswork.

### 2.1 Tasks

**S0.1 — Logging Layer**

Create `hive/logs/` with an NDJSON event logger. Every component writes structured events:

```python
# hive/logs/event_logger.py
class EventLogger:
    """
    Appends one JSON line per event to a daily-rotated NDJSON file.
    Schema: {ts, component, event_type, payload, latency_ms}
    """
    def log(self, component: str, event_type: str, payload: dict, latency_ms: float = 0):
        entry = {
            "ts": datetime.utcnow().isoformat() + "Z",
            "component": component,
            "event_type": event_type,
            "payload": payload,
            "latency_ms": round(latency_ms, 2)
        }
        self._append(entry)
```

Event types per component:

| Component | event_type | payload keys |
|---|---|---|
| router | task_classified | query_hash, complexity_score, routed_to, latency_ms |
| ultra_small | relevance_scored | chunk_ids[], scores[], confidence[], latency_ms |
| medium | uncertainty_validated | chunk_ids[], original_scores[], revised_scores[], latency_ms |
| decay | decay_applied | chunk_id, previous_score, new_score, multiplier, age_turns |
| remembrance | remembrance_pass | chunk_id, relevance_at_eviction, topic_similarity, saved: bool |
| assembly | context_assembled | total_tokens, chunk_count, chunk_ids[], budget_used, budget_total |
| drift | topic_drift_detected | current_topic_hash, drift_score, threshold, context_reset: bool |
| congestion | congestion_detected | queue_depth, avg_latency_ms, spike_factor, action_taken |
| oracle | relevance_labeled | chunk_id, predicted_relevant: bool, actually_relevant: bool, turn_gap |
| efficiency | score_computed | composite_score, breakdown: {retrieval, routing, latency, throughput} |

**S0.2 — Efficiency Scoring System**

Define the **Pipeline Efficiency Score (PES)** — a composite 0–100 metric that indicates
overall system health. Computed after every N turns (configurable, default every 5 turns).

```
PES = w1 × RetrievalPrecision
    + w2 × RoutingAccuracy
    + w3 × LatencyHealth
    + w4 × ThroughputHealth
    + w5 × ContextUtilization
```

**Default weights (tunable from logged data):**
- w1 = 0.30 (RetrievalPrecision: % of retrieved context that was actually used)
- w2 = 0.20 (RoutingAccuracy: % of correct drone routing decisions)
- w3 = 0.20 (LatencyHealth: 100 - normalized_latency_ms, where 50ms = 100, 200ms = 0)
- w4 = 0.15 (ThroughputHealth: actual_tokens_per_sec / baseline_tokens_per_sec × 100)
- w5 = 0.15 (ContextUtilization: budget_used / budget_total × 100, penalize <60% and >95%)

**Alert thresholds:**
- PES ≥ 80: GREEN (system healthy)
- PES 60–79: YELLOW (investigate; log a warning event)
- PES < 60: RED (trigger shadow-mode A/B test of last known good config)

**S0.3 — Congestion Detection**

Monitor three signals, each with independent thresholds:

1. **Queue depth:** number of context chunks waiting for drone processing.
   - Normal: 0–5
   - Warning: 6–15 (trigger batching optimization)
   - Critical: >15 (trigger graceful degradation — skip medium drone, use ultra-small only)

2. **Latency spikes:** rolling average of last 10 drone query latencies.
   - Normal: <20ms average
   - Warning: 20–100ms average (log; investigate GPU contention)
   - Critical: >100ms average (trigger fallback to cached embeddings)

3. **Processing backlog:** time since last context assembly completed vs. time since user input.
   - Normal: assembly completes before user's next message arrives
   - Warning: assembly still running when next message arrives (queue the message)
   - Critical: 2+ messages queued (trigger aggressive context compression)

**Congestion response actions (escalating):**
1. Batch similar chunks (cluster by embedding similarity, process clusters not individuals)
2. Skip medium drone for low-confidence chunks (accept ultra-small scores)
3. Reduce context budget to 2k tokens (aggressive compression)
4. Fall back to truncation (FIFO) with a warning event logged

**S0.4 — Test Harness Scaffold**

Create `hive/tests/` with the test runner and fixture infrastructure:

```
hive/tests/
├── run_hive_tests.py          # Entry point, runs all test stages
├── fixtures/
│   ├── synthetic_conversations/  # 50+ labeled conversations
│   ├── ground_truth_labels.json  # Manually/LLM-labeled relevance data
│   └── expected_outputs/         # Baseline outputs for regression
├── unit/
│   ├── test_logger.py
│   ├── test_efficiency.py
│   ├── test_congestion.py
│   └── test_routing.py
├── integration/
│   ├── test_drone_ultra_small.py
│   ├── test_drone_medium.py
│   ├── test_context_assembly.py
│   └── test_pipeline_e2e.py
└── benchmarks/
    ├── bench_latency.py
    └── bench_throughput.py
```

**S0.5 — Baseline Measurements**

Before building any hive component, measure the baseline:

1. **LM Studio baseline (no hive):** Run 20 test conversations through LM Studio + Qwen3.6
   with default rolling context. Record: tokens/sec, context window utilization, task
   completion rate (human-judged), OOM events. This is the "before" measurement.

2. **Naive truncation baseline:** Run the same 20 conversations with a simple FIFO
   truncation to 4k tokens. Record the same metrics.

3. **Record baseline PES:** Compute what the PES would be for both baselines (most metrics
   will be 0 or N/A, but latency and throughput are measurable).

### 2.2 Definition of Done
- Event logger writes valid NDJSON (parse test: every line is valid JSON with required fields).
- PES computation produces a 0–100 score with correct weight application.
- Congestion detector fires at correct thresholds (unit tests with synthetic data).
- Test harness runs all stages and reports pass/fail with exit code 0/1.
- Baseline measurements recorded in `hive/logs/baseline_*.json`.
- Full conversation history logging works for 100+ turns without file corruption.
- All unit tests pass: `python hive/tests/run_hive_tests.py` → exit 0.

---

## 3. S1 — Drone Fleet: Ultra-Small + Medium Drones & Routing

**Goal:** implement the two-tier drone system with heuristic routing. By the end of this
section, the system can analyze context relevance using both drones and route between them
based on task complexity.

### 3.1 Tasks

**S1.1 — Ultra-Small Drone (all-MiniLM-L6-v2)**

Load and wrap the ultra-small embedding model:

```python
# hive/drones/ultra_small.py
class UltraSmallDrone:
    """
    Wraps sentence-transformers/all-MiniLM-L6-v2 (384-dim, ~80MB, CPU-capable).
    Produces:
      - relevance_score: cosine similarity between query embedding and chunk embedding
      - confidence: prediction variance (run 3x with dropout, measure std dev)
    """
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.model.eval()

    def score(self, query: str, chunks: list[str]) -> list[ChunkScore]:
        """
        Returns list of ChunkScore(chunk_id, relevance_score, confidence).
        Confidence = 1.0 - normalized_std_dev across 3 forward passes with dropout.
        """
        query_emb = self.model.encode(query, convert_to_numpy=True)
        chunk_embs = self.model.encode(chunks, convert_to_numpy=True)
        scores = cosine_similarity([query_emb], chunk_embs)[0]

        confidences = []
        for chunk in chunks:
            passes = [self.model.encode(chunk, convert_to_numpy=True) for _ in range(3)]
            std = np.std(passes, axis=0).mean()
            confidence = max(0, 1.0 - std / 0.1)  # normalize; 0.1 = max expected std
            confidences.append(confidence)

        return [ChunkScore(i, s, c) for i, (s, c) in enumerate(zip(scores, confidences))]
```

**Targeted masking vocabulary (static, per domain):**

For the initial implementation, use a static domain vocabulary rather than dynamic hot-swapping:

- **Code domain:** Python/JS/Rust keywords, common library names (numpy, pandas, torch,
  react, godot), variable naming patterns, structural patterns (function signatures, class
  hierarchies), Gatekeeper-specific terms (confidence, gate, audit, drone, schema).
- **General domain:** Fallback vocabulary for non-code conversations.

The vocabulary is a JSON file (`hive/vocab/code.json`, `hive/vocab/general.json`) loaded at
startup. Chunks containing vocabulary terms get a relevance score boost (configurable weight,
default +0.15 added to raw cosine similarity).

**Improvement — context fingerprinting:** Hash each chunk's content and cache its embedding.
If the same chunk appears in multiple turns (e.g., system prompt re-injected), reuse the
cached embedding instead of recomputing. This saves significant latency in long conversations.

```python
class EmbeddingCache:
    def __init__(self, max_size=10000):
        self._cache = OrderedDict()  # hash -> (embedding, timestamp)
        self._max_size = max_size

    def get_or_compute(self, text: str, compute_fn) -> np.ndarray:
        h = hashlib.md5(text.encode()).hexdigest()
        if h in self._cache:
            self._cache.move_to_end(h)
            return self._cache[h]
        emb = compute_fn(text)
        if len(self._cache) >= self._max_size:
            self._cache.popitem(last=False)
        self._cache[h] = emb
        return emb
```

**S1.2 — Medium Drone (Domain-Specific)**

Load and wrap the domain-specific model:

```python
# hive/drones/medium.py
class MediumDrone:
    """
    Wraps a domain-specific model (e.g., microsoft/codebert-base for code,
    or microsoft/deberta-v3-base fine-tuned for the target domain).
    ~400MB, GPU-preferred, 20-50ms per query.
    Only invoked for chunks the ultra-small drone flagged as uncertain.
    """
    def __init__(self, model_name="microsoft/codebert-base", device="cuda"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(device).eval()
        self.device = device

    def score(self, query: str, chunks: list[str]) -> list[ChunkScore]:
        """Same interface as UltraSmallDrone but with domain-aware scoring."""
        results = []
        for i, chunk in enumerate(chunks):
            inputs = self.tokenizer(
                query, chunk, truncation=True, max_length=512,
                return_tensors="pt"
            ).to(self.device)
            with torch.no_grad():
                outputs = self.model(**inputs)
                cls_emb = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            score = float(cls_emb.mean())  # simplified; real impl uses cross-encoder
            results.append(ChunkScore(i, score, confidence=0.85))
        return results
```

**S1.3 — Routing Layer**

Start with heuristic routing, then add confidence-based escalation:

```python
# hive/routing/router.py
class DroneRouter:
    """
    Phase 1 (this section): Heuristic routing based on task-type signals.
    Phase 2 (S4): Lightweight classifier trained on logged data.

    Routes each user query to:
      - "ultra_small": simple context retrieval (90% of queries)
      - "medium": complex analysis requiring domain awareness (10% of queries)
      - "escalation": ultra-small first, medium validates uncertain results
    """

    HEURISTIC_RULES = {
        "complex_keywords": [
            "refactor", "architecture", "debug", "explain", "analyze",
            "compare", "design", "optimize", "review", "audit"
        ],
        "complex_patterns": [
            r"how does .+ work",
            r"why is .+ (broken|failing|slow)",
            r"what (caused|causes) .+",
            r"(connect|relate|depend).+ between",
        ],
        "code_density_threshold": 3,  # 3+ code blocks = complex
    }

    def route(self, query: str, conversation_history: list) -> RoutingDecision:
        """
        Returns RoutingDecision(route_to: str, confidence: float, reason: str).
        """
        score = 0
        reasons = []

        # Keyword matching
        for kw in self.HEURISTIC_RULES["complex_keywords"]:
            if kw in query.lower():
                score += 1
                reasons.append(f"keyword:{kw}")

        # Pattern matching
        for pat in self.HEURISTIC_RULES["complex_patterns"]:
            if re.search(pat, query.lower()):
                score += 1
                reasons.append(f"pattern:{pat}")

        # Code density in query
        code_blocks = query.count("```")
        if code_blocks >= self.HEURISTIC_RULES["code_density_threshold"]:
            score += 2
            reasons.append(f"code_density:{code_blocks}")

        # Message length (longer = more complex, generally)
        if len(query) > 500:
            score += 1
            reasons.append(f"length:{len(query)}")

        # Conversation depth (deeper conversations need more context awareness)
        if len(conversation_history) > 30:
            score += 1
            reasons.append(f"depth:{len(conversation_history)}")

        # Route decision
        if score >= 3:
            return RoutingDecision("escalation", confidence=0.8, reasons=reasons)
        elif score >= 2:
            return RoutingDecision("medium", confidence=0.7, reasons=reasons)
        else:
            return RoutingDecision("ultra_small", confidence=0.9, reasons=reasons)
```

**Confidence-based escalation (refinement of Option 2 from our discussion):**

When the router chooses "escalation" or when the ultra-small drone returns low-confidence
scores, the medium drone validates ONLY the uncertain chunks — not the entire context:

```python
class EscalationHandler:
    """
    Runs ultra-small first. For chunks where:
      - relevance_score > 0.7 AND confidence < 0.6
    Re-scores ONLY those chunks with the medium drone.
    """
    UNCERTAINTY_THRESHOLD_SCORE = 0.7
    UNCERTAINTY_THRESHOLD_CONFIDENCE = 0.6

    def process(self, query, chunks, ultra_small: UltraSmallDrone, medium: MediumDrone):
        initial_scores = ultra_small.score(query, chunks)

        uncertain_indices = [
            i for i, s in enumerate(initial_scores)
            if s.relevance_score > self.UNCERTAINTY_THRESHOLD_SCORE
            and s.confidence < self.UNCERTAINTY_THRESHOLD_CONFIDENCE
        ]

        if not uncertain_indices:
            return initial_scores  # All confident, no escalation needed

        uncertain_chunks = [chunks[i] for i in uncertain_indices]
        medium_scores = medium.score(query, uncertain_chunks)

        # Merge: replace ultra-small scores with medium scores for uncertain chunks
        for idx, medium_score in zip(uncertain_indices, medium_scores):
            initial_scores[idx].relevance_score = medium_score.relevance_score
            initial_scores[idx].confidence = medium_score.confidence
            initial_scores[idx].source = "medium_validated"

        return initial_scores
```

**S1.4 — Testing for S1**

**Unit tests:**
- Ultra-small drone: encode known query/chunk pairs, assert cosine similarity within 0.05
  of expected values. Test confidence scoring with known-similar and known-dissimilar pairs.
- Medium drone: same interface tests, assert domain-aware scoring differentiates code
  from natural language.
- Router: feed 50 labeled queries (25 simple, 25 complex), assert ≥85% correct routing.
- Escalation handler: feed chunks with known uncertain scores, assert medium drone is
  invoked only for uncertain chunks and scores are merged correctly.
- Embedding cache: assert cache hit returns same embedding, cache miss computes new,
  eviction works at capacity.

**Integration tests:**
- Full pipeline: query → router → ultra-small → (optional escalation) → scored chunks.
  Assert output shape, latency <100ms total for simple queries, <200ms for escalated queries.
- Run 20 test conversations through the drone system alone (no LLM), measure:
  - Average latency per turn (target: <50ms for ultra-small-only, <150ms for escalated)
  - Routing distribution (target: 85–95% ultra-small, 5–15% medium/escalation)
  - Confidence distribution (target: >70% of chunks have confidence >0.7)

**Benchmark tests:**
- Latency benchmark: 1000 random query/chunk pairs, record p50/p95/p99 latency.
- Throughput benchmark: sustained processing of chunks at maximum rate, record chunks/sec.
- Memory benchmark: monitor VRAM/RAM usage during sustained processing.

### 3.2 Definition of Done
- Ultra-small drone loads, encodes, and scores with confidence output.
- Medium drone loads and scores with domain awareness.
- Router correctly classifies ≥85% of 50 labeled test queries.
- Escalation handler invokes medium drone only for uncertain chunks.
- Embedding cache reduces redundant computation (verified by cache hit rate >50% on
  repeated conversations).
- All unit tests pass, all integration tests pass.
- Latency benchmarks recorded: p50 <20ms (ultra-small), p95 <100ms (escalation).
- Event logger records every routing decision and drone score.
- PES computation includes routing accuracy and latency health from this section.

---

## 4. S2 — The Hive: Context Management, Decay & Assembly

**Goal:** implement the core context management logic — the remembrance pass, decay matrix,
context deduplication, adaptive budgeting, and topic drift detection. By the end of this
section, the hive can manage an arbitrarily long conversation and produce a curated,
compressed context window for every turn.

### 4.1 Tasks

**S2.1 — Context Store**

The central repository for all conversation context, with chunk-level granularity:

```python
# hive/context/store.py
class ContextStore:
    """
    Stores conversation context as discrete chunks with metadata.
    Each chunk has:
      - id: unique hash
      - content: the text
      - turn: which conversation turn it belongs to
      - timestamp: when it was added
      - relevance_history: list of (turn, score) pairs
      - decay_multiplier: current decay factor
      - times_saved: how many remembrance passes saved this chunk
      - fingerprint: content hash for deduplication
    """
    def __init__(self):
        self.chunks: dict[str, ContextChunk] = {}
        self.turn_index: dict[int, list[str]] = {}  # turn -> chunk_ids

    def add_chunk(self, turn: int, content: str) -> str:
        chunk_id = hashlib.md5(f"{turn}:{content}".encode()).hexdigest()[:12]
        fingerprint = hashlib.md5(content.encode()).hexdigest()[:12]
        self.chunks[chunk_id] = ContextChunk(
            id=chunk_id, content=content, turn=turn,
            fingerprint=fingerprint, decay_multiplier=1.0, times_saved=0
        )
        self.turn_index.setdefault(turn, []).append(chunk_id)
        return chunk_id

    def apply_refresh(self, refresh_map: dict[str, int]) -> None:
        """
        Membrane-before-Retention fix: when a duplicate merge keeps the densest
        chunk but a fresher copy triggered it, refresh the kept chunk's decay
        state so it is not penalized for stale age it no longer deserves.
        """
        for chunk_id, freshest_turn in refresh_map.items():
            chunk = self.chunks.get(chunk_id)
            if chunk:
                chunk.last_referenced_turn = max(
                    chunk.last_referenced_turn, freshest_turn
                )
```

**S2.2 — Remembrance Pass**

As chunks approach the deletion boundary (context window overflow), intercept them for
relevance checking:

```python
# hive/context/remembrance.py
class RemembrancePass:
    """
    Intercepts chunks moving toward the deletion cliff.
    For each candidate chunk:
      1. Score relevance against current conversation topic (via drone)
      2. If relevance > threshold: strip conversational fluff, re-inject at front
      3. Increment decay multiplier (exponential: 1.8x per save, starting point)
      4. Log the decision
    """
    REMEMBRANCE_THRESHOLD = 0.65  # Relevance score above which a chunk is saved

    def process(self, candidates: list[ContextChunk], current_topic: str,
                drone: UltraSmallDrone) -> list[RemembranceResult]:
        results = []
        for chunk in candidates:
            score = drone.score(current_topic, [chunk.content])[0]

            if score.relevance_score >= self.REMEMBRANCE_THRESHOLD:
                # Save: compress and re-inject
                compressed = self._compress(chunk.content)
                new_decay = chunk.decay_multiplier * self._decay_factor(chunk.times_saved)
                chunk.times_saved += 1
                chunk.decay_multiplier = new_decay

                results.append(RemembranceResult(
                    chunk_id=chunk.id,
                    saved=True,
                    compressed_content=compressed,
                    new_decay=new_decay,
                    relevance_score=score.relevance_score
                ))
            else:
                results.append(RemembranceResult(
                    chunk_id=chunk.id,
                    saved=False,
                    relevance_score=score.relevance_score
                ))

        return results

    def _compress(self, content: str) -> str:
        """
        Strip conversational fluff (greetings, filler, hedging language).
        Keep: technical terms, variable names, decisions, rules, parameters.
        Implementation: regex-based removal of common filler patterns,
        then sentence-level importance scoring.
        """
        filler_patterns = [
            r"^(I think|I believe|I feel|basically|essentially|actually),?\s*",
            r"(just|really|very|quite|rather)\s+",
            r"(kind of|sort of|a bit|a little)\s+",
            r"(as I mentioned|as discussed|per our conversation),?\s*",
            r"^(so|well|okay|right|sure),?\s*",
        ]
        cleaned = content
        for pattern in filler_patterns:
            cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE)

        # Sentence-level: keep sentences containing domain vocabulary
        sentences = re.split(r'(?<=[.!?])\s+', cleaned)
        important = [s for s in sentences if self._has_domain_terms(s)]
        return " ".join(important) if important else cleaned

    def _decay_factor(self, times_saved: int) -> float:
        """
        Exponential decay factor. Starts at 1.8x, increases with each save.
        This creates mathematical friction: chunks that have been saved many times
        require increasingly strong relevance to survive future passes.
        Initial value (1.8) is a placeholder — S4 tunes this from logged data.
        """
        return 1.8 + (times_saved * 0.3)  # 1.8, 2.1, 2.4, 2.7, ...
```

**S2.3 — Sharp Decay Matrix**

Apply decay to all chunks on every context assembly pass:

```python
# hive/context/decay.py
class DecayMatrix:
    """
    Applies decay to chunk relevance scores on every assembly pass.
    Decay formula:
      effective_score = raw_relevance / (decay_multiplier ^ age_factor)

    age_factor = min(turns_since_last_reference / 10, 3.0)
      (caps at 3.0 to prevent total erasure of very old but foundational context)

    decay_multiplier increases each time a chunk is saved by remembrance pass.
    Chunks that are never referenced decay naturally.
    Chunks saved by remembrance decay faster on subsequent passes.
    """

    def apply(self, chunks: list[ContextChunk], current_turn: int,
              raw_scores: dict[str, float],
              drift_penalties: dict[str, float] | None = None) -> dict[str, float]:
        """
        Applies decay to the SURVIVING (post-dedup) chunks on every assembly pass.
        Optional drift_penalties from the Membrane drift reset (Membrane runs
        before Retention): {chunk_id: multiplier} for old-topic chunks.
        """
        drift_penalties = drift_penalties or {}
        effective_scores = {}
        for chunk in chunks:
            raw = raw_scores.get(chunk.id, 0.0)
            age = current_turn - chunk.last_referenced_turn
            age_factor = min(age / 10.0, 3.0)
            decayed = raw / (chunk.decay_multiplier ** age_factor)
            decayed *= drift_penalties.get(chunk.id, 1.0)
            effective_scores[chunk.id] = max(0.0, decayed)
        return effective_scores
```

**S2.4 — Context Deduplication**

When the same concept appears in multiple turns, keep only the densest version:

```python
# hive/context/dedup.py
class ContextDeduplicator:
    """
    Detects semantic duplicates and keeps the most information-dense version.
    Uses embedding cosine similarity > 0.92 as the duplicate threshold.
    Information density = (domain_term_count / total_words) × avg_sentence_length.
    Returns:
      - chunks: surviving (deduplicated) chunks
      - refresh_map: {kept_chunk_id: freshest_turn} so the kept chunk's
        decay state can be refreshed (Membrane-before-Retention rule)
    """
    DUPLICATE_THRESHOLD = 0.92

    def deduplicate(self, chunks: list[ContextChunk],
                    embeddings: dict[str, np.ndarray]
                    ) -> tuple[list[ContextChunk], dict[str, int]]:
        if len(chunks) <= 1:
            return chunks, {}

        emb_list = [embeddings[c.id] for c in chunks]
        sim_matrix = cosine_similarity(emb_list)

        keep = set(range(len(chunks)))
        refresh_map: dict[str, int] = {}
        for i in range(len(chunks)):
            if i not in keep:
                continue
            for j in range(i + 1, len(chunks)):
                if j not in keep:
                    continue
                if sim_matrix[i][j] > self.DUPLICATE_THRESHOLD:
                    # Keep the denser one; refresh its decay state if a
                    # fresher copy triggered the merge
                    density_i = self._info_density(chunks[i].content)
                    density_j = self._info_density(chunks[j].content)
                    if density_i >= density_j:
                        keep_idx, discard = i, j
                    else:
                        keep_idx, discard = j, i
                    keep.discard(discard)
                    freshest = max(chunks[i].turn, chunks[j].turn)
                    refresh_map[chunks[keep_idx].id] = max(
                        refresh_map.get(chunks[keep_idx].id, 0), freshest
                    )

        return [chunks[i] for i in sorted(keep)], refresh_map
```

**S2.5 — Adaptive Context Budget**

Instead of a fixed 4k token limit, dynamically adjust based on task complexity:

```python
# hive/context/budget.py
class AdaptiveBudget:
    """
    Computes the context token budget based on:
      1. Task complexity (from router classification)
      2. Number of high-relevance chunks available
      3. VRAM headroom (if queryable from the LLM backend)

    Budget ranges:
      - Simple Q&A: 1,000–2,000 tokens
      - Standard coding: 3,000–4,000 tokens
      - Complex architecture/debugging: 4,000–6,000 tokens

    Never exceeds the LLM's max context window minus generation headroom.
    """
    BUDGET_RANGES = {
        "ultra_small": (1000, 3000),
        "medium": (3000, 5000),
        "escalation": (4000, 6000),
    }
    GENERATION_HEADROOM = 2048  # tokens reserved for output

    def compute(self, route: str, high_relevance_count: int,
                max_context: int = 8192) -> int:
        lo, hi = self.BUDGET_RANGES.get(route, (2000, 4000))

        # Scale within range based on how many chunks scored highly
        fill_factor = min(high_relevance_count / 10.0, 1.0)
        budget = int(lo + (hi - lo) * fill_factor)

        # Never exceed max_context minus generation headroom
        return min(budget, max_context - self.GENERATION_HEADROOM)
```

**S2.6 — Topic Drift Detection**

Detect when the conversation topic changes significantly and trigger context reset:

```python
# hive/context/drift.py
class TopicDriftDetector:
    """
    Monitors cosine similarity between recent turns and historical context.
    When drift exceeds threshold, triggers aggressive decay of old context
    and rebuilds the context window around the new topic.

    Drift score = 1.0 - cosine_similarity(recent_embedding, historical_embedding)
    "Recent" = last 3 turns. "Historical" = average embedding of all stored chunks.
    """
    DRIFT_THRESHOLD = 0.6  # 60% dissimilarity triggers reset

    def check(self, recent_chunks: list[ContextChunk],
              all_chunks: list[ContextChunk],
              drone: UltraSmallDrone) -> DriftResult:
        if len(all_chunks) < 5 or len(recent_chunks) < 1:
            return DriftResult(drift_score=0.0, should_reset=False)

        recent_text = " ".join(c.content for c in recent_chunks)
        historical_text = " ".join(c.content for c in all_chunks[:len(all_chunks)//2])

        recent_emb = drone.model.encode(recent_text)
        hist_emb = drone.model.encode(historical_text)

        similarity = cosine_similarity([recent_emb], [hist_emb])[0][0]
        drift_score = 1.0 - float(similarity)

        return DriftResult(
            drift_score=drift_score,
            should_reset=drift_score > self.DRIFT_THRESHOLD
        )
```

**Improvement — stale context detection:** Context that hasn't been referenced in N turns
(configurable, default 20) gets an additional decay acceleration, even if its relevance
score was historically high. This prevents "zombie context" that was important early but
is no longer relevant:

```python
# Integrated into DecayMatrix.apply():
if age > STALE_THRESHOLD:  # default 20 turns
    decayed *= 0.5  # Extra 50% decay for stale chunks
```

**S2.7 — Context Assembly (the full pipeline)**

Bring all components together:

```python
# hive/context/assembly.py
class ContextAssembler:
    """
    The full context assembly pipeline, invoked once per user turn:
      1. Run remembrance pass on chunks approaching deletion
      2. Score all chunks against current query (via drone fleet)
      3. Deduplicate semantically similar chunks (Membrane)
      4. Detect topic drift, reset if needed (Membrane)
      5. Apply decay matrix to surviving chunks (Retention)
      6. Compute adaptive budget
      7. Sort by (confidence × effective_score), take top chunks within budget
      8. Return assembled context string
    """
    def assemble(self, query: str, current_turn: int,
                 store: ContextStore, router: DroneRouter,
                 ultra_small: UltraSmallDrone, medium: MediumDrone,
                 escalation: EscalationHandler,
                 dedup: ContextDeduplicator,
                 drift_detector: TopicDriftDetector,
                 budget: AdaptiveBudget) -> AssembledContext:
        # 1. Remembrance pass
        deletion_candidates = store.get_deletion_candidates()
        current_topic = self._extract_topic(query, store)
        remembrance_results = RemembrancePass().process(
            deletion_candidates, current_topic, ultra_small
        )

        # 2. Route and score
        routing = router.route(query, store.get_turns())
        if routing.route_to == "escalation":
            scores = escalation.process(query, store.all_contents(), ultra_small, medium)
        elif routing.route_to == "medium":
            scores = medium.score(query, store.all_contents())
        else:
            scores = ultra_small.score(query, store.all_contents())

        # 3. Deduplicate (Membrane FIRST: collapse duplicates, refresh decay state)
        all_chunks = store.all_chunks()
        chunks, refresh_map = dedup.deduplicate(all_chunks, store.all_embeddings())
        store.apply_refresh(refresh_map)

        # 4. Topic drift (Membrane: reset decision -> drift penalties)
        recent = store.get_recent_chunks(3)
        drift = drift_detector.check(recent, all_chunks, ultra_small)
        drift_penalties = {}
        if drift.should_reset:
            drift_penalties = self._apply_drift_reset(chunks, recent, store)

        # 5. Apply decay to surviving chunks (Retention)
        raw_scores = {s.chunk_id: s.relevance_score for s in scores}
        effective_scores = DecayMatrix().apply(
            chunks, current_turn, raw_scores, drift_penalties
        )

        # 6. Budget
        high_relevance = sum(1 for s in effective_scores.values() if s > 0.6)
        token_budget = budget.compute(routing.route_to, high_relevance)

        # 7. Sort and select
        scored_chunks = sorted(
            [(cid, effective_scores[cid]) for cid in effective_scores],
            key=lambda x: x[1], reverse=True
        )
        selected = self._select_within_budget(scored_chunks, store, token_budget)

        return AssembledContext(
            content=self._format_context(selected),
            token_count=self._count_tokens(selected),
            budget=token_budget,
            chunks_used=len(selected),
            routing_decision=routing,
            drift_detected=drift.should_reset
        )
```

**S2.8 — Testing for S2**

**Unit tests:**
- ContextStore: add/retrieve chunks, verify turn index, fingerprint deduplication.
- RemembrancePass: feed chunks with known relevance scores, assert correct save/discard
  decisions. Verify decay multiplier increases correctly (1.8 → 2.1 → 2.4). Verify
  compression strips filler while keeping domain terms.
- DecayMatrix: verify effective scores decrease with age, verify stale detection at 20 turns.
- ContextDeduplicator: feed known-duplicate pairs (cosine >0.92), assert only denser version
  kept. Feed non-duplicates, assert all kept.
- AdaptiveBudget: verify budget ranges per route type, verify cap at max_context - headroom.
- TopicDriftDetector: feed known-similar and known-dissimilar chunk sets, assert drift
  score and reset decision.
- ContextAssembler: end-to-end test with a 50-turn synthetic conversation, assert output
  is within budget, no duplicate concepts, high-relevance chunks present.

**Integration tests:**
- Run 20 full conversations through the full assembly pipeline (drones + context management).
  Measure:
  - Context window utilization (target: 70–90% of adaptive budget used)
  - Duplicate rate (target: <5% semantic duplicates in assembled context)
  - Remembrance save rate (target: 10–30% of deletion candidates saved)
  - Decay convergence (chunks that are saved multiple times eventually decay out)
  - Topic drift detection (inject deliberate topic changes, assert detection within 3 turns)

**Regression tests:**
- Run the same 20 baseline conversations from S0.5 through the hive.
  Compare assembled context quality against naive truncation baseline.
  Target: ≥20% improvement in retrieval precision (measured by oracle in S4).

### 4.2 Definition of Done
- ContextStore correctly stores and retrieves chunks with full metadata.
- RemembrancePass saves relevant chunks, compresses them, and increments decay correctly.
- DecayMatrix applies exponential decay with stale detection.
- ContextDeduplicator removes semantic duplicates while preserving the densest version.
- AdaptiveBudget produces correct ranges per route type.
- TopicDriftDetector fires at correct threshold.
- Full assembly pipeline produces context within budget, no duplicates, high relevance.
- All unit tests pass. All integration tests pass.
- Event logger records every assembly decision.
- PES includes context utilization score from this section.

---

## 5. S3 — Integration: LLM Backend & Pipeline Health

**Goal:** connect the hive to the LLM inference backend (vLLM **and** LM Studio / llama.cpp on
Windows 11), implement KV-cache management where the backend supports it, and build the
pipeline health monitoring system with congestion detection and auto-scaling. Where the
existing Gatekeeper logic already solves a problem (endpoint resolution, confidence
calibration, reliability tracking, rollback), reuse it instead of re-implementing.

### 5.1 Tasks

**S3.1 — LLM Backend Integration (dual backend)**

The hive must drive **two** local backends on Windows 11 (16 GB+ GPU):

- **vLLM** — PagedAttention enables surgical, page-level KV-cache edits without full
  recomputation. This is the backend that can realize the hive's cache-manipulation design.
- **LM Studio (llama.cpp)** — the primary inference host already in use. It exposes an
  OpenAI-compatible API, but any context change forces a full prompt re-process (no
  surgical KV-cache edits). It still benefits from the hive because the *compressed* context
  is smaller, so even a full re-process is cheaper than feeding raw history.

Both backends speak the same OpenAI-compatible chat-completion contract, so the hive uses a
single abstraction and swaps the transport:

```python
# hive/backend/base.py
class LLMBackend:
    """
    Abstract chat-completion backend. Both vLLM and LM Studio (llama.cpp) expose
    OpenAI-compatible endpoints, so one contract serves both. The hive only ever
    talks to an LLMBackend; the concrete class decides how context is delivered.

    - vLLM:  surgical KV-cache page edits via the vLLM cache API (see S3.2).
    - LM Studio: no surgical cache API; always send the (compressed) context.
    """
    def generate(self, assembled_context: AssembledContext, user_query: str,
                 sampling_params: dict) -> str:
        raise NotImplementedError

# hive/backend/openai_compat.py
class OpenAICompatBackend(LLMBackend):
    """
    Baseline transport for any local OpenAI-compatible server.
    Passes the hive-assembled context as the system message and the user
    query as the user message, replacing the naive rolling window.
    Used by both vLLM and LM Studio when no cache API is available.
    """
    def __init__(self, base_url: str, model: str, api_key: str = "lm-studio"):
        self.base_url = base_url
        self.model = model
        self.api_key = api_key

    def generate(self, assembled_context, user_query, sampling_params):
        messages = [
            {"role": "system", "content": assembled_context.content},
            {"role": "user", "content": user_query}
        ]
        resp = requests.post(f"{self.base_url}/v1/chat/completions", json={
            "model": self.model,
            "messages": messages,
            **sampling_params
        }, headers={"Authorization": f"Bearer {self.api_key}"})
        return resp.json()["choices"][0]["message"]["content"]

# hive/backend/vllm.py
class VLLMBackend(OpenAICompatBackend):
    """
    vLLM backend. Inherits OpenAI-compatible transport; additionally exposes the
    KV-cache API for surgical page-level edits (see S3.2). Default vLLM serving
    port is 8000.
    """
    def __init__(self, base_url="http://localhost:8000", model="qwen3.8-27b"):
        super().__init__(base_url=base_url, model=model)

# hive/backend/lmstudio.py
class LMStudioBackend(OpenAICompatBackend):
    """
    LM Studio (llama.cpp) backend. Default port 1234 (matches the running
    Gatekeeper LM Studio host). No surgical KV-cache API; sends compressed
    context. Reuses Gatekeeper's endpoint normalization to resolve the URL.
    """
    def __init__(self, base_url="http://localhost:1234", model=""):
        # model resolved via Gatekeeper Resolve-LmEndpoint / loaded-model list
        super().__init__(base_url=base_url, model=model, api_key="lm-studio")
```

**S3.2 — KV-Cache Management (vLLM only; LM Studio falls back to compressed-context)**

Only vLLM exposes a page-level KV-cache API. LM Studio (llama.cpp) does not, so its mode is
"always send the compressed context" — still faster than feeding raw history because the
hive has shrunk it. The two modes are tested side-by-side (see S3.6).

```python
# hive/backend/cache_manager.py
class KVCacheManager:
    """
    Manages the vLLM KV-cache to avoid redundant recomputation.
    - Pinned pages: system prompt and persistent rules (never evict)
    - Cached pages: recent context (evict on topic drift)
    - Dynamic pages: current turn's assembled context (replace every turn)

    vLLM mode: surgical page edits (no full recomputation).
    LM Studio mode: cache management is a no-op; the OpenAICCompatBackend
    always sends the compressed context (still cheaper than raw history).
    """
    def __init__(self, backend: LLMBackend):
        self.backend = backend
        self.supports_surgical_edits = isinstance(backend, VLLMBackend)

    def update_cache(self, assembled: AssembledContext, persistent_prefix: str):
        if not self.supports_surgical_edits:
            # LM Studio / llama.cpp: nothing to manage — send full compressed
            # context each turn. Log the mode for the S3.6 side-by-side test.
            return
        # vLLM mode:
        # 1. Keep persistent_prefix pages (system prompt, core rules)
        # 2. Replace dynamic pages with new assembled context
        # 3. Invalidate any pages from evicted context
        pass  # implementation depends on the vLLM cache API
```

**S3.3 — Pipeline Health Monitor**

```python
# hive/monitor/health.py
class PipelineHealthMonitor:
    """
    Continuously monitors pipeline health and triggers alerts/actions.
    Runs on a background thread, polling every 500ms.

    Metrics tracked:
      - Queue depth (chunks pending drone processing)
      - Average drone latency (rolling window of last 10 queries)
      - Assembly latency (time from user input to context ready)
      - Generation latency (time from context ready to first token)
      - Total turn latency (user input to first token)
      - PES composite score
      - Error rate (failed drone queries, failed assemblies)
    """
    def __init__(self, logger: EventLogger):
        self.logger = logger
        self.metrics = RollingMetrics(window_size=100)

    def check_congestion(self) -> CongestionReport:
        """
        Evaluates congestion signals and recommends actions.
        Returns a report with severity level and recommended actions.
        """
        queue_depth = self.metrics.current_queue_depth
        avg_drone_latency = self.metrics.avg_drone_latency_ms
        assembly_pending = self.metrics.pending_assemblies

        actions = []
        severity = "normal"

        if queue_depth > 15:
            severity = "critical"
            actions.append("fallback_to_truncation")
        elif queue_depth > 5:
            severity = "warning"
            actions.append("batch_similar_chunks")

        if avg_drone_latency > 100:
            severity = max(severity, "critical")
            actions.append("use_cached_embeddings")
        elif avg_drone_latency > 20:
            severity = max(severity, "warning")
            actions.append("skip_medium_drone")

        if assembly_pending > 2:
            severity = max(severity, "critical")
            actions.append("aggressive_compression")
        elif assembly_pending > 0:
            severity = max(severity, "warning")
            actions.append("queue_messages")

        report = CongestionReport(
            severity=severity,
            queue_depth=queue_depth,
            avg_drone_latency_ms=avg_drone_latency,
            pending_assemblies=assembly_pending,
            recommended_actions=actions
        )

        if severity != "normal":
            self.logger.log("congestion", "congestion_detected", report.to_dict())

        return report
```

**S3.4 — Graceful Degradation**

```python
# hive/monitor/degradation.py
class GracefulDegradation:
    """
    When the pipeline is under stress, degrade gracefully rather than
    blocking or crashing. Degradation levels:

    Level 0 (Normal): Full pipeline — drones, remembrance, decay, dedup, adaptive budget.
    Level 1 (Warning): Skip medium drone, use ultra-small only. Skip deduplication.
    Level 2 (Critical): Skip remembrance pass. Use cached embeddings only.
                        Reduce budget to 2k tokens.
    Level 3 (Emergency): Skip all hive processing. Fall back to naive FIFO truncation.
                         Log emergency event for post-mortem analysis.
    """
    def __init__(self):
        self.current_level = 0

    def update(self, congestion: CongestionReport):
        severity_map = {"normal": 0, "warning": 1, "critical": 2}
        new_level = severity_map.get(congestion.severity, 0)

        if new_level > self.current_level:
            self.current_level = new_level
        elif new_level < self.current_level:
            # Recover slowly: drop one level at a time, with cooldown
            self.current_level = max(self.current_level - 1, new_level)

    def should_skip_medium(self) -> bool:
        return self.current_level >= 1

    def should_skip_dedup(self) -> bool:
        return self.current_level >= 1

    def should_skip_remembrance(self) -> bool:
        return self.current_level >= 2

    def should_use_cached_only(self) -> bool:
        return self.current_level >= 2

    def should_fallback_fifo(self) -> bool:
        return self.current_level >= 3
```

**Improvement — auto-scaling drone instances:** If the host machine has sufficient VRAM
(check via `nvidia-smi` or `torch.cuda.mem_get_info()`), spawn additional ultra-small drone
instances to parallelize chunk processing. Each instance handles a partition of the chunk
queue:

```python
class DronePool:
    """
    Manages multiple drone instances for parallel processing.
    Auto-scales based on queue depth and available VRAM.
    """
    def __init__(self, max_instances=3):
        self.instances: list[UltraSmallDrone] = []
        self.max_instances = max_instances

    def scale_if_needed(self, queue_depth: int, available_vram_mb: int):
        needed = min(queue_depth // 10, self.max_instances)
        while len(self.instances) < needed and available_vram_mb > 500:
            self.instances.append(UltraSmallDrone())
            available_vram_mb -= 100  # approximate per-instance cost

        while len(self.instances) > 1 and queue_depth < 3:
            self.instances.pop()
```

**S3.5 — Gatekeeper Logic Reuse**

Gatekeeper Studio already ships production-tested logic that maps directly onto hive needs.
Reuse it (via its engine contract / `EngineClient` seam) rather than re-implementing, to save
engineering time and keep behavior consistent across both projects:

| Gatekeeper capability | Hive consumer | What it saves |
|---|---|---|
| `Resolve-LmEndpoint` / `NormalizeLmEndpoint` | `LMStudioBackend` / `VLLMBackend` URL + model resolution | Handles host-only → `/v1/completions`, empty → default, full-URL passthrough |
| `Get-LmStatus` / `Get-LmModels` / `Invoke-LmModelLoad` | Backend readiness probe + loaded-model discovery (default port 1234) | Avoids duplicate server/health logic |
| Confidence system (`Get-ConfidenceScore`, calibration, thresholds) | Sieve confidence calibration — aligns drone confidence with measured pass rate | Reuses the calibrated-gate design instead of re-deriving thresholds |
| `Get-DroneReliability` (pass rate, over-confidence penalty) | PES `RoutingAccuracy` + a drift-reliability input | Reuses the reliability rollup math for drone tiers |
| `Add-RunHistoryEntry` / `Get-RunHistory` | A/B and ablation result logging | Consistent experiment provenance |
| `Set-PendingHumanGate` / `Clear-PendingHumanGate` | Human-in-the-loop gate for low-confidence evictions | Reuses the gate lifecycle (avoid silent wrong evictions) |
| Automated rollback (confidence-gated) | `AutomatedRollback` in S5 | Same "revert to last-known-good" pattern |
| `Get-GatekeeperConfig` fallback + `Set-GatekeeperConfigJson` | Hive config load/persist with backward-compatible defaults | Safe config merge without clobbering |

**Interop:** the hive reads/writes Gatekeeper state through the documented `HOST-SEAM.md`
contract (JSON shapes), never by reaching into Gatekeeper internals. Where a capability does
not yet exist in Gatekeeper (e.g., targeted-masking fine-tuning), the hive owns it; where it
does, the hive delegates.

**S3.6 — Testing for S3**

**Unit tests:**
- OpenAICompatBackend / VLLMBackend / LMStudioBackend: mock HTTP responses, verify correct
  message formatting, model/URL resolution, error handling, and that `KVCacheManager`
  detects backend type (surgical vs. no-op).
- KVCacheManager: verify vLLM page-management logic and the LM Studio no-op path.
- PipelineHealthMonitor: feed synthetic metrics, verify congestion detection at thresholds.
- GracefulDegradation: verify level transitions, verify recovery cooldown.
- DronePool: verify scaling decisions based on queue depth and VRAM.
- Gatekeeper interop: mock the Gatekeeper endpoint/confidence/reliability contract; verify
  the hive consumes the JSON shapes without reaching into internals.

**Integration tests:**
- Full pipeline with a real vLLM backend **and** a real LM Studio (llama.cpp) backend
  (or mock if a backend is unavailable):
  - Send 50 queries through each, verify assembled context reaches the LLM correctly.
  - Verify generation latency is not degraded by hive overhead (target: <50ms added latency).
  - Verify VRAM usage stays within bounds (primary model + drones).
  - **Side-by-side vLLM vs. LM Studio:** run the same 50-turn conversation on both backends;
    record per-turn latency and PES. Expect vLLM to show lower KV-cache re-process cost;
    LM Studio still benefits from the compressed context. This data feeds the backend
    recommendation in the white paper.

**Congestion simulation tests:**
- Inject artificial latency into drone processing (sleep 50ms, 100ms, 200ms).
- Verify congestion detector fires at correct thresholds.
- Verify graceful degradation activates and recovers correctly.
- Verify PES drops proportionally and recovers after degradation ends.

**End-to-end test:**
- Run a 100-turn conversation with the full pipeline (hive + LLM backend).
- Verify: no OOM events, constant generation speed, PES stays above 70.
- Compare against LM Studio baseline from S0.5:
  - Task completion rate (human-judged): target ≥20% improvement.
  - Context relevance (oracle-judged in S4): target ≥30% improvement.

### 5.2 Definition of Done
- OpenAICompatBackend, VLLMBackend, and LMStudioBackend all send hive-assembled context to the LLM and return generation.
- KVCacheManager runs vLLM page management when the backend supports it, and cleanly no-ops for LM Studio.
- PipelineHealthMonitor detects congestion at correct thresholds.
- GracefulDegradation activates and recovers correctly under simulated stress.
- DronePool scales instances based on load and available VRAM.
- Gatekeeper interop consumed through the HOST-SEAM contract (no internal reach-in).
- Full pipeline (hive + vLLM and hive + LM Studio) runs 100-turn conversations without OOM or crashes.
- Side-by-side vLLM vs. LM Studio latency/PES recorded.
- All tests pass. PES recorded for full pipeline.
- Event logger records all health events, congestion events, degradation events.

---

## 6. S4 — Oracle, Optimization & A/B Testing

**Goal:** implement the async oracle for ground truth labeling, build the A/B testing
framework, and use logged data to optimize all tunable parameters (decay multiplier,
routing thresholds, drone confidence thresholds, budget ranges, drift threshold).

### 6.1 Tasks

**S4.1 — Async Batch Oracle**

```python
# hive/oracle/batch_oracle.py
class AsyncOracle:
    """
    Runs in a background process (never on the hot path).
    Evaluates past conversation turns to create ground truth labels.

    For each turn N:
      1. Reconstruct what context the hive provided at turn N.
      2. Ask the LLM: "Given ONLY this context, could you answer the query
         at turn N? What additional context would have helped?"
      3. Parse the LLM's response into structured relevance labels.
      4. Store labels in the ground truth database.

    These labels are used to:
      - Tune the decay multiplier (which chunks were wrongly evicted?)
      - Tune routing thresholds (were complex tasks routed correctly?)
      - Tune drone confidence thresholds (were uncertain chunks identified?)
      - Tune the context budget (was there enough/too much context?)
    """

    EVALUATION_PROMPT = """
    You are evaluating whether an AI assistant had sufficient context.

    The assistant was given this context:
    ---
    {context}
    ---

    The user asked: {query}

    The assistant responded: {response}

    Answer these questions:
    1. Was the context sufficient? (yes/no)
    2. Which specific pieces of context were used? (quote them)
    3. What additional context would have helped? (describe)
    4. Rate context sufficiency 1-5: ___

    Respond in JSON format.
    """

    def evaluate_turn(self, turn_data: TurnRecord) -> OracleLabel:
        prompt = self.EVALUATION_PROMPT.format(
            context=turn_data.assembled_context,
            query=turn_data.user_query,
            response=turn_data.llm_response
        )
        result = self.llm.generate(prompt)
        parsed = json.loads(result)

        return OracleLabel(
            turn=turn_data.turn,
            context_sufficient=parsed["sufficient"],
            context_used=parsed["used_pieces"],
            missing_context=parsed["missing"],
            sufficiency_score=parsed["score"],
            chunk_labels=self._map_to_chunks(parsed, turn_data.chunk_ids)
        )

    def run_batch(self, conversation_log: list[TurnRecord],
                  sample_rate: float = 0.1) -> list[OracleLabel]:
        """
        Evaluate a sample of turns from a conversation.
        Default: 10% sampling (evaluate every 10th turn).
        """
        sampled = conversation_log[::int(1/sample_rate)]
        labels = []
        for turn_data in sampled:
            label = self.evaluate_turn(turn_data)
            labels.append(label)
        return labels
```

**S4.2 — Ground Truth Database**

```python
# hive/oracle/ground_truth.py
class GroundTruthDB:
    """
    SQLite database storing all oracle labels and hive decisions.
    Tables:
      - oracle_labels: (turn, chunk_id, predicted_relevant, actually_relevant, score)
      - hive_decisions: (turn, decision_type, parameters, outcome)
      - parameter_versions: (version, decay_multiplier, routing_threshold, ...)

    Queries:
      - retrieval_precision: % of retrieved chunks that were actually relevant
      - retrieval_recall: % of relevant chunks that were retrieved
      - false_eviction_rate: % of evicted chunks that were needed later
      - routing_accuracy: % of correct routing decisions
      - decay_quality: correlation between decay score and actual relevance
    """
    def retrieval_precision(self, window: int = 100) -> float:
        """Of the last N retrieved chunks, what % were actually relevant?"""
        ...

    def retrieval_recall(self, window: int = 100) -> float:
        """Of the last N relevant chunks, what % were retrieved?"""
        ...

    def false_eviction_rate(self, window: int = 100) -> float:
        """Of the last N evicted chunks, what % were needed later?"""
        ...
```

**S4.3 — A/B Testing Framework**

```python
# hive/testing/ab_test.py
class ABTestRunner:
    """
    Runs two configurations of the hive in parallel on the same conversations.
    Config A: current production parameters.
    Config B: experimental parameters.

    Both configs process the same input but only Config A's output goes to the user.
    Config B's output is logged for comparison.

    Metrics compared:
      - Retrieval precision
      - Retrieval recall
      - Context utilization
      - Latency
      - PES score
    """
    def run(self, conversations: list, config_a: HiveConfig,
            config_b: HiveConfig, turns: int = 50) -> ABTestResult:
        results_a = []
        results_b = []

        for conv in conversations:
            hive_a = Hive(config_a)
            hive_b = Hive(config_b)

            for turn_data in conv[:turns]:
                result_a = hive_a.process_turn(turn_data.query)
                result_b = hive_b.process_turn(turn_data.query)
                results_a.append(result_a)
                results_b.append(result_b)

        return ABTestResult(
            config_a_metrics=self._compute_metrics(results_a),
            config_b_metrics=self._compute_metrics(results_b),
            winner=self._determine_winner(results_a, results_b)
        )
```

**S4.4 — Parameter Optimization**

Systematically search for optimal parameters using logged data:

**Decay multiplier sweep:**
```python
def optimize_decay(db: GroundTruthDB, candidates=[1.2, 1.4, 1.6, 1.8, 2.0, 2.2, 2.5]):
    """
    For each candidate decay multiplier:
      1. Replay the conversation log with that multiplier.
      2. Measure: false eviction rate, context bloat, retrieval precision.
      3. Plot the trade-off curve.
    Return the multiplier that minimizes: false_evictions + context_bloat_penalty.
    """
```

**Routing threshold sweep:**
```python
def optimize_routing_threshold(db, candidates=[1, 2, 3, 4, 5]):
    """
    For each candidate threshold (score needed to route to medium drone):
      1. Replay routing decisions.
      2. Measure: routing accuracy, unnecessary medium invocations, missed complex tasks.
    Return the threshold that maximizes routing accuracy while minimizing compute cost.
    """
```

**Budget range sweep:**
```python
def optimize_budget_ranges(db, budget_candidates):
    """
    For each set of budget ranges:
      1. Replay assemblies.
      2. Measure: context utilization, sufficiency score from oracle.
    Return ranges that maximize sufficiency while minimizing token waste.
    """
```

**S4.5 — Ablation Studies**

Run the system with components disabled to measure each component's contribution:

| Config | Components Active | Purpose |
|---|---|---|
| Full | Drones + Remembrance + Decay + Dedup + Adaptive Budget + Drift | Baseline |
| No Decay | All except decay (FIFO eviction) | Measure decay's contribution |
| No Drones | Random filtering instead of drone scoring | Measure drones' contribution |
| No Remembrance | No remembrance pass | Measure remembrance's contribution |
| No Dedup | Skip deduplication | Measure dedup's contribution |
| No Adaptive | Fixed 4k budget | Measure adaptive budget's contribution |
| No Drift | No topic drift detection | Measure drift detection's contribution |
| Baseline | No hive (naive truncation) | Lower bound |

Run each config on the same 20 test conversations. Compare PES, retrieval precision/recall,
task completion rate, and latency.

**S4.6 — Lightweight Classifier Training**

Using logged routing decisions and oracle labels, train a tiny classifier to replace
heuristic routing:

```python
# hive/routing/classifier.py
class RoutingClassifier:
    """
    Trained on logged data from heuristic routing.
    Features: message_length, keyword_density, code_block_count,
              conversation_depth, avg_chunk_age, topic_drift_score.
    Output: "ultra_small" | "medium" | "escalation".

    Model: logistic regression or small decision tree (<10MB, <20ms inference).
    Trained offline, loaded at startup.
    """
    def train(self, training_data: list[RoutingRecord]):
        """Train on logged routing decisions + oracle labels."""
        features = [self._extract_features(r) for r in training_data]
        labels = [r.optimal_route for r in training_data]
        self.model = DecisionTreeClassifier(max_depth=5)
        self.model.fit(features, labels)

    def predict(self, query, conversation_history) -> str:
        features = self._extract_features_live(query, conversation_history)
        return self.model.predict([features])[0]
```

**S4.7 — Testing for S4**

**Oracle tests:**
- Feed known-sufficient and known-insufficient context to the oracle.
  Assert it correctly identifies sufficiency (≥90% accuracy on labeled test set).
- Verify batch processing doesn't interfere with main pipeline (separate process).
- Verify sampling rate is respected (10% of turns evaluated).

**Ground truth tests:**
- Verify precision/recall/false-eviction calculations against hand-computed values.
- Verify database handles 10,000+ labels without performance degradation.

**A/B test tests:**
- Run an A/B test where config A and config B are identical.
  Assert metrics are within 2% of each other (statistical noise).
- Run an A/B test where config B has a deliberately bad parameter (e.g., decay=5.0).
  Assert config B metrics are significantly worse.

**Ablation tests:**
- Run all 8 ablation configs. Verify each produces valid results.
- Record the contribution of each component to PES.

**Classifier tests:**
- Train on 1000 logged routing decisions.
- Test on 200 held-out decisions. Assert ≥90% agreement with oracle-optimal routing.
- Verify inference latency <20ms.

### 6.2 Definition of Done
- Async oracle runs in background, produces valid labels, never blocks the main pipeline.
- Ground truth DB stores and queries labels correctly.
- A/B test framework runs two configs in parallel and produces comparative metrics.
- At least 3 parameter sweeps completed: decay multiplier, routing threshold, budget ranges.
- Ablation study completed: all 8 configs run, contribution of each component measured.
- Routing classifier trained and achieves ≥90% accuracy on held-out data.
- All tests pass. Optimized parameters documented and applied to production config.
- PES improvement from S3 baseline to S4 optimized config recorded.

---

## 7. S5 — Production Hardening & Shadow Testing

**Goal:** harden the system for production use with shadow mode testing, predictive
pre-loading, performance benchmarking, and automated rollback.

### 7.1 Tasks

**S5.1 — Shadow Mode Testing**

```python
# hive/testing/shadow_mode.py
class ShadowMode:
    """
    Runs an experimental configuration alongside production without
    affecting the user. The production config's output goes to the user;
    the shadow config's output is logged only.

    After N turns (configurable, default 100), compare shadow metrics
    against production metrics. If shadow is better, promote to production.
    If worse, discard.

    This enables safe, continuous optimization without user impact.
    """
    def __init__(self, production_config: HiveConfig, shadow_config: HiveConfig):
        self.production = Hive(production_config)
        self.shadow = Hive(shadow_config)
        self.shadow_log = []

    def process_turn(self, query: str, turn: int) -> str:
        # Production: output goes to user
        production_result = self.production.process_turn(query, turn)

        # Shadow: output logged only
        shadow_result = self.shadow.process_turn(query, turn)
        self.shadow_log.append({
            "turn": turn,
            "production": production_result.metrics,
            "shadow": shadow_result.metrics
        })

        return production_result.content  # Only production reaches user

    def evaluate_after(self, n_turns: int = 100) -> ShadowEvaluation:
        """Compare accumulated metrics. Recommend promote/discard/continue."""
        ...
```

**S5.2 — Predictive Pre-loading**

```python
# hive/context/predictive.py
class PredictivePreloader:
    """
    Based on conversation patterns, predict what context will be needed
    next and pre-compute embeddings / pre-fetch from the vault.

    Pattern matching:
      - If the user frequently alternates between two topics, pre-load both.
      - If the user just asked about function X, pre-load callers of X.
      - If the conversation is in "debugging mode" (multiple error-related
        queries), pre-load recent error logs and stack traces.
    """
    def predict_next_context(self, recent_queries: list[str],
                             store: ContextStore) -> list[str]:
        """Returns a list of chunk IDs likely to be needed next."""
        patterns = self._detect_patterns(recent_queries)
        predicted = []

        if "debugging" in patterns:
            predicted.extend(self._find_error_related_chunks(store))
        if "alternating_topics" in patterns:
            predicted.extend(self._find_alternating_topic_chunks(store, recent_queries))
        if "function_exploration" in patterns:
            predicted.extend(self._find_caller_callee_chunks(store, recent_queries))

        return predicted
```

**S5.3 — Automated Rollback**

```python
# hive/monitor/rollback.py
class AutomatedRollback:
    """
    If the PES drops below a threshold for N consecutive turns,
    automatically revert to the last known good configuration.

    Thresholds:
      - PES < 50 for 10 consecutive turns: immediate rollback
      - PES < 60 for 25 consecutive turns: rollback after warning period
      - PES trending downward for 50 turns (linear regression): rollback

    Rollback restores:
      - All tunable parameters to last known good values
      - Decay matrix state (revert to checkpoint)
      - Routing classifier to previous version

    Logs the rollback event for post-mortem analysis.
    """
    def check(self, recent_pes: list[float]) -> RollbackDecision:
        if len(recent_pes) < 10:
            return RollbackDecision(should_rollback=False)

        # Immediate rollback
        if all(p < 50 for p in recent_pes[-10:]):
            return RollbackDecision(should_rollback=True, reason="PES < 50 for 10 turns")

        # Warning-period rollback
        if len(recent_pes) >= 25 and all(p < 60 for p in recent_pes[-25:]):
            return RollbackDecision(should_rollback=True, reason="PES < 60 for 25 turns")

        # Trend-based rollback
        if len(recent_pes) >= 50:
            slope = np.polyfit(range(50), recent_pes[-50:], 1)[0]
            if slope < -0.5:  # Declining more than 0.5 points per turn
                return RollbackDecision(should_rollback=True, reason="PES declining trend")

        return RollbackDecision(should_rollback=False)
```

**Improvement — checkpoint system:** Save the hive state (context store, decay matrix,
parameter config) periodically so rollback can restore a known-good state:

```python
class HiveCheckpoint:
    def save(self, state: HiveState, tag: str):
        """Save full hive state to disk."""
        ...

    def restore(self, tag: str) -> HiveState:
        """Restore hive state from checkpoint."""
        ...

    def auto_checkpoint(self, state: HiveState, pes: float):
        """Auto-save when PES is high (known-good state)."""
        if pes > 80:
            self.save(state, f"pes_{pes:.0f}_{datetime.now():%Y%m%d_%H%M%S}")
```

**S5.4 — Performance Benchmarks**

Establish and record performance benchmarks for the final system:

```python
# hive/benchmarks/full_benchmark.py
def run_full_benchmark():
    """
    Runs a comprehensive benchmark suite:

    1. Latency benchmarks:
       - Ultra-small drone: p50/p95/p99 over 10,000 queries
       - Medium drone: p50/p95/p99 over 10,000 queries
       - Full assembly pipeline: p50/p95/p99 over 1,000 turns
       - End-to-end (hive + LLM): p50/p95/p99 over 100 turns

    2. Throughput benchmarks:
       - Chunks processed per second (ultra-small only)
       - Chunks processed per second (with escalation)
       - Full turns per minute (including LLM generation)

    3. Quality benchmarks:
       - Retrieval precision over 500 labeled chunks
       - Retrieval recall over 500 labeled chunks
       - False eviction rate over 200 labeled evictions
       - Routing accuracy over 200 labeled queries
       - Task completion rate over 50 conversations

    4. Resource benchmarks:
       - Peak VRAM usage (primary model + drones)
       - Peak RAM usage
       - Disk I/O (logging)
       - CPU utilization

    5. Stability benchmarks:
       - 500-turn conversation: PES over time (plot)
       - 1000-turn conversation: OOM events, crash events
       - Congestion recovery: inject load, measure recovery time

    Outputs: JSON report + plots in hive/benchmarks/results/
    """
```

**S5.5 — Testing for S5**

**Shadow mode tests:**
- Run shadow mode for 200 turns with an identical config.
  Assert production and shadow metrics are within 5% (noise margin).
- Run shadow mode with a deliberately better config.
  Assert shadow metrics improve and promotion is recommended.

**Predictive pre-loader tests:**
- Feed known alternating-topic conversations.
  Assert pre-loaded chunks match the next topic ≥60% of the time.
- Measure latency improvement from pre-loading vs. on-demand computation.

**Automated rollback tests:**
- Inject a bad parameter config (e.g., decay=10.0).
  Assert PES drops and rollback fires within the configured turn window.
- Assert rollback restores previous parameters and PES recovers.

**Checkpoint tests:**
- Save and restore checkpoints. Assert hive state is identical after restore.
- Verify auto-checkpoint only saves at PES > 80.

**Stability tests:**
- Run a 500-turn conversation. Assert no crashes, no OOM, PES stays above 60.
- Run a 1000-turn conversation. Assert same.
- Inject congestion (artificial drone latency). Assert degradation activates and recovers.

### 7.2 Definition of Done
- Shadow mode runs without affecting user output, produces valid comparative metrics.
- Predictive pre-loader improves latency for predicted contexts (measured).
- Automated rollback fires correctly under simulated PES degradation.
- Checkpoint save/restore works correctly.
- Full benchmark suite completed, results recorded.
- 500-turn and 1000-turn stability tests pass (no crashes, no OOM).
- All tests pass. Final PES recorded and compared against S0 baseline.

---

## 8. Appendix A — Definition-of-Done Checklist Template

Use for EVERY section:

- [ ] All unit tests pass (`python hive/tests/run_hive_tests.py` → exit 0).
- [ ] All integration tests pass for this section's components.
- [ ] Event logger records every decision for this section's components.
- [ ] PES computation includes this section's metrics and produces correct 0–100 score.
- [ ] Congestion detector correctly fires at this section's thresholds.
- [ ] Baseline comparison recorded (improvement over previous section's metrics).
- [ ] No regressions in previous sections' tests (all earlier tests still pass).
- [ ] Logged data from this section's tests is valid NDJSON and queryable.
- [ ] Performance within latency budget (see §1.2 component table).
- [ ] VRAM/RAM usage within bounds (no OOM during testing).

---

## 9. Appendix B — Efficiency Scoring System (Full Specification)

### B.1 PES Component Metrics

| Component | Weight | Formula | Source |
|---|---|---|---|
| RetrievalPrecision | 0.30 | (relevant_retrieved / total_retrieved) × 100 | Oracle labels |
| RoutingAccuracy | 0.20 | (correct_routes / total_routes) × 100 | Oracle labels |
| LatencyHealth | 0.20 | max(0, 100 - (avg_latency_ms - 50) × 0.67) | Event logger |
| ThroughputHealth | 0.15 | (actual_tps / baseline_tps) × 100 | Event logger |
| ContextUtilization | 0.15 | see §B.2 | Event logger |

### B.2 ContextUtilization Formula

```
utilization = budget_used / budget_total

if utilization < 0.60:
    score = utilization × 100  # Under-utilization: context too sparse
elif utilization > 0.95:
    score = 100 - (utilization - 0.95) × 1000  # Over-utilization: risk of truncation
else:
    score = 100  # Sweet spot: 60-95% utilization
```

### B.3 PES Alert Actions

| PES Range | Color | Action |
|---|---|---|
| 80–100 | GREEN | Normal operation. Log periodic snapshot. |
| 60–79 | YELLOW | Log warning. Trigger shadow-mode A/B test of optimized config. |
| 40–59 | RED | Trigger automated rollback to last known good config. |
| 0–39 | CRITICAL | Emergency fallback to FIFO truncation. Alert operator. |

### B.4 PES Trend Analysis

Track PES over a rolling window of 50 turns. Compute linear regression slope.
If slope < -0.5 (declining more than 0.5 points per turn), trigger proactive
investigation even if current PES is above threshold.

---

## 10. Appendix C — Congestion Detection (Full Specification)

### C.1 Signals and Thresholds

| Signal | Normal | Warning | Critical |
|---|---|---|---|
| Queue depth (chunks) | 0–5 | 6–15 | >15 |
| Drone latency (ms, avg of 10) | <20 | 20–100 | >100 |
| Assembly backlog (pending) | 0 | 1 | ≥2 |
| VRAM usage (%) | <80 | 80–90 | >90 |
| Error rate (per 100 queries) | <1 | 1–5 | >5 |

### C.2 Response Actions (Escalating)

| Level | Trigger | Action |
|---|---|---|
| 0 — Normal | All signals normal | Full pipeline active |
| 1 — Warning | Any signal at warning | Skip medium drone for low-confidence chunks; batch similar chunks |
| 2 — Critical | Any signal at critical | Skip medium drone entirely; skip remembrance; use cached embeddings only; reduce budget to 2k |
| 3 — Emergency | Multiple critical signals | Fall back to FIFO truncation; log emergency event |

### C.3 Recovery

Degradation levels recover one level at a time, with a cooldown of 10 turns between
recoveries. This prevents oscillation between degradation levels.

---

## 11. Appendix D — Testing Matrix

### D.1 Per-Section Test Requirements

| Section | Unit Tests | Integration Tests | Benchmark Tests | Oracle Tests | Stability Tests |
|---|---|---|---|---|---|
| S0 | Logger, PES, congestion | — | Baseline latency/throughput | — | — |
| S1 | Drone scoring, routing, cache, escalation | Drone pipeline E2E | Latency p50/p95/p99 | — | — |
| S2 | Store, remembrance, decay, dedup, budget, drift | Assembly pipeline E2E | Assembly latency | — | 100-turn assembly |
| S3 | Backend, cache mgr, health monitor, degradation | Full pipeline with LLM | End-to-end latency | — | 100-turn with LLM |
| S4 | Oracle accuracy, GT queries, classifier | A/B framework, ablation | Parameter sweep | Full oracle batch | — |
| S5 | Shadow mode, preloader, rollback, checkpoint | Shadow E2E, rollback E2E | Full benchmark suite | — | 500/1000-turn |

### D.2 Test Data Requirements

- **50 synthetic conversations** (generated by LLM, labeled by human/oracle):
  - 10 short (10–20 turns, single topic)
  - 20 medium (30–50 turns, 2–3 topic shifts)
  - 15 long (80–100 turns, multiple topic shifts, complex dependencies)
  - 5 edge cases (very long single turns, rapid topic switching, contradictory instructions)

- **200 labeled query-chunk pairs** for retrieval precision/recall testing.
- **200 labeled routing decisions** for routing accuracy testing.
- **100 labeled eviction decisions** for false eviction rate testing.

---

## 12. Appendix E — Pitfalls & Gotchas

1. **Never run the oracle inline.** It adds 200–500ms per turn. Always async/batch.
2. **Decay multiplier is not universal.** The optimal value depends on domain and conversation
   patterns. Start with 1.8, tune from logged data in S4.
3. **all-MiniLM-L6-v2 is a starting point, not a final answer.** If retrieval precision is
   below 80% after S4 optimization, swap in a domain-specific embedding model.
4. **Dynamic vocabulary hot-swapping is complex.** Don't attempt it until the static
   vocabulary system is proven. One vocabulary per domain is enough to validate.
5. **PagedAttention is a capability differentiator, not a hard dependency.** The hive works on
   both backends: vLLM enables surgical page-level KV-cache edits (no full recomputation on
   context change); LM Studio (llama.cpp) has no such API and re-processes the prompt each turn,
   but still wins because the hive's compressed context is much smaller than raw history. Run the
   S3.6 side-by-side to quantify the gap and decide which backend to standardize on.
6. **Embedding cache invalidation.** If a chunk's content changes (e.g., user edits a
   previous message), the cached embedding is stale. Hash includes content, so edits
   produce new hashes automatically.
7. **VRAM contention.** The medium drone competes with the primary LLM for VRAM. Monitor
   usage and fall back to CPU for the drone if VRAM is tight.
8. **Logging disk I/O.** NDJSON files grow over long conversations. Rotate daily, compress
   old files, and ensure the logger doesn't block on disk writes (buffered, async flush).
9. **Graceful degradation is not invisible.** When the system degrades, log the event AND
   the reason. Post-mortem analysis of degradation events reveals systemic issues.
10. **A/B test contamination.** If config B's logging affects the system's resource usage
    enough to impact config A's performance, the comparison is invalid. Run shadow configs
    with minimal logging overhead.
11. **Oracle bias.** The oracle uses the same LLM it's evaluating. If the LLM can answer
    correctly despite bad context (by hallucinating or using training data), the oracle
    may rate context as "sufficient" when it isn't. Mitigate by asking specifically about
    context utilization, not just answer correctness.
12. **Don't over-optimize early.** S0–S2 prove the architecture works. S4 optimizes.
    Running parameter sweeps before the pipeline is stable wastes time on parameters that
    may change structurally.

---

## 13. Status Table (update after every section)

| Section | Tag | Commit SHA | Date | Status |
|---|---|---|---|---|
| 0 — Foundation | S0 | — | 2026-08-20 | DONE (harness + tests + mock baselines; live LM Studio baseline pending) |
| 1 — Drone Fleet | S1 | — | 2026-08-20 | DONE (drones, router, escalation, cache; real all-MiniLM verified; 56 unit + 4 integration tests pass) |
| 2 — Hive Context | S2 | — | 2026-08-20 | DONE (store, remembrance, decay, dedup, budget, drift, assembler; 88 unit + 6 integration tests pass) |
| 3 — LLM Integration | S3 | — | 2026-08-20 | DONE (backends, KV-cache mgr, health, degradation, drone pool, interop; mock-verified 100-turn; live vLLM/LM Studio side-by-side pending real backends) |
| 4 — Oracle & Optimization | S4 | — | 2026-08-20 | DONE (oracle, ground-truth DB, A/B, sweeps, ablation, routing classifier; 148 unit + 13 integration tests pass) |
| 5 — Production Hardening | S5 | — | 2026-08-20 | DONE (shadow mode, preloader, rollback, checkpoints, full benchmark, 500-turn stability; 173 unit + 16 integration tests pass) |

---

## 14. Technology Stack (Recommended 2026)

| Component | Technology | Reason |
|---|---|---|
| Host OS | Windows 11 (native; WSL2 optional for vLLM) | Target deployment machine |
| GPU | ≥16 GB VRAM (e.g., RTX 4090) | Fits primary LLM + medium drone |
| Orchestration | Python 3.11+ | Ecosystem (transformers, torch, vllm) |
| Ultra-Small Drone | `sentence-transformers/all-MiniLM-L6-v2` | 80MB, 5ms, CPU-capable, good quality |
| Medium Drone | `microsoft/codebert-base` or `deberta-v3-base` | Domain-aware, 400MB, GPU |
| Primary LLM | Qwen 3.8-27B (IQ4_XS quant) or Qwen 3.6-35B-A3B MoE | High reasoning, fits consumer GPU |
| Inference Backend (primary) | LM Studio (llama.cpp), OpenAI-compatible API on localhost:1234 | Already the host; compressed context still wins |
| Inference Backend (advanced) | vLLM (PagedAttention) | Surgical KV-cache page edits; side-by-side vs. LM Studio in S3 |
| Gatekeeper interop | Consume via `HOST-SEAM.md` contract (`EngineClient`/JSON shapes) | Reuse endpoint, confidence, reliability, rollback logic |
| Logging | NDJSON (one file per day) | Simple, appendable, queryable with jq |
| Ground Truth DB | SQLite | Zero-config, single-file, sufficient scale |
| Testing | pytest + custom harness | Standard Python testing |
| Monitoring | Custom (EventLogger + PipelineHealthMonitor) | No external dependencies |

---

## 15. Key Metrics Summary (Targets)

| Metric | Baseline (LM Studio) | S3 Target | S5 Target |
|---|---|---|---|
| Retrieval precision | N/A (FIFO) | ≥70% | ≥85% |
| Retrieval recall | N/A (FIFO) | ≥75% | ≥90% |
| False eviction rate | ~30% (blind FIFO) | <15% | <5% |
| Routing accuracy | N/A | ≥80% | ≥92% |
| Avg turn latency added | 0ms (no hive) | <50ms | <30ms |
| Context utilization | ~40% (lots of fluff) | 60–80% | 70–90% |
| PES | ~30 (baseline) | ≥65 | ≥80 |
| 100-turn stability | Degrades over time | Stable | Stable |
| 500-turn OOM events | Likely | 0 | 0 |
| Task completion rate | ~60% (long convos) | ≥75% | ≥85% |
