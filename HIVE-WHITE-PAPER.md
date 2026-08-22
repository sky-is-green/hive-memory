# Hive Memory: A Managed-Decay Context Curation Architecture for Long-Horizon LLM Conversations

**A white paper for the open-source LLM and testing community**

---

## Abstract

Large language models (LLMs) exhibit well-documented degradation over long-horizon interactions: performance decays as conversation length grows, relevant information "falls off" rolling context windows, and generation speed slows with growing KV-cache state. We propose **Hive Memory**, an external, multi-agent context curation architecture that decouples *context comprehension* from *context generation*. A fleet of small bidirectional encoder models ("drones") continuously scores, filters, compresses, and reassembles conversation history into a bounded, high-relevance context window that is then delivered to the primary autoregressive model via a cache-managed inference backend.

This paper states the architectural theory, formalizes the mechanisms (targeted masking, tiered routing, managed decay, remembrance passes, deduplication, drift detection), and—critically—presents **labeled, falsifiable predictions** (P1–P10) with explicit logic chains and measurement protocols, so the community can reproduce, challenge, or extend the work. All claims are designed to be testable on consumer hardware with open-weight models.

---

## 1. Introduction & Motivation

### 1.1 The problem

Modern open-weight LLMs (e.g., Qwen 3 27B, Gemma 3) exhibit three compounding failure modes in long conversations:

1. **Context loss (recall failure).** Standard local inference uses a first-in-first-out (FIFO) rolling window. When the window fills, old text is discarded blindly—including foundational instructions, system rules, and early architectural decisions. The model must then guess or hallucinate missing information.

2. **Lost-in-the-middle (attention failure).** Even within a large context, models systematically under-utilize information in the middle of the input relative to the beginning and end (Liu et al., 2023). Large raw context windows therefore do not guarantee large *usable* context.

3. **Quadratic cost growth (compute failure).** Per-token cost grows with total prompt length; generation slows as conversations lengthen, and memory pressure eventually causes out-of-memory (OOM) crashes on consumer hardware.

### 1.2 The proposed remedy

We propose that context management should be **externalized** from the generative model. A small fleet of bidirectional encoders—which can run concurrently and cheaply—pre-processes the entire conversation history and produces a *curated, bounded, high-density context stream*. The primary model receives only what is predicted to matter, at a bounded size, for every turn.

This is a division of labor: the generative model does what it does best (generation); the encoder fleet does what it does best (comparison, similarity, filtering). We hypothesize this decomposition yields strictly better long-horizon behavior than feeding the primary model raw, unbounded context.

---

## 2. Related Work

| Work | Relevance | How Hive differs |
|---|---|---|
| Lost in the Middle (Liu et al., 2023) | Documents the attention-failure phenomenon | Hive treats it as a *design problem to engineer around*, not an irreducible limit |
| Retrieval-Augmented Generation (Lewis et al., 2020) | Retrieval before generation improves groundedness | Hive retrieves from *its own conversation*, not an external corpus, and does so continuously, not per-query |
| Don't Stop Pretraining (Gururangan et al., 2020); SciBERT/BioBERT/CodeBERT | Domain-adaptive pretraining improves downstream performance | Basis for targeted masking and domain-optimized drones |
| PagedAttention / vLLM (Kwon et al., 2023) | Page-level KV-cache management avoids memory fragmentation and enables surgical edits | Hive's KV-cache manipulation depends on this primitive |
| MemGPT (Packer et al., 2023) | OS-inspired memory paging between main and external context | Hive uses *learned relevance scoring* (encoder fleet) rather than OS-style paging heuristics |
| LLMLingua (Jiang et al., 2023) | Prompt compression accelerates inference | Hive compresses *persistent* memory, not just the current prompt, with decay-aware retention |
| Generative Agents (Park et al., 2023) | Agents maintain memory with importance scoring and reflection | Hive formalizes the forgetting side with an explicit, tunable decay matrix |
| Ebbinghaus forgetting curve (1885) | Exponential forgetting over time | The *Sharp Decay Matrix* is a computational analogue, with *escalating* friction on re-saved items |

---

## 3. Architectural Overview

Hive Memory is organized into five functional layers. (Full implementation spec in the companion document, `HIVE-MEMORY-PLAN.md`.)

| Layer | Function | Core mechanisms |
|---|---|---|
| **Cortex** | Orchestration & routing | Task classification, drone fleet dispatch, congestion detection, graceful degradation, auto-scaling |
| **Sieve** | Relevance scoring | Ultra-small encoder (≈80 MB) for fast similarity; medium encoder (≈400 MB, domain-optimized) for uncertain cases; ≥4-character content-word filter; confidence estimation via prediction variance |
| **Membrane** | Selective filtering | Semantic deduplication (cosine > 0.92, keep densest), topic-drift detection and reset |
| **Retention** | Memory & decay | Remembrance pass (eviction interception), Sharp Decay Matrix (exponential, escalating friction), stale-context acceleration |
| **Focal** | Assembly | Adaptive token budget (1k–6k), confidence-weighted sorting, final compressed context construction |

**The data flow per turn:** user input → Cortex classifies complexity → Sieve scores all chunks → Membrane deduplicates and checks drift → Retention applies decay to the surviving chunks → Focal assembles a bounded context → delivered to the primary model via a local inference backend (vLLM or LM Studio / llama.cpp) → response generated → async oracle (offline) labels ground-truth relevance → parameters tuned.

---

## 4. Theoretical Foundations

We ground the architecture in four explicit postulates. Each is stated so it can be accepted, challenged, or refined independently.

### Postulate 1 — The Context Curation Postulate
> For a fixed token budget *B*, a context window assembled by relevance-ranked selection from the full conversation **dominates** a contiguous window of the same size *B* on downstream task quality, and this gap **widens monotonically** with conversation length.

*Rationale:* The contiguous window contains an increasing fraction of stale, off-topic, or low-information tokens as the conversation grows. Relevance-ranked selection concentrates the budget on tokens with high predicted mutual information with the current query.

### Postulate 2 — The Managed Decay Postulate
> Forgetting is a **design parameter**, not a failure mode. Explicitly modeling forgetting with an escalating-friction decay matrix yields better long-horizon recall than either no forgetting (unbounded growth) or uniform forgetting (FIFO).

*Rationale:* Unbounded context causes both quadratic cost and lost-in-the-middle dilution. Uniform forgetting destroys foundational content as readily as trivial content. Escalating friction on re-saved items (the "Sharp Decay Matrix") encodes the intuition that items which must be re-saved repeatedly are either transient (correctly forgotten) or contextually mis-scored (correctly penalized).

### Postulate 3 — The Separation Postulate
> Decomposing context comprehension (bidirectional encoders) from context generation (autoregressive decoder) achieves a more favorable accuracy-per-compute allocation than performing both in the primary model.

*Rationale:* Bidirectional encoders are strictly better at similarity/comparison tasks than causal decoders of similar size, and are orders of magnitude cheaper at the sizes used. The generative model's expensive attention budget should be spent on generation, not on re-discovering which earlier tokens matter.

### Postulate 4 — The Ground-Truth Bootstrap Postulate
> An LLM-as-oracle, prompted to judge **context utilization** rather than answer correctness, can generate relevance labels whose agreement with human labels is sufficient (≥90%) to drive parameter optimization without human annotation.

*Rationale:* Answer-correctness is confounded (models may answer from parametric knowledge despite bad context). Framing the oracle question around "which context was actually used" isolates the retrieval-quality signal.

---

## 5. Hypotheses and Predictions

Each prediction below is **labeled** with its identifier, a falsifiable statement, the **logic chain** (premises → prediction), the **measurement protocol**, and the **falsification condition**. Repeating the protocol with the same hardware/model setup must reproduce the measured outcome.

### P1 — Constant-Throughput Hypothesis
**Prediction:** Tokens-per-second of the primary model remain within ±10% of the turn-10 value across a 500-turn conversation, when fed hive-curated context of bounded size.

**Logic chain:**
1. Generation cost is dominated by total prompt size (KV-cache state) in naive systems.
2. Hive bounds prompt size at the adaptive budget (< 6k tokens) for every turn.
3. Therefore prompt size—and per-token generation cost—is approximately constant.
4. *Conclusion:* throughput is flat.

**Measurement:** Record tokens/sec at turns 10, 50, 100, 200, 500 under (a) naive FIFO and (b) hive-curated context, same model, same hardware, same conversation.
Tokens/sec is the model's *decode* rate, recorded from the backend's `usage.completion_tokens` divided by generation time — not turns/sec, which conflates prefill and hive overhead. Turns whose wall-clock generation time is an extreme outlier (>5× the median) are excluded, since on laptops such spans correspond to OS sleep / idle suspend and would distort the comparison.

**Falsification:** Tokens/sec drops >10% between turn 10 and turn 500 under the hive condition, OR hive throughput is not meaningfully higher than naive at turn 500.

### P2 — Retrieval Precision Hypothesis
**Prediction:** Hive achieves ≥85% retrieval precision and ≥90% recall on a labeled test set of 200 query–chunk pairs extracted from long conversations, versus the near-chance precision of FIFO for chunks older than the window boundary.

**Logic chain:**
1. FIFO retains chunks by recency, not relevance.
2. Sieve scores chunks by semantic similarity to the query.
3. Relevance-ranked selection therefore concentrates on relevant chunks.
4. *Conclusion:* precision/recall exceed recency-based selection.

**Measurement:** Use oracle-labeled ground truth (Postulate 4). Compute precision = relevant_retrieved/total_retrieved; recall = relevant_retrieved/total_relevant over the test set.

**Falsification:** Either metric falls below target on two independent labeled sets.

### P3 — Context Sufficiency Hypothesis
**Prediction:** For equal token budgets, oracle-rated "context sufficiency" is higher for hive-assembled context than for the last-B-tokens FIFO window, on ≥80% of turns sampled in long conversations.

**Logic chain:**
1. Equal budget ⟹ equal compute cost of generation.
2. Hive selects chunks with higher predicted mutual information with the query.
3. Higher mutual information ⟹ higher sufficiency, on average.
4. *Conclusion:* sufficiency is strictly higher for equal cost.

**Measurement:** Paired A/B on the same conversations; oracle rates sufficiency (1–5) blinded to condition; report % of turns where hive ≥ FIFO.

**Falsification:** Hive wins on <80% of turns (i.e., the selection advantage is not reliably realizable).

### P4 — Domain-Dependent Decay Curve
**Prediction:** The optimal initial decay multiplier is domain-dependent: code-heavy conversations and prose conversations yield different optima (estimated: prose 1.4–1.8; code 1.8–2.2), and each is discoverable by replay search.

**Logic chain:**
1. Code context is more modular—dependencies cluster in time (imports, function definitions).
2. Prose context has longer-range topical dependencies.
3. The optimal friction multiplier tracks these dependency structures.
4. *Conclusion:* a single universal multiplier is suboptimal.

**Measurement:** Replay logged conversations under candidate multipliers (1.2, 1.4, 1.6, 1.8, 2.0, 2.2, 2.5) on pre-computed ground truth; plot false-eviction rate vs. multiplier per domain.

**Falsification:** The optima for both domains fall within the same 0.2-wide band (i.e., no meaningful domain separation).

### P5 — Targeted-Masking Hypothesis
**Prediction:** A drone fine-tuned with domain-targeted masked-language modeling (masking only domain vocabulary) achieves higher relevance-scoring accuracy on in-domain text than the same model fine-tuned with uniform random masking, at equal training compute.

**Logic chain:**
1. Random masking spends prediction capacity on trivial tokens (articles, conjunctions).
2. Targeted masking concentrates learning signal on content-bearing, domain-defining tokens.
3. Content-bearing tokens carry most of the relevance signal.
4. *Conclusion:* targeted masking yields better in-domain embeddings.

**Measurement:** Fine-tune identical architectures with random vs. targeted masking on the same domain corpus; compare downstream retrieval precision on a held-out in-domain set.

**Falsification:** No statistically significant precision difference at equal compute, or targeted masking loses on an out-of-domain generalization check.

### P6 — Confidence-Escalation Hypothesis
**Prediction:** Escalating uncertain chunks (high predicted relevance, low confidence) to a medium drone improves recall ≥5% over accepting low-confidence small-drone scores, while escalating <15% of chunks.

**Logic chain:**
1. Small encoders are occasionally wrong on difficult pairs but know when they are uncertain (variance signal).
2. A larger, domain-aware encoder is more accurate on those pairs.
3. Escalating only the uncertain minority concentrates the extra compute where it helps.
4. *Conclusion:* recall improves with bounded compute overhead.

**Measurement:** On labeled pairs, compare recall of (a) small-drone-only vs. (b) escalate-when-uncertain; record escalation rate.

**Falsification:** Recall improves <5%, or the escalation rate exceeds 15% for a 5% gain.

### P7 — Oracle-Agreement Hypothesis
**Prediction:** LLM-as-oracle relevance labels, framed as context-utilization questions, agree with human labels on ≥90% of a 500-item sample.

**Logic chain:**
1. Utilization-framing removes the parametric-knowledge confound.
2. The oracle's own relevance judgments are internally consistent.
3. Human raters agree with each other ≥90% on clear relevance.
4. *Conclusion:* oracle–human agreement reaches 90%.

**Measurement:** Dual annotation (oracle + 2 human raters) on 500 sampled chunks; report agreement; resolve ties by a third rater.

**Falsification:** Oracle–human agreement <90%, or human–human agreement < oracle–human agreement (indicating the oracle is not measuring what humans measure).

### P8 — Heuristic-Routing Floor Hypothesis
**Prediction:** Heuristic routing (keywords, patterns, message length, conversation depth) achieves ≥85% of the routing accuracy of a classifier trained on 1,000 logged decisions, and both exceed 85% absolute accuracy.

**Logic chain:**
1. Task complexity correlates with observable surface signals (keyword density, length).
2. These signals carry most of the variance in optimal drone choice.
3. A classifier adds marginal signal beyond the heuristics.
4. *Conclusion:* simple rules capture most of the value.

**Measurement:** Compare heuristic vs. trained-classifier routing accuracy against oracle-optimal routes on held-out data.

**Falsification:** The classifier beats heuristics by >15 points of accuracy (i.e., surface signals are weakly predictive and the "start simple" strategy is wrong).

### P9 — Densest-Duplicate Hypothesis
**Prediction:** When semantically duplicate chunks are found (cosine > 0.92), retaining the information-densest version—rather than the most recent—improves downstream task quality per token, measured by oracle sufficiency.

**Logic chain:**
1. Recent duplicates are often restatements or conversational recaps (low density).
2. The densest version carries the same information in fewer tokens.
3. Fewer tokens ⟹ higher information density ⟹ better budget use.
4. *Conclusion:* densest retention dominates recency retention.

**Measurement:** Paired A/B on conversations with engineered duplicates; measure sufficiency per 1,000 tokens of context.

**Falsification:** Recency retention wins on ≥55% of turns, or no measurable difference.

### P10 — Drift-Reset Hypothesis
**Prediction:** An explicit topic-drift-triggered context reset improves oracle-rated task quality within 3 turns after a topic change, relative to decay-without-reset.

**Logic chain:**
1. After a hard topic change, old-topic context is systematically irrelevant.
2. Gradual decay removes it slowly; an explicit reset removes it immediately.
3. Immediate removal frees budget for new-topic context sooner.
4. *Conclusion:* reset accelerates recovery.

**Measurement:** Inject engineered topic changes into test conversations; compare oracle sufficiency at turns +1..+3 after the change, reset vs. no-reset.

**Falsification:** No sufficiency improvement within 3 turns, or reset causes a regression (e.g., discarding genuinely cross-cutting context).

---

## 6. The Pipeline Efficiency Score (PES)

To make optimization measurable rather than impressionistic, we define a single composite metric. PES is a weighted sum of five normalized components:

```
PES = 0.30·RetrievalPrecision
    + 0.20·RoutingAccuracy
    + 0.20·LatencyHealth
    + 0.15·ThroughputHealth
    + 0.15·ContextUtilization
```

- **RetrievalPrecision:** % of retrieved chunks the oracle confirms were actually relevant.
- **RoutingAccuracy:** % of routing decisions that matched the oracle-optimal drone tier.
- **LatencyHealth:** `max(0, 100 − (avg_latency_ms − 50) × 0.67)` — 50 ms = 100, 200 ms = 0.
- **ThroughputHealth:** actual tokens/sec ÷ baseline tokens/sec × 100.
- **ContextUtilization:** 100 at 60–95% budget fill; penalized below 60% (under-use) and above 95% (truncation risk).

**Interpretation bands:** ≥80 GREEN (healthy) · 60–79 YELLOW (investigate) · <60 RED (auto-rollback). The weights are *not* claimed to be optimal a priori; they are the initial configuration and are themselves subject to sensitivity analysis.

---

## 7. Experimental Protocol (Reproducibility)

To make all predictions reproducible on consumer hardware with open weights, we fix the following protocol.

**Hardware baseline:** Windows 11 (native; WSL2 optional for vLLM) on a single consumer GPU
with ≥16 GB VRAM (e.g., RTX 4090); 32 GB system RAM.

**Models (fixed):**
- Primary: Qwen 3 27B (IQ4_XS quantization).
- Ultra-small drone: `sentence-transformers/all-MiniLM-L6-v2` (CPU-capable).
- Medium drone: domain-optimized encoder (e.g., `microsoft/codebert-base` for code; `deberta-v3-base` for prose).

**Backend (dual):**
- **LM Studio (llama.cpp)** — OpenAI-compatible API on `localhost:1234`; the primary host. No
  surgical KV-cache API; benefits from the hive solely via the smaller compressed context.
- **vLLM (PagedAttention)** — OpenAI-compatible API on `localhost:8000`; enables surgical
  page-level KV-cache edits. Both backends are run on the same conversations in a side-by-side
  comparison; all predictions are evaluated on each.

**Gatekeeper interop:** where the existing Gatekeeper Studio engine already solves a problem
(endpoint resolution, confidence calibration, reliability tracking, rollback, gate lifecycle),
the hive consumes it through the documented `HOST-SEAM.md` contract rather than re-implementing
it.

**Test corpus (fixed):**
- 50 synthetic conversations: 10 short (10–20 turns, single topic), 20 medium (30–50 turns, 2–3 topic shifts), 15 long (80–100 turns, multiple shifts, cross-cutting dependencies), 5 edge cases (rapid topic switching, contradictory instructions).
- 200 labeled query–chunk pairs (precision/recall).
- 200 labeled routing decisions.
- 100 labeled eviction decisions (false-eviction rate).

**Labeling:** dual human annotation for the ground-truth sets; oracle (Postulate 4) for bulk labeling; agreement checks per P7.

**Baselines (both must be measured):**
1. *Naive FIFO:* last-B-tokens rolling window, no hive.
2. *No-hive truncation:* same model, fixed 4k truncation, no curation.

**Ablation set (component attribution):** full system; minus decay; minus drones (random filtering); minus remembrance; minus dedup; minus adaptive budget (fixed 4k); minus drift; baseline. Each run on the full test corpus.

**Metric recording:** every decision logged to NDJSON (timestamped, structured); PES computed every 5 turns; full log retention for replay-based parameter sweeps (P4).

---

## 8. Predicted Outcome Summary

| Metric | Naive FIFO (baseline) | Hive S3 target | Hive S5 target |
|---|---|---|---|
| Retrieval precision | near-chance past window | ≥70% | ≥85% |
| Retrieval recall | n/a (FIFO) | ≥75% | ≥90% |
| False eviction rate | ~30% | <15% | <5% |
| Routing accuracy | n/a | ≥80% | ≥92% |
| Added per-turn latency | 0 ms | <50 ms | <30 ms |
| Context utilization | ~40% (fluff) | 60–80% | 70–90% |
| PES | ~30 | ≥65 | ≥80 |
| OOM events (500-turn) | likely | 0 | 0 |

---

## 9. Threats to Validity & Known Limitations

1. **Oracle circularity.** The oracle is the same model family being served. P7 mitigates via agreement checks, but agreement is not ground truth; a systematic shared bias between oracle and model is possible.
2. **Synthetic-corpus bias.** Generated conversations may not capture real user messiness. Real-world validation after S5 is mandatory before generalizing.
3. **Hardware ceiling.** VRAM contention between the medium drone and the primary model may force CPU fallback, altering latency predictions.
4. **Decay realism.** The Sharp Decay Matrix is an engineering heuristic with cognitive-science inspiration, *not* a validated model of human forgetting; its value is empirical, not neuroscientific.
5. **Confounded ablations.** Disabling a component changes downstream decisions of other components; ablation results should be read as upper bounds on component contribution.
6. **Small-model ceiling.** The drones are deliberately small; the entire architecture's ceiling may be bounded by encoder quality, which the medium-drone tier only partially addresses.
7. **Prefix-cache attribution.** On backends with automatic prefix caching (LM Studio / llama.cpp), flat throughput is co-produced by the hive's bounded context *and* a byte-stable pinned system prefix whose KV is reused every turn. The naive FIFO baseline's window shifts each turn and never reuses KV, so the P1 comparison is "curation + stable prefix" vs. "shifting window" rather than raw context length. The falsification conditions are unchanged, but the throughput claim is attributable to both mechanisms, and replication on a backend without prefix caching (e.g., vLLM without pinned pages) may observe a smaller gap.

---

## 10. Open Questions

1. Does the optimal decay curve converge across domains, or is per-domain tuning mandatory (P4)?
2. At what conversation length does hive-curated context *stop* dominating FIFO (i.e., is there a crossover point below which the added latency is pure overhead)?
3. Does the remembrance pass interact constructively with the drift reset, or do they fight (saved old-topic content vs. reset-to-new-topic)?
4. Can the oracle labels be fed back to *distill* the medium drone's capability into the ultra-small drone, eliminating the escalation tier over time?
5. Does hive-curated context measurably reduce hallucination on factual consistency checks, or only improve relevance?

---

## 11. Licensing, Reproducibility, and Community Invitation

- All code, logs, and protocols will be released open-source.
- All test corpora are synthetic and will be published for reuse.
- We explicitly invite: (a) independent replication of P1–P10 on different hardware, (b) adversarial construction of conversations that defeat the architecture, and (c) contribution of domain-specific masked vocabularies and drone checkpoints to the project repository.

---

## References

1. Ebbinghaus, H. (1885). *Über das Gedächtnis* (On Memory).
2. Lewis, P., et al. (2020). Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks. *NeurIPS*.
3. Gururangan, S., et al. (2020). Don't Stop Pretraining: Adapt Language Models to Domains and Tasks. *ACL*.
4. Liu, N., et al. (2023). Lost in the Middle: How Language Models Use Long Contexts. *TACL*.
5. Kwon, W., et al. (2023). Efficient Memory Management for Large Language Model Serving with PagedAttention. *SOSP*.
6. Jiang, H., et al. (2023). LLMLingua: Compressing Prompts for Accelerated Inference of Large Language Models. *EMNLP*.
7. Packer, C., et al. (2023). MemGPT: Towards LLMs as Operating Systems.
8. Park, J., et al. (2023). Generative Agents: Interactive Simulacra of Human Behavior. *UIST*.