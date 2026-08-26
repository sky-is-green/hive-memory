# P12 (DRAFT) — Store-Time Fact Distillation for Comb Resurrection

**Status:** protocol proposal — NOT yet a numbered prediction. No implementation.
Promotion path: human approves spec → implement corpus + probe → first measurement →
then the prediction graduates to P12 with its baseline frozen.

---

## Prediction statement (draft)

> Distilling store-evicted chunks into **atomic fact statements** before archiving them
> to the comb raises lexical retrievability of return-turn candidates from the measured
> **45% baseline** to **≥ 75%**, without displacing relevant active-store chunks
> (crowding regression ≤ 2 points on non-return turns).

## Logic chain

1. Return turns fail when archived chunks do not *lexically name* their facts:
   comb_probe measured only **45% of return turns lexically retrievable**, and lexical
   ranking beat semantic ranking on retrievable turns (76.4% vs 69.8% recall@3).
2. Same-domain raw chunks score near-identically under embedding similarity (Threat 6,
   six-encoder ceiling) — but fact statements *are* their keywords, so distillation
   moves content across the lexical gap that defeats cosine ranking.
3. Therefore: distill-at-archive converts the comb's weakest stage (candidate naming)
   into its strongest stage's input, at one-time CPU/LLM cost per archived chunk.

## Corpus

- Reuse `--return` fixture corpus and the comb_probe harness unchanged (comparability).
- Distiller: two arms, measured separately — (a) extractive (sentence scoring by the
  existing ultra drone, zero tokens), (b) abstractive via the served model offline
  (one call per archived chunk, cost counted in the report).

## Metric

Primary: % of return turns where the expected fact string is present in the assembled
context with comb enabled vs disabled (existing P11 diagnostic math).
Secondary: crowding on non-return turns; distillation token/call cost per chunk;
false-fact rate (distiller hallucination check against source chunk).

## Falsification

- Extractive arm fails to clear 60% (halfway) → premise wrong: facts do not survive
  sentence-scoring distillation either; close option-b for extractive approaches.
- Both arms < 75% with no crowding improvement → Threat 6 option-b closed entirely;
  record as a negative result in §10.
- Crowding regresses > 2 points → distillation changes chunk identity enough to break
  the comb gate; redesign before any claim.

## Confound controls

- Same budget (1000), same drone, same seed corpus, isolation per store (Threat 8a),
  hedge filtering unchanged, budget fixed not adaptive (P4 lesson).
- Blind scoring: retrieval diagnostic remains deterministic; no queen involvement.

## Relation to existing results

- Extends P11 (comb resurrection) rather than replacing it: P11 proved resurrection
  works when candidates are lexically named; P12 asks whether distillation can *make*
  them named.
- Directly motivated by the 2026-08-26 paired A/B: strict losses concentrate on
  recency/drift-heavy turns where fidelity decides winners — precision, not presence,
  is the binding constraint (Threat 6 option b).

## Open design questions (for review before implementation)

1. Extractive-only vs LLM-assisted arms — run both or sequence?
2. Distill at eviction time vs lazily at gate-fire time?
3. Does distillation interact with remembrance-pass decay multipliers?
