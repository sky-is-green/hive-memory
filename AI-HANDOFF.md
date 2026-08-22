# AI Handoff — Hive Memory / HiveBench Project State

This document summarizes everything built so another AI (or human) with file
access to this repo can understand the project, run it, and continue the work.

---

## 1. Project overview

**Hive Memory** is an *external, multi-agent context-curation layer* that sits
between a user and a local LLM backend. It filters, scores, compresses, and
reassembles conversation history into a bounded, high-relevance context window
for every turn, so a generative model performs well over arbitrarily long
conversations on consumer hardware.

- `HIVE-MEMORY-PLAN.md` — the executable implementation plan (sections S0–S5).
- `HIVE-WHITE-PAPER.md` — the theory + 10 falsifiable predictions (**P1–P10**).
- `README.md` — the benchmark usage guide (named **HiveBench**).
- `HIVE-DIAGRAMS.md` — diagrams: why HIVE beats no-HIVE, why testing is thorough,
  and the improvement action plan.

**Core idea:** small bidirectional "drone" encoders (fast, cheap) do the
*comprehension* (scoring/filtering context) while the primary generative LLM does
the *generation*. This is the paper's "Separation Postulate."

---

## 2. Environment & hardware constraints

- **OS:** Windows 11, PowerShell. **Python:** 3.14.7, venv at `.venv`.
- **GPU:** AMD Radeon RX 7900 XT. **No NVIDIA GPU.**
- **Consequence:** vLLM (needs CUDA or ROCm-on-Linux) is **NOT usable** here.
  **LM Studio (llama.cpp) is the sole live backend** (OpenAI-compat API on
  `localhost:1234`).
- Installed: `torch 2.13+cpu`, `sentence-transformers 6.0`, `transformers 5.15`,
  `numpy`, `scipy`, `scikit-learn`, `requests`, `pytest`, `psutil`,
  `sentencepiece`. See `requirements.txt` / `requirements-dev.txt`.

---

## 3. Architecture & repo layout

The code is organized by the white paper's layers (not the plan's `hive/`
naming). `__init__.py` files exist for all packages.

| Path | Contents |
|---|---|
| `cortex/` | routing, PES (`efficiency.py`), congestion, health, degradation, drone_pool, interop (Gatekeeper seam), config (`HiveConfig`), checkpoint, rollback, classifier, **hive** (unified orchestrator), e2e, sanitize, tokenizer, `baselines/` |
| `sieve/` | **drones**: `ultra_small.py`, `medium.py`; `scores.py`, `vocabulary.py`, `embedding_cache.py` |
| `membrane/` | `dedup.py`, `drift.py` |
| `retention/` | `store.py`, `remembrance.py`, `decay.py` |
| `focal/` | `budget.py`, `assembly.py` (ContextAssembler), `predictive.py` |
| `backend/` | `base.py`, `openai_compat.py`, `vllm.py` (dormant), `lmstudio.py`, `cache_manager.py` |
| `oracle/` | `async_oracle.py`, `ground_truth.py` (SQLite), `labeling.py` |
| `logs/` | `event_logger.py`, `query.py` |
| `testing/` | `ab_test.py`, `ablation.py`, `optimization.py`, `shadow_mode.py` |
| `experiments/` | `generate_data.py` (live benchmark), `run_p1_p10.py` (protocol), `p5_targeted_masking.py`, `dashboard.py` (KeepAwake + terminal dashboard), `launcher.py` (Tkinter run app) |
| `tests/` | `run_hive_tests.py` (grouped runner), `unit/`, `integration/`, `benchmarks/`, `fixtures/` |
| `vocab/` | `code.json`, `general.json` (domain relevance vocab) |

---

## 4. What was built (by section + extras)

### S0 — Foundation
- `logs/event_logger.py`: NDJSON logger with **daily + size rotation, gzip
  archiving, retention, secret redaction, correlation IDs** (run/conversation/turn),
  async buffered writes, model-tagged.
- `cortex/efficiency.py`: **Pipeline Efficiency Score (PES)** — weighted composite
  (0.30 retrieval, 0.20 routing, 0.20 latency, 0.15 throughput, 0.15 utilization)
  with missing-component renormalization and GREEN/YELLOW/RED/CRITICAL bands.
- `cortex/congestion.py`: queue-depth/latency/backlog thresholds.
- Synthetic corpus: 50 conversations (`tests/fixtures/generated/`), generator at
  `tests/fixtures/synthetic_conversations/generate.py`.
- Baselines harness (`cortex/baselines/`): LM-Studio rolling + FIFO runners.
- Benchmarks (`tests/benchmarks/`).

### S1 — Drone fleet
- `sieve/ultra_small.py`: `UltraSmallDrone` (all-MiniLM-L6-v2), cosine scoring,
  **confidence modes** (`mcdropout`/`single`/`off`), vocab relevance boost.
- `sieve/medium.py`: `MediumDrone` (graphcodebert-base), **cross-encoder AND
  bi-encoder (`mode="bi"`) scoring**, `trust_remote_code`/`add_eos_token` options.
- `sieve/embedding_cache.py`: LRU cache, **model-tagged + persistent to disk**
  (returns empty on model mismatch), `namespace(model, dir)`.
- `sieve/vocabulary.py`: domain vocab loader/booster.
- `cortex/routing.py`: `DroneRouter` (heuristics) + `EscalationHandler`.
- `cortex/classifier.py`: `RoutingClassifier` (shallow decision tree).

### S2 — The Hive (context management)
- `retention/store.py` (`ContextStore`), `retention/remembrance.py`,
  `retention/decay.py`, `membrane/dedup.py`, `membrane/drift.py`,
  `focal/budget.py` (`AdaptiveBudget`), `focal/assembly.py` (`ContextAssembler`,
  with **skip_remembrance/skip_dedup flags + per-stage timing**).

### S3 — Integration & health
- `backend/`: `LLMBackend` abstraction, `OpenAICompatBackend` (**single leading
  system message** — required by strict templates like bonsai-27b),
  `VLLMBackend` (dormant), `LMStudioBackend`.
- `backend/cache_manager.py`: `KVCacheManager` — **surgical (vLLM)** or
  **automatic prefix-caching (LM Studio)**: keeps a stable pinned prefix first so
  llama.cpp reuses KV.
- `cortex/health.py` (`PipelineHealthMonitor`), `cortex/degradation.py`
  (levels 0–3), `cortex/drone_pool.py`, `cortex/interop.py` (Gatekeeper seam).

### S4 — Oracle & optimization
- `oracle/async_oracle.py` (LLM-as-judge), `oracle/ground_truth.py` (SQLite
  precision/recall/false-eviction/routing metrics), `oracle/labeling.py`
  (ground-truth label generation).
- `testing/ab_test.py` (**Mann-Whitney statistical significance**),
  `testing/ablation.py` (8 configs), `testing/optimization.py` (replay-based
  sweeps), `testing/shadow_mode.py`.

### S5 — Hardening
- `testing/shadow_mode.py`, `focal/predictive.py`, `cortex/rollback.py`
  (AutomatedRollback), `cortex/checkpoint.py` (**path-traversal-safe**),
  `tests/benchmarks/full_benchmark.py` (with **peak RSS** via psutil),
  500-turn stability test.

### Extras / cross-cutting
- `cortex/hive.py` — **unified `Hive` orchestrator**: `process_turn()` wires
  store + drones + router + assembler + health + degradation + KV-cache + logger,
  with latency breakdown, degradation-driven behavior, and **resilient per-turn
  error handling** (logs + continues instead of crashing).
- `cortex/config.py` — **`HiveConfig`** (all tunables incl. `ultra_model`,
  `medium_model`, `enable_medium`, `sanitize_context`), save/load, Gatekeeper
  overrides.
- `cortex/sanitize.py` — prompt-injection / context-poisoning sanitizer.
- `cortex/tokenizer.py` — heuristic + optional real tokenizer.
- `cortex/e2e.py` — lightweight end-to-end runner.
- `experiments/generate_data.py` — **live data-generation benchmark** with
  per-phase progress bar (%, elapsed, ETA), **reply caps, checkpoint/resume,
  run-dir lock, `--confidence`, `--no-thinking`, `--term`, post-run PES** (see §7).
- `experiments/run_p1_p10.py` — **P1–P10 protocol driver** (mock or live).
- `experiments/p5_targeted_masking.py` — targeted-vs-random MLM training.
- `experiments/dashboard.py` — **`KeepAwake`** (Windows `SetThreadExecutionState`
  `ES_SYSTEM_REQUIRED`, auto on for live runs) and **`TermDashboard`** (ANSI
  terminal live dashboard via `--term`: phase, progress, ETA, rolling stats,
  recent-turn feed; no-op when stdout isn't a TTY).
- `experiments/launcher.py` — **Tkinter run-configurator app** (`python -m
  experiments.launcher`): checkboxes/dropdowns → builds & runs the `generate_data`
  command with output streamed in-window; hover tooltips on every control;
  auto-fills the loaded LM Studio model and the newest resume checkpoint; runs the
  benchmark with the **venv python from the repo root** (`run_python()` + `cwd=REPO_ROOT`).

---

## 5. Key technical decisions & lessons learned

1. **AMD/no-NVIDIA → LM Studio is the sole live backend.** vLLM stays dormant but
   mock-tested. LM Studio gets *automatic prefix caching* (not surgical KV edits).
2. **Real-model template strictness:** `bonsai-27b` rejected a second system
   message (`"System message must be at the beginning"`). Fix: merge pinned
   prefix + assembled context into **one leading system message**.
3. **Drones run in Python, not LM Studio.** Only the ultra-small drone is active;
   the medium drone is **opt-in** (`enable_medium=False`) because it's heavy and
   VRAM-contending and rarely fires (stock embeddings give confidence ≈ 1.0, so
   escalation never triggers).
4. **Drone/model swaps are now drop-in:** `HiveConfig` fields, model-tagged
   embedding cache (safe across dimension changes), bi-encoder support.
5. **Mock validates code; live validates science.** The test suite proves
   correctness; P1–P10 need live data. In mock, some predictions report FAIL
   honestly (fake drone ≠ real retrieval) — that's expected, not a bug.
6. **`confidence_mode=off` is the correct default.** The stock all-MiniLM drone
   disables dropout at inference → all MC-dropout passes are identical →
   confidence is always 1.0, so `mcdropout`/`single` add encode cost with no
   signal (live scoring grew ~5s → ~12s/turn). Enable `mcdropout` only with a
   dropout-active encoder (custom drone / P6 escalation).
7. **Reasoning models must have thinking disabled to be usable fast.** bonsai-27b
   (and similar) spend their output budget on chain-of-thought first, so a small
   `--max-tokens` cap yields empty replies. Disable thinking via LM Studio's
   "thinking" toggle and/or `--no-thinking` (`enable_thinking=false`); then reply
   caps work and turns are much faster. Reasoning is pure overhead for P1–P10 —
   no prediction depends on it.
8. **Run tooling (all added post-handoff):** `--max-tokens`/`--baseline-max-tokens`
   reply caps, `--checkpoint-every`/`--resume` (interruption-safe runs), run-dir
   lock, keep-awake (ES_SYSTEM_REQUIRED), `--term` terminal dashboard, post-run
   ground-truth PES in `run_report.json`, and `experiments.launcher` (a Tkinter
   app that builds & runs the benchmark command).
9. **Live per-turn PES is structurally depressed — read `post_run_pes` instead.**
   The in-process per-turn PES (`Hive.process_turn`) only sees latency +
   utilization; `LatencyHealth` is ms-calibrated (50ms=100, 200ms=0) while live
   generation is seconds, so it's always 0 and live per-turn PES ≈ ~5–15 (mostly
   context-utilization, which climbs as the store fills). The meaningful headline
   is `run_report.json` → `post_run_pes`, computed after the run from oracle
   retrieval/routing + measured latency/tps/utilization.
10. **Oracle robustness:** `AsyncOracle._extract_json` parses real LLM output
    (markdown fences, prose-wrapped JSON, trailing garbage; raises `ValueError` on
    truly empty). `_populate_ground_truth` wraps each call in try/except so one bad
    oracle label logs `oracle/label_failed` and continues instead of killing the
    run. `_live_oracle` clears the E2E pinned prefix, frames JSON as the system
    message, and uses a reasoning-safe budget.
11. **Launcher must use the venv python + repo-root cwd.** Launching the launcher
    with the system Python (`pythoncore-3.14-64`) and no `cwd` made the benchmark
    subprocess fail with `ModuleNotFoundError: No module named 'experiments'`.
    Fixed via `run_python()` (prefers `.venv\Scripts\python.exe`) and
    `cwd=REPO_ROOT` in `subprocess.Popen`.
12. **Model speed matters more than any harness tweak.** bonsai-27b is ~14 t/s with
    ~500 mandatory reasoning tokens (~40s min/turn); capping below ~512 yields
    empty replies. For iteration, load a faster model (the loaded MoE family
    `qwen3.6-35b-a3b-*` should be ~5–10x faster) and pass `--model <id>`.

---

## 6. Test suite (grouped runner)

`tests/run_hive_tests.py` groups tests by what they measure, each with an
estimated + measured duration:

| Group | Measures | Time |
|---|---|---|
| `speed` | performance (latency/throughput/memory/drones) | ~80s |
| `intelligence` | accuracy (retrieval, classifier, oracle, A/B, P1–P10, P5) | ~10s |
| `skills` | functionality (drones, hive, backends, security, E2E) | ~35s |
| `maximum` | everything (default) | ~2 min |

```powershell
.\.venv\Scripts\python tests\run_hive_tests.py --group speed|intelligence|skills|maximum
```

~283 unit + integration tests; all groups pass. Benchmarks include:
per-turn assembly p50 ≈ **3.4ms**, throughput ≈ **285 turns/s**, peak RSS ≈
**34.7 MB**, real all-MiniLM per-pair p50 ≈ **13–15ms**, classifier p50 ≈ **0.04ms**.

**Model-swap guidance:** `skills` + retrieval/routing are LLM-independent — re-run
only on drone/code changes. On an LLM swap, re-run the live benchmark instead.

---

## 7. Live data-generation benchmark

```powershell
# Fast iteration / validate pipeline (thinking off, resumable, watchable):
.\.venv\Scripts\python -m experiments.generate_data --live --no-thinking --confidence off --term --checkpoint-every 5 --max-convs 3 --max-turns 10

# Full evidence run (overnight, no reply cap; point --model at a fast MoE if loaded):
.\.venv\Scripts\python -m experiments.generate_data --live --model <id> --max-convs 20 --protocol --baselines --confidence off --checkpoint-every 5

# Resume an interrupted run from its checkpoint:
.\.venv\Scripts\python -m experiments.generate_data --live --resume runs/<ts>

# Offline / synthetic:
.\.venv\Scripts\python -m experiments.generate_data --mock --max-convs 5 --protocol

# No-command launcher app (builds & runs the command for you):
.\.venv\Scripts\python -m experiments.launcher
```

Flags: `--live`/`--mock`, `--max-convs`, `--max-turns`, `--protocol` (P1–P10),
`--baselines` (rolling + FIFO comparisons), `--output`, `--base-url`, `--model`,
`--pinned-prefix`, `--max-tokens` (reply cap; blank = uncapped 4096 ceiling),
`--baseline-max-tokens`, `--confidence mcdropout|single|off`, `--no-thinking`
(`enable_thinking=false`), `--checkpoint-every N`, `--resume DIR`, `--term`
(terminal dashboard). Phases: `1/3 E2E` → `ground truth` → `2/3 protocol` →
`3/3 baselines`.

Each run writes `runs/<ts>/`: `run_report.json`, `ground_truth.sqlite`,
`baseline_lm_studio.json`, `baseline_fifo.json`, `logs/events-*.ndjson`,
`checkpoint.json` (every N turns, for `--resume`), `run.lock` (double-launch guard).

---

## 8. Reading & quantifying results

- **`run_report.json`** — `aggregate` (PES min/avg, turns, latency, fallbacks),
  per-conversation turns (incl. per-turn `completion_tokens`), `protocol`,
  `baselines`, `ground_truth`, and **`post_run_pes`** (the headline).
- **PES** (0–100): `0.30·Retrieval + 0.20·Routing + 0.20·Latency +
  0.15·Throughput + 0.15·Utilization`. Bands: ≥80 GREEN, 60–79 YELLOW, 40–59 RED,
  <40 CRITICAL. **Per-turn PES is NOT meaningful live** (ms-calibrated latency vs
  seconds-scale generation → floors near 0); read `post_run_pes` for the real
  score, and its `notes` field explains any floors.
- **`post_run_pes`** — the paper's PES computed post-run via
  `experiments.generate_data._compute_post_run_pes`: oracle `retrieval_precision`
  (from `ground_truth.sqlite`) + `routing_accuracy` + measured
  latency/throughput/utilization, with component scores in `components` and the
  raw measurements in `measurements`.
- **`ground_truth`** — `oracle_labels`, `retrieval_precision`, `retrieval_recall`,
  `false_eviction_rate`, `routing_accuracy` (from `oracle.ground_truth.GroundTruthDB`).
- **P1–P10** each return `{id, title, status (PASS/FAIL/SKIP/REPORT), evidence,
  note}`. SKIP = needs live data / P5 training / human labels. P1 now measures
  real decode tps from `usage.completion_tokens` and excludes OS-sleep-contaminated
  turns (>5× median generation time).
- **Baselines** — compare hive PES vs rolling/FIFO PES to show hive benefit.
- **`logs.query`**: `python -m logs.query --dir runs/<ts>/logs --include-archive`.

---

## 9. Current status & recommended next steps

**Status:** All S0–S5 + the post-handoff run tooling (below) is built. Full offline
suite green (~283 tests). **No complete successful full evidence run exists yet.**

**Live-run history (each informed fixes):**
- `runs/20260820_222808` — very first live attempt; failed instantly (real-model
  template bug, pre-resilience). Empty logs.
- `runs/20260820_223616` — long run, killed after **308 turns**; wall-clock was
  inflated by the PC sleeping overnight (not a hang — sleep suspension pauses the
  process). Marked `status: interrupted` with **`timing_analysis.json`**: 308 turns
  of flat generation (~62s median, 1 sleep-contaminated turn at 179) — useful
  P1-relevant evidence that the bounded context keeps decode time constant.
- `runs/20260821_164839` — 13-turn run with `--max-tokens 128` → **empty replies**
  and empty oracle labels, because bonsai-27b is a reasoning model that spent the
  whole budget on chain-of-thought. Fixed: no cap by default (4096 ceiling),
  `--no-thinking`, reasoning-safe oracle, empty-reply warning.

**Key decisions made this session:**
1. No reply cap by default (`DEFAULT_MAX_TOKENS = 4096`); `--max-tokens` is opt-in.
2. `--no-thinking` (and LM Studio's "thinking" toggle) to disable reasoning for
   speed — no white-paper prediction depends on reasoning.
3. `confidence_mode=off` default (stock drone yields no variance).
4. Keep the **terminal dashboard** (`--term`); the Tkinter *live-progress window*
   was removed at the user's request, replaced by the **launcher app**.
5. White paper updated (P1 measurement = real decode tps + sleep-outlier exclusion;
   Threats item 7 = prefix-cache attribution caveat). Falsification conditions
   unchanged.

**Next steps (in order):**
1. **Run a clean live iteration** — `--live --no-thinking --confidence off --term
   --checkpoint-every 5 --max-convs 3 --max-turns 10` (ideally with a fast MoE
   model via `--model`) and read `post_run_pes` + `ground_truth` (oracle should now
   label; if a label still fails it logs `oracle/label_failed` and continues).
2. **Validate P1–P10 with live data** (P1 now measures real decode tps and skips
   sleep-contaminated turns).
3. **P5 targeted-masking training** (`experiments.p5_targeted_masking`) and
   optionally **P7 human labeling**.
4. **Speed the live test**: load a fast model for iteration (MoE qwen3.6-35b-a3b);
   skip `--protocol --baselines` while iterating; the final evidence run stays
   full-fidelity and is best run overnight (keep-awake is automatic).
5. **Optional: package** — `pyproject.toml`, CLI entry points, LICENSE, CI — only
   after live evidence.

**Long runs are slow because** generation dominates (bonsai ~14 t/s + ~500
mandatory reasoning tokens ≈ ~40s/turn minimum). Disable thinking and/or use a
faster model for iteration; run the full canonical run only for final evidence.

---

## 10. Command cheat sheet

```powershell
# run a test group
.\.venv\Scripts\python tests\run_hive_tests.py --group maximum

# fast live iteration (thinking off, resumable, watchable)
.\.venv\Scripts\python -m experiments.generate_data --live --no-thinking --confidence off --term --checkpoint-every 5 --max-convs 3 --max-turns 10

# full evidence run (overnight; add --model <fast-moe-id> if loaded)
.\.venv\Scripts\python -m experiments.generate_data --live --max-convs 20 --protocol --baselines --confidence off --checkpoint-every 5

# resume an interrupted run
.\.venv\Scripts\python -m experiments.generate_data --live --resume runs/<ts>

# offline check
.\.venv\Scripts\python -m experiments.generate_data --mock --max-convs 5 --protocol

# launcher app (builds & runs the command)
.\.venv\Scripts\python -m experiments.launcher

# P1-P10 protocol only
.\.venv\Scripts\python -m experiments.run_p1_p10 --mock   # or --live

# P5 training smoke
.\.venv\Scripts\python -m experiments.p5_targeted_masking --quick

# generate ground-truth labels
.\.venv\Scripts\python -m oracle.labeling

# read event logs
.\.venv\Scripts\python -m logs.query --dir runs/<ts>/logs --include-archive
```
