# HiveBench — The Hive Memory Evaluation Suite

**HiveBench** is the test + benchmark harness for the **Hive Memory** architecture
(a managed-decay context-curation layer for long-horizon LLM conversations). It
validates that the system is correct, measures its performance, and — crucially —
drives the white paper's falsifiable predictions (P1–P10) against a live or mock
LLM backend.

It is two things in one:

1. **A grouped test runner** (`tests/run_hive_tests.py`) — fast, targeted test
   groups by what they measure.
2. **A live data-generation benchmark** (`experiments/generate_data.py`) — runs
   real conversations through the full pipeline and writes a self-contained,
   queryable result bundle.

---

## Quick start

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements.txt -r requirements-dev.txt

# Run the full offline suite (~2 min):
.\.venv\Scripts\python tests\run_hive_tests.py --group maximum

# No-command launcher app (select tests, build & run):
.\.venv\Scripts\python -m experiments.launcher
```

> `--live` commands need **LM Studio** running with a model loaded on
> `localhost:1234`. `--mock` works entirely offline (fake drone + mock backend).

---

## 1. The grouped test runner

Tests are grouped by **what they actually measure**, so you only run what you need.
Each group reports PASS/FAIL, the measured duration, and an estimated time.

| Group | What it measures | Est. |
|---|---|---|
| `speed` | Performance: latency, throughput, memory, drone/classifier/assembly latency | ~80s |
| `intelligence` | Accuracy/reasoning: retrieval precision, routing & classifier accuracy, oracle, ground truth, A/B statistics, P1–P10, P5 training | ~10s |
| `skills` | Functionality/correctness: logger, drones, hive context, backends, security, resilience, E2E, property/fuzz | ~35s |
| `maximum` | Everything (full coverage / hardware min-maxing) | ~2 min |

```powershell
.\.venv\Scripts\python tests\run_hive_tests.py --group speed          # perf
.\.venv\Scripts\python tests\run_hive_tests.py --group intelligence   # accuracy
.\.venv\Scripts\python tests\run_hive_tests.py --group skills         # functionality
.\.venv\Scripts\python tests\run_hive_tests.py --group maximum        # everything (default)
```

**Model-swap guidance:** the `skills` group and retrieval/routing tests are
**LLM-independent** — re-run them only when you change a *drone* or the *code*.
When you swap the *primary LLM*, re-run the live benchmark instead (below).

---

## 2. The live data-generation benchmark

The main way to collect real data and measure the white paper's claims.

```powershell
# Offline validation / synthetic data (~seconds):
.\.venv\Scripts\python -m experiments.generate_data --mock --max-convs 5 --protocol

# Live against LM Studio — full evidence run (2–3.5 hrs at 20 convs):
.\.venv\Scripts\python -m experiments.generate_data --live --max-convs 20 --protocol --baselines

# Fast iteration / validate the pipeline (~10–15 min):
.\.venv\Scripts\python -m experiments.generate_data --live --max-convs 3 --max-turns 10
```

**Flags**

| Flag | Effect |
|---|---|
| `--live` / `--mock` | Real LM Studio + all-MiniLM, or offline (fake) |
| `--max-convs N` | Number of conversations to run |
| `--max-turns N` | Cap user turns per conversation |
| `--protocol` | Also run the P1–P10 predictions |
| `--baselines` | Also run LM-Studio-rolling + FIFO comparisons |
| `--output DIR` | Write to a specific run directory |
| `--base-url`, `--model`, `--pinned-prefix` | Backend endpoint / model / pinned system prefix |
| `--max-tokens N` | Cap E2E reply length (iteration/stability runs; P1–P10 stay uncapped) |
| `--baseline-max-tokens N` | Cap baseline reply length (baseline tps is a decode measurement) |
| `--checkpoint-every N` | Write a resume checkpoint every N turns (default 10) |
| `--resume DIR` | Resume a run directory from its `checkpoint.json` |
| `--term` | Render a live dashboard inside the terminal (phase, progress, ETA, real-time feed; ANSI, no dependencies) |
| `--confidence mcdropout\|single\|off` | Drone confidence mode; `off` skips MC-dropout passes (stock model yields confidence ≈1.0 anyway) |
| `--no-thinking` | Send `enable_thinking=false` with every request so reasoning models skip chain-of-thought (faster; makes reply caps yield real output). Pair with LM Studio's "thinking" toggle for models that only honor the GUI setting |

> **Reasoning models:** they spend their output budget on chain-of-thought *before*
> the visible answer, so a small `--max-tokens` cap yields empty replies unless
> thinking is disabled (`--no-thinking` or LM Studio's "thinking" toggle). The
> harness detects empty-reply starvation and warns once. Live runs auto-keep the
> system awake (`ES_SYSTEM_REQUIRED` on Windows) so the OS can't sleep mid-benchmark
> and inflate wall-clock timing. Each run dir takes a `run.lock` so a second process
> can't start into a live run. Combine `--live --no-thinking --confidence off
> --term --checkpoint-every 5` for a resumable, watchable iteration run.

**Phases** (with a live progress bar per phase, showing %/elapsed/ETA):
`1/3 E2E conversations` → `2/3 P1-P10 protocol` → `3/3 Baselines`.

---

## 3. Reading & quantifying the results

Every live run writes a self-contained bundle to `runs/<timestamp>/`:

```
runs/<ts>/
    run_report.json          # the main report (everything below)
    ground_truth.sqlite      # queryable retrieval/routing/oracle data
    baseline_lm_studio.json  # no-hive raw rolling comparison
    baseline_fifo.json       # naive 4k-truncation comparison
    logs/events-*.ndjson     # correlation-tagged, redacted event log
```

### 3.1 `run_report.json` — the headline

```json
{
  "run_id": "16d081e5", "mode": "live", "backend": "LMStudioBackend",
  "ground_truth_db": "runs/<ts>/ground_truth.sqlite",
  "aggregate": {
    "conversations": 20, "user_turns": 340,
    "avg_pes": 78.55, "min_pes": 71.2,
    "avg_total_ms": 9200.0, "fifo_fallbacks": 0, "drift_events": 3
  },
  "conversations": [ ...per-turn PES/latency/reply... ],
  "protocol": [ ...P1-P10 results... ],
  "baselines": { ... }
}
```

**Quantify the headline with PES (Pipeline Efficiency Score, 0–100):**

```
PES = 0.30·RetrievalPrecision + 0.20·RoutingAccuracy + 0.20·LatencyHealth
    + 0.15·ThroughputHealth + 0.15·ContextUtilization
```

Bands: `≥80` GREEN · `60–79` YELLOW · `40–59` RED · `<40` CRITICAL.

### 3.2 The P1–P10 protocol (the scientific claims)

Each prediction returns `{id, title, status, evidence, note}` where status is
`PASS` / `FAIL` / `SKIP` / `REPORT`:

| Prediction | Claims | Status meaning |
|---|---|---|
| P1 | Constant throughput over 500 turns | `PASS` if tps stays within ±10% |
| P2 | Retrieval precision ≥85% / recall ≥90% | `PASS` if both met on labeled pairs |
| P3 | Hive context sufficiency ≥ FIFO on ≥80% of turns | oracle-rated |
| P4 | Domain-dependent decay curve | `REPORT` (replay sweep) |
| P5 | Targeted masking beats random | `SKIP` (run `experiments.p5_targeted_masking`) |
| P6 | Escalation improves recall ≥5% at <15% escalation | |
| P7 | Oracle–human agreement ≥90% | `SKIP` (needs human annotation) |
| P8 | Routing accuracy ≥85% | |
| P9 | Densest-duplicate retention | `SKIP` (needs live A/B) |
| P10 | Drift reset speeds recovery | `SKIP` (needs live A/B) |

`SKIP` predictions aren't failures — they're measurements that need live data,
the P5 training run, or human labeling.

### 3.3 The ground-truth DB (SQLite)

```powershell
.\.venv\Scripts\python -c "import sqlite3; c=sqlite3.connect('runs/<ts>/ground_truth.sqlite'); \
print(c.execute('SELECT COUNT(*) FROM oracle_labels').fetchone())"
```

Tables: `oracle_labels`, `hive_decisions`, `parameter_versions`. Metrics (via
`oracle.ground_truth.GroundTruthDB`): `retrieval_precision()`,
`retrieval_recall()`, `false_eviction_rate()`, `routing_accuracy()`.

### 3.4 Comparing to baselines (did the hive help?)

The baselines give the **before** numbers. Compare `aggregate.avg_pes` (hive) vs
`baseline_lm_studio.json`/`baseline_fifo.json` `aggregate.avg_pes` on the same
conversations. That comparison is the evidence that hive-curated context helps.

### 3.5 Event-log summary

```powershell
.\.venv\Scripts\python -m logs.query --dir runs/<ts>/logs --include-archive
```

Prints per-component/event counts, latency percentiles, route distribution, and
mean PES from the NDJSON logs.

---

## 4. Models & drones (how the benchmark is configured)

`cortex/config.py` → `HiveConfig` controls the models; swap without code changes:

```python
HiveConfig(
    ultra_model="BAAI/bge-small-en-v1.5",        # ultra-small retrieval drone
    medium_model="microsoft/graphcodebert-base", # medium validator (downloaded)
    enable_medium=False,                          # opt-in (heavy, VRAM-contending)
    confidence_mode="off",                        # mcdropout | single | off (see below)
)
```

- The primary LLM is whatever is loaded in LM Studio; the hive talks to it via
  the OpenAI-compatible `backend/` layer.
- The medium drone is only invoked on escalation; with the stock embedding model
  confidence ≈ 1.0, so it rarely fires — leave `enable_medium=False` unless
  testing P6.
- **Confidence mode (`mcdropout | single | off`):** the drone's confidence is
  prediction variance across forward passes (MC-dropout). The stock
  `all-MiniLM-L6-v2` disables dropout at inference, so every pass is identical →
  variance is zero → confidence is always 1.0, and `mcdropout`/`single` only add
  encode time with no signal (this is what made drone scoring grow ~5s → ~12s per
  turn in live runs). Use `off` (the launcher default). `mcdropout` (3 passes) is
  only meaningful with a dropout-active encoder — e.g. a custom fine-tuned
  drone — where it powers the P6 escalation tier.
- Drone embeddings are **model-tagged in the cache**, so swapping drones never
  reuses incompatible (different-dimension) embeddings.

---

## 5. Repository layout (relevant to the benchmark)

```
tests/run_hive_tests.py            # grouped test runner
tests/benchmarks/                  # speed-group benchmarks
experiments/generate_data.py       # live data-generation benchmark
experiments/run_p1_p10.py          # P1-P10 protocol driver
experiments/p5_targeted_masking.py # targeted-masking training experiment
oracle/                            # async oracle, ground-truth DB, labeling
logs/event_logger.py, logs/query.py# logging + log report tool
cortex/                            # hive, config, health, degradation, rollback
```

---

## 6. Notes & limitations

- **Mock vs live:** mock mode validates the *code*; only live runs produce real
  evidence for P1–P10. Treat mock `PASS` as engineering validation, not science.
- **Oracle circularity (P7):** the oracle is an LLM; keep it fixed/independent
  when comparing models.
- **Synthetic corpus:** data comes from generated conversations; real
  conversations are needed before generalizing.
- **Hardware:** targets a local OpenAI-compatible backend (`localhost:1234`).
  Some models enforce strict chat templates (a leading system message is always
  used).

---

## License

Open-source research tooling. See the project repo for license details.
