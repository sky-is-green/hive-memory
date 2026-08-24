# Hive Memory / HiveBench

**Hive Memory** is an external, multi-agent context-curation layer for
long-horizon LLM conversations. It sits between a user and a local LLM backend,
filtering, scoring, compressing, and reassembling conversation history into a
bounded, high-relevance context window for every turn — so a generative model
performs well over arbitrarily long conversations on consumer hardware.

**HiveBench** is its evaluation suite: unit/integration/benchmark tests, a live
benchmark harness, and the white paper's falsifiable predictions (P1–P12) with
measured verdicts.

## Why Hive

The core idea is the white paper's *Separation Postulate*: small bidirectional
"drone" encoders (fast, cheap, CPU-friendly) do the *comprehension* — scoring,
filtering, and routing context — while the primary generative LLM does the
*generation*.

- ~5 ms/query CPU drone scoring; no GPU required to run the system
- Adaptive token budget, dedup, decay, and drift detection keep the context
  window small and relevant
- Full offline test suite — no LLM calls needed to verify the pipeline
- Deterministic, replayable evaluation for every claim (P1–P12)

## Repo layout

| Path | Contents |
|---|---|
| `hive/` | The system: cortex (routing, PES, congestion), sieve (drones), retention, focal (budget/assembly), backend (LM Studio / OpenAI-compat), queen (async ground truth) |
| `hivebench/` | The evaluation suite: `tests/` (grouped runner), `testing/` (A/B, ablation, shadow mode), `experiments/` (live benchmark, protocol, probes) |
| `harness/` | HiveBench Studio sidecar (FastAPI service over the hive) |
| `HIVE-HANDOFF.md` | **The single master doc** — project state, roadmap (S0–S6), how to run everything, next steps |

## Install

Requires Python 3.10+.

```powershell
# Windows (PowerShell)
python -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements.txt -r requirements-dev.txt
.\.venv\Scripts\python -m pip install -e .
```

```bash
# Linux / macOS
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt -r requirements-dev.txt
.venv/bin/python -m pip install -e .
```

## Run the test suite

The suite is grouped by what it measures (offline; no LLM required):

```powershell
.\.venv\Scripts\python -m tests.run_hive_tests --group maximum   # full suite (default)
.\.venv\Scripts\python -m tests.run_hive_tests --group speed     # latency/PES
.\.venv\Scripts\python -m tests.run_hive_tests --group intelligence  # retrieval/assembly
.\.venv\Scripts\python -m tests.run_hive_tests --group skills    # pipeline/backends
```

## Try it live

The live benchmark talks to an OpenAI-compatible backend (e.g. LM Studio on
`localhost:1234`). A quick resumable iteration run:

```powershell
.\.venv\Scripts\python -m experiments.generate_data --live --no-thinking --confidence off --max-convs 3 --max-turns 10
```

See `HIVE-HANDOFF.md` §9 (live benchmark) and §15 (command cheat sheet) for the
full run matrix and the overnight evidence protocol.

## Documentation

- **`HIVE-HANDOFF.md`** — master document: state, roadmap (S0–S6), lessons, measured results, next steps
- **`HIVE-WHITE-PAPER.md`** — the theory, falsifiable predictions P1–P12, threats, evaluation scope
- **`HIVE-DIAGRAMS.md`** — visuals and measured charts
- **`HARNESS-SPEC.md`** — build brief for the HiveBench Studio sidecar