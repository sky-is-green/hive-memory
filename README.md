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

**The headline measurement** — same 308+ turn conversations, same model,
hive vs naive FIFO windowing (live run `20260822_211131`):

- **Hive ≥ FIFO on 85.1% of retrievable turns** (P3) — the direct head-to-head
  against the current standard
- **90.3% of the facts the model stated made it into context** — deterministic
  P2 diagnostic, ≥90% target; FIFO truncates and drops facts at its window
  limit
- **Flat generation speed across 308+ turns** — 14.5 → 15.5 decode tps
  (+6.7%), no context-bloat slowdown
- All at ~3.4 ms assembly + ~15 ms drone scoring overhead per turn

![Post-run PES: hive 80.0 GREEN vs rolling 12.2 / FIFO 11.6](figures/pes.png)

*PES is the system's own pipeline-efficiency score (retrieval/routing/
latency/throughput/utilization) — a health signal, not a measure of answer
quality. The head-to-head evidence above is what the claims rest on.*

- Full offline test suite — no LLM calls needed to verify the pipeline;
  deterministic, replayable evaluation for every claim (P1–P12)

## Repo layout

| Path | Contents |
|---|---|
| `hive/` | The system: cortex (routing, PES, congestion), sieve (drones), retention, focal (budget/assembly), backend (LM Studio / OpenAI-compat), queen (async ground truth) |
| `hivebench/` | The evaluation suite: `tests/` (grouped runner), `testing/` (A/B, ablation, shadow mode), `experiments/` (live benchmark, protocol, probes) |
| `harness/` | HiveBench Studio sidecar (FastAPI service over the hive) |
| `HIVE-HANDOFF.md` | **The single master doc** — project state, roadmap (S0–S6), how to run everything, next steps |

## Install

Requires Python 3.10+. Pick the layer you need — they install cleanly:

```powershell
# Windows (PowerShell)
python -m venv .venv
.\.venv\Scripts\python -m pip install -e ".[harness,bench]"
```

```bash
# Linux / macOS
python3 -m venv .venv
.venv/bin/python -m pip install -e ".[harness,bench]"
```

| Install target | What you get |
|---|---|
| `pip install hive-memory` | The system: drones, cortex, retention, backends |
| `pip install "hive-memory[harness]"` | + HiveBench Studio (FastAPI sidecar) |
| `pip install "hive-memory[bench]"` | + the evaluation suite (pytest, ST trainer) |

`requirements.txt` / `requirements-dev.txt` remain the full pinned manifest.

## Use the system in your own project

`hive/` is self-contained — it never imports from the bench or the harness:

```python
from hive import Hive, HiveConfig, UltraSmallDrone, LMStudioBackend

hive = Hive(
    config=HiveConfig(),
    ultra=UltraSmallDrone(),
    backend=LMStudioBackend(base_url="http://localhost:1234"),
)
result = hive.process_turn("what did we decide about auth?")
print(result.reply)
```

## Run the studio (HiveBench Studio)

One guided setup, then one command:

```powershell
.\.venv\Scripts\python -m harness --setup   # creates config, checks backend
.\.venv\Scripts\python -m harness           # starts http://127.0.0.1:8765
```

`--setup` copies `providers.example.json` → `providers.local.json` if missing,
probes for a reachable backend (LM Studio on `:1234`, or auto-starts the local
`llama-server` from `models/gguf`), and prints the next step. The studio serves
the hive over a FastAPI API (see `HARNESS-SPEC.md`).

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