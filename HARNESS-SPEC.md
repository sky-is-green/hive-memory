# HARNESS-SPEC — The Hive Model Harness (working title: HiveBench Studio)

**Status:** PROPOSED — design locked 2026-08-24. Build track is separate from the
research track (research lives in `HIVE-WHITE-PAPER.md` + `HIVE-HANDOFF.md` and is
owned by the research session; this spec is the complete build brief for the
harness).

---

## 1. Vision

An **open-source local-model testing studio** built on the Hive Memory research.
It is the "LM Studio alternative with the science": load any local or hosted
model, chat with it, evaluate it, and watch a falsifiable measurement layer
(curation, retrieval, P1–P10) run live under it. Two capabilities, both
first-class:

1. **Agentic-task evaluation** — run AI agents on tasks, with the hive curating
   every agent step's context, and score task completion from the session log.
2. **Protocol / measurement testing** — the existing HiveBench pipeline
   (P1–P10, deterministic diagnostics, comb probe, run reports) as a service,
   with results rendered in the web UI.

Everything is open-sourced (MIT), local-first, and vendor-neutral: local
backends (LM Studio / llama.cpp, later vLLM) and hosted OpenAI-compatible APIs
(DeepSeek, OpenAI, OpenRouter, Groq, …) behind one seam.

---

## 2. Decisions (locked)

| Decision | Choice | Why |
|---|---|---|
| Scaffold | **Fork `deepseek-ai/deepseek-harness` (`dsh`)** | MIT; everything-is-a-plugin (Cordis); the `agent/pre-step` seam *is* a context-rewrite hook — the hive's exact function; session log enforces "model-visible means logged" (auditability); headless profile exists for batch runs; web UI included |
| Fork strategy | Fork + **pin a commit**; treat as owned code | dsh is developer preview ("compatibility-breaking changes"); owning the fork stabilizes the seam |
| Language split | **Node/TS shell (fork) + Python sidecar (existing repo)** | The entire research stack (~130 modules, 400+ tests) is Python; never rewrite it |
| Sidecar transport | HTTP on `127.0.0.1` (FastAPI) | Language-neutral seam; local-only binding by default; CORS for the web UI |
| Provider layer | Extend the existing Python `backend/` (OpenAI-compat) with a **provider config** | `OpenAICompatBackend` already speaks any OpenAI-compatible endpoint; hosted APIs are OpenAI-compatible |
| The "LM Studio customizability" | Model management (load/unload/settings) is a **new UI + llama.cpp-server control layer** in the fork | None of it exists in dsh; build it in the shell |
| License of scaffold | MIT (dsh) — compatible with the repo | Crush was rejected partly for FSL-1.1 (source-available) |

---

## 3. Architecture

```
dsh fork (Node/Cordis) — the shell
├── dsh-hive plugin @ agent/pre-step  →  POST /v1/hive/turn    (curate context per agent step)
├── llm adapter (ctx.llm)             →  local (LM Studio) or hosted providers
├── session log (SessionEventMap)     →  agentic-evaluation record (durable, replayable)
├── benchmark plugin (web + headless) →  runs the Python protocol pipeline, renders reports
└── model-management UI (new)         →  load/unload/settings for llama.cpp servers

Python sidecar (this repo — the science, unchanged)
├── hive/   cortex (Hive orchestrator), sieve (drones), retention (store+comb),
│           membrane, focal, backend, queen (oracle), logs, vocab
├── hivebench/  tests (400+), experiments (generate_data, run_p1_p10, model_probe,
│               comb_probe, retrieval_diagnostic, human_label, launcher, dashboard)
└── NEW: harness_service/  (or hive/harness/ — see §5) FastAPI service
```

### 3.1 Seam A — agentic evaluation (hive-curated agent context)

dsh's documented extension points (source: `docs/architecture.md` of the fork):

- **`agent/pre-step`** — "decides what the model sees. Listeners may rewrite the
  claimed messages." The hive plugin listens here: for each step it calls the
  sidecar `process_turn(step_input, conversation_id)` and rewrites the system
  prompt with the curated context (single leading system message — the strict
  template requirement, see `HIVE-HANDOFF.md` §6.0 note 2).
- **`ctx.llm`** — register model adapters (local OpenAI-compat + hosted); the
  hive-curated context flows through this seam exactly like the existing
  `backend/` layer does today.
- **Session log** — the durable `SessionEventMap` is the evaluation record:
  task prompts, steps, tool calls, outcomes, latencies. Agentic scoring
  (completion rate, tool success, cost) is derived from it.
- **`ctx.tools`, agent presets, subagents** — compose task definitions for
  agentic benchmarks.

### 3.2 Seam B — protocol / measurement testing

- **Headless profile** (`dsh --profile headless`) is the one-shot batch runner:
  the benchmark plugin shells out to the Python CLI
  (`python -m experiments.generate_data --live|--mock …`, `comb_probe`,
  `retrieval_diagnostic`, `run_p1_p10`) and stores/renders `run_report.json`.
- **Web profile** renders the same reports: PES breakdown, P1–P10 verdicts,
  retrieval diagnostic, comb stats, baselines comparison, NDJSON event streams
  (`logs.query` for summaries).

### 3.3 The Python sidecar contract (v1)

FastAPI on `127.0.0.1:8765` (configurable). No auth by default (local-only
binding); CORS allowlist for the dsh web UI origin.

| Endpoint | Body → Response |
|---|---|
| `POST /v1/hive/turn` | `{query, conversation_id, model?, config?}` → `{reply, assembled_content, token_count, budget, mode, error?, timings, pes, degradation_level}` |
| `POST /v1/hive/reset` | `{conversation_id}` → `{ok}` (fresh store + comb per conversation) |
| `GET /v1/hive/state` | → `{store_chunks, comb_stats, turn}` (live dashboard feed) |
| `GET /v1/models` | → model probe results (reuse `experiments/model_probe`) |
| `POST /v1/protocol/run` | `{mode: live\|mock, args…}` → `{run_dir}` (launches `generate_data`, returns the run bundle path) |
| `GET /v1/report/{run_dir}` | → the `run_report.json` bundle (for web rendering) |
| `POST /v1/provider/config` | `{providers: [{name, base_url, api_key, model, headers}]}` — the provider layer (§4) |

Implementation notes: the sidecar is stateless per request except the hive
instances keyed by `conversation_id` (per-conversation store + comb isolation is
mandatory — `HIVE-HANDOFF.md` §6.0 #14). Threads, not processes, for local
instances; generation calls are blocking (streaming is a v2).

---

## 4. Provider layer (small, do it first)

Extend the Python side so both the sidecar and the CLI can talk to any provider:

- New `HiveConfig`-adjacent **provider config** (JSON): list of
  `{name, base_url, api_key, model, extra_headers}`.
- `OpenAICompatBackend` gains: api-key header injection, optional
  `extra_headers`, and `--provider NAME` flags in `generate_data`,
  `model_probe`, `run_p1_p10` (default remains `localhost:1234` LM Studio).
- Hosted models are OpenAI-compatible (DeepSeek, OpenAI, OpenRouter, Groq);
  reasoning models need the same `--no-thinking` caveats already documented.

---

## 5. Repository & packaging

- The sidecar lives in this repo (e.g. `harness/` package next to `hive/` and
  `hivebench/`) — or a sibling repo that depends on `hive-memory` via
  `pip install -e .`. **Decision needed at M1 start** (prefer sibling repo:
  keeps the research repo pure, harness is a product).
- The dsh fork lives in its own repo; the `dsh-hive` plugin package is
  published separately.
- Conventions (from this repo): flat import names, docstrings, no gratuitous
  comments, pytest groups (`tests/run_hive_tests.py`), NDJSON structured logs,
  redaction of secrets (`logs/event_logger.py`), keep-awake for long runs.

---

## 6. Milestones

| M | Deliverable | Exit criteria |
|---|---|---|
| M1 | Provider config + FastAPI sidecar | `curl /v1/hive/turn` against LM Studio returns a hive-curated reply; existing 400+ tests still green |
| M2 | dsh fork + `dsh-hive` plugin at `agent/pre-step` | An agent session runs through the hive; session log contains the curated context per step |
| M3 | Benchmark plugin + report views (web + headless) | `--profile headless` runs P1–P10 and writes a report; web UI renders it |
| M4 | Model management UI + packaging | Load/unload/configure llama.cpp servers from the web UI; installable build (npm + pip) |

Milestones are independent of the research track; nothing here blocks P11 or
publication.

---

## 7. Constraints & known issues the builder must respect

1. **AMD-only rig, no NVIDIA**: vLLM is dormant (`backend/vllm.py`, mock-tested);
   LM Studio/llama.cpp is the live local backend. Don't gate anything on CUDA.
2. **Reasoning models**: `enable_thinking=false` is ignored by most qwen
   variants; bonsai-27b honors it. Keep the `--no-thinking` plumbing and the
   reasoning-starve warning (see `HIVE-HANDOFF.md` §6.0 #16).
3. **Strict chat templates**: a single leading system message is mandatory.
4. **Per-conversation isolation**: one store + one comb per conversation —
   never share across conversations.
5. **Secrets**: API keys live in the provider config, never in logs
   (`logs/event_logger.py` redacts); provider config is gitignored
   (`*.local.json`).
6. **The comb is opt-in** (`comb_dir` + `comb_enabled`); its retrieval is
   lexical-overlap ranking (measured 2–3× better than the drone on return
   turns — `HIVE-HANDOFF.md` §5 comb section); don't reintroduce the drone pass.
7. **Re-run `pip install -e .`** after adding/renaming top-level packages.

---

## 8. Handoff checklist for the building AI

1. Read `HIVE-HANDOFF.md` (project state + commands), this spec.
2. Verify: `.venv\Scripts\python -m tests.run_hive_tests --group skills` PASS.
3. M1 first: provider config + sidecar, per §4/§3.3, with tests.
4. Then fork dsh (pin a commit), build `dsh-hive`, then the benchmark plugin.
5. Keep the research repo untouched by harness code (sibling repo for the
   sidecar unless decided otherwise at M1).
6. Report progress against the M1–M4 exit criteria, not against the paper.