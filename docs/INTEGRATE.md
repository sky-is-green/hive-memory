# Using hive-memory inside another harness

hive-memory is an **external** context-curation layer: it wraps your existing
LLM backend and curates every conversation into a bounded, high-relevance
context window. It does not replace your harness — it sits in front of the
model. Three integration modes, from zero-code to deep.

## Mode A — OpenAI-compatible swap (zero code, minutes)

Start the studio once; it serves an OpenAI-compatible endpoint that curates
every request through the hive:

```powershell
.\.venv\Scripts\python -m harness --setup
.\.venv\Scripts\python -m harness        # serves http://127.0.0.1:8765/v1/chat/completions
```

Any client that accepts a `baseURL` + model id can point at it. The API key is
ignored locally (send anything, e.g. `lm-studio`).

### OpenCode

Add a provider to your `opencode.json` (project or `~/.config/opencode/`), then
restart opencode — config is loaded once at startup:

```json
{
  "$schema": "https://opencode.ai/config.json",
  "provider": {
    "hive-memory": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "Hive Memory (curated context)",
      "options": {
        "baseURL": "http://127.0.0.1:8765/v1"
      },
      "models": {
        "prism-ml/bonsai-27b": { "name": "bonsai-27b via Hive" }
      }
    }
  },
  "model": "hive-memory/prism-ml/bonsai-27b"
}
```

Select the model with `/models` (or set `model` in config). The model id must
be one the studio's backend can serve (list them at `http://127.0.0.1:8765/v1/models`).

### dsh / other OpenAI-compatible harnesses

Same pattern: set the LLM provider's base URL to `http://127.0.0.1:8765/v1` and
an api key of `lm-studio`. The studio passes sampling parameters through
(`temperature`, `top_p`, `max_tokens`, …).

### How it behaves

- Every user turn goes through the hive: classify → route → score → assemble →
  generate, inside a per-conversation store.
- Conversations are isolated by `conversation_id`; reset with `POST /v1/hive/reset`.
- Reasoning models: pass `enable_thinking=false` in client params if the model
  honors it, and note that a small `max_tokens` cap yields empty visible replies
  on models that burn output on hidden CoT (see `docs/INSTALL.md` §8).
- Watch it curate in real time: `GET /v1/hive/state` (or the studio UI).

## Mode B — Python facade (build your own)

The system is a normal pip package with one import surface:

```python
from hive import Hive, HiveConfig, UltraSmallDrone, LMStudioBackend

hive = Hive(
    config=HiveConfig(),                      # budgets, decay, drift thresholds
    ultra=UltraSmallDrone(),                  # ~60 MB CPU drone (~5 ms/query)
    backend=LMStudioBackend(base_url="http://localhost:1234"),
)

result = hive.process_turn("what did we decide about auth?")
print(result.reply)          # the model's answer, generated under curated context
print(result.assembled)      # exactly what the model saw (content + token budget)
```

Per-conversation isolation is manual: call `hive.reset_conversation()` between
conversations (one store per conversation). Any OpenAI-compatible backend works
(`OpenAICompatBackend`), including hosted providers — keys live in
`providers.local.json` (gitignored) or the `HARNESS_PROVIDERS_FILE` env var.
`hive/` never imports from the bench or the studio, so it drops into any
project cleanly.

## Mode C — dsh plugin (deepest integration)

For the dsh fork (`deepseek-harness`, pinned `b150a551b8`), the harness spec
defines a full plugin contract — see `HARNESS-SPEC.md` §3.1:

1. The plugin listens at **`agent/pre-step`** — the documented extension point
   for "decides what the model sees".
2. It calls the sidecar `POST /v1/hive/turn` with
   `{query, conversation_id, model?}` and rewrites the system prompt with the
   curated context (single leading system message — strict chat templates
   require it).
3. Model adapters register through **`ctx.llm`**, so the curated context flows
   through the same seam as any provider.
4. The session log (`SessionEventMap`) is the evaluation record: task prompts,
   steps, tool calls, outcomes — agentic completion can be scored from it.

`setup.ps1` automates the whole dsh toolchain on a fresh machine (fork build,
node carrier, SDK editable installs, llama-server) — run it once, then
`python -m harness`.

## Config & gotchas

- **Secrets**: provider keys live in `providers.local.json` (gitignored); the
  NDJSON event logger redacts `api_key`/`token`-like values before writing.
- **Local-only by default**: the studio binds `127.0.0.1`; set
  `HARNESS_TOKEN` to require a console password on the web UI.
- **Reasoning models** and the **AMD/no-NVIDIA** notes from `docs/INSTALL.md`
  §8 apply to every integration mode.