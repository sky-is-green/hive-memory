# Studio Console

[![CI](https://github.com/sky-is-green/hive-memory/actions/workflows/ci.yml/badge.svg)](https://github.com/sky-is-green/hive-memory/actions/workflows/ci.yml)

**Studio Console** is the operator UI for the hive-memory stack: one local web
UI that takes you from a bare machine to a served model — GGUF library,
Linux/Docker setup, fit math, engine profiles, agent profiles, Hive tuning,
live chat, and the evaluation suite — all against backends you host.

It sits on the same FastAPI sidecar (`python -m harness`) that serves the
Hive context-curation pipeline, so everything you click is also an API call
your own tooling can make. Nothing phones home: it binds `127.0.0.1`, stores
stay in local files, and hosted APIs only enter the picture if you add keys
yourself.

## Quickstart

Requires Python 3.10+. No model, no GPU, no keys needed to look around.

```powershell
git clone https://github.com/sky-is-green/hive-memory.git
cd hive-memory
python -m venv .venv
.\.venv\Scripts\python -m pip install -e ".[harness,bench]"
.\.venv\Scripts\python -m harness --setup   # creates config, probes backend
.\.venv\Scripts\python -m harness           # console on http://127.0.0.1:8765/server
```

Linux/macOS: `python3 -m venv .venv && .venv/bin/python -m pip install -e ".[harness,bench]"`.
The full fresh-machine walkthrough is `docs/INSTALL.md`.

## Why Studio

Most model UIs serve a chat box and stop. Studio covers the whole job —
finding room for the model, fitting it into VRAM/RAM/disk, picking its
settings, giving the agent its toolbox, and checking the answers — in one
place, with every number re-computed live from your machine:

| Tab | What it does |
|---|---|
| Linux/Docker setup | One-click path to a Linux AI server: pick a drive → Create Drive (sparse VHDX) → Mount AI Drive → Bootstrap Docker → WebUI talks to it |
| Sys Calc | Fit math for your hardware: VRAM/RAM/spill split, KV cost, max context, bandwidth, disk warnings — plus a real uncached drive-speed test |
| Agent Profiles | DSH toolboxes (`standard`, `ptc`, `minimal`, `cordis`) plus Hive-aware variants; one radio picks the active profile the LLM actually gets |
| Engines | Engine profiles: model, endpoint, sampling, load, KV presets (`f16` → `q4_1`, `iq4_nl`), head-to-head A/B bench |
| Local Library | GGUFs on Windows + Linux side, right-click (or double-tap) to move, delete, look up the README, or jump to Engines |
| Hive | Tuning for new conversations (budgets, decay, drones, comb) |
| Providers | OpenAI-compatible endpoints, keys masked |
| Hub | Live Hugging Face GGUF search, file browser, downloads |
| Inspector | Last-turn prompt inspector: selected/dropped chunks, assembled context, timings |
| Settings | Theme and console preferences |

**The headline convenience**: the setup tab turns ~10 manual steps
(WSL mount, ext4 format, compose files, endpoint wiring) into 3 clicks, and
Sys Calc answers *will this model fit* before you wait through a load.

## Tabs in detail

**Linux/Docker setup.** Built for extra-large models on a Windows + WSL2 +
AMD/ROCm rig. Drive/VHDX mismatch is caught by a red banner before it bites.
`Refresh Status` merges health, Docker, and WebUI checks into one verdict.

**Sys Calc.** Reads your VRAM, RAM, and free disk, then splits the model
across tiers. The context slider and model-size box re-run the math without
saving anything. `Test drive speed` runs a real uncached 128 MB read/write
(deleted after) — rated specs lie, this doesn't. Per-drive detection means a
plugged-in HDD honestly reports ~0.15 GB/s instead of an NVMe guess.

**Agent Profiles.** Each profile is a DSH toolbox, not a pasted prompt. The
Hive variants (`Hive Standard`, `Hive PTC`, `Hive Minimal`) pair DSH
tooling with Hive memory; the radio selects exactly one, and the sidecar
passes it to the LLM via `DSH_CORDIS_CONFIG` on the next Agent turn. `Edit`
opens the preset in your system editor, `Open File Location` shows the
folder, `Duplicate` authors your own, `Creator` authors new Hive-aware ones.

**Engines.** Every load and sampling field carries an operational default —
nothing loads blank. KV cache types cover the full official llama.cpp list
(`f32`/`f16`/`bf16`/`q8_0`/`q5_0`/`q5_1`/`q4_0`/`q4_1`/`iq4_nl`) with one-click
presets (Lossless, Balanced, Asymmetric q8/q4, Extreme). `Auto` fits layers
and context to your GPU; `Bench` runs two profiles head-to-head in tok/s.

**Local Library.** Both homes for your weights: Windows folder and
`/mnt/dsh_storage/models`. Right-click moves GGUFs between them over WSL —
no terminal. `View README` resolves the actual Hub repo (verified against
the file list, best-guess labeled when unsure); `Edit Settings` jumps to
Engines with that model pre-selected.

**Chat.** Session tabs, Hive (curated) and Agent (full dsh tool loop) modes,
slash commands, durable transcripts. The top bar shows only what's live:
loaded model · active profile.

## Ideas / roadmap

Where Studio goes next (proposed, not yet built):

- **Swappable engines.** Per-engine server binary/image + accepted KV types,
  so a TurboQuant fork (or any custom `llama-server`) drops in as just
  another engine to A/B — no fork of llama.cpp to maintain while upstream
  settles on a format.
- **One-click evidence runs.** Fire a paired A/B or protocol run from the
  Runs page and watch verdicts land, instead of the terminal.
- **GGUF-suggested settings.** Prefill sampling from the model card README
  the Hub resolver already finds (some authors publish exact temps).
- **Profile sharing.** Export/import agent presets as single files.

## FAQ

**Does my data leave my machine?**
No. Same answer as the main README: local backend, local files,
`127.0.0.1` bind. The Hub tab is the only part that talks to the internet,
and only when you search.

**How is this different from LM Studio / Open WebUI?**
Those are excellent chat frontends. Studio is an operator console: it
provisions the Linux/Docker backend, fiscally fits models to hardware,
versions engine and agent profiles, and wires in the HiveBench evidence
pipeline. Use them to chat; use Studio to run the shop.

**Does the Linux/Docker setup work on macOS?**
No — VHDX/WSL/ROCm are Windows+Linux constructs. On a Mac, skip that tab:
run Docker Desktop with a plain models folder. The rest of Studio (library,
engines, profiles, chat, bench) is platform-neutral.

**Why not TurboQuant cache types yet?**
`turbo3`/`turbo4` exist only in unmerged PRs and forks, with three competing
names. Our launcher tracks official llama.cpp, where those flags error out.
The swappable-engines plan above is the honest route to them.

## Repo layout (studio parts)

| Path | Contents |
|---|---|
| `harness/harness/app.py` | The sidecar: every `/v1/*` endpoint the console calls |
| `harness/harness/reports.py` | The console page itself (HTML + JS, UTF-8) |
| `harness/harness/studio.css` | Theme, layout, mobile rules |
| `harness/harness/models.py` | Model manager: library scan, llama-server lifecycle, Hub client |
| `harness/harness/agent.py` | dsh agent bridge behind Agent chat mode |
| `harness_state/` | Local state: selected preset, sessions, model dir |

## Use the API in your own project

Every button is a call you can script:

```powershell
# which agent profile the next Agent turn gets
Invoke-WebRequest http://127.0.0.1:8765/v1/agent-presets/selected -UseBasicParsing
# resolve a GGUF to its Hub README
Invoke-WebRequest "http://127.0.0.1:8765/v1/models/hf-link?file=Qwen3-4B-Q4_K_M.gguf" -UseBasicParsing
# serve a model
Invoke-WebRequest http://127.0.0.1:8765/v1/server/start -Method Post `
  -Body '{"model":"Qwen3-4B-Q4_K_M.gguf","ctx_size":8192}' -ContentType "application/json" -UseBasicParsing
```

## Documentation

- **`README.md`**, the system + evaluation suite (HiveMemory / HiveBench protocol)
- **`docs/INSTALL.md`**, full-stack install guide (fresh machine)
- **`docs/INTEGRATE.md`**, using hive-memory inside OpenCode, dsh, or your own harness
- **`HIVE-WHITE-PAPER.md`**, theory, P1-P11 verdicts (§8), PES (§6), threats (§9)
