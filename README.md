# Hive Memory / HiveBench

**Hive Memory** is an external, multi-agent context-curation layer for
long-horizon LLM conversations. **HiveBench** is its evaluation suite (tests +
live benchmark + the white paper's falsifiable predictions P1–P11).

> **All project documentation lives in `HIVE-HANDOFF.md`** — the single master
> document: project state, roadmap (S0–S6), what was built, lessons learned,
> measured results, how to run everything, and next steps.
>
> Companion docs: `HIVE-WHITE-PAPER.md` (theory + predictions + threats),
> `HIVE-DIAGRAMS.md` (visuals), `HARNESS-SPEC.md` (studio sidecar build brief).

Quick start:

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements.txt -r requirements-dev.txt
.\.venv\Scripts\python -m pip install -e .
.\.venv\Scripts\python -m tests.run_hive_tests --group maximum   # full offline suite
```

See `HIVE-HANDOFF.md` §9 for the live benchmark and §15 for the command cheat
sheet.