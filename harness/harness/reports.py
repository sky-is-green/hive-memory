"""Server-rendered report views (Seam B) + the Studio console page.

Plain HTML, no JS dependencies. Renders one ``run_report.json`` bundle: the
post-run PES headline with its weighted components, the P1–P11 verdict table,
the deterministic P2 retrieval diagnostic, comb (P11) totals, and the
baselines comparison. Every dynamic value is HTML-escaped; missing blocks
render as em dashes rather than erroring, so partial bundles from in-flight
runs stay viewable.

This file is UTF-8 and must stay that way — it contains arrows, stars and
ellipses in the console UI. Never round-trip it through PowerShell text
cmdlets (they decode as ANSI on Windows and mojibake every non-ASCII char).
"""

from __future__ import annotations

import html
from pathlib import Path

from fastapi import HTTPException

# Paper weights for the PES composite (README §3.1).
_PES_COMPONENTS = [
    ("retrieval_precision", 0.30),
    ("routing_accuracy", 0.20),
    ("latency_health", 0.20),
    ("throughput_health", 0.15),
    ("context_utilization", 0.15),
]
_BAND_FLOORS = [(80, "band-green"), (60, "band-yellow"), (40, "band-red"),
                (0, "band-critical")]

_CSS = """
 body { font-family: Segoe UI, system-ui, sans-serif; margin: 2rem auto; max-width: 60rem;
        color: #1c2733; background: #f7f9fb; padding: 0 1rem; }
 h1 { font-size: 1.4rem; } h2 { font-size: 1.05rem; margin-top: 2rem;
      border-bottom: 1px solid #dde5ec; padding-bottom: .3rem; }
 table { border-collapse: collapse; width: 100%; background: #fff; }
 th, td { text-align: left; padding: .45rem .7rem; border-bottom: 1px solid #e8edf2;
          font-size: .92rem; }
 th { background: #eef2f6; }
 .cards { display: flex; gap: 1rem; flex-wrap: wrap; }
 .card { background: #fff; border-radius: 8px; padding: 1rem 1.4rem;
         box-shadow: 0 1px 3px rgba(16,32,48,.08); min-width: 150px; }
 .card span { color: #5a6b7d; font-size: .85rem; }
 .card b { display: block; font-size: 1.6rem; }
 .kv-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(230px, 1fr));
            gap: .4rem 1.4rem; background: #fff; padding: 1rem; border-radius: 8px; }
 .kv span { color: #5a6b7d; } .kv b { float: right; font-variant-numeric: tabular-nums; }
 .band-green { color: #157a3e; } .band-yellow { color: #9a6b00; }
 .band-red { color: #b3372c; } .band-critical { color: #8a1f1f; font-weight: 700; }
 .st-pass { color: #157a3e; font-weight: 600; } .st-fail { color: #b3372c; font-weight: 600; }
 .st-skip { color: #9a6b00; } .st-report { color: #456; }
 .note { color: #5a6b7d; font-size: .85rem; }
 code { background: #eef2f6; padding: .1rem .35rem; border-radius: 4px; }
 ul.runs { list-style: none; padding: 0; } ul.runs li { margin: .35rem 0; }
"""


def _esc(value: object) -> str:
    return html.escape(str(value))


def _fmt(value: object, nd: int = 1) -> str:
    if value is None or value == "":
        return "&mdash;"
    try:
        return f"{float(value):.{nd}f}"
    except (TypeError, ValueError):
        return _esc(value)


def _pes_band_class(score: object) -> str:
    try:
        s = float(score)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return ""
    for floor, cls in _BAND_FLOORS:
        if s >= floor:
            return cls
    return ""


def _status_cell(status: object) -> str:
    s = _esc(status or "")
    return f"<span class='st-{s.lower()}'>{s}</span>"


def resolve_run_dir(runs_root: Path, run_dir: str) -> Path:
    """Resolve + confine a run_dir under runs_root (path-traversal safe)."""
    root = runs_root.resolve()
    target = (root / run_dir).resolve()
    if target != root and root not in target.parents:
        raise HTTPException(400, "run_dir must stay inside the runs directory")
    return target


def _comb_totals(comb: dict) -> dict:
    totals = {"archived": 0, "resurrected": 0, "comb_hits": 0, "gate_fired": 0}
    conversations = comb.get("conversations")
    if isinstance(conversations, list):
        for entry in conversations:
            if not isinstance(entry, dict):
                continue
            for key in totals:
                value = entry.get(key)
                if isinstance(value, (int, float)):
                    totals[key] += int(value)
    return totals


def _baseline_rows(report: dict, composite: object) -> list[tuple[str, object]]:
    rows: list[tuple[str, object]] = [("Hive (post-run PES)", composite)]
    for label, key in (("LM-Studio rolling", "baseline_lm_studio"),
                       ("FIFO truncation", "baseline_fifo")):
        blob = report.get(key)
        if not isinstance(blob, dict):
            nested = report.get("baselines")
            blob = nested.get(key) if isinstance(nested, dict) else None
        if isinstance(blob, dict):
            aggregate = blob.get("aggregate")
            if isinstance(aggregate, dict):
                rows.append((label, aggregate.get("avg_pes")))
    return rows


def render_report_page(report: dict, run_name: str) -> str:
    """One run_report.json bundle as a standalone HTML page."""
    aggregate = report.get("aggregate") if isinstance(report.get("aggregate"), dict) else {}
    post = report.get("post_run_pes") if isinstance(report.get("post_run_pes"), dict) else {}
    components = post.get("components") if isinstance(post.get("components"), dict) else {}
    diag = report.get("retrieval_diagnostic") \
        if isinstance(report.get("retrieval_diagnostic"), dict) else {}
    comb = report.get("comb") if isinstance(report.get("comb"), dict) else {}
    comb_t = _comb_totals(comb)
    # generate_data has used two shapes across versions: composite/components
    # and pes/breakdown — accept both.
    composite = post.get("composite", post.get("pes"))
    components = post.get("components")
    if not isinstance(components, dict):
        components = post.get("breakdown")
    if not isinstance(components, dict):
        components = {}
    band = str(post.get("band") or "")

    def _band_class() -> str:
        if band:
            for token in ("green", "yellow", "red", "critical"):
                if token in band.lower():
                    return f"band-{token}"
            return ""
        return _pes_band_class(composite)

    protocol_rows = "".join(
        "<tr>"
        f"<td>{_esc(p.get('id', ''))}</td>"
        f"<td>{_status_cell(p.get('status'))}</td>"
        f"<td>{_esc(p.get('title', ''))}</td>"
        f"<td class='note'>{_esc(p.get('note', ''))}</td>"
        "</tr>"
        for p in (report.get("protocol") or [])
        if isinstance(p, dict)
    ) or "<tr><td colspan='4' class='note'>no protocol block</td></tr>"

    component_rows = "".join(
        "<tr>"
        f"<td>{_esc(name.replace('_', ' ').title())}</td>"
        f"<td>{weight:.2f}</td>"
        f"<td>{_fmt(components.get(name))}</td>"
        "</tr>"
        for name, weight in _PES_COMPONENTS
    )

    baseline_rows = "".join(
        "<tr>"
        f"<td>{_esc(label)}</td>"
        f"<td class='{_pes_band_class(pes)}'><b>{_fmt(pes)}</b></td>"
        "</tr>"
        for label, pes in _baseline_rows(report, composite)
    )

    def kv(label: str, value: object, nd: int = 1) -> str:
        return f"<div class='kv'><span>{label}</span><b>{_fmt(value, nd)}</b></div>"

    avg_ms = aggregate.get("avg_total_ms")
    return f"""<!doctype html>
<html><head><meta charset="utf-8">
<title>HiveBench report &mdash; {_esc(run_name)}</title>
<style>{_CSS}</style></head><body>
<h1>HiveBench report &mdash; <code>{_esc(run_name)}</code></h1>
<p class="note">mode <b>{_esc(report.get('mode') or '&mdash;')}</b> &middot;
backend <b>{_esc(report.get('backend') or '&mdash;')}</b> &middot; run
<b>{_esc(report.get('run_id') or '&mdash;')}</b></p>
<div class="cards">
<div class="card"><span>PES (post-run)</span><b class="{_band_class()}">{_fmt(composite)}</b></div>
<div class="card"><span>Avg per-turn PES</span><b>{_fmt(aggregate.get('avg_pes'))}</b></div>
<div class="card"><span>User turns</span><b>{_fmt(aggregate.get('user_turns'), 0)}</b></div>
<div class="card"><span>Avg turn</span><b>{_fmt(avg_ms / 1000 if isinstance(avg_ms, (int, float)) else None)}s</b></div>
</div>

<h2>PES components (post-run)</h2>
<table><tr><th>Component</th><th>Weight</th><th>Score</th></tr>
{component_rows}
</table>

<h2>P1&ndash;P11 verdicts</h2>
<table><tr><th>ID</th><th>Status</th><th>Prediction</th><th>Note</th></tr>
{protocol_rows}
</table>

<h2>Deterministic retrieval diagnostic (P2)</h2>
<div class="kv-grid">
{kv('Recall (all turns)', diag.get('retrieval_recall'))}
{kv('Recall (retrievable turns)', diag.get('retrieval_recall_retrievable'))}
{kv('Ingestion rate', diag.get('ingestion_rate'))}
{kv('Perfect-hive ceiling', diag.get('perfect_hive_ceiling'))}
{kv('Precision (sentence proxy)', diag.get('retrieval_precision'))}
</div>

<h2>Comb (P11 surplus tier)</h2>
<div class="kv-grid">
{kv('Archived', comb_t['archived'], 0)}
{kv('Resurrected', comb_t['resurrected'], 0)}
{kv('Comb hits', comb_t['comb_hits'], 0)}
{kv('Gate fired', comb_t['gate_fired'], 0)}
</div>

<h2>Baselines comparison</h2>
<table><tr><th>System</th><th>PES</th></tr>
{baseline_rows}
</table>

<p class="note"><a href="/runs">&larr; all runs</a></p>
</body></html>"""


def render_runs_page(entries: list[dict]) -> str:
    """Index of available run bundles."""
    items = "".join(
        f"<li><a href='/view/{_esc(e['name'])}'><code>{_esc(e['name'])}</code></a> "
        f"<span class='note'>modified {_esc(e['modified'])}"
        f"{'' if e['has_report'] else ' · no run_report.json yet'}</span></li>"
        for e in entries
    ) or "<li class='note'>no runs yet</li>"
    return f"""<!doctype html>
<html><head><meta charset="utf-8"><title>HiveBench runs</title>
<style>{_CSS}</style></head><body>
<h1>HiveBench run bundles</h1>
<p><a href="/server"><button>← Studio console</button></a> <a href="/docs"><button>API docs</button></a></p>
<ul class="runs">{items}</ul>
</body></html>"""


_STUDIO_CSS_PATH = Path(__file__).with_name("studio.css")

def _get_server_css() -> str:
    """Studio CSS: base + studio overrides, re-read on each request so refresh picks up edits without restart."""
    try:
        extra = _STUDIO_CSS_PATH.read_text(encoding="utf-8")
    except OSError:
        extra = ""
    return _CSS + extra

# Backwards compat: keep _SERVER_CSS as snapshot at import, but renderers call _get_server_css()
_SERVER_CSS = _get_server_css()


def tip(text: str) -> str:
    """One inline help glyph whose hover/focus tooltip explains the control it follows."""
    return f'<span class="hint" tabindex="0" data-tip="{html.escape(text, quote=True)}">?</span>'


def render_server_page() -> str:
    """Model-management console: server + settings | loaded-model chat | hub.

    Discovery is live against the Hugging Face hub API — no model catalog is
    hardcoded here."""
    return f"""<!doctype html>
<html><head><meta charset="utf-8"><meta http-equiv="Cache-Control" content="no-cache, no-store, must-revalidate"><meta http-equiv="Pragma" content="no-cache"><meta http-equiv="Expires" content="0"><title>Studio server &amp; models</title>
<style>{_get_server_css()}</style></head><body>
<div style="background:#000;color:#FFDD00;padding:.4rem .8rem;margin:-1rem -1.2rem 1rem -1.2rem;text-align:center;font-weight:700;letter-spacing:.04rem">STUDIO v4 — stronger shadow + stable library — {len(_get_server_css())}b CSS</div>
<h1>Hive Studio console</h1>
<p style="margin:.3rem 0"><button id="afkbtn" onclick="toggleAfk(this)">AFK</button>{tip('AFK mode: human away - QUEEN runs expanded autonomy (GREEN/YELLOW fixes, regen, HIVE-PLAN orders). Public pushes/merges/policy changes queue for return; RED defects contained and logged.')} <a href="/runs"><button>Runs →</button></a> <a href="/docs"><button>API docs</button></a></p>
<form style="display:inline" onsubmit="event.preventDefault(); return false"><input id="researchq" placeholder="deep-research question..." size="30" autocomplete="off" onkeydown="if (event.key === &quot;Enter&quot;) researchAdd(this)"> <button id="researchsubmit" onclick="researchAdd(this)">Research</button> <span id="researchcount" class="meta"></span>{tip('Queues a deep-research question for QUEEN. Execution is master-only; reports land in RESEARCH/<slug>.md and are summarized on wake.')}</form>

<div class="grid">

<!-- ==================== LEFT: tabs ==================== -->
<div class="col">
<div class="tabs">
<button class="tab active" data-tab="tab-engines">Engines</button>
<button class="tab" data-tab="tab-hive">Hive</button>
<button class="tab" data-tab="tab-providers">Providers</button>
<button class="tab" data-tab="tab-hub">Hub</button>
<button class="tab" data-tab="tab-inspect">Inspector</button>
</div>

<div id="tab-engines" class="tabpane">
<section>
<h2 style="margin-top:0">Engine profiles — unified</h2>
<form id="engine-profile-form" class="engine-grid" onsubmit="event.preventDefault(); saveEngineProfile();">
  <div class="row" style="gap:.6rem; flex-wrap:wrap">
    <select id="eng-select" style="min-width:180px" onchange="engineSelected()"></select>
    <label class="inline">default {tip('Profile applied when a conversation names no engine.')} <input id="eng-default" type="checkbox" onchange="engDirty=true"></label>
    <button type="button" onclick="engineAdd()">+ Add</button>
  </div>

  <!-- Model -->
  <div class="engine-group">
    <h3>Model {tip('Local GGUF model — dropdown populated from the local library')}</h3>
    <div class="grid-2col">
      <label>Model {tip('Local GGUF from library — dropdown populated from /v1/models/local')} <select id="eng-model" onchange="engDirty=true; updateFit();"><option value="">— choose local model —</option></select></label>
      <span class="note">Selected: <b id="eng-model-display">—</b> <span id="eng-model-note" class="note" style="margin-left:.5rem"></span></span>
    </div>
  </div>

  <!-- Endpoint -->
  <div class="engine-group">
    <h3>Endpoint</h3>
    <div class="grid-2col">
      <label>name {tip('Display name of this engine profile.')} <input id="eng-name" size="16" placeholder="local-bonsai" oninput="engDirty=true"></label>
      <label>displayName {tip('Human name for this profile')} <input id="eng-displayName" placeholder="local-bonsai" oninput="engDirty=true"></label>
      <label>baseURL {tip('OpenAI-compatible endpoint root, e.g. http://localhost:1234/v1')} <input id="eng-url" placeholder="http://localhost:1234/v1" oninput="engDirty=true"></label>
      <label>apiKey {tip('Bearer token; leave blank for none')} <input id="eng-apikey" type="password" placeholder="(none)" oninput="engDirty=true"></label>
    </div>
  </div>

  <!-- Context -->
  <div class="engine-group">
    <h3>Context</h3>
    <div class="grid-2col">
      <label>contextLength {tip('Context window tokens')} <input id="eng-ctxlen" type="number" value="8192" min="256" step="256" oninput="engDirty=true; updateFit();"></label>
      <div id="eng-fit" class="fit-panel" style="display:block">
        <div class="fit-grid">
          <span id="fit-needs" class="fit-needs">needs —</span>
          <span id="fit-has" class="fit-has">you have —</span>
          <span id="fit-pct" class="fit-pct"></span>
        </div>
        <div class="fit-bar"><div id="fit-fill" class="fit-fill" style="width:0%"></div></div>
        <div class="row fit-controls">
          <label class="inline">context <input id="fit-ctx" type="range" min="2048" max="131072" step="1024" value="8192"> <span id="fit-ctx-label">8k</span> → <b id="fit-needs-val">—</b> {tip('VRAM estimate: model file + KV cache (≈0.25GB per 1k tokens for 7B q4). 32k → 12GB total for a 4GB model.')}</label>
          <button type="button" id="eng-load-fit" onclick="engineLoadFromFit()" disabled>Load</button>
        </div>
      </div>
    </div>
  </div>

  <!-- Sampling 4x2 -->
  <div class="engine-group">
    <h3>Sampling</h3>
    <div class="grid-2col sampling-grid">
      <label>temp {tip('Randomness: higher = more varied, lower = more focused.')} <input id="s-temp" type="number" step="0.05" min="0" max="2" oninput="engDirty=true"></label>
      <label>top_p {tip('Nucleus sampling: keep only tokens covering this cumulative probability.')} <input id="s-topp" type="number" step="0.05" min="0" max="1" oninput="engDirty=true"></label>
      <label>top_k {tip('Sample only from the K most likely tokens.')} <input id="s-topk" type="number" oninput="engDirty=true"></label>
      <label>min_p {tip('Drop tokens below this fraction of the top token probability.')} <input id="s-minp" type="number" step="0.01" oninput="engDirty=true"></label>
      <label>repeat_penalty {tip('Penalty on tokens already present; higher = less repetition.')} <input id="s-rep" type="number" step="0.05" oninput="engDirty=true"></label>
      <label>presence_penalty {tip('Flat penalty once a token appears at all.')} <input id="s-pres" type="number" step="0.1" oninput="engDirty=true"></label>
      <label>frequency_penalty {tip('Penalty that grows with each repetition of a token.')} <input id="s-freq" type="number" step="0.1" oninput="engDirty=true"></label>
      <label>seed {tip('Fixed RNG seed for reproducible output; blank = random.')} <input id="s-seed" type="number" oninput="engDirty=true"></label>
    </div>
  </div>

  <!-- Load -->
  <div class="engine-group">
    <h3>Load</h3>
    <div class="grid-2col load-grid">
      <label>threads {tip('CPU threads for inference; blank = automatic.')} <input id="eng-threads" type="number" placeholder="auto" oninput="engDirty=true"></label>
      <label>gpu_layers {tip('Model layers offloaded to the GPU. 999 = every layer')} <input id="eng-gpu" type="number" value="999" oninput="engDirty=true"></label>
      <label>flash_attn {tip('FlashAttention kernels: faster attention and lower VRAM at long context.')} <select id="eng-flash" onchange="engDirty=true"><option value="">off</option><option value="on">on</option><option value="auto">auto</option></select></label>
      <label>parallel {tip('Requests decoded concurrently; each slot shares the context window.')} <input id="eng-parallel" type="number" placeholder="1" oninput="engDirty=true"></label>
      <label>batch {tip('Logical prompt-processing batch size.')} <input id="eng-batch" type="number" placeholder="512" oninput="engDirty=true"></label>
      <label>ubatch {tip('Physical micro-batch fed to the model per step.')} <input id="eng-ubatch" type="number" placeholder="512" oninput="engDirty=true"></label>
      <label>cache K {tip('Quantize the attention key cache to save VRAM')} <select id="eng-ctk" onchange="engDirty=true"><option value="">f16</option><option>q8_0</option><option>q4_0</option></select></label>
      <label>cache V {tip('Same quantization for the value cache.')} <select id="eng-ctv" onchange="engDirty=true"><option value="">f16</option><option>q8_0</option><option>q4_0</option></select></label>
    </div>
  </div>

  <!-- Advanced -->
  <div class="engine-group">
    <h3>Advanced</h3>
    <div class="grid-2col advanced-grid">
      <label>mirostat {tip('Adaptive perplexity control: 0 = off, 1 = v1, 2 = v2.')} <input id="s-miro" type="number" min="0" max="2" oninput="engDirty=true"></label>
      <label>mirostat_tau {tip('Mirostat target entropy: higher = more surprising text.')} <input id="s-tau" type="number" step="0.1" oninput="engDirty=true"></label>
      <label>mirostat_eta {tip('How fast mirostat adapts toward its target.')} <input id="s-eta" type="number" step="0.01" oninput="engDirty=true"></label>
      <label>stop {tip('Comma-separated strings that end generation early.')} <input id="s-stop" placeholder="a,b" oninput="engDirty=true"></label>
      <label>alias {tip('Model id exposed on /v1/models instead of the file path.')} <input id="eng-alias" placeholder="model id" oninput="engDirty=true"></label>
      <label>kind {tip('Backend family the harness talks to.')} <select id="eng-kind" onchange="engDirty=true"><option>llama_cpp</option><option>lmstudio</option><option>vllm</option><option>ollama</option><option>hosted</option></select></label>
    </div>
  </div>

  <!-- A/B compare — grid cell with two profile selects, Bench button, winner badge -->
  <div class="engine-group" id="ab-compare">
    <h3>A/B compare</h3>
    <div class="grid-2col">
      <label>A profile {tip('First engine profile to compare')} <select id="ab-select-a"></select></label>
      <label>B profile {tip('Second engine profile to compare')} <select id="ab-select-b"></select></label>
      <label>basePort {tip('Base port for A (B runs on basePort+1)')} <input id="ab-baseport" type="number" value="1234" min="1024" max="65534"></label>
      <div class="row" style="gap:.5rem; align-items:center">
        <button type="button" id="ab-bench-btn" onclick="benchAb()">Bench</button>
        <span id="ab-winner" class="winner-badge">—</span>
        <span id="ab-badge" style="display:none"></span>
      </div>
      <span id="ab-result-a" class="note"></span>
      <span id="ab-result-b" class="note"></span>
      <!-- aliases for alternative test ids -->
      <select id="ab-profile-a" style="display:none"></select>
      <select id="ab-profile-b" style="display:none"></select>
    </div>
  </div>

  <div class="row" style="margin-top:.6rem; gap:.5rem; align-items:center; flex-wrap:wrap"><span id="eng-msg" class="note"></span><button type="submit" id="eng-save">Save</button> <button type="button" id="eng-auto" onclick="engineAuto()" title="Auto-set GPU layers & context from hardware + model size">Auto</button> <span id="eng-auto-msg" class="note" style="margin-left:.5rem"></span> <button type="button" onclick="exportEngineProfiles()">Export</button><button type="button" onclick="document.getElementById('import-engine-profiles').click()">Import</button><input type="file" id="import-engine-profiles" accept=".json" style="display:none" onchange="importEngineProfiles(this)"><span id="import-export-msg" class="note"></span></div>
  <pre id="eng-loadopts" style="display:none"></pre>
</form>
</section>

<section>
<h2 style="margin-top:0">Local library <span class="note" id="library-path-note"></span></h2>
<div class="row"><input type="text" id="library-path" placeholder="C:/Users/you/.lmstudio/models" size="38" style="flex:1"><button onclick="setLibraryPath()">Use this folder</button><span id="import-status" class="note" style="margin-left:.5rem"></span></div>
<div class="row" style="margin-top:.6rem"><input type="text" id="library-filter" placeholder="Filter by name…" size="28" style="flex:1" oninput="filterLibrary()"><span class="note" id="library-filter-count" style="margin-left:.5rem"></span></div>
<div id="local" class="liblist">loading…</div>
</section>
</div>

<div id="tab-hive" class="tabpane" style="display:none">
<section>
<h2 style="margin-top:0">Hive tuning <span class="note">(new conversations)</span></h2>
<div class="note">Applied when a conversation is created — hit
"New conversation" in the chat pane after changing.</div>
<div class="row">
<label class="inline">max_context {tip('Token ceiling for the prompt the hive assembles each turn.')} <input id="h-maxctx" type="number" size="6"></label>
<label class="inline">max_tokens {tip('Hard cap on generated tokens per reply.')} <input id="h-maxtok" type="number" size="5" placeholder="4096 ceiling"></label>
</div>
<div class="row">
<label class="inline">stale wall {tip('Turns a fact may sit unreferenced before it ages out of the store.')} <input id="h-stale" type="number" size="3"></label>
<label class="inline">dedup {tip('Similarity above which new text counts as a duplicate and is not stored again.')} <input id="h-dedup" type="number" step="0.01" size="4"></label>
<label class="inline">drift {tip('Similarity drop between turns that marks a topic change.')} <input id="h-drift" type="number" step="0.05" size="4"></label>
<label class="inline">remem {tip('Recall threshold: how similar content must be to resurface from memory.')} <input id="h-remem" type="number" step="0.05" size="4"></label>
</div>
<div class="row">
<label class="inline">vocab boost {tip('Bonus added to relevance scores on exact keyword hits.')} <input id="h-vocab" type="number" step="0.05" size="4"></label>
<label class="inline">confidence {tip('How the drone estimates its own certainty; mcdropout uses multiple stochastic passes.')} <select id="h-conf">
<option>off</option><option>single</option><option>mcdropout</option></select></label>
</div>
<div class="row">
<label class="inline"><input id="h-sanitize" type="checkbox"> sanitize context {tip('Security scrub of the assembled context before it reaches the model.')}</label>
<label class="inline"><input id="h-hedge" type="checkbox"> filter hedge replies {tip('Never store refusal/hedge replies; they pollute the store and resurface as bad context.')}</label>
<label class="inline"><input id="h-medium" type="checkbox"> medium drone {tip('Second-pass encoder for harder queries: better recall, heavier and VRAM-hungry.')}</label>
</div>
<details><summary>Comb (P11 surplus tier)</summary>
<div class="row">
<label class="inline"><input id="h-comb" type="checkbox"> enabled {tip('Freeze evicted chunks to disk so old topics can be resurrected later (P11).')}</label>
<label class="inline">top_k {tip('Archived candidates allowed to compete for context each turn.')} <input id="h-combk" type="number" size="3"></label>
<label class="inline">gate {tip('Comb is consulted only when the store scores below this; normal turns pay nothing.')} <input id="h-combgate" type="number" step="0.05" size="4"></label>
<label class="inline">max records {tip('Cap on archived records kept on disk.')} <input id="h-combmax" type="number" size="5"></label>
<label class="inline"><input id="h-combrel" type="checkbox"> curated-only {tip('Archive only chunks the hive previously selected as relevant.')}</label>
</div>
</details>
<div class="row"><span id="hive-msg" class="note"></span>
<button onclick="resetHiveDefaults()">Reset to defaults</button></div>
</section>
</div>

<div id="tab-providers" class="tabpane" style="display:none">
<section>
<h2 style="margin-top:0">Providers <span class="note">(OpenAI-compatible)</span></h2>
<div id="prov-list"></div>




<div class="row">
<button onclick="providerAdd()">+ Add provider</button>
<button onclick="saveProviders()">Save providers</button>
<span id="prov-msg" class="note"></span></div>
<div class="note">Keys echo as *** — leave untouched rows as-is to keep the
stored secret; type a new key to replace it. Saved to
providers.local.json (gitignored).</div>
</section>
</div>

<div id="tab-hub" class="tabpane" style="display:none">
<section>
<h2 style="margin-top:0">Hugging Face hub <span class="note">(live)</span></h2>
<div class="row"><span class="sugwrap" style="width:100%"><input id="q" placeholder="search gguf repos…" style="width:100%">
<div class="sugbox" id="sug-q"></div></span></div>
<datalist id="repo-suggestions"></datalist>
<pre id="hub">(search above)</pre>
<div class="row"><span class="sugwrap" style="width:100%"><input id="drepo" placeholder="repo id (type for suggestions)" style="width:100%">
<div class="sugbox" id="sug-drepo"></div></span><br>
<input id="dfile" placeholder="file.gguf" size="24">{tip('Exact GGUF filename inside that repo (copy it from the search results).')}
<button onclick="download(this)">Download</button></div>
<pre id="downloads"></pre>
</section>
</div>

<div id="tab-inspect" class="tabpane" style="display:none">
<section>
<h2 style="margin-top:0">Prompt inspector <span class="note">(last turn)</span></h2>
<div id="inspect-summary" class="kv-grid"></div>
<h2 style="font-size:.95rem;margin-top:1rem">Selected chunks (in the context)</h2>
<div id="inspect-selected" class="inspector-list"></div>
<h2 style="font-size:.95rem;margin-top:1rem">Dropped chunks <span class="note" id="inspect-dropped-n"></span></h2>
<div id="inspect-dropped" class="inspector-list"></div>
<h2 style="font-size:.95rem;margin-top:1rem">Assembled context (preview)</h2>
<pre id="inspect-assembled" style="max-height:200px"></pre>
<h2 style="font-size:.95rem;margin-top:1rem">Stage timings</h2>
<pre id="inspect-timings"></pre>
</section>
</div>


</div>

<!-- ==================== MIDDLE: loaded AI chat ==================== -->
<div class="col mid">
<section class="chatpane">
<div id="sess-tabs" class="sesstabs"></div>
<div class="chat-head">
<h2 id="chat-title">Loaded model</h2>
<div class="chat-controls">
<span class="modesel">
<label class="inline">model <select id="chat-provider" onchange="saveConvProvider(this.value)" title="Inference target for this conversation"></select></label>
<label class="inline"><input type="radio" name="chatmode" value="hive" checked> Hive {tip('Curated generation: the hive assembles the context, then generates directly.')}</label>
<label class="inline"><input type="radio" name="chatmode" value="agent"> Agent (dsh) {tip('The full DeepSeek Harness agent loop — tools, multi-step turns, session log — pointed at the loaded model.')}</label>
</span>
<button onclick="newConversation()">New conversation</button>{tip('Opens a fresh session tab; the current one stays in the tab strip.')}
</div>
</div>
<div id="chatlog" class="chatlog"></div>
<div class="composer">
<span class="sugwrap composer-input"><input id="chatin" placeholder="Talk to the loaded AI…  (/ for commands)"
       onkeydown="if (event.key === 'Enter') chatSubmit()" autocomplete="off">
<div class="sugbox" id="sug-chat"></div></span>
<button id="sendbtn" onclick="chatSubmit()">Send</button>
<button id="stopbtn" onclick="cancelStream()">Stop</button>
<button id="savebtn" onclick="saveSession()">Save session</button>{tip('Name and keep this session as a tab. Tabs and transcripts survive page reloads.')}
<button onclick="newConversation()">+ New session</button>{tip('Opens a fresh session tab right away; the current one stays in the tab strip.')}
</div>
<div class="note"><b>Hive</b>: direct curated generation. <b>Agent (dsh)</b>:
the full DeepSeek Harness agent loop — bash/files/code tools, multi-step
turns, durable session log.</div>
</section>
</div>

<!-- ==================== RIGHT: Server (standalone, no tabs) ==================== -->
<div class="col">
<section>
<h2 style="margin-top:0">Launch</h2>
<div class="row"><button onclick="api('/v1/server/status').then(s => show('status', s))">Refresh</button>
<button onclick="api('/v1/server/stop', 'POST').then(() => refresh())">Stop</button></div>
<div class="row">
<label class="inline" style="flex:1">Model <select id="launch-model-select" style="flex:1;min-width:220px"><option value="">— choose local model —</option></select></label>
<span class="sugwrap"><input id="model" placeholder="or type path" size="18" list="local-suggestions">
<datalist id="local-suggestions"></datalist></span>{tip('Pick from Local library dropdown or type a GGUF path; blank uses Hugging Face repo/file below.')}
<label class="inline">ctx {tip('Context window in tokens llama-server serves; caps prompt + reply together.')} <input id="ctx" type="number" value="8192" size="4"></label>
<label class="inline">gpu {tip('Model layers offloaded to the GPU. 999 = every layer (needs enough VRAM); lower it if you run out.')} <input id="ngl" type="number" value="999" size="3"></label>
<label class="inline">api-key {tip('Bearer token llama-server expects on requests; leave blank when none is set.')} <input id="l-apikey" size="10" placeholder="(none)"></label><br>
<span class="sugwrap"><input id="hfrepo" placeholder="--hf-repo (type to search)" size="30">
<div class="sugbox" id="sug-hfrepo"></div></span>
<input id="hffile" placeholder="--hf-file" size="18">{tip('Hugging Face source: repo id plus the exact GGUF filename inside it.')}
<button onclick="startServer(this)">Start</button></div>
<details><summary>Advanced launch flags</summary>
<div class="row">
<label class="inline">threads {tip('CPU threads for inference; blank = automatic.')} <input id="l-threads" type="number" size="3" placeholder="auto"></label>
<label class="inline"><input id="l-fa" type="checkbox"> flash-attn {tip('FlashAttention kernels: faster attention and lower VRAM at long context.')}</label>
<label class="inline">parallel {tip('Requests decoded concurrently; each slot shares the context window.')} <input id="l-parallel" type="number" size="2" placeholder="1"></label>
<label class="inline"><input id="l-mlock" type="checkbox"> mlock {tip('Lock model weights in RAM so they never page to disk; slower startup.')}</label>
<label class="inline"><input id="l-nommap" type="checkbox"> no-mmap {tip('Read weights fully into memory instead of memory-mapping the file.')}</label>
</div>
<div class="row">
<label class="inline">kv-K {tip('Quantize the attention key cache to save VRAM (small quality cost).')} <select id="l-ctk"><option value="">f16</option><option>q8_0</option><option>q4_0</option></select></label>
<label class="inline">kv-V {tip('Same quantization for the value cache.')} <select id="l-ctv"><option value="">f16</option><option>q8_0</option><option>q4_0</option></select></label>
<label class="inline">batch {tip('Logical prompt-processing batch size.')} <input id="l-batch" type="number" size="4" placeholder="512"></label>
<label class="inline">ubatch {tip('Physical micro-batch fed to the model per step.')} <input id="l-ubatch" type="number" size="4" placeholder="512"></label>
<label class="inline">alias {tip('Model id exposed on /v1/models instead of the file path.')} <input id="l-alias" size="14" placeholder="model id"></label>
</div>
</details>
<pre id="status">loading…</pre>
<div id="instances"></div>
<pre id="srvlog" title="llama_server.log tail">log: loading…</pre>
</section>

</div></div>

<script>
let convId = localStorage.getItem('hive-console-conv');
if (!convId) {{
  convId = 'console-' + crypto.randomUUID().slice(0, 8);
  localStorage.setItem('hive-console-conv', convId);
}}
let hiveOverrides = {{}};
// engDirty declared in unified grid block above

async function api(path, method, body) {{
  const headers = {{'content-type': 'application/json'}};
  const token = localStorage.getItem('hive-token');
  if (token) headers['x-hive-token'] = token;
  const r = await fetch(path, {{method: method || 'GET', headers,
    body: body === undefined ? (method === 'POST' ? '{{}}' : undefined)
                             : JSON.stringify(body)}});
  if (r.status === 401) {{
    const t = prompt('This server requires an access token (HARNESS_TOKEN):');
    if (t !== null) {{ localStorage.setItem('hive-token', t); }}
    throw new Error('unauthorized — token saved, retry');
  }}
  const t = await r.text();
  if (!r.ok) throw new Error(r.status + ': ' + t.slice(0, 400));
  return t.startsWith('{{') ? JSON.parse(t) : t;
}}
function show(id, obj) {{ document.getElementById(id).textContent =
  typeof obj === 'string' ? obj : JSON.stringify(obj, null, 1); }}
function val(id) {{ return document.getElementById(id).value.trim(); }}
function num(id) {{ const v = val(id); return v === '' ? null : +v; }}

/* ------------------------------ tabs -------------------------------- */
for (const btn of document.querySelectorAll('.tab')) {{
  btn.addEventListener('click', () => {{
    for (const b of document.querySelectorAll('.tab')) b.classList.remove('active');
    for (const p of document.querySelectorAll('.tabpane')) p.style.display = 'none';
    btn.classList.add('active');
    document.getElementById(btn.dataset.tab).style.display = '';
    if (btn.dataset.tab === 'tab-engines') loadEngines();
    if (btn.dataset.tab === 'tab-hive') loadHiveDefaults();
    if (btn.dataset.tab === 'tab-providers') loadProviders();
  }});
}}

/* ---------------- typeahead (hub repos + local library) --------------- */
let suggestTimer = null;
function hideSuggestions() {{
  for (const id of ['sug-q', 'sug-drepo', 'sug-hfrepo'])
    document.getElementById(id).innerHTML = '';
}}

async function suggestRepos(value, targetId) {{
  if (!value || value.length < 2) {{ hideSuggestions(); return; }}
  let results = [];
  try {{
    const res = await api('/v1/models/hub?q=' + encodeURIComponent(value) + '&limit=8');
    results = res.results || [];
  }} catch (e) {{ return; }}
  const dl = document.getElementById('repo-suggestions');
  dl.innerHTML = '';
  for (const r of results) {{
    const opt = document.createElement('option');
    opt.value = r.repo;
    dl.appendChild(opt);
  }}
  const box = document.getElementById('sug-' + targetId);
  if (!box) return;
  box.innerHTML = '';
  for (const r of results.slice(0, 8)) {{
    const item = document.createElement('div');
    item.className = 'sugitem';
    const name = document.createElement('span');
    name.textContent = r.repo;
    const meta = document.createElement('span');
    meta.className = 'meta';
    meta.textContent = '↓' + r.downloads;
    item.appendChild(name);
    item.appendChild(meta);
    item.addEventListener('mousedown', ev => {{
      ev.preventDefault();
      document.getElementById(targetId).value = r.repo;
      hideSuggestions();
      if (targetId === 'q') searchHub();
      else hubFiles(r.repo).catch(() => {{}});
    }});
    box.appendChild(item);
  }}
}}

async function hubFiles(repo) {{
  const res = await api('/v1/models/hub/files/' + encodeURIComponent(repo));
  document.getElementById('drepo').value = repo;
  document.getElementById('dfile').value = res.files.length ? res.files[0].file : '';
  const lines = res.files.map(f => `${{f.file}} — ${{f.size_gb}} GB`);
  show('hub', `files in ${{repo}}:\\n${{lines.join('\\n') || '(none)'}}`);
}}

async function suggestLocal() {{
  try {{
    const l = await api('/v1/models/local');
    const dl = document.getElementById('local-suggestions');
    dl.innerHTML = '';
    for (const m of l.models) {{
      const opt = document.createElement('option');
      opt.value = m.file;
      opt.label = `${{m.size_gb}} GB`;
      dl.appendChild(opt);
    }}
  }} catch (e) {{ /* best-effort */ }}
}}

for (const id of ['q', 'drepo', 'hfrepo']) {{
  const el = document.getElementById(id);
  if (!el) continue;
  el.addEventListener('input',
    e => {{ clearTimeout(suggestTimer);
           suggestTimer = setTimeout(() => suggestRepos(e.target.value.trim(), id), 350); }});
  el.addEventListener('blur', () => setTimeout(hideSuggestions, 150));
}}
document.getElementById('model').addEventListener('focus', suggestLocal);

/* --------------------- sessions (OpenCode-style tabs) ----------------- */
const SESS_KEY = 'hive-console-sessions';
let sessions = {{}};
try {{ sessions = JSON.parse(localStorage.getItem(SESS_KEY) || '{{}}') || {{}}; }}
catch (e) {{ sessions = {{}}; }}
function persistSessions() {{
  localStorage.setItem(SESS_KEY, JSON.stringify(sessions));
}}
function ensureSession(id) {{
  if (!sessions[id]) sessions[id] = {{ title: null, updated: Date.now(), transcript: [] }};
  return sessions[id];
}}
function record(role, text, meta) {{
  const s = ensureSession(convId);
  s.transcript.push({{role: role, text: text, meta: meta || null}});
  if (s.transcript.length > 400) s.transcript.splice(0, s.transcript.length - 400);
  s.updated = Date.now();
  persistSessions();
}}
function restoreTranscript(id) {{
  const log = document.getElementById('chatlog');
  log.innerHTML = '';
  const s = sessions[id];
  if (!s) return;
  for (const m of s.transcript) {{
    if (m.role === 'tool') continue;   // tool steps are agent-run noise on replay
    bubble(m.role, m.text, m.meta);
  }}
}}
function tabLabel(id) {{
  const s = sessions[id];
  return (s && s.title) ? s.title : 'Session ' + id.slice(-5);
}}
function renderTabs() {{
  const wrap = document.getElementById('sess-tabs');
  wrap.innerHTML = '';
  const ids = Object.keys(sessions).sort((a, b) =>
    (sessions[b].updated || 0) - (sessions[a].updated || 0));
  for (const id of ids) {{
    const t = document.createElement('span');
    t.className = 'sesstab' + (id === convId ? ' active' : '')
      + (sessions[id].title ? '' : ' unsaved');
    const name = document.createElement('span');
    name.className = 'name';
    name.textContent = tabLabel(id);
    t.appendChild(name);
    const x = document.createElement('span');
    x.className = 'x';
    x.textContent = '\\u00d7';
    x.title = 'Close session';
    x.addEventListener('click', ev => {{ ev.stopPropagation(); closeSession(id); }});
    t.appendChild(x);
    t.addEventListener('click', () => switchSession(id));
    wrap.appendChild(t);
  }}
}}
async function switchSession(id) {{
  if (id === convId) return;
  convId = id;
  localStorage.setItem('hive-console-conv', convId);
  restoreTranscript(id);
  renderTabs();
  applyConvProvider();
  try {{ await api('/v1/hive/state?conversation_id=' + encodeURIComponent(id)); }} catch (e) {{}}
}}
function closeSession(id) {{
  delete sessions[id];
  persistSessions();
  api('/v1/hive/reset', 'POST', {{conversation_id: id}}).catch(() => {{}});
  if (id !== convId) {{ renderTabs(); return; }}
  const rest = Object.keys(sessions).sort((a, b) =>
    (sessions[b].updated || 0) - (sessions[a].updated || 0));
  if (rest.length) switchSession(rest[0]);
  else newConversation();
}}
function saveSession() {{
  const cur = ensureSession(convId);
  const name = prompt('Name this session:', cur.title || '');
  if (name === null) return;
  cur.title = name.trim() || ('Session ' + new Date().toLocaleString());
  cur.updated = Date.now();
  persistSessions();
  renderTabs();
}}
/* per-conversation inference target (the header model dropdown) */
function convProviderStore() {{
  try {{ return JSON.parse(localStorage.getItem('hive-console-convprov') || '{{}}'); }}
  catch (e) {{ return {{}}; }}
}}
function saveConvProvider(value) {{
  const per = convProviderStore();
  if (value) per[convId] = value;
  else delete per[convId];
  localStorage.setItem('hive-console-convprov', JSON.stringify(per));
}}
function applyConvProvider() {{
  const v = convProviderStore()[convId];
  if (!v) return;
  const sel = document.getElementById('chat-provider');
  if ([...sel.options].some(o => o.value === v)) sel.value = v;
}}

/* --------------------------- afk toggle ------------------------------ */
async function toggleAfk(btn) {{
  const on = btn.dataset.afk === '1';
  const r = await api('/v1/hive/mode', 'POST', {{afk: !on, note: on ? 'operator returned' : 'operator away'}});
  applyAfk(r.afk);
}}
function applyAfk(on) {{
  const b = document.getElementById('afkbtn');
  if (!b) return;
  b.dataset.afk = on ? '1' : '0';
  b.textContent = on ? 'AFK ON - click to end' : 'AFK';
  b.classList.toggle('afk-on', !!on);
}}
(async function initAfk() {{
  try {{ const m = await api('/v1/hive/mode'); applyAfk(!!m.afk); }} catch (e) {{}}
}})();
/* --------------------------- research queue -------------------------- */
async function researchAdd(btn) {{
  const inp = document.getElementById('researchq');
  const q = inp.value.trim();
  if (!q) return;
  btn.disabled = true;
  try {{
    await api('/v1/research/queue', 'POST', {{question: q}});
    inp.value = '';
    loadResearchCount();
  }} finally {{ btn.disabled = false; }}
}}
async function loadResearchCount() {{
  try {{
    const r = await api('/v1/research/queue');
    const b = document.getElementById('researchcount');
    if (b) b.textContent = r.items.length ? '(' + r.items.length + ' queued)' : '';
  }} catch (e) {{}}
}}
loadResearchCount();
/* ------------------------------ chat -------------------------------- */
function bubble(role, text, meta) {{
  const log = document.getElementById('chatlog');
  const div = document.createElement('div');
  div.className = 'msg ' + role;
  div.textContent = text;
  if (meta) {{
    const m = document.createElement('span');
    m.className = 'meta';
    m.textContent = meta;
    div.appendChild(m);
  }}
  log.appendChild(div);
  log.scrollTop = log.scrollHeight;
  // Cap DOM growth: a long-lived tab must not accumulate every bubble.
  while (log.children.length > 200) log.removeChild(log.firstChild);
  return div;
}}

function chatMode() {{
  return document.querySelector('input[name="chatmode"]:checked').value;
}}

/* ------------------------- slash commands ---------------------------- */
let chatCommands = [];

async function loadChatCommands() {{
  try {{
    chatCommands = (await api('/v1/commands')).commands;
  }} catch (e) {{ chatCommands = []; }}
}}

function chatSubmit() {{
  const input = document.getElementById('chatin');
  const text = input.value;
  if (text.trim().startsWith('/')) {{
    input.value = '';
    hideSuggestions();
    runCommand(text.trim());
    return;
  }}
  sendChat();
}}

async function runCommand(line) {{
  bubble('user', line);
  record('user', line);
  try {{
    const r = await api('/v1/commands/run', 'POST',
                        {{line: line, conversation_id: convId}});
    bubble('sys', (r.kind === 'error' ? '⚠ ' : '') + (r.text || r.kind));
    record('sys', (r.kind === 'error' ? '⚠ ' : '') + (r.text || r.kind));
    if (r.new_conversation_id) {{
      convId = r.new_conversation_id;
      localStorage.setItem('hive-console-conv', convId);
      ensureSession(convId);
      persistSessions();
      renderTabs();
      document.getElementById('chatlog').innerHTML = '';
    }}
    if (r.mode) {{
      const radio = document.querySelector(
        `input[name="chatmode"][value="${{r.mode}}"]`);
      if (radio) radio.checked = true;
    }}
  }} catch (e) {{
    bubble('sys', 'command failed: ' + e.message);
  }}
}}

document.getElementById('chatin').addEventListener('input', e => {{
  const v = e.target.value;
  const box = document.getElementById('sug-chat');
  if (!v.startsWith('/')) {{ box.innerHTML = ''; return; }}
  const query = v.slice(1).toLowerCase();
  const matches = chatCommands.filter(c => c.name.startsWith(query));
  box.innerHTML = '';
  for (const cmd of matches) {{
    const item = document.createElement('div');
    item.className = 'sugitem';
    const name = document.createElement('span');
    name.textContent = '/' + cmd.name;
    const meta = document.createElement('span');
    meta.className = 'meta';
    meta.textContent = cmd.description + (cmd.input ? ' ' + cmd.input.hint : '');
    item.appendChild(name);
    item.appendChild(meta);
    item.addEventListener('mousedown', ev => {{
      ev.preventDefault();
      e.target.value = '/' + cmd.name + ' ';
      box.innerHTML = '';
      e.target.focus();
    }});
    box.appendChild(item);
  }}
}});

loadChatCommands();

async function sendChat() {{
  if (streamBusy) return;              /* one stream at a time */
  const input = document.getElementById('chatin');
  const query = input.value.trim();
  if (!query) return;
  input.value = '';
  if (chatMode() === 'agent') return sendAgent(query);
  bubble('user', query);
  record('user', query);
  const ai = bubble('ai', '');
  ai.classList.add('typing');
  setBusy(true);
  const ctrl = new AbortController();
  streamAbort = ctrl;
  const body = {{query: query, conversation_id: convId}};
  const provSel = document.getElementById('chat-provider').value;
  if (provSel) body.provider = provSel;
  if (Object.keys(hiveOverrides).length) body.config = hiveOverrides;
  let text = '';
  let turn = '?';
  let metaText = '';
  try {{
    const r = await fetch('/v1/hive/stream', {{method: 'POST',
      headers: {{'content-type': 'application/json'}},
      body: JSON.stringify(body), signal: ctrl.signal}});
    if (!r.ok || !r.body) throw new Error('HTTP ' + r.status);
    const reader = r.body.getReader();
    const dec = new TextDecoder();
    let buf = '';
    while (true) {{
      const {{done, value}} = await reader.read();
      if (done) break;
      buf += dec.decode(value, {{stream: true}});
      let idx;
      while ((idx = buf.indexOf('\\n\\n')) >= 0) {{
        const frame = buf.slice(0, idx); buf = buf.slice(idx + 2);
        if (!frame.startsWith('data: ')) continue;
        const ev = JSON.parse(frame.slice(6));
        if (ev.type === 'delta') {{
          if (!text) ai.classList.remove('typing');
          text += ev.text;
          ai.textContent = text;
          log_scroll();
        }} else if (ev.type === 'meta') {{
          turn = ev.turn;
          metaText = `curated ${{ev.token_count}}/${{ev.budget}} tokens`;
        }} else if (ev.type === 'done') {{
          metaText = `hive-curated · turn ${{turn}}`
            + (ev.tokens ? ` · ${{ev.tokens}} tok` : '')
            + (ev.tokens_per_sec ? ` · ${{ev.tokens_per_sec}} tok/s` : '')
            + (ev.stored ? '' : ' · not stored');
        }} else if (ev.type === 'error') {{
          metaText = 'error: ' + friendlyError(ev.error);
        }}
      }}
    }}
    ai.classList.remove('typing');
    if (!text && metaText) {{
      /* error-only turn: show the message itself, not an empty bubble */
      ai.textContent = metaText;
    }} else {{
      ai.textContent = text || '(empty reply)';
      const m = document.createElement('span');
      m.className = 'meta';
      m.textContent = metaText;
      ai.appendChild(m);
    }}
    if (body.inspection !== undefined && body.inspection) {{
      renderInspection(body.inspection);
    }} else {{
      fetchInspection();
    }}
  }} catch (e) {{
    ai.classList.remove('typing');
    const cancelled = e && e.name === 'AbortError';
    const msg = cancelled ? 'cancelled.'
      : 'request failed: ' + friendlyError(e.message);
    ai.textContent = msg;
    record('ai', msg);
    if (!input.value) input.value = query;   /* give the draft back */
  }} finally {{
    streamAbort = null;
    setBusy(false);
  }}
}}

let streamBusy = false;
let streamAbort = null;
function setBusy(b) {{
  streamBusy = b;
  document.getElementById('stopbtn').style.display = b ? '' : 'none';
  document.getElementById('sendbtn').disabled = b;
}}

/* Map backend/sidecar failure text to something a user can act on. The
   SSE `error` frame carries raw Python exception strings today; this is
   the client-side translation layer until the server emits clean copy. */
function friendlyError(msg) {{
  const s = String(msg || '');
  const low = s.toLowerCase();
  if (low.includes(':1234') || low.includes('max retries') ||
      low.includes('failed to establish') || low.includes('connection refused'))
    return 'No backend reachable on :1234 — start a server from the Models '
         + 'tab, or pick another provider.';
  if (low.includes('failed to fetch') || low.includes('networkerror') ||
      low.includes('load failed'))
    return 'Sidecar unreachable — is the harness process still running?';
  if (s.length > 300) return s.slice(0, 300) + '…';
  return s;
}}

function cancelStream() {{
  if (chatMode() === 'agent') return cancelAgent();
  if (streamAbort) streamAbort.abort();
}}

async function cancelAgent() {{
  try {{ await api('/v1/agent/cancel', 'POST'); }} catch (e) {{}}
}}

async function sendAgent(query) {{
  bubble('user', query);
  record('user', query);
  bubble('sys', 'dsh agent starting…');
  const ai = bubble('ai', '');
  setBusy(true);
  let text = '';
  let finish = '';
  try {{
    const r = await fetch('/v1/agent/stream', {{method: 'POST',
      headers: {{'content-type': 'application/json'}},
      body: JSON.stringify({{message: query, conversation_id: convId}})}});
    if (!r.ok || !r.body) {{
      const t = await r.text();
      throw new Error('HTTP ' + r.status + ' ' + t.slice(0, 200));
    }}
    const reader = r.body.getReader();
    const dec = new TextDecoder();
    let buf = '';
    while (true) {{
      const {{done, value}} = await reader.read();
      if (done) break;
      buf += dec.decode(value, {{stream: true}});
      let idx;
      while ((idx = buf.indexOf('\\n\\n')) >= 0) {{
        const frame = buf.slice(0, idx); buf = buf.slice(idx + 2);
        if (!frame.startsWith('data: ')) continue;
        const ev = JSON.parse(frame.slice(6));
        if (ev.type === 'assistant') {{
          text = ev.text;                       // committed per agent step
          ai.textContent = text;
          log_scroll();
        }} else if (ev.type === 'tool') {{
          /* persistent collapsible step: stays in the transcript so a long
             agent run reads like dsh's own tool view instead of toasts */
          const det = document.createElement('details');
          det.className = 'toolstep';
          const sum = document.createElement('summary');
          sum.textContent = (ev.phase === 'end' ? '\\u2713 ' : '\\u25b6 ')
            + ev.tool;
          det.appendChild(sum);
          const pre = document.createElement('pre');
          pre.textContent = JSON.stringify(ev, null, 1).slice(0, 2000);
          det.appendChild(pre);
          document.getElementById('chatlog').appendChild(det);
          record('tool', ev.phase + ': ' + ev.tool);
          log_scroll();
        }} else if (ev.type === 'lifecycle') {{
          /* turn/step boundaries stay quiet */
        }} else if (ev.type === 'done') {{
          finish = ev.finish_reason || 'end';
          if (ev.final && ev.final !== text) {{
            text = ev.final;
            ai.textContent = text;
          }}
        }} else if (ev.type === 'error') {{
          ai.textContent = 'agent error: ' + ev.error;
          return;
        }}
      }}
    }}
    ai.textContent = text || '(no response)';
    const m = document.createElement('span');
    m.className = 'meta';
    m.textContent = 'dsh agent · ' + finish;
    ai.appendChild(m);
    record('ai', text || '(no response)', 'dsh agent · ' + finish);
  }} catch (e) {{
    ai.textContent = 'agent request failed: ' + e.message;
    record('ai', 'agent request failed: ' + e.message);
  }} finally {{
    setBusy(false);
  }}
}}
function log_scroll() {{
  const log = document.getElementById('chatlog');
  log.scrollTop = log.scrollHeight;
}}

function newConversation() {{
  /* OpenCode semantics: the current session keeps its tab; this opens a
     fresh conversation id and a fresh (empty) tab beside it. */
  convId = 'console-' + crypto.randomUUID().slice(0, 8);
  localStorage.setItem('hive-console-conv', convId);
  ensureSession(convId);
  persistSessions();
  document.getElementById('chatlog').innerHTML = '';
  bubble('ai', 'Fresh session started.'
    + (Object.keys(hiveOverrides).length
       ? ' Hive overrides apply from the next message.' : ''));
  renderTabs();
}}

/* boot: rebuild the tab strip and the last session's transcript */
ensureSession(convId);
persistSessions();
renderTabs();
restoreTranscript(convId);
/* the model dropdown populates async from /v1/provider/config */
setTimeout(applyConvProvider, 2000);

/* --------------------------- server panel ---------------------------- */
function launchBody() {{
  return {{
    model: val('model') || null,
    hf_repo: val('hfrepo') || null,
    hf_file: val('hffile') || null,
    ctx_size: +val('ctx') || 8192,
    ngl: +val('ngl') || 999,
    api_key: val('l-apikey') || null,
    threads: num('l-threads'),
    flash_attn: document.getElementById('l-fa').checked,
    parallel_slots: num('l-parallel'),
    cache_type_k: val('l-ctk') || null,
    cache_type_v: val('l-ctv') || null,
    batch_size: num('l-batch'),
    ubatch_size: num('l-ubatch'),
    alias: val('l-alias') || null,
    mlock: document.getElementById('l-mlock').checked,
    no_mmap: document.getElementById('l-nommap').checked,
  }};
}}

async function startServer(btn) {{
  if (btn) btn.disabled = true;          /* slow POST — no double submits */
  try {{
    const r = await fetch('/v1/server/start', {{method: 'POST',
      headers: {{'content-type': 'application/json'}},
      body: JSON.stringify(launchBody())}});
    show('status', await r.text());
    refresh();
  }} finally {{ if (btn) btn.disabled = false; }}
}}

async function searchHub() {{
  show('hub', 'searching…');
  try {{
    const res = await api('/v1/models/hub?q=' +
      encodeURIComponent(val('q')) + '&limit=12');
    show('hub', res.results.map(r =>
      `${{r.repo}} · ↓${{r.downloads}} · ★${{r.likes}} · ${{r.last_modified}}`).join('\\n'));
  }} catch (e) {{ show('hub', String(e)); }}
}}

async function download(btn) {{
  if (btn) btn.disabled = true;          /* slow POST — no double submits */
  try {{
    const r = await fetch('/v1/models/hub/download', {{method: 'POST',
      headers: {{'content-type': 'application/json'}},
      body: JSON.stringify({{repo: val('drepo'), file: val('dfile')}})}});
    show('downloads', await r.text());
  }} finally {{ if (btn) btn.disabled = false; }}
}}

async function setLibraryPath() {{
  const input = document.getElementById('library-path');
  const folder = input.value.trim();
  const status = document.getElementById('import-status');
  if (!folder) {{ status.textContent = 'enter a folder path'; setTimeout(()=>status.textContent='',2000); return; }}
  status.textContent = `switching to ${{folder}}…`;
  try {{
    const r = await fetch('/v1/models/local/path', {{method: 'POST', headers: {{'content-type':'application/json'}}, body: JSON.stringify({{folder}})}});
    const j = await r.json();
    if (!r.ok) throw new Error(j.detail || j.error || await r.text());
    status.textContent = `now using ${{j.models_dir}} (${{j.models?.length || 0}} models)`;
    document.getElementById('library-path-note').textContent = j.models_dir;
    refresh();
  }} catch (e) {{
    status.textContent = 'failed: ' + e.message;
  }} finally {{
    setTimeout(()=>status.textContent='',4000);
  }}
}}

function filterLibrary() {{
  const q = (document.getElementById('library-filter').value || '').toLowerCase().trim();
  const wrap = document.getElementById('local');
  let visible = 0;
  for (const row of wrap.children) {{
    const name = (row.dataset.file || '').toLowerCase();
    const show = !q || name.includes(q);
    row.style.display = show ? '' : 'none';
    if (show) visible++;
  }}
  const count = document.getElementById('library-filter-count');
  if (count) count.textContent = q ? `${{visible}} / ${{wrap.children.length}}` : '';
  // Also filter the Engine and Launch dropdowns
  const engSel = document.getElementById('eng-model-select');
  const launchSel = document.getElementById('launch-model-select');
  const filterSelect = (sel) => {{
    if (!sel) return;
    for (const opt of sel.options) {{
      if (!opt.value) continue;
      opt.style.display = !q || opt.value.toLowerCase().includes(q) ? '' : 'none';
      opt.hidden = !q || opt.value.toLowerCase().includes(q) ? false : true;
    }}
  }};
  filterSelect(engSel);
  filterSelect(launchSelect);
}}

/* --------------------------- engines tab ----------------------------- */
let engines = [];
let engDefault = '';

function samplingToForm(s) {{
  const set = (id, v) => document.getElementById(id).value =
    (v === undefined || v === null) ? '' : v;
  set('s-temp', s.temperature); set('s-topp', s.top_p); set('s-topk', s.top_k);
  set('s-minp', s.min_p); set('s-rep', s.repeat_penalty);
  set('s-pres', s.presence_penalty); set('s-freq', s.frequency_penalty);
  set('s-seed', s.seed); set('s-miro', s.mirostat);
  set('s-tau', s.mirostat_tau); set('s-eta', s.mirostat_eta);
  set('s-stop', Array.isArray(s.stop) ? s.stop.join(',') : (s.stop || ''));
}}

function formToSampling() {{
  const out = {{}};
  const put = (key, id, parse) => {{
    const raw = val(id);
    if (raw !== '') out[key] = parse ? +raw : raw;
  }};
  put('temperature', 's-temp', true); put('top_p', 's-topp', true);
  put('top_k', 's-topk', true); put('min_p', 's-minp', true);
  put('repeat_penalty', 's-rep', true); put('presence_penalty', 's-pres', true);
  put('frequency_penalty', 's-freq', true); put('seed', 's-seed', true);
  put('mirostat', 's-miro', true); put('mirostat_tau', 's-tau', true);
  put('mirostat_eta', 's-eta', true);
  const stop = val('s-stop');
  if (stop) out.stop = stop.includes(',') ? stop.split(',').map(x => x.trim())
                                          : stop;
  return out;
}}

function currentEngineIndex() {{
  return +document.getElementById('eng-select').value;
}}

function engineSelected() {{
  const e = engines[currentEngineIndex()];
  if (!e) return;
  document.getElementById('eng-name').value = e.name;
  document.getElementById('eng-kind').value = e.kind;
  document.getElementById('eng-url').value = e.base_url || '';
  document.getElementById('eng-default').checked =
    e.name.toLowerCase() === engDefault.toLowerCase();
  samplingToForm(e.sampling || {{}});
  show('eng-loadopts', e.load_options && Object.keys(e.load_options).length
    ? e.load_options : '(none recorded)');
  engDirty = false;
  
  // merged grid: populate uniform fields
  const mSel = document.getElementById('eng-model');
  if (mSel && e.model) mSel.value = e.model;
  const ctxLenInput = document.getElementById('eng-ctxlen');
  if (ctxLenInput) ctxLenInput.value = String(e.load_options?.context ?? e.load_options?.contextLength ?? 8192);
  const fitCtx = document.getElementById('fit-ctx');
  if (fitCtx) fitCtx.value = String(e.load_options?.context ?? 8192);
  const thr = document.getElementById('eng-threads');
  if (thr) thr.value = e.load_options?.threads ?? '';
  const gpu = document.getElementById('eng-gpu');
  if (gpu) gpu.value = String(e.load_options?.gpu_layers ?? 999);
  const flash = document.getElementById('eng-flash');
  if (flash) flash.value = e.load_options?.flash_attn ? 'on' : '';
  const par = document.getElementById('eng-parallel');
  if (par) par.value = e.load_options?.parallel_slots ?? '';
  const bat = document.getElementById('eng-batch');
  if (bat) bat.value = e.load_options?.batch_size ?? '';
  const ubat = document.getElementById('eng-ubatch');
  if (ubat) ubat.value = e.load_options?.ubatch_size ?? '';
  const ctk = document.getElementById('eng-ctk');
  if (ctk) ctk.value = e.load_options?.cache_type_k ?? '';
  const ctv = document.getElementById('eng-ctv');
  if (ctv) ctv.value = e.load_options?.cache_type_v ?? '';
  const alias = document.getElementById('eng-alias');
  if (alias) alias.value = e.load_options?.alias ?? '';
  const disp = document.getElementById('eng-displayName');
  if (disp) disp.value = e.displayName || e.name || '';
  const apik = document.getElementById('eng-apikey');
  if (apik) apik.value = '';
document.getElementById('eng-msg').textContent = '';
}}

function engineAdd() {{
  engines.push({{name: 'engine-' + (engines.length + 1), kind: 'llama_cpp',
                base_url: '', load_options: {{}}, capabilities: ['streaming'],
                sampling: {{}}}});
  renderEngineSelect(engines.length - 1);
  engineSelected();
  engDirty = true;
}}

function renderEngineSelect(selectIndex) {{
  const sel = document.getElementById('eng-select');
  sel.innerHTML = '';
  engines.forEach((e, i) => {{
    const o = document.createElement('option');
    o.value = i;
    o.textContent = e.name + (e.name.toLowerCase() === engDefault.toLowerCase()
                              ? '  (default)' : '');
    sel.appendChild(o);
  }});
  sel.value = selectIndex;
  refreshAbSelects();
}}

function refreshAbSelects() {{
  const ids = ['ab-select-a','ab-select-b','ab-profile-a','ab-profile-b'];
  for (const id of ids) {{
    const sel = document.getElementById(id);
    if (!sel) continue;
    const cur = sel.value;
    sel.innerHTML = '';
    engines.forEach(e => {{
      const o = document.createElement('option');
      o.value = e.name;
      o.textContent = e.name;
      sel.appendChild(o);
    }});
    if (cur && [...sel.options].some(o => o.value === cur)) sel.value = cur;
    else if (sel.options.length) {{
      if (id.endsWith('-a') && sel.options.length >= 1) sel.value = sel.options[0].value;
      if (id.endsWith('-b') && sel.options.length >= 2) sel.value = sel.options[1].value;
      else if (id.endsWith('-b') && sel.options.length >= 1) sel.value = sel.options[0].value;
    }}
  }}
}}

async function benchAb() {{
  const a = document.getElementById('ab-select-a')?.value || document.getElementById('ab-profile-a')?.value || '';
  const b = document.getElementById('ab-select-b')?.value || document.getElementById('ab-profile-b')?.value || '';
  const basePort = parseInt(document.getElementById('ab-baseport')?.value || '1234', 10);
  const btn = document.getElementById('ab-bench-btn');
  const winnerEl = document.getElementById('ab-winner');
  const badgeEl = document.getElementById('ab-badge');
  const ra = document.getElementById('ab-result-a');
  const rb = document.getElementById('ab-result-b');
  if (!a || !b) {{ if (winnerEl) winnerEl.textContent = 'select two profiles'; return; }}
  if (btn) btn.disabled = true;
  if (winnerEl) winnerEl.textContent = 'benching…';
  if (badgeEl) badgeEl.textContent = 'benching…';
  if (ra) ra.textContent = '';
  if (rb) rb.textContent = '';
  try {{
    let r;
    try {{
      r = await api('/v1/engines/ab/bench', 'POST', {{profile_a: a, profile_b: b, basePort: basePort, base_port: basePort}});
    }} catch (e) {{
      // fallback aliases
      try {{ r = await api('/v1/engines/bench', 'POST', {{profile_a: a, profile_b: b, basePort: basePort}}); }}
      catch (e2) {{ r = await api('/v1/ab/bench', 'POST', {{a: a, b: b, basePort: basePort}}); }}
    }}
    const winner = r.winner || (r.a_tok_per_sec > r.b_tok_per_sec ? 'A' : r.b_tok_per_sec > r.a_tok_per_sec ? 'B' : 'tie');
    const atps = (r.a_tok_per_sec ?? r.a?.tok_per_sec ?? r.tokens_per_sec_a ?? 0).toFixed(1);
    const btps = (r.b_tok_per_sec ?? r.b?.tok_per_sec ?? r.tokens_per_sec_b ?? 0).toFixed(1);
    const txt = `winner: ${{winner}} — A ${{atps}} tok/s vs B ${{btps}} tok/s`;
    if (winnerEl) {{ winnerEl.textContent = txt; winnerEl.className = 'winner-badge winner-' + winner.toLowerCase(); }}
    if (badgeEl) {{ badgeEl.textContent = txt; badgeEl.style.display = ''; badgeEl.className = 'winner-badge'; }}
    if (ra) ra.textContent = `A (${{a}}) ${{atps}} tok/s`;
    if (rb) rb.textContent = `B (${{b}}) ${{btps}} tok/s`;
  }} catch (e) {{
    const msg = 'error: ' + (e.message || String(e));
    if (winnerEl) winnerEl.textContent = msg;
    if (badgeEl) {{ badgeEl.textContent = msg; badgeEl.style.display = ''; }}
  }} finally {{
    if (btn) btn.disabled = false;
  }}
}}

function collectEngines() {{
  const e = engines[currentEngineIndex()];
  if (e) {{
    e.name = val('eng-name') || e.name;
    e.kind = document.getElementById('eng-kind').value;
    e.base_url = val('eng-url');
    e.sampling = formToSampling();
  }}
  return engines;
}}

async function loadEngines() {{
  try {{
    const data = await api('/v1/engines');
    engines = data.engines.map(e => ({{...e}}));
    engDefault = data.default || '';
    engRevision = data.revision ?? 0;
    // also fetch provider revision for atomic guard
    try {{ const p = await api('/v1/provider/config'); provRevision = p.revision ?? 0; provDefault = p.default || provDefault; }} catch(e) {{}}
    renderEngineSelect(0);
    engineSelected();
  }} catch (e) {{ show('eng-loadopts', String(e)); }}
}}

// unified Engine profiles grid — single form edits both namespaces atomically (ctx.models lifecycle + ctx.llm provider profile)
let engRevision = 0;
let provRevision = 0;
let engDirty = false;

// settings façade mirroring dsh-settings: mutate(ns, payload, expectedRevision)
const settings = {{
  mutate: async (ns, payload, expectedRevision) => {{
    const path = ns === 'llm' ? '/v1/provider/config' : '/v1/engines';
    const body = ns === 'llm'
      ? {{ providers: payload.providers, default: payload.default, persist: true, expectedRevision }}
      : {{ engines: payload.engines, default: payload.default, persist: true, expectedRevision }};
    const r = await api(path, 'POST', body);
    if (ns === 'llm') provRevision = r.revision ?? provRevision + 1;
    else engRevision = r.revision ?? engRevision + 1;
    return r;
  }}
}};

async function requestLoad(opts) {{
  // ctx.models lifecycle: spawn /v1/server/start with the profile's load_options
  return api('/v1/server/start', 'POST', opts);
}}

async function saveEngines() {{
  // legacy alias — now delegates to the unified saver
  return saveEngineProfile();
}}

async function saveEngineProfile() {{
  const form = document.getElementById('engine-profile-form');
  if (!form) return saveEnginesLegacy();
  const idx = currentEngineIndex();
  const eng = engines[idx];
  if (!eng) return;
  // collect Endpoint + Model + Context into the two namespaces
  eng.name = val('eng-name') || eng.name;
  eng.kind = document.getElementById('eng-kind').value;
  eng.base_url = val('eng-url');
  eng.displayName = val('eng-displayName') || eng.name;
  // Model dropdown from local library
  const selModel = val('eng-model') || document.getElementById('launch-model-select')?.value || '';
  eng.model = selModel;
  // Context
  const ctxLen = parseInt(val('eng-ctxlen') || document.getElementById('fit-ctx')?.value || '8192', 10);
  eng.load_options = eng.load_options || {{}};
  eng.load_options.context = ctxLen;
  eng.load_options.contextLength = ctxLen;
  // Load grid
  const threadsVal = val('eng-threads'); if (threadsVal) eng.load_options.threads = parseInt(threadsVal,10); else delete eng.load_options.threads;
  const gpuVal = val('eng-gpu'); if (gpuVal) eng.load_options.gpu_layers = parseInt(gpuVal,10);
  const flashVal = document.getElementById('eng-flash')?.value; if (flashVal) eng.load_options.flash_attn = flashVal === 'on'; else delete eng.load_options.flash_attn;
  const parVal = val('eng-parallel'); if (parVal) eng.load_options.parallel_slots = parseInt(parVal,10); else delete eng.load_options.parallel_slots;
  const batchVal = val('eng-batch'); if (batchVal) eng.load_options.batch_size = parseInt(batchVal,10); else delete eng.load_options.batch_size;
  const ubatchVal = val('eng-ubatch'); if (ubatchVal) eng.load_options.ubatch_size = parseInt(ubatchVal,10); else delete eng.load_options.ubatch_size;
  eng.load_options.cache_type_k = val('eng-ctk') || undefined;
  eng.load_options.cache_type_v = val('eng-ctv') || undefined;
  eng.load_options.alias = val('eng-alias') || undefined;
  if (val('eng-alias')) eng.load_options.alias = val('eng-alias');
  // Sampling 4x2
  eng.sampling = formToSampling();
  // Advanced (mirostat etc already in sampling)
  // persist expectedRevision guard — 409 conflict if stale
  const msgEl = document.getElementById('eng-msg');
  const saveBtn = document.getElementById('eng-save');
  if (saveBtn) saveBtn.disabled = true;
  if (msgEl) msgEl.textContent = 'saving…';
  // build llm provider profile (ctx.llm) from Endpoint section
  const llmPayload = {{
    providers: [{{ name: eng.name, base_url: eng.base_url, api_key: val('eng-apikey') || '', model: selModel || eng.model || '', displayName: eng.displayName || eng.name }}],
    default: document.getElementById('eng-default')?.checked ? eng.name : provDefault,
  }};
  const modelsPayload = {{
    engines: engines,
    default: document.getElementById('eng-default')?.checked ? eng.name : engDefault,
  }};
  const loadOpts = {{
    model: selModel || eng.model || null,
    ctx_size: ctxLen,
    ngl: eng.load_options.gpu_layers ?? 999,
    threads: eng.load_options.threads ?? null,
    flash_attn: eng.load_options.flash_attn ?? false,
    parallel_slots: eng.load_options.parallel_slots ?? null,
    cache_type_k: eng.load_options.cache_type_k ?? null,
    cache_type_v: eng.load_options.cache_type_v ?? null,
    batch_size: eng.load_options.batch_size ?? null,
    ubatch_size: eng.load_options.ubatch_size ?? null,
    alias: eng.load_options.alias ?? null,
    mlock: eng.load_options.mlock ?? false,
    no_mmap: eng.load_options.no_mmap ?? false,
  }};
  try {{
    // single Save that atomically mutates both namespaces + lifecycle
    // ctx.models lifecycle + ctx.llm provider profile
    await Promise.all([

    // single Save: Promise.all([settings.mutate(llm), settings.mutate(models), requestLoad]) with expectedRevision guard      settings.mutate('llm', llmPayload, provRevision),
      settings.mutate('models', modelsPayload, engRevision),
      requestLoad(loadOpts),
    ]);
    if (msgEl) msgEl.textContent = 'saved + loaded';
    engDirty = false;
  }} catch (e) {{
    const txt = String(e.message || e);
    if (txt.includes('409') || txt.toLowerCase().includes('revision')) {{
      if (msgEl) msgEl.textContent = 'conflict: stale revision — reload and retry (expectedRevision guard)';
    }} else {{
      if (msgEl) msgEl.textContent = 'save failed: ' + txt;
    }}
    throw e;
  }} finally {{
    if (saveBtn) saveBtn.disabled = false;
    setTimeout(() => {{ if (msgEl && msgEl.textContent.startsWith('saved')) msgEl.textContent = ''; }}, 3000);
  }}
}}

async function saveEnginesLegacy() {{

  const list = collectEngines();
  const defName = document.getElementById('eng-default').checked
    ? (val('eng-name') || (engines[currentEngineIndex()] || {{}}).name || '')
    : engDefault;
  try {{
    const r = await api('/v1/engines', 'POST',
      {{engines: list, default: defName, persist: true}});
    engDefault = r.default;
    renderEngineSelect(currentEngineIndex());
    document.getElementById('eng-msg').textContent =
      'saved to ' + r.persisted_to;
    engDirty = false;
  }} catch (e) {{ document.getElementById('eng-msg').textContent = String(e); }}
}}

/* ---------------- Auto presets: hardware + model size + gguf-metadata → load_options ---------------- */
function computeAutoPreset(size_gb, hw, file, ggufMeta) {{
  const avail = Number(hw.available_gb ?? hw.vram_gb ?? hw.total_ram_gb ?? 8) || 8;
  const name = (file || '').toLowerCase();
  const meta = ggufMeta || {{}};
  // gguf-metadata: prefer declared parameter count / block_count when available
  const paramsB = Number(meta.parameter_count) ? Number(meta.parameter_count)/1e9
    : Number(meta['general.parameter_count']) ? Number(meta['general.parameter_count'])/1e9
    : null;
  const blockCount = Number(meta.block_count ?? meta['llama.block_count'] ?? meta['qwen.block_count']) || null;
  // Heuristic layers for known families when metadata missing
  let estLayers = blockCount;
  if (!estLayers) {{
    if (name.includes('32b') || name.includes('30b') || (paramsB && paramsB>=30) || size_gb>15) estLayers = 62;
    else if (name.includes('14b') || name.includes('13b') || (paramsB && paramsB>=13)) estLayers = 40;
    else if (name.includes('7b') || name.includes('8b') || (paramsB && paramsB>=7)) estLayers = 32;
    else if (name.includes('4b') || name.includes('3b') || (paramsB && paramsB>=3)) estLayers = 36;
    else if (name.includes('qwen3-4b')) estLayers = 36;
    else if (name.includes('qwen3-32b')) estLayers = 64;
    else estLayers = 32;
  }}
  let gpu_layers, ctx, threads, flash_attn, cache_k, cache_v;
  const low = name;
  const is32 = low.includes('32b') || low.includes('30b') || (paramsB && paramsB>=30) || size_gb>15;
  const is4 = low.includes('4b') || low.includes('3b') || low.includes('qwen3-4b') || (paramsB && paramsB>=3 && paramsB<6) || (size_gb>=2 && size_gb<6);
  if (is32) {{
    // qwen3-32b example: on 8GB → offload 12 + 4k
    if (avail <= 9) {{ gpu_layers = 12; ctx = 4096; }}
    else if (avail <= 16) {{ gpu_layers = 20; ctx = 8192; }}
    else if (avail <= 24) {{ gpu_layers = Math.min(40, estLayers); ctx = 8192; }}
    else {{ gpu_layers = 999; ctx = 16384; }}
  }} else if (is4) {{
    // qwen3-4b example: on 8GB → gpu_layers 28 + 8k ctx
    if (avail <= 9) {{ gpu_layers = 28; ctx = 8192; }}
    else if (avail <= 16) {{ gpu_layers = 999; ctx = 16384; }}
    else {{ gpu_layers = 999; ctx = 32768; }}
  }} else {{
    if (size_gb + 2.0 <= avail * 0.9) {{ gpu_layers = 999; ctx = 8192; }}
    else if (size_gb + 1.0 <= avail * 1.1) {{ gpu_layers = Math.min(28, estLayers); ctx = 8192; }}
    else {{ gpu_layers = Math.min(12, estLayers); ctx = 4096; }}
  }}
  // Clamp to real layer count
  if (gpu_layers !== 999) gpu_layers = Math.min(gpu_layers, estLayers);
  // Other load options tuned with context
  threads = null;
  flash_attn = ctx >= 8192;
  cache_k = null; cache_v = null;
  if (ctx > 16384) {{ cache_k = 'q8_0'; cache_v = 'q8_0'; }}
  return {{ gpu_layers, context: ctx, ctx_size: ctx, threads, flash_attn, cache_type_k: cache_k, cache_type_v: cache_v, block_count: estLayers, gguf_metadata: meta }};
}}
async function engineAuto() {{
  const btn = document.getElementById('eng-auto');
  const msg = document.getElementById('eng-auto-msg');
  const selFile = window._selectedFile || (document.getElementById('launch-model-select') ? document.getElementById('launch-model-select').value : '');
  if (!selFile) {{
    if (msg) msg.textContent = 'select a model first';
    setTimeout(()=>{{ if(msg) msg.textContent=''; }}, 2500);
    return;
  }}
  if (btn) btn.disabled = true;
  if (msg) msg.textContent = 'computing preset…';
  try {{
    // Prefer server-side preset API, fallback to client-side hardware/size_gb/gguf-metadata
    let preset = null;
    try {{
      const data = await api('/v1/engines/preset?file=' + encodeURIComponent(selFile));
      preset = data.load_options || data.preset || data;
      if (preset.gpu_layers == null && preset.gpu_layers !== 0) throw new Error('no preset');
    }} catch (_e) {{
      // Client-side: GET /v1/server/status hardware + GET /v1/models/local size_gb + gguf-metadata
      const [status, local] = await Promise.all([api('/v1/server/status'), api('/v1/models/local')]);
      const hardware = status.hardware || status;
      const entry = (local.models||[]).find(m=>m.file===selFile);
      const size_gb = entry ? Number(entry.size_gb)||0 : 0;
      const gguf_metadata = entry ? (entry.gguf_metadata || entry.metadata || {{}}) : {{}};
      preset = computeAutoPreset(size_gb, hardware, selFile, gguf_metadata);
    }}
    const ctxVal = preset.context ?? preset.ctx_size ?? 8192;
    const nglVal = preset.gpu_layers ?? 999;
    // Update Engine profile load_options and display
    const idx = currentEngineIndex();
    if (engines[idx]) {{
      engines[idx].load_options = {{ ...(engines[idx].load_options||{{}}), context: ctxVal, gpu_layers: nglVal }};
      if (preset.threads) engines[idx].load_options.threads = preset.threads;
      if (preset.flash_attn) engines[idx].load_options.flash_attn = true;
      if (preset.cache_type_k) engines[idx].load_options.cache_type_k = preset.cache_type_k;
      if (preset.cache_type_v) engines[idx].load_options.cache_type_v = preset.cache_type_v;
      show('eng-loadopts', engines[idx].load_options);
    }}
    // Directly set values in the form (Launch panel + Engine fit slider) and then save
    const ctxInput = document.getElementById('ctx');
    const nglInput = document.getElementById('ngl');
    if (ctxInput) ctxInput.value = String(ctxVal);
    if (nglInput) nglInput.value = String(nglVal);
    const fitCtx = document.getElementById('fit-ctx');
    if (fitCtx) {{ fitCtx.value = String(ctxVal); try{{ updateFit(); }}catch(e){{}} }}
    // Optional: reflect other load options in Launch panel
    if (preset.threads != null) {{ const el=document.getElementById('l-threads'); if(el) el.value=preset.threads; }}
    if (preset.flash_attn != null) {{ const el=document.getElementById('l-fa'); if(el) el.checked=!!preset.flash_attn; }}
    if (preset.cache_type_k) {{ const el=document.getElementById('l-ctk'); if(el) el.value=preset.cache_type_k; }}
    if (preset.cache_type_v) {{ const el=document.getElementById('l-ctv'); if(el) el.value=preset.cache_type_v; }}
    engDirty = true;
    if (msg) msg.textContent = `preset: ${{nglVal}} layers / ${{ctxVal}} ctx — saving…`;
    await saveEngines();
    if (msg) msg.textContent = `Auto: ${{nglVal}} layers / ${{ctxVal}} ctx — saved`;
  }} catch (e) {{
    if (msg) msg.textContent = 'auto failed: ' + e.message;
  }} finally {{
    if (btn) btn.disabled = false;
    setTimeout(()=>{{ if(msg && msg.textContent.startsWith('Auto:')) msg.textContent=''; }}, 4000);
  }}
}}

async function exportEngineProfiles() {{
  const msg = document.getElementById('import-export-msg');
  try {{
    const data = {{ engines, exported_at: new Date().toISOString(), version: 1 }};
    const blob = new Blob([JSON.stringify(data, null, 2)], {{type: 'application/json'}});
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'engine-profiles.json';
    a.click();
    URL.revokeObjectURL(url);
    if (msg) msg.textContent = `exported ${{engines.length}} profiles`;
  }} catch (e) {{
    if (msg) msg.textContent = 'export failed: ' + e.message;
  }} finally {{
    setTimeout(()=>{{ if(msg) msg.textContent=''; }}, 3000);
  }}
}}

async function importEngineProfiles(input) {{
  const file = input.files[0];
  if (!file) return;
  const msg = document.getElementById('import-export-msg');
  try {{
    const text = await file.text();
    const data = JSON.parse(text);
    const incoming = data.engines || data.profiles || data;
    if (!Array.isArray(incoming)) throw new Error('invalid bundle: missing engines array');
    for (const eng of incoming) {{
      if (!eng.name) throw new Error('engine missing name');
      const idx = engines.findIndex(e => e.name === eng.name);
      if (idx >= 0) engines[idx] = eng;
      else engines.push(eng);
    }}
    renderEngineSelect(0);
    engineSelected();
    if (msg) msg.textContent = `imported ${{incoming.length}} profiles — save to persist`;
  }} catch (e) {{
    if (msg) msg.textContent = 'import failed: ' + e.message;
  }} finally {{
    input.value = '';
    setTimeout(()=>{{ if(msg) msg.textContent=''; }}, 4000);
  }}
}}

/* ---------------- Fit estimator S2: VRAM needs vs has + context slider ---------------- */
let _fitHardware = null;
let _fitHardwarePromise = null;
let _fitModelsCache = null;
async function getFitHardware() {{
  if (_fitHardware) return _fitHardware;
  if (_fitHardwarePromise) return _fitHardwarePromise;
  _fitHardwarePromise = api('/v1/server/status').then(s => {{
    const hw = s.hardware || s;
    const has = hw.available_gb ?? hw.available_ram_gb ?? hw.total_ram_gb ?? hw.totalRamGb ?? 8;
    const total = hw.total_ram_gb ?? hw.totalRamGb ?? has;
    return {{ available_gb: Number(has) || 8, total_ram_gb: Number(total) || 8, vram_gb: hw.vram_gb ?? null, devices: hw.devices || [] }};
  }}).catch(()=>({{available_gb:8, total_ram_gb:8, vram_gb:null, devices:[]}})).then(h=>{{_fitHardware=h; return h;}});
  return _fitHardwarePromise;
}}
async function getFitModels() {{
  if (_fitModelsCache && Date.now()-_fitModelsCache.ts < 5000) return _fitModelsCache.data;
  const data = await api('/v1/models/local');
  _fitModelsCache = {{data, ts: Date.now()}};
  return data;
}}
function formatCtx(v) {{
  const n = Number(v);
  if (n>=1024) return (n/1024)|0 + 'k';
  return String(n);
}}
async function updateFit() {{
  const selFile = window._selectedFile || (document.getElementById('launch-model-select') ? document.getElementById('launch-model-select').value : '');
  const box = document.getElementById('eng-fit');
  const ctxEl = document.getElementById('fit-ctx');
  if (!box || !ctxEl) return;
  if (!selFile) {{ box.style.display='none'; return; }}
  box.style.display='';
  let size_gb = 0;
  try {{
    const local = await getFitModels();
    const entry = (local.models||[]).find(m=>m.file===selFile);
    if (entry && entry.size_gb!=null) size_gb = Number(entry.size_gb)||0;
  }} catch(e) {{}}
  const ctx = parseInt(ctxEl.value,10)||8192;
  const label = document.getElementById('fit-ctx-label');
  if (label) label.textContent = formatCtx(ctx);
  let hw;
  try {{ hw = await getFitHardware(); }} catch(e) {{ hw = {{available_gb:8}}; }}
  const has_gb = Number(hw.available_gb)||8;
  const kv_gb = (ctx/1024)*0.25;
  const needs_gb = size_gb + kv_gb;
  const ratio = has_gb>0 ? needs_gb/has_gb : 1;
  const needsEl = document.getElementById('fit-needs');
  const hasEl = document.getElementById('fit-has');
  const needsVal = document.getElementById('fit-needs-val');
  const pctEl = document.getElementById('fit-pct');
  const fill = document.getElementById('fit-fill');
  const btn = document.getElementById('eng-load-fit');
  if (needsEl) needsEl.textContent = `needs ${{needs_gb.toFixed(1)}}GB`;
  if (hasEl) hasEl.textContent = `/ ${{has_gb.toFixed(1)}}GB`;
  if (needsVal) needsVal.textContent = `${{needs_gb.toFixed(1)}}GB`;
  if (pctEl) {{
    const fits = needs_gb <= has_gb;
    pctEl.textContent = fits ? (ratio>0.9 ? 'Tight' : 'Fits') : 'Too large';
    pctEl.className = 'fit-pct ' + (fits ? (ratio>0.7 ? 'warn' : 'ok') : 'bad');
  }}
  if (fill) {{
    fill.style.width = Math.min(100, ratio*100).toFixed(1)+'%';
    fill.className = 'fit-fill ' + (ratio>1 ? 'red' : ratio>0.7 ? 'amber' : 'green');
  }}
  if (btn) {{
    const fits = needs_gb <= has_gb;
    btn.disabled = !fits;
    btn.title = fits ? `Fits — load ${{selFile.split('/').pop().split('\\\\').pop()}} at ${{formatCtx(ctx)}}` : `needs ${{needs_gb.toFixed(1)}}GB but you have ${{has_gb.toFixed(1)}}GB — reduce context or free VRAM`;
  }}
}}
function engineLoadFromFit() {{
  const selFile = window._selectedFile || (document.getElementById('launch-model-select') ? document.getElementById('launch-model-select').value : '');
  const ctx = parseInt(document.getElementById('fit-ctx')?.value||'8192',10);
  if (!selFile) return;
  const modelInput = document.getElementById('model');
  const ctxInput = document.getElementById('ctx');
  if (modelInput) modelInput.value = selFile;
  if (ctxInput) ctxInput.value = String(ctx);
  const sel = document.getElementById('launch-model-select');
  if (sel) sel.value = selFile;
  startServer(document.querySelector('button[onclick="startServer(this)"]') || null);
}}
(function initFitHooks(){{
  const ctxEl = document.getElementById('fit-ctx');
  if (ctxEl) ctxEl.addEventListener('input', ()=>{{ updateFit(); }});
  let lastSel = null;
  setInterval(()=>{{
    const cur = window._selectedFile || (document.getElementById('launch-model-select') ? document.getElementById('launch-model-select').value : '');
    if (cur !== lastSel) {{ lastSel = cur; updateFit(); }}
  }}, 400);
  const launchSel = document.getElementById('launch-model-select');
  if (launchSel) launchSel.addEventListener('change', ()=>{{ setTimeout(updateFit, 50); }});
  setTimeout(()=>{{ getFitHardware().then(()=>updateFit()); }}, 600);
}})();

/* ----------------------------- hive tab ------------------------------ */
const HIVE_NUMERIC = [['max_context', 'h-maxctx'], ['max_tokens', 'h-maxtok'],
  ['stale_threshold', 'h-stale'], ['dedup_threshold', 'h-dedup'],
  ['drift_threshold', 'h-drift'], ['remembrance_threshold', 'h-remem'],
  ['vocab_boost', 'h-vocab']];

function hiveToForm(cfg) {{
  for (const [key, id] of HIVE_NUMERIC)
    document.getElementById(id).value =
      (cfg[key] === undefined || cfg[key] === null) ? '' : cfg[key];
  document.getElementById('h-conf').value = cfg.confidence_mode || 'off';
  document.getElementById('h-sanitize').checked = !!cfg.sanitize_context;
  document.getElementById('h-hedge').checked = !!cfg.filter_hedge_replies;
  document.getElementById('h-medium').checked = !!cfg.enable_medium;
  document.getElementById('h-comb').checked = !!cfg.comb_enabled;
  document.getElementById('h-combk').value = cfg.comb_top_k ?? '';
  document.getElementById('h-combgate').value = cfg.comb_gate_threshold ?? '';
  document.getElementById('h-combmax').value = cfg.comb_max_records ?? '';
  document.getElementById('h-combrel').checked = !!cfg.comb_relevant_only;
}}

function collectHiveOverrides() {{
  const out = {{}};
  const defaults = window.__hiveDefaults || {{}};
  const changed = (key, value) => defaults[key] === undefined
    || JSON.stringify(defaults[key]) !== JSON.stringify(value);
  for (const [key, id] of HIVE_NUMERIC) {{
    const v = num(id);
    if (v !== null && changed(key, v)) out[key] = v;
  }}
  const conf = document.getElementById('h-conf').value;
  if (changed('confidence_mode', conf)) out.confidence_mode = conf;
  const checks = [['h-sanitize', 'sanitize_context'],
    ['h-hedge', 'filter_hedge_replies'], ['h-medium', 'enable_medium']];
  for (const [id, key] of checks) {{
    const v = document.getElementById(id).checked;
    if (changed(key, v)) out[key] = v;
  }}
  if (document.getElementById('h-comb').checked) {{
    out.comb_enabled = true;
    const k = num('h-combk'); if (k !== null) out.comb_top_k = k;
    const g = num('h-combgate'); if (g !== null) out.comb_gate_threshold = g;
    const m = num('h-combmax'); if (m !== null) out.comb_max_records = m;
    out.comb_relevant_only = document.getElementById('h-combrel').checked;
    out.comb_dir = 'harness_comb';
  }}
  return out;
}}

async function loadHiveDefaults() {{
  try {{
    const cfg = await api('/v1/hive/defaults');
    window.__hiveDefaults = cfg;
    hiveToForm(cfg);
    hiveOverrides = collectHiveOverrides();
    document.getElementById('hive-msg').textContent =
      Object.keys(hiveOverrides).length + ' override(s) active for new conversations';
  }} catch (e) {{ document.getElementById('hive-msg').textContent = String(e); }}
}}

function resetHiveDefaults() {{
  if (window.__hiveDefaults) hiveToForm(window.__hiveDefaults);
  hiveOverrides = collectHiveOverrides();
  document.getElementById('hive-msg').textContent = 'defaults restored';
}}

for (const key of ['h-maxctx','h-maxtok','h-stale','h-dedup','h-drift',
                   'h-remem','h-vocab','h-conf','h-sanitize','h-hedge',
                   'h-medium','h-comb','h-combk','h-combgate','h-combmax',
                   'h-combrel']) {{
  const el = document.getElementById(key);
  if (el) el.addEventListener('change', () => {{
    hiveOverrides = collectHiveOverrides();
    document.getElementById('hive-msg').textContent =
      Object.keys(hiveOverrides).length + ' override(s) active for new conversations';
  }});
}}

/* --------------------------- providers tab ---------------------------- */
let providers = [];
let provDefault = '';

async function loadProviders() {{
  try {{
    const data = await api('/v1/provider/config');
    providers = data.providers.map(p => ({{...p}}));
    provDefault = data.default || '';
    renderProviders();
  }} catch (e) {{ document.getElementById('prov-msg').textContent = String(e); }}
}}

function renderProviders() {{
  const wrap = document.getElementById('prov-list');
  wrap.innerHTML = '';
  providers.forEach((p, i) => {{
    const row = document.createElement('div');
    row.className = 'row';
    row.style.borderTop = '1px solid #e8edf2';
    row.style.paddingTop = '.4rem';
    const mk = (placeholder, value, size, onChange) => {{
      const inp = document.createElement('input');
      inp.placeholder = placeholder; inp.size = size; inp.value = value || '';
      inp.addEventListener('input', onChange);
      return inp;
    }};
    const name = mk('name', p.name, 12, v => p.name = v);
    const url = mk('base_url', p.base_url, 34, v => p.base_url = v);
    const model = mk('model', p.model, 18, v => p.model = v);
    const key = mk('api_key', p.api_key, 16, v => p.api_key = v);
    const def = document.createElement('input');
    def.type = 'radio'; def.name = 'prov-default'; def.checked =
      p.name.toLowerCase() === provDefault.toLowerCase();
    def.title = 'default provider';
    def.addEventListener('change', () => provDefault = p.name);
    const del = document.createElement('button');
    del.textContent = '✕'; del.title = 'remove';
    del.addEventListener('click', () => {{
      providers.splice(i, 1); renderProviders(); }});
    const lbl = document.createElement('label');
    lbl.className = 'inline'; lbl.title = 'default';
    lbl.appendChild(def);
    row.appendChild(lbl);
    row.appendChild(name); row.appendChild(url); row.appendChild(model);
    row.appendChild(key); row.appendChild(del);
    wrap.appendChild(row);
  }});
  if (!providers.length)
    wrap.innerHTML = '<div class="note">(none — add one)</div>';
}}

function providerAdd() {{
  providers.push({{name: '', base_url: 'http://', api_key: '', model: '',
                  headers: {{}}}});
  renderProviders();
}}

async function saveProviders() {{
  const list = providers
    .filter(p => p.name && p.base_url)
    .map(p => ({{name: p.name, base_url: p.base_url,
                api_key: p.api_key || '', model: p.model || '',
                headers: p.headers || {{}}}}));
  try {{
    const r = await api('/v1/provider/config', 'POST',
      {{providers: list, default: provDefault, persist: true}});
    providers = r.providers.map(p => ({{...p}}));
    provDefault = r.default;
    renderProviders();
    document.getElementById('prov-msg').textContent =
      'saved to ' + (r.persisted_to || 'memory');
  }} catch (e) {{ document.getElementById('prov-msg').textContent = String(e); }}
}}

/* --------------------------- inspector ------------------------------- */
for (const btn of document.querySelectorAll('[data-rtab]')) {{
  btn.addEventListener('click', () => {{
    for (const b of document.querySelectorAll('[data-rtab]')) b.classList.remove('active');
    for (const p of document.querySelectorAll('[data-rtab] + .tabpane, .tabpane[data-rtab]')) {{}}
    document.querySelectorAll('.col .tabpane').forEach(p => p.style.display = 'none');
    btn.classList.add('active');
    document.getElementById(btn.dataset.rtab).style.display = '';
    if (btn.dataset.rtab === 'rtab-inspect') fetchInspection();
  }});
}}

async function fetchInspection() {{
  try {{
    const data = await api('/v1/hive/inspect/' + encodeURIComponent(convId));
    renderInspection(data);
  }} catch (e) {{
    document.getElementById('inspect-summary').textContent =
      'no inspection data: ' + e.message;
  }}
}}

function renderInspection(d) {{
  const kv = (label, v) => `<div class='kv'><span>${{label}}</span><b>${{v}}</b></div>`;
  document.getElementById('inspect-summary').innerHTML =
    kv('Turn', d.turn) + kv('Route', d.routing?.route_to || '-') +
    kv('Budget', `${{d.budget?.used ?? 0}} / ${{d.budget?.total ?? 0}}`) +
    kv('Utilization', `${{((d.budget?.utilization ?? 0) * 100).toFixed(1)}}%`) +
    kv('Top score', d.top_raw_score) +
    kv('Store chunks', d.store_chunks) +
    kv('Drift', d.drift_detected ? '⚠ yes' : 'no');
  const selWrap = document.getElementById('inspect-selected');
  selWrap.innerHTML = '';
  for (const c of d.selected_chunks || []) {{
    selWrap.insertAdjacentHTML('beforeend',
      `<div class='chunkrow sel'><span class='score'>${{c.raw_score}}</span>${{c.id}}` +
      `<div class='preview'>${{c.preview}}</div></div>`);
  }}
  if (!(d.selected_chunks || []).length)
    selWrap.textContent = '(none selected)';
  const dropWrap = document.getElementById('inspect-dropped');
  dropWrap.innerHTML = '';
  for (const c of d.dropped_chunks || []) {{
    dropWrap.insertAdjacentHTML('beforeend',
      `<div class='chunkrow drop'><span class='score'>${{c.raw_score}}</span>${{c.id}}` +
      `<div class='preview'>${{c.preview}}</div></div>`);
  }}
  document.getElementById('inspect-dropped-n').textContent =
    d.dropped_count ? `(${{d.dropped_count}} total)` : '';
  if (!(d.dropped_chunks || []).length)
    dropWrap.textContent = '(none dropped)';
  document.getElementById('inspect-assembled').textContent =
    d.assembled_preview || '(empty)';
  document.getElementById('inspect-timings').textContent =
    JSON.stringify(d.timings ?? {{}}, null, 1);
}}

/* --------------------------- status poll ------------------------------ */
async function refresh() {{
  api('/v1/server/status').then(s => {{
    show('status', s);
    const title = document.getElementById('chat-title');
    if (s.running && s.healthy) {{
      title.textContent = s.instances.length > 1
        ? `${{s.instances.length}} models loaded`
        : 'Loaded: ' + (s.model || 'model');
      title.className = 'ok';
    }} else {{
      title.textContent = s.running ? 'Loading…' : 'No model loaded';
      title.className = 'bad';
    }}
    // Loaded-instance cards: per-model unload, port shown.
    const wrap = document.getElementById('instances');
    wrap.innerHTML = '';
    for (const inst of s.instances || []) {{
      const row = document.createElement('div');
      row.className = 'librow' + (inst.healthy ? ' loaded' : '');
      const label = document.createElement('span');
      label.textContent = `${{inst.key}} — port ${{inst.port}}`
        + `${{inst.adopted ? ' (adopted)' : ''}}`;
      const unload = document.createElement('button');
      unload.textContent = 'unload';
      unload.title = 'stop this model server';
      unload.addEventListener('click', async () => {{
        if (!confirm('Unload ' + inst.key + '?')) return;
        try {{
          await api('/v1/server/unload', 'POST', {{key: inst.key}});
          refresh();
        }} catch (e) {{ alert(String(e)); }}
      }});
      row.appendChild(label);
      row.appendChild(unload);
      wrap.appendChild(row);
    }}
    // Chat model picker: every loaded instance's provider + remote providers.
    api('/v1/provider/config').then(pc => {{
      const sel = document.getElementById('chat-provider');
      const current = sel.value;
      sel.innerHTML = '';
      const locals = (s.instances || []).map(i => 'local-' + i.key);
      for (const p of pc.providers) {{
        if (!locals.includes(p.name) && p.name.startsWith('local-')) continue;
        const o = document.createElement('option');
        o.value = p.name;
        o.textContent = p.name + (p.model ? ' — ' + p.model.split('\\\\').pop().split('/').pop() : '');
        sel.appendChild(o);
      }}
      sel.value = pc.default && locals.concat(
        pc.providers.filter(p => !p.name.startsWith('local-')).map(p => p.name)
      ).includes(pc.default) ? pc.default : (sel.firstElementChild ? sel.firstElementChild.value : '');
      if (current && [...sel.options].some(o => o.value === current)) sel.value = current;
    }}).catch(() => {{}});
  }}).catch(e => show('status', String(e)));
  api('/v1/server/log?tail=30').then(l => {{
    if (l.lines.length) show('srvlog', l.lines.join('\\n'));
  }}).catch(() => {{}});
  api('/v1/models/local').then(l => {{
    console.log('local fetch ok', l.models_dir, l.models?.length);
    const pathInput = document.getElementById('library-path');
    const pathNote = document.getElementById('library-path-note');
    if (l.models_dir) {{
      if (!pathInput.value) pathInput.value = l.models_dir;
      pathNote.textContent = `(${{l.models_dir}})`;
    }}
    const wrap = document.getElementById('local');
    const engDisplay = document.getElementById('eng-model-display');
    const launchSelect = document.getElementById('launch-model-select');
    const engNote = document.getElementById('eng-model-note');
    if (launchSelect) launchSelect.innerHTML = '<option value="">— choose local model —</option>';
    const engModelSel = document.getElementById('eng-model');
    if (engModelSel) engModelSel.innerHTML = '<option value="">— choose local model —</option>';
    for (const m of l.models) {{
      const shortName = m.file.split('/').pop().split('\\\\').pop();
      if (launchSelect) {{
        const o2 = document.createElement('option');
        o2.value = m.file;
        o2.textContent = shortName;
        if (engModelSel) {{ const oE = document.createElement('option'); oE.value = m.file; oE.textContent = shortName; engModelSel.appendChild(oE); }}
        launchSelect.appendChild(o2);
      }}
    }}
    const prevLaunch = launchSelect ? launchSelect.value : '';
    if (launchSelect) {{
      if (prevLaunch && l.models.some(m => m.file === prevLaunch)) launchSelect.value = prevLaunch;
      else if (l.models.length === 1) launchSelect.value = l.models[0].file;
    }}
    const updateEngModel = () => {{
      const v = window._selectedFile || (launchSelect ? launchSelect.value : '');
      const short = v ? v.split('/').pop().split('\\\\').pop() : '—';
      if (engDisplay) engDisplay.textContent = short;
      if (engNote) engNote.textContent = v ? `selected: ${{short}}` : '';
      // Keep launchSelect in sync with _selectedFile
      if (launchSelect && v && launchSelect.value !== v) {{
        // Ensure option exists even if filtered
        let opt = [...launchSelect.options].find(o=>o.value===v);
        if (!opt) {{
          opt = document.createElement('option'); opt.value=v; opt.textContent=short; launchSelect.appendChild(opt);
        }}
        launchSelect.value = v;
      }}
      for (const row of wrap.children) {{
        const isSel = row.dataset.file === v;
        row.classList.toggle('selected', isSel);
        row.style.background = isSel ? '#000000' : '';
        row.style.color = isSel ? '#FFDD00' : '';
        row.style.borderColor = isSel ? '#FFDD00' : '';
        row.style.boxShadow = isSel ? '0 0 10px rgba(255,221,0,0.5)' : '';
      }}
      const launchModel = document.getElementById('model');
      if (launchModel && v) launchModel.value = v;
      console.log('updateEngModel v', v, 'selected', v ? 'yes' : 'none');
      try {{ if (typeof updateFit==='function') updateFit(); }} catch(e){{}}
    }};
    if (launchSelect) launchSelect.onchange = () => {{ window._selectedFile = launchSelect.value || null; updateEngModel(); try{{ if(typeof updateFit==='function') updateFit(); }}catch(e){{}} }};
    wrap.innerHTML = '';
    if (!l.models.length) {{
      wrap.textContent = '(no .gguf files yet)';
      if (engNote) engNote.textContent = '';
      return;
    }}
    for (const m of l.models) {{
      const row = document.createElement('div');
      row.className = 'librow';
      row.dataset.file = m.file;
      const label = document.createElement('span');
      const short = m.file.split('/').pop().split('\\\\').pop();
      label.textContent = `${{short}} — ${{m.size_gb}} GB`;
      const del = document.createElement('button');
      del.textContent = 'delete';
      del.title = 'remove from disk';
      del.addEventListener('click', async (e) => {{
        e.stopPropagation();
        if (!confirm('Delete ' + m.file + ' from disk?')) return;
        try {{
          await api('/v1/models/local?file=' + encodeURIComponent(m.file),
                    'DELETE');
          refresh();
        }} catch (e) {{ alert(String(e)); }}
      }});
      // HoverCard: delayed hover preview with gguf-metadata + sizeGb + lastModified inline
      let hoverTimer = null;
      let hoverCard = null;
      const showHover = () => {{
        if (hoverCard) return;
        const arch = m.architecture || m.gguf_metadata?.['general.architecture'] || m.ggufMetadata?.['general.architecture'] || '—';
        const quant = m.quantization || m.gguf_metadata?.quantization || m.ggufMetadata?.quantization || '—';
        const ctx = m.context_length ?? m.contextLength ?? m.gguf_metadata?.context_length ?? m.ggufMetadata?.context_length ?? (() => {{
          const gm = m.gguf_metadata || m.ggufMetadata || {{}};
          for (const k in gm) {{ if (k.endsWith('.context_length')) return gm[k]; }}
          return '—';
        }})();
        const size = m.size_gb != null ? `${{m.size_gb}} GB` : (m.sizeGb != null ? `${{m.sizeGb}} GB` : '—');
        const mod = m.modified || m.lastModified || '—';
        hoverCard = document.createElement('div');
        hoverCard.className = 'hovercard';
        hoverCard.setAttribute('role', 'tooltip');
        hoverCard.textContent = `${{arch}} · ${{quant}} · ${{ctx}} · ${{size}} · ${{mod}}`;
        document.body.appendChild(hoverCard);
        const rect = row.getBoundingClientRect();
        const cardRect = hoverCard.getBoundingClientRect();
        let left = rect.right + 8;
        let top = rect.top;
        if (top + cardRect.height > window.innerHeight - 8) top = window.innerHeight - cardRect.height - 8;
        if (left + cardRect.width > window.innerWidth - 8) left = rect.left - cardRect.width - 8;
        if (left < 8) left = 8;
        if (top < 8) top = 8;
        hoverCard.style.left = left + 'px';
        hoverCard.style.top = top + 'px';
      }};
      const hideHover = () => {{
        if (hoverTimer) {{ clearTimeout(hoverTimer); hoverTimer = null; }}
        if (hoverCard) {{ hoverCard.remove(); hoverCard = null; }}
      }};
      // Click row = select model (also long-press visual)
      let pressTimer = null;
      const setPressed = (on) => row.classList.toggle('pressed', on);
      row.addEventListener('mousedown', () => {{ pressTimer = setTimeout(()=>setPressed(true), 400); }});
      row.addEventListener('mouseup', () => {{ clearTimeout(pressTimer); setPressed(false); }});
      row.addEventListener('mouseleave', () => {{ clearTimeout(pressTimer); setPressed(false); hideHover(); }});
      row.addEventListener('mouseenter', () => {{ hoverTimer = setTimeout(showHover, 1000); }});
      row.addEventListener('touchstart', () => {{ pressTimer = setTimeout(()=>setPressed(true), 400); }}, {{passive:true}});
      row.addEventListener('touchend', () => {{ clearTimeout(pressTimer); setPressed(false); }});
      row.addEventListener('click', (e) => {{
        hideHover();
        e.stopPropagation();
        console.log('row click', m.file);
        if (launchSelect) {{
          launchSelect.value = m.file;
          // Ensure the option is visible even if filtered
          for (const opt of launchSelect.options) {{ if (opt.value === m.file) {{ opt.hidden = false; opt.style.display = ''; }} }}
        }}
        if (window._selectedFile !== m.file) window._selectedFile = m.file;
        else window._selectedFile = null; // toggle off if same file clicked again? No, keep selected
        // Actually, toggle behaviour: if already selected, keep it selected (don't unselect)
        // So we set _selectedFile to m.file and call update
        window._selectedFile = m.file;
        updateEngModel();
      }});
      row.appendChild(label);
      row.appendChild(del);
      wrap.appendChild(row);
    }}
    // Restore selection from _selectedFile if exists
    if (window._selectedFile && launchSelect) launchSelect.value = window._selectedFile;
    updateEngModel();
  }}).catch(e => {{ console.error('local fetch failed', e); const w=document.getElementById('local'); if(w) w.textContent = 'load failed: ' + String(e).slice(0,200); }});
  api('/v1/models/hub/downloads').then(d => {{
    const lines = d.downloads.map(j => `${{j.filename}}: ${{j.state}} (${{j.elapsed_s}}s)`);
    if (lines.length) show('downloads', lines.join('\\n'));
  }}).catch(() => {{}});
}}

refresh();
setInterval(refresh, 15000);
loadHiveDefaults();
document.getElementById('chatin').addEventListener('blur',
  () => setTimeout(() => {{ document.getElementById('sug-chat').innerHTML = ''; }}, 150));
document.getElementById('chatin').focus();
</script>
</body></html>"""
