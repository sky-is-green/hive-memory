"""Server-rendered report views (Seam B) â€” plain HTML, no JS dependencies.

Renders one ``run_report.json`` bundle: the post-run PES headline with its
weighted components, the P1â€“P10 verdict table, the deterministic P2 retrieval
diagnostic, comb (P11) totals, and the baselines comparison. Every dynamic
value is HTML-escaped; missing blocks render as em dashes rather than erroring,
so partial bundles from in-flight runs stay viewable.
"""

from __future__ import annotations

import html
from pathlib import Path

from fastapi import HTTPException

# Paper weights for the PES composite (README Â§3.1).
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
    # and pes/breakdown â€” accept both.
    composite = post.get("composite", post.get("pes"))
    components = post.get("components")
    if not isinstance(components, dict):
        components = post.get("breakdown")
    if not isinstance(components, dict):
        components = {}
    band = str(post.get("band") or "")

    def _band_class() -> str:
        if band:
            cls = _BAND_FLOORS[0][1]
            for token in ("green", "yellow", "red", "critical"):
                if token in band.lower():
                    return f"band-{token}"
            return cls
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
        f"{'' if e['has_report'] else ' Â· no run_report.json yet'}</span></li>"
        for e in entries
    ) or "<li class='note'>no runs yet</li>"
    return f"""<!doctype html>
<html><head><meta charset="utf-8"><title>HiveBench runs</title>
<style>{_CSS}</style></head><body>
<h1>HiveBench run bundles</h1>
<ul class="runs">{items}</ul>
</body></html>"""


_SERVER_CSS = _CSS + """
 /* console overrides: full-bleed three-pane layout */
 body { max-width: none; margin: 0; padding: 1.2rem 1.6rem; }
 h1 { margin: 0 0 1rem 0; }
 section { background: #fff; border-radius: 8px; padding: 1rem 1.4rem;
          margin: 0 0 1rem 0; box-shadow: 0 1px 3px rgba(16,32,48,.08); }
 .grid { display: grid;
         grid-template-columns: minmax(320px, 1fr) minmax(420px, 1.3fr)
                                minmax(340px, 1fr);
         gap: 1rem; align-items: start; }
 @media (max-width: 1150px) { .grid { grid-template-columns: 1fr; } }
 .col { min-width: 0; }
 input, button { font: inherit; padding: .35rem .6rem; margin: .15rem .3rem .15rem 0; }
 button { cursor: pointer; }
 pre { background: #eef2f6; padding: .7rem; border-radius: 6px;
       white-space: pre-wrap; word-break: break-word; font-size: .82rem;
       max-height: 340px; overflow-y: auto; }
 .row { margin: .4rem 0; } .ok { color: #157a3e; font-weight: 600; }
 .bad { color: #b3372c; font-weight: 600; }
 .sugwrap { position: relative; display: inline-block; }
 .sugbox { position: absolute; z-index: 10; left: 0; right: 0; top: 100%;
           background: #fff; border: 1px solid #cfd8e0; border-radius: 6px;
           box-shadow: 0 4px 14px rgba(16,32,48,.15); max-height: 260px;
           overflow-y: auto; margin-top: 2px; }
 .sugbox:empty { display: none; }
 .sugitem { padding: .45rem .6rem; cursor: pointer; font-size: .9rem;
            display: flex; justify-content: space-between; gap: 1rem; }
 .sugitem:hover, .sugitem.active { background: #eef4fb; }
 .sugitem .meta { color: #5a6b7d; white-space: nowrap; font-size: .8rem; }
 /* chat pane */
 .chatlog { height: 480px; overflow-y: auto; background: #f2f6fa;
            border: 1px solid #e3eaf1; border-radius: 8px; padding: .8rem;
            display: flex; flex-direction: column; gap: .5rem; }
 .msg { max-width: 85%; padding: .5rem .75rem; border-radius: 10px;
        font-size: .92rem; white-space: pre-wrap; word-break: break-word; }
 .msg.user { align-self: flex-end; background: #1f6feb; color: #fff; }
 .msg.ai { align-self: flex-start; background: #fff; border: 1px solid #dde5ec; }
 .msg .meta { display: block; font-size: .72rem; color: #8a99a8;
              margin-top: .3rem; }
 .msg.user .meta { color: rgba(255,255,255,.75); }
 #chatin { flex: 1; }
 /* settings tabs */
 .tabs { display: flex; gap: .25rem; margin-bottom: .6rem; }
 .tab { border: 1px solid #cfd8e0; background: #eef2f6; border-radius: 6px 6px 0 0; }
 .tab.active { background: #fff; font-weight: 600; border-bottom-color: #fff; }
 .tabpane { animation: fadein .15s ease; }
 @keyframes fadein { from { opacity: .4; } to { opacity: 1; } }
 label.inline { white-space: nowrap; color: #5a6b7d; font-size: .85rem; }
 label.inline input[type="number"], label.inline input[type="text"],
 label.inline select { margin-left: .15rem; }
 label.inline input[type="checkbox"] { vertical-align: middle; }
 details { margin: .3rem 0; }
 summary { cursor: pointer; color: #456; font-size: .88rem; }
 #prov-list input { margin: .12rem .15rem; }
"""


def render_server_page() -> str:
    """Model-management console: server + settings | loaded-model chat | hub.

    Discovery is live against the Hugging Face hub API â€” no model catalog is
    hardcoded here."""
    return f"""<!doctype html>
<html><head><meta charset="utf-8"><title>Studio server &amp; models</title>
<style>{_SERVER_CSS}</style></head><body>
<h1>HiveBench Studio console</h1>

<div class="grid">

<!-- ==================== LEFT: tabs (server/engines/hive/providers) === -->
<div class="col">
<div class="tabs">
<button class="tab active" data-tab="tab-server">Server</button>
<button class="tab" data-tab="tab-engines">Engines</button>
<button class="tab" data-tab="tab-hive">Hive</button>
<button class="tab" data-tab="tab-providers">Providers</button>
</div>

<div id="tab-server" class="tabpane">
<section>
<h2 style="margin-top:0">Launch</h2>
<div class="row"><button onclick="api('/v1/server/status').then(s => show('status', s))">Refresh</button>
<button onclick="api('/v1/server/stop', 'POST').then(() => refresh())">Stop</button></div>
<div class="row">
<span class="sugwrap"><input id="model" placeholder="local model (blank = ok w/ hf)" size="26" list="local-suggestions">
<datalist id="local-suggestions"></datalist></span>
<label class="inline">ctx <input id="ctx" type="number" value="8192" size="4"></label>
<label class="inline">gpu <input id="ngl" type="number" value="999" size="3"></label><br>
<span class="sugwrap"><input id="hfrepo" placeholder="--hf-repo (type to search)" size="30"
       list="repo-suggestions">
<div class="sugbox" id="sug-hfrepo"></div></span>
<input id="hffile" placeholder="--hf-file" size="18">
<button onclick="startServer()">Start</button></div>
<details><summary>Advanced launch flags</summary>
<div class="row">
<label class="inline">threads <input id="l-threads" type="number" size="3" placeholder="auto"></label>
<label class="inline"><input id="l-fa" type="checkbox"> flash-attn</label>
<label class="inline">parallel <input id="l-parallel" type="number" size="2" placeholder="1"></label>
<label class="inline"><input id="l-mlock" type="checkbox"> mlock</label>
<label class="inline"><input id="l-nommap" type="checkbox"> no-mmap</label>
</div>
<div class="row">
<label class="inline">kv-K <select id="l-ctk"><option value="">f16</option><option>q8_0</option><option>q4_0</option></select></label>
<label class="inline">kv-V <select id="l-ctv"><option value="">f16</option><option>q8_0</option><option>q4_0</option></select></label>
<label class="inline">batch <input id="l-batch" type="number" size="4" placeholder="512"></label>
<label class="inline">ubatch <input id="l-ubatch" type="number" size="4" placeholder="512"></label>
<label class="inline">alias <input id="l-alias" size="14" placeholder="model id"></label>
</div>
</details>
<pre id="status">loadingâ€¦</pre>
</section>

<section>
<h2 style="margin-top:0">Local library <span class="note">(models/gguf)</span></h2>
<pre id="local">loadingâ€¦</pre>
</section>
</div>

<div id="tab-engines" class="tabpane" style="display:none">
<section>
<h2 style="margin-top:0">Engine profiles</h2>
<div class="row">
<select id="eng-select" style="min-width:180px" onchange="engineSelected()"></select>
<label class="inline">default <input id="eng-default" type="checkbox" onchange="engDirty=true"></label>
<button onclick="engineAdd()">+ Add</button>
</div>
<div class="row">
<label class="inline">name <input id="eng-name" size="16" oninput="engDirty=true"></label>
<label class="inline">kind <select id="eng-kind" onchange="engDirty=true">
<option>llama_cpp</option><option>lmstudio</option><option>vllm</option>
<option>ollama</option><option>hosted</option></select></label>
</div>
<div class="row"><input id="eng-url" placeholder="base_url" style="width:95%" oninput="engDirty=true"></div>
<details open><summary>Sampling defaults (every request)</summary>
<div class="row">
<label class="inline">temp <input id="s-temp" type="number" step="0.05" min="0" max="2" size="4" oninput="engDirty=true"></label>
<label class="inline">top_p <input id="s-topp" type="number" step="0.05" min="0" max="1" size="4" oninput="engDirty=true"></label>
<label class="inline">top_k <input id="s-topk" type="number" size="4" oninput="engDirty=true"></label>
<label class="inline">min_p <input id="s-minp" type="number" step="0.01" size="4" oninput="engDirty=true"></label>
</div>
<div class="row">
<label class="inline">repeat <input id="s-rep" type="number" step="0.05" size="4" oninput="engDirty=true"></label>
<label class="inline">presence <input id="s-pres" type="number" step="0.1" size="3" oninput="engDirty=true"></label>
<label class="inline">freq <input id="s-freq" type="number" step="0.1" size="3" oninput="engDirty=true"></label>
<label class="inline">seed <input id="s-seed" type="number" size="7" oninput="engDirty=true"></label>
</div>
<div class="row">
<label class="inline">mirostat <input id="s-miro" type="number" min="0" max="2" size="2" oninput="engDirty=true"></label>
<label class="inline">tau <input id="s-tau" type="number" step="0.1" size="4" oninput="engDirty=true"></label>
<label class="inline">eta <input id="s-eta" type="number" step="0.01" size="4" oninput="engDirty=true"></label>
<label class="inline">stop <input id="s-stop" size="12" placeholder="a,b" oninput="engDirty=true"></label>
</div>
</details>
<details><summary>Load options (advisory record)</summary>
<pre id="eng-loadopts"></pre>
</details>
<div class="row"><span id="eng-msg" class="note"></span>
<button onclick="saveEngines()" id="eng-save">Save engines</button></div>
</section>
</div>

<div id="tab-hive" class="tabpane" style="display:none">
<section>
<h2 style="margin-top:0">Hive tuning <span class="note">(new conversations)</span></h2>
<div class="note">Applied when a conversation is created â€” hit
"New conversation" in the chat pane after changing.</div>
<div class="row">
<label class="inline">max_context <input id="h-maxctx" type="number" size="6"></label>
<label class="inline">max_tokens <input id="h-maxtok" type="number" size="5" placeholder="4096 ceiling"></label>
</div>
<div class="row">
<label class="inline">stale wall <input id="h-stale" type="number" size="3"></label>
<label class="inline">dedup &ge; <input id="h-dedup" type="number" step="0.01" size="4"></label>
<label class="inline">drift &ge; <input id="h-drift" type="number" step="0.05" size="4"></label>
<label class="inline">remem &ge; <input id="h-remem" type="number" step="0.05" size="4"></label>
</div>
<div class="row">
<label class="inline">vocab boost <input id="h-vocab" type="number" step="0.05" size="4"></label>
<label class="inline">confidence <select id="h-conf">
<option>off</option><option>single</option><option>mcdropout</option></select></label>
</div>
<div class="row">
<label class="inline"><input id="h-sanitize" type="checkbox"> sanitize context</label>
<label class="inline"><input id="h-hedge" type="checkbox"> filter hedge replies</label>
<label class="inline"><input id="h-medium" type="checkbox"> medium drone</label>
</div>
<details><summary>Comb (P11 surplus tier)</summary>
<div class="row">
<label class="inline"><input id="h-comb" type="checkbox"> enabled</label>
<label class="inline">top_k <input id="h-combk" type="number" size="3"></label>
<label class="inline">gate &ge; <input id="h-combgate" type="number" step="0.05" size="4"></label>
<label class="inline">max records <input id="h-combmax" type="number" size="5"></label>
<label class="inline"><input id="h-combrel" type="checkbox"> curated-only</label>
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
<div class="note">Keys echo as *** â€” leave untouched rows as-is to keep the
stored secret; type a new key to replace it. Saved to
providers.local.json (gitignored).</div>
</section>
</div>
</div>

<!-- ==================== MIDDLE: loaded AI chat ==================== -->
<div class="col">
<section style="display:flex; flex-direction:column;">
<div class="row" style="display:flex; justify-content:space-between; align-items:center;">
<h2 style="margin:0" id="chat-title">Loaded model</h2>
<button onclick="newConversation()">New conversation</button></div>
<div id="chatlog" class="chatlog"></div>
<div class="row" style="display:flex; gap:.4rem;">
<input id="chatin" placeholder="Talk to the loaded AIâ€¦" style="flex:1;"
       onkeydown="if (event.key === 'Enter') sendChat()">
<button onclick="sendChat()">Send</button></div>
<div class="note">Every message runs through the hive: context curation,
store, decay â€” the same pipeline the benchmarks measure. Hive-tab settings
apply to new conversations.</div>
</section>
</div>

<!-- ==================== FAR RIGHT: hub ==================== -->
<div class="col">
<section>
<h2 style="margin-top:0">Hugging Face hub <span class="note">(live)</span></h2>
<div class="row"><span class="sugwrap" style="width:100%"><input id="q" placeholder="search gguf reposâ€¦" style="width:100%"
       list="repo-suggestions">
<div class="sugbox" id="sug-q"></div></span></div>
<datalist id="repo-suggestions"></datalist>
<pre id="hub">(search above)</pre>
<div class="row"><span class="sugwrap" style="width:100%"><input id="drepo" placeholder="repo id (type for suggestions)" style="width:100%"
       list="repo-suggestions">
<div class="sugbox" id="sug-drepo"></div></span><br>
<input id="dfile" placeholder="file.gguf" size="24">
<button onclick="download()">Download</button></div>
<pre id="downloads"></pre>
</section>
</div>

</div>

<script>
let convId = localStorage.getItem('hive-console-conv');
if (!convId) {{
  convId = 'console-' + crypto.randomUUID().slice(0, 8);
  localStorage.setItem('hive-console-conv', convId);
}}
let hiveOverrides = {{}};
let engDirty = false;

async function api(path, method, body) {{
  const r = await fetch(path, {{method: method || 'GET',
    headers: {{'content-type': 'application/json'}},
    body: body === undefined ? (method === 'POST' ? '{{}}' : undefined)
                             : JSON.stringify(body)}});
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
    meta.textContent = 'â†“' + r.downloads;
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
  const lines = res.files.map(f => `${{f.file}} â€” ${{f.size_gb}} GB`);
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
}}

async function sendChat() {{
  const input = document.getElementById('chatin');
  const query = input.value.trim();
  if (!query) return;
  input.value = '';
  bubble('user', query);
  bubble('ai', 'â€¦');
  const body = {{query: query, conversation_id: convId}};
  if (Object.keys(hiveOverrides).length) body.config = hiveOverrides;
  try {{
    const r = await api('/v1/hive/turn', 'POST', body);
    const log = document.getElementById('chatlog');
    log.lastChild.remove();
    const who = r.mode === 'error' ? 'error' : 'hive-curated';
    bubble('ai', r.reply || '(empty reply)',
           `${{who}} Â· turn ${{r.turn}} Â· ${{r.token_count}}/${{r.budget}} tokens Â· pes ${{r.pes}}`
           + (r.error ? ' Â· ' + r.error : ''));
  }} catch (e) {{
    const log = document.getElementById('chatlog');
    log.lastChild.remove();
    bubble('ai', 'request failed: ' + e.message);
  }}
}}

async function newConversation() {{
  try {{ await api('/v1/hive/reset', 'POST', {{conversation_id: convId}}); }} catch (e) {{}}
  document.getElementById('chatlog').innerHTML = '';
  bubble('ai', 'Fresh conversation â€” the store was reset.'
    + (Object.keys(hiveOverrides).length
       ? ' Hive overrides apply from the next message.' : ''), null);
}}

/* --------------------------- server panel ---------------------------- */
function launchBody() {{
  return {{
    model: val('model') || null,
    hf_repo: val('hfrepo') || null,
    hf_file: val('hffile') || null,
    ctx_size: +val('ctx') || 8192,
    ngl: +val('ngl') || 999,
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

async function startServer() {{
  const r = await fetch('/v1/server/start', {{method: 'POST',
    headers: {{'content-type': 'application/json'}},
    body: JSON.stringify(launchBody())}});
  show('status', await r.text());
  refresh();
}}

async function searchHub() {{
  show('hub', 'searchingâ€¦');
  try {{
    const res = await api('/v1/models/hub?q=' +
      encodeURIComponent(val('q')) + '&limit=12');
    show('hub', res.results.map(r =>
      `${{r.repo}} Â· â†“${{r.downloads}} Â· â˜…${{r.likes}} Â· ${{r.last_modified}}`).join('\\n'));
  }} catch (e) {{ show('hub', String(e)); }}
}}

async function download() {{
  const r = await fetch('/v1/models/hub/download', {{method: 'POST',
    headers: {{'content-type': 'application/json'}},
    body: JSON.stringify({{repo: val('drepo'), file: val('dfile')}})}});
  show('downloads', await r.text());
}}

/* --------------------------- engines tab ----------------------------- */
let engines = [];
let engDefault = '';

function samplingToForm(s) {{
  const set = (id, v) => document.getElementById(id).value = (v === undefined || v === null) ? '' : v;
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
}}

function collectEngines() {{
  const i = currentEngineIndex();
  const e = engines[i];
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
    renderEngineSelect(0);
    engineSelected();
  }} catch (e) {{ show('eng-loadopts', String(e)); }}
}}

async function saveEngines() {{
  const list = collectEngines();
  const defName = document.getElementById('eng-default').checked
    ? (val('eng-name') || engines[currentEngineIndex()]?.name || '') : engDefault;
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

/* ----------------------------- hive tab ------------------------------ */
const HIVE_FIELDS = ['max_context', 'max_tokens', 'stale_threshold',
  'dedup_threshold', 'drift_threshold', 'remembrance_threshold',
  'vocab_boost'];

function hiveToForm(cfg) {{
  const set = (id, v) => document.getElementById(id).value =
    (v === undefined || v === null) ? '' : v;
  set('h-maxctx', cfg.max_context); set('h-maxtok', cfg.max_tokens);
  set('h-stale', cfg.stale_threshold); set('h-dedup', cfg.dedup_threshold);
  set('h-drift', cfg.drift_threshold); set('h-remem', cfg.remembrance_threshold);
  set('h-vocab', cfg.vocab_boost);
  document.getElementById('h-conf').value = cfg.confidence_mode || 'off';
  document.getElementById('h-sanitize').checked = !!cfg.sanitize_context;
  document.getElementById('h-hedge').checked = !!cfg.filter_hedge_replies;
  document.getElementById('h-medium').checked = !!cfg.enable_medium;
  document.getElementById('h-comb').checked = !!cfg.comb_enabled;
  set('h-combk', cfg.comb_top_k); set('h-combgate', cfg.comb_gate_threshold);
  set('h-combmax', cfg.comb_max_records);
  document.getElementById('h-combrel').checked = !!cfg.comb_relevant_only;
}}

function collectHiveOverrides() {{
  const out = {{}};
  const defaults = window.__hiveDefaults || {{}};
  const changed = (key, value) => defaults[key] === undefined
    || JSON.stringify(defaults[key]) !== JSON.stringify(value);
  for (const key of HIVE_FIELDS) {{
    const v = num('h-' + (key === 'max_context' ? 'maxctx'
      : key === 'max_tokens' ? 'maxtok' : key));
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
    del.textContent = 'âœ•'; del.title = 'remove';
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
  if (!providers.length) wrap.innerHTML = '<div class="note">(none â€” add one)</div>';
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

/* --------------------------- status poll ------------------------------ */
async function refresh() {{
  api('/v1/server/status').then(s => {{
    show('status', s);
    const title = document.getElementById('chat-title');
    if (s.running && s.healthy) {{
      title.textContent = 'Loaded: ' + (s.model || 'model');
      title.className = 'ok';
    }} else {{
      title.textContent = s.running ? 'Loadingâ€¦' : 'No model loaded';
      title.className = 'bad';
    }}
  }}).catch(e => show('status', String(e)));
  api('/v1/models/local').then(l => show('local',
    l.models.length ? l.models.map(m => `${{m.file}} â€” ${{m.size_gb}} GB`).join('\\n')
                    : '(no .gguf files yet)')).catch(e => show('local', String(e)));
  api('/v1/models/hub/downloads').then(d => {{
    const lines = d.downloads.map(j => `${{j.filename}}: ${{j.state}} (${{j.elapsed_s}}s)`);
    if (lines.length) show('downloads', lines.join('\\n'));
  }}).catch(() => {{}});
}}

refresh();
setInterval(refresh, 15000);
loadHiveDefaults();
document.getElementById('chatin').focus();
</script>
</body></html>"""
