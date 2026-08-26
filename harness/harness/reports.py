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
<ul class="runs">{items}</ul>
</body></html>"""


_SERVER_CSS = _CSS + """
 /* console overrides: full-bleed three-pane layout */
 body { max-width: none; margin: 0; padding: .8rem 1.2rem;
        height: 100vh; box-sizing: border-box;
        display: flex; flex-direction: column; overflow: hidden; }
 h1 { margin: 0 0 .7rem 0; flex: none; }
 section { background: #fff; border-radius: 8px; padding: 1rem 1.4rem;
           margin: 0 0 1rem 0; box-shadow: 0 1px 3px rgba(16,32,48,.08); }
 /* three panes in fixed proportion — sides 1 part, chat 1.4 parts.
     Pure fr tracks: every pane keeps its share of the row at ANY
     viewport width, and each pane scrolls internally instead of
     forcing the others wider or narrower. */
 .grid { display: grid;
         grid-template-columns: 1fr 1.4fr 1fr;
         gap: 1rem; align-items: stretch;
         flex: 1 1 auto; min-height: 0; }
 @media (max-width: 900px) {
    /* stacked layout: the page scrolls again and the chat pane keeps the
       lion's share of the height instead of being squeezed into a sliver */
    body { overflow-y: auto; }
    .grid { display: flex; flex-direction: column; }
    .col { overflow: visible; }
    .mid { min-height: 75vh; overflow: visible; }
    .chatpane .chatlog { min-height: 50vh; } }
 .col { min-width: 0; min-height: 0; overflow-y: auto; padding-bottom: 1rem; }
 /* the chat column never scrolls itself: its pane fills the exact space */
 .mid { overflow: hidden; display: flex; flex-direction: column;
        padding-bottom: 0; }
 .mid .chatpane { margin-bottom: 0; }
 input, button, select { font: inherit; padding: .35rem .6rem;
        margin: .15rem .3rem .15rem 0; }
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
 @keyframes hiveblink { 0%,100% { opacity: .15; } 50% { opacity: 1; } }
 .msg.ai.typing::after { content: '···'; letter-spacing: .18rem;
        animation: hiveblink 1.1s infinite; }
 .msg.sys { align-self: center; background: #eef2f6; color: #5a6b7d;
        font-size: .8rem; padding: .25rem .7rem; border-radius: 999px; }
 #chatin { flex: 1; }
 #stopbtn { background: #b3372c; color: #fff; border: none; }
 #afkbtn { min-width: 5.5rem; }
 #afkbtn.afk-on { background: #b3372c; color: #fff; border-color: #b3372c;
        animation: hiveblink 2.4s infinite; }
 /* inspector */
 .inspector-list { max-height: 300px; overflow-y: auto; }
 .chunkrow { background: #fff; border: 1px solid #e3eaf1; border-radius: 6px;
        padding: .45rem .7rem; margin: .3rem 0; font-size: .85rem; }
 .chunkrow.sel { border-left: 3px solid #157a3e; }
 .chunkrow.drop { border-left: 3px solid #d4d9de; opacity: .7; }
 .chunkrow .score { float: right; font-weight: 600; font-variant-numeric: tabular-nums; }
 .chunkrow .preview { color: #456; margin-top: .2rem; word-break: break-word; }
 /* settings tabs */
 .tabs { display: flex; gap: .25rem; margin-bottom: .6rem; }
 .tab { border: 1px solid #cfd8e0; background: #eef2f6;
        border-radius: 6px 6px 0 0; }
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
 .liblist .librow { display: flex; justify-content: space-between;
        align-items: center; gap: .6rem; background: #eef2f6;
        border-radius: 6px; padding: .35rem .6rem; margin: .25rem 0;
        font-size: .88rem; }
 .liblist .librow.loaded { outline: 2px solid #157a3e; }
 .librow button { padding: .05rem .5rem; }
 .hint {
   display: inline-flex; align-items: center; justify-content: center;
   width: 14px; height: 14px; margin-left: .15rem;
   border: 1px solid #98a0b3; border-radius: 50%;
   font-size: 10px; line-height: 1; color: #556;
   background: #fff; cursor: help; position: relative;
   user-select: none; flex: none; vertical-align: 1px; }
 .hint:hover, .hint:focus { outline: none; border-color: #22263a; color: #000; }
 .hint::after {
   content: attr(data-tip);
   position: absolute; left: -6px; top: calc(100% + 7px); z-index: 40;
   width: max-content; max-width: 300px;
   padding: .45rem .6rem; border-radius: 6px;
   background: #1c1e26; color: #f2f3f7;
   font-size: .78rem; line-height: 1.35; text-align: left;
   white-space: normal; letter-spacing: normal;
   opacity: 0; visibility: hidden; transform: translateY(-4px);
   transition: opacity .12s ease .18s, transform .12s ease .18s;
   pointer-events: none;
   box-shadow: 0 4px 14px rgba(0, 0, 0, .28); }
  .hint:hover::after, .hint:focus::after {
    opacity: 1; visibility: visible; transform: none; }
 /* chat pane: one flex column filling its card; transcript grows,
    every other row stays fixed */
 .chatpane { flex: 1 1 auto; min-height: 0; display: flex;
             flex-direction: column; }
 .mid .chatpane { margin-bottom: 0; padding-bottom: .9rem; }
 .chatpane .chatlog { flex: 1 1 auto; height: auto; min-height: 220px;
                      margin-bottom: .6rem; }
 .chat-head { display: flex; justify-content: space-between;
              align-items: center; gap: .6rem; flex: none;
              margin: .2rem 0 .6rem; }
 .chat-head h2 { margin: 0; font-size: 1.05rem; white-space: nowrap; }
 .chat-controls { display: flex; align-items: center; gap: .45rem;
                  flex-wrap: wrap; justify-content: flex-end; }
 .composer { display: flex; gap: .4rem; align-items: center;
             margin-top: auto; flex: none; }
 .composer-input { flex: 1 1 auto; display: flex; min-width: 0; }
 .composer-input input { width: 100%; min-width: 0; flex: 1; }
 #stopbtn { display: none; }
 /* OpenCode-style session tabs */
 .sesstabs { display: flex; gap: .3rem; flex-wrap: wrap; margin: 0 0 .5rem; }
 .sesstab { border: 1px solid #cfd8e0; background: #eef2f6;
            border-radius: 6px; padding: .18rem .55rem; font-size: .84rem;
            cursor: pointer; display: inline-flex; align-items: center;
            gap: .45rem; max-width: 15rem; }
 .sesstab .name { overflow: hidden; text-overflow: ellipsis;
                  white-space: nowrap; }
 .sesstab.active { background: #1f6feb; border-color: #1f6feb; color: #fff; }
 .sesstab.unsaved .name::after { content: " •"; opacity: .7; }
 .sesstab .x { opacity: .55; font-weight: 700; padding: 0 .1rem; }
 .sesstab .x:hover { opacity: 1; }
 /* persistent agent tool steps (collapsible, stay in the transcript) */
 .toolstep { align-self: flex-start; max-width: 85%; background: #fff;
             border: 1px dashed #cfd8e0; border-radius: 8px;
             font-size: .82rem; }
 .toolstep summary { padding: .3rem .6rem; cursor: pointer; color: #456; }
 .toolstep pre { margin: .35rem .6rem; max-height: 180px; background: #f7fafc; }
 /* ---- HiveBench palette --------------------------------------------
    black canvas · amber containers · black text inside containers ·
    yellow text outside containers · red-orange borders --------------- */
 body { background: #000000; color: #FFDD00; }
 h1 { color: #FFDD00; }
 section { background: #FFB703; border: 2.4px solid #FB8500; color: #000000;
           box-shadow: 0 1px 6px rgba(251,133,0,.22); }
 section h2 { color: #000000; }
 section .note, section label.inline, section summary { color: rgba(0,0,0,.72); }
 section b { color: #000000; }
 input, select, textarea { background: #ffffff; color: #000000;
        border: 1.2px solid #FB8500; border-radius: 4px; }
 button { background: #ffffff; color: #000000;
          border: 1.2px solid #FB8500; border-radius: 4px; }
 button:hover:not(:disabled) { background: #ffedd1; }
 #stopbtn { background: #b3372c; color: #fff; border: none; }
 pre { background: #fff8ec; color: #000000; border: 1.2px solid #FB8500; }
 .chatlog { background: #fffdf7; border: 1.2px solid #FB8500; }
 .msg { transition: box-shadow .15s ease; }
 .msg.user { background: #000000; color: #FFDD00;
             box-shadow: 0 6px 16px rgba(0,0,0,.55),
                         0 0 0 2px #FFDD00,
                         0 0 20px rgba(0,0,0,.30); }
 .msg.ai { background: #ffffff; border: 1.2px solid #FB8500;
           box-shadow: 0 6px 16px rgba(251,133,0,.45),
                       0 0 18px rgba(251,133,0,.30); }
 .msg.sys { background: rgba(0,0,0,.08); color: #000000; }
 .tab { border-color: #FB8500; border-width: 1.2px; background: #ffe9c8; color: #000; }
 .tab.active { background: #ffffff; border-bottom-color: #ffffff; }
 .sesstab { border-color: #FB8500; border-width: 1.2px; background: #ffe9c8; color: #000;
            transition: box-shadow .15s ease, transform .15s ease; }
 .sesstab:hover { transform: translateY(-1px); }
 .sesstab.active { background: #000000; border-color: #000000; color: #FFDD00;
                   transform: translateY(-2px);
                   box-shadow: 0 6px 16px rgba(0,0,0,.55),
                               0 0 0 2.5px #FFDD00,
                               0 0 24px rgba(0,0,0,.30); }
 .hint { background: #ffffff; color: #553300; border-color: #7a4a00; }
 .toolstep { background: #ffffff; border: 1.2px dashed #FB8500; }
 """


def tip(text: str) -> str:
    """One inline help glyph whose hover/focus tooltip explains the control it follows."""
    return f'<span class="hint" tabindex="0" data-tip="{html.escape(text, quote=True)}">?</span>'


def render_server_page() -> str:
    """Model-management console: server + settings | loaded-model chat | hub.

    Discovery is live against the Hugging Face hub API — no model catalog is
    hardcoded here."""
    return f"""<!doctype html>
<html><head><meta charset="utf-8"><title>Studio server &amp; models</title>
<style>{_SERVER_CSS}</style></head><body>
<h1>Hive Studio console</h1>
<p style="margin:.3rem 0"><button id="afkbtn" onclick="toggleAfk(this)">AFK</button>{tip('AFK mode: human away - QUEEN runs expanded autonomy (GREEN/YELLOW fixes, regen, HIVE-PLAN orders). Public pushes/merges/policy changes queue for return; RED defects contained and logged.')}</p>
<form style="display:inline" onsubmit="event.preventDefault(); return false"><input id="researchq" placeholder="deep-research question..." size="30" autocomplete="off" onkeydown="if (event.key === &quot;Enter&quot;) researchAdd(this)"> <button id="researchsubmit" onclick="researchAdd(this)">Research</button> <span id="researchcount" class="meta"></span>{tip('Queues a deep-research question for QUEEN. Execution is master-only; reports land in RESEARCH/<slug>.md and are summarized on wake.')}</form>

<div class="grid">

<!-- ==================== LEFT: tabs ==================== -->
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
<datalist id="local-suggestions"></datalist></span>{tip('Local GGUF (models/gguf) to serve; leave blank to rely on the hub fields below.')}
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

<section>
<h2 style="margin-top:0">Local library <span class="note">(models/gguf)</span></h2>
<div id="local" class="liblist">loading…</div>
</section>
</div>

<div id="tab-engines" class="tabpane" style="display:none">
<section>
<h2 style="margin-top:0">Engine profiles</h2>
<div class="row">
<select id="eng-select" style="min-width:180px" onchange="engineSelected()"></select>
<label class="inline">default {tip('Profile applied when a conversation names no engine.')} <input id="eng-default" type="checkbox" onchange="engDirty=true"></label>
<button onclick="engineAdd()">+ Add</button>
</div>
<div class="row">
<label class="inline">name {tip('Display name of this engine profile.')} <input id="eng-name" size="16" oninput="engDirty=true"></label>
<label class="inline">kind {tip('Backend family the harness talks to.')} <select id="eng-kind" onchange="engDirty=true">
<option>llama_cpp</option><option>lmstudio</option><option>vllm</option>
<option>ollama</option><option>hosted</option></select></label>
</div>
<div class="row"><input id="eng-url" placeholder="base_url" style="width:95%" oninput="engDirty=true">{tip('OpenAI-compatible endpoint root, e.g. http://localhost:1234/v1')}</div>
<details open><summary>Sampling defaults (every request)</summary>
<div class="row">
<label class="inline">temp {tip('Randomness: higher = more varied, lower = more focused.')} <input id="s-temp" type="number" step="0.05" min="0" max="2" size="4" oninput="engDirty=true"></label>
<label class="inline">top_p {tip('Nucleus sampling: keep only tokens covering this cumulative probability.')} <input id="s-topp" type="number" step="0.05" min="0" max="1" size="4" oninput="engDirty=true"></label>
<label class="inline">top_k {tip('Sample only from the K most likely tokens.')} <input id="s-topk" type="number" size="4" oninput="engDirty=true"></label>
<label class="inline">min_p {tip('Drop tokens below this fraction of the top token probability.')} <input id="s-minp" type="number" step="0.01" size="4" oninput="engDirty=true"></label>
</div>
<div class="row">
<label class="inline">repeat {tip('Penalty on tokens already present; higher = less repetition.')} <input id="s-rep" type="number" step="0.05" size="4" oninput="engDirty=true"></label>
<label class="inline">presence {tip('Flat penalty once a token appears at all.')} <input id="s-pres" type="number" step="0.1" size="3" oninput="engDirty=true"></label>
<label class="inline">freq {tip('Penalty that grows with each repetition of a token.')} <input id="s-freq" type="number" step="0.1" size="3" oninput="engDirty=true"></label>
<label class="inline">seed {tip('Fixed RNG seed for reproducible output; blank = random.')} <input id="s-seed" type="number" size="7" oninput="engDirty=true"></label>
</div>
<div class="row">
<label class="inline">mirostat {tip('Adaptive perplexity control: 0 = off, 1 = v1, 2 = v2.')} <input id="s-miro" type="number" min="0" max="2" size="2" oninput="engDirty=true"></label>
<label class="inline">tau {tip('Mirostat target entropy: higher = more surprising text.')} <input id="s-tau" type="number" step="0.1" size="4" oninput="engDirty=true"></label>
<label class="inline">eta {tip('How fast mirostat adapts toward its target.')} <input id="s-eta" type="number" step="0.01" size="4" oninput="engDirty=true"></label>
<label class="inline">stop {tip('Comma-separated strings that end generation early.')} <input id="s-stop" size="12" placeholder="a,b" oninput="engDirty=true"></label>
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

<!-- ==================== FAR RIGHT: hub + inspector tabs ============ -->
<div class="col">
<div class="tabs">
<button class="tab active" data-rtab="rtab-hub">Hub</button>
<button class="tab" data-rtab="rtab-inspect">Inspector</button>
</div>
<div id="rtab-hub" class="tabpane">
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
<div id="rtab-inspect" class="tabpane" style="display:none">
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
    renderEngineSelect(0);
    engineSelected();
  }} catch (e) {{ show('eng-loadopts', String(e)); }}
}}

async function saveEngines() {{
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
    const wrap = document.getElementById('local');
    wrap.innerHTML = '';
    if (!l.models.length) {{
      wrap.textContent = '(no .gguf files yet)';
      return;
    }}
    for (const m of l.models) {{
      const row = document.createElement('div');
      row.className = 'librow';
      const label = document.createElement('span');
      label.textContent = `${{m.file}} — ${{m.size_gb}} GB`;
      const del = document.createElement('button');
      del.textContent = 'delete';
      del.title = 'remove from disk';
      del.addEventListener('click', async () => {{
        if (!confirm('Delete ' + m.file + ' from disk?')) return;
        try {{
          await api('/v1/models/local?file=' + encodeURIComponent(m.file),
                    'DELETE');
          refresh();
        }} catch (e) {{ alert(String(e)); }}
      }});
      row.appendChild(label);
      row.appendChild(del);
      wrap.appendChild(row);
    }}
  }}).catch(e => {{ document.getElementById('local').textContent = String(e); }});
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
