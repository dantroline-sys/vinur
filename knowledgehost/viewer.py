"""A tiny, dependency-free browser for inspecting the knowledge base.

Six tabs (two-level nav — see VINUR-UI-01_panel_redesign_plan.md) let you
*peruse* (not just search) everything the pipeline produces, so you can debug it:
  * **Ask**        — ONE query box, three modes: Answer (kb_ask) · Passages
                     (kb_search with scores) · Library (document FTS); the
                     query text survives mode switches.
  * **Distilled**  — the pipeline's layers in order: Sources (registry: trust +
                     epistemic regime per doc) → Raw (ingested chunks; spot
                     OCR/boilerplate problems) → Concepts (distilled nodes) →
                     Relations (typed edges w/ mechanism, regime, meta links —
                     the window into reconciliation) → Cards (procedures/criteria).
  * **Curation**   — the clean-up queues: Adjudication (ambiguous node merges)
                     · Gaps (queries the KB couldn't answer).
  * **Operations** — the authed maintenance-job runner + import formats.
  * **Serving**    — models this box hosts, weights-on-disk, swap control.
  * **Stats**      — graphed telemetry (GPU / vLLM queue / throughput) with
                     op + mark event lines, for tuning and A/B runs.
  * **Settings**   — General (scalar tunables) · Bundles · Library · Prioritizer.

No CDN/asset dependencies; everything is inline.
"""
from __future__ import annotations

INDEX_HTML = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Knowledge Host — viewer</title>
<style>
  :root { color-scheme: light dark; }
  body { font: 15px/1.5 system-ui, sans-serif; margin: 0; }
  header { padding: 12px 20px 0; border-bottom: 1px solid #8884;
           position: sticky; top: 0; background: Canvas; z-index: 2; }
  h1 { font-size: 17px; margin: 0 0 6px; }
  #stats { font-size: 13px; opacity: .85; margin-bottom: 6px; }
  #live { display: flex; gap: 16px; flex-wrap: wrap; align-items: baseline;
          font-size: 13px; margin-bottom: 8px; }
  #live .stat { display: inline-flex; gap: 5px; align-items: baseline; }
  #live .lbl { opacity: .6; }
  #live .val { font-weight: 600; font-variant-numeric: tabular-nums; }
  #live .rate { font-size: 11px; font-variant-numeric: tabular-nums; }
  #live .rate.up { color: #22aa66; }
  #live .rate.zero { opacity: .35; }
  #live .upd { opacity: .5; font-size: 11px; margin-left: auto; }
  .badge { display: inline-block; padding: 1px 7px; margin: 0 6px 2px 0;
           border: 1px solid #8886; border-radius: 10px; font-size: 12px; }
  .tabs { display: flex; gap: 2px; flex-wrap: wrap; }
  .tabs button { font: inherit; font-size: 13px; padding: 6px 12px; cursor: pointer;
           border: 1px solid #8886; border-bottom: none; border-radius: 6px 6px 0 0;
           background: #8881; color: CanvasText; }
  .tabs button.active { background: Canvas; font-weight: 600; position: relative; top: 1px; }
  .subtabs { display: flex; gap: 6px; flex-wrap: wrap; padding: 0 0 12px; }
  .subtabs:empty { display: none; }
  .subtabs button { font: inherit; font-size: 12px; padding: 3px 12px; cursor: pointer;
           border: 1px solid #8886; border-radius: 14px; background: transparent; color: CanvasText; }
  .subtabs button.active { background: #4a90d922; border-color: #4a90d9; font-weight: 600; }
  /* ONE centred column — the page used to hug the left edge of a big screen */
  .hwrap, main { max-width: 1240px; margin: 0 auto; }
  main { padding: 16px 20px; }
  @media (min-width: 1500px) {
    .hwrap, main { max-width: 1460px; }
    #results.grid2 { display: grid; grid-template-columns: 1fr 1fr; gap: 0 20px; align-items: start; }
  }
  /* ── Stats tab: small-multiple SVG time-series (validated 4-slot palette;
        light steps re-stepped for dark under prefers-color-scheme) ── */
  :root { --s1:#2a78d6; --s2:#008300; --s3:#e87ba4; --s4:#eda100;
          --cgrid:#e1e0d9; --caxis:#c3c2b7; --cmuted:#898781; }
  @media (prefers-color-scheme: dark) {
    :root { --s1:#3987e5; --s3:#d55181; --s4:#c98500;
            --cgrid:#2c2c2a; --caxis:#383835; }
  }
  .charts { display: grid; grid-template-columns: repeat(auto-fill, minmax(360px, 1fr));
            gap: 14px; align-items: start; }
  .chartcard { border: 1px solid #8883; border-radius: 8px; padding: 10px 12px; }
  .chartcard .ct { font-weight: 600; font-size: 13px; margin-bottom: 2px; }
  .chartcard .cu { opacity: .55; font-weight: 400; }
  .chartcard svg { display: block; width: 100%; }
  .chartcard svg:focus { outline: 1px solid var(--s1); outline-offset: 2px; }
  .clegend { display: flex; gap: 12px; flex-wrap: wrap; font-size: 12px;
             opacity: .85; margin: 2px 0 4px; }
  .clegend i { display: inline-block; width: 14px; height: 0; border-top: 2px solid;
               border-radius: 1px; vertical-align: middle; margin-right: 5px; }
  .ctable { margin-top: 6px; font-size: 12px; }
  .ctable > summary { cursor: pointer; opacity: .55; list-style: none; }
  .ctable table { font-size: 11px; }
  .ctable td, .ctable th { padding: 2px 6px; font-variant-numeric: tabular-nums; }
  #cktip { position: fixed; z-index: 10; pointer-events: none; display: none;
           background: Canvas; border: 1px solid #8886; border-radius: 6px;
           padding: 6px 9px; font-size: 12px; box-shadow: 0 2px 8px #0003; }
  #cktip .tv { font-weight: 600; font-variant-numeric: tabular-nums; }
  #cktip .tn { opacity: .7; margin-left: 6px; }
  #cktip .tt { opacity: .55; font-size: 11px; margin-bottom: 3px; }
  #cktip i { display: inline-block; width: 10px; height: 0; border-top: 2px solid;
             border-radius: 1px; vertical-align: middle; margin-right: 5px; }
  .evchips { display: flex; gap: 6px; flex-wrap: wrap; margin: 0 0 10px; font-size: 12px; }
  .evchips .badge { margin: 0; }
  .abdelta td { font-weight: 600; border-top: 2px solid #8886; }
  /* Distilled → Overview: stat tiles (proportional figures, per the contract) */
  .tiles { display: grid; grid-template-columns: repeat(auto-fill, minmax(150px, 1fr));
           gap: 10px; margin-bottom: 14px; }
  .tile { border: 1px solid #8883; border-radius: 8px; padding: 10px 12px; }
  .tile .tl { font-size: 12px; opacity: .65; }
  .tile .tvv { font-size: 26px; font-weight: 600; margin: 2px 0; }
  .tile .ts { font-size: 11px; opacity: .55; }
  /* per-source distillation progress (Sources view) */
  .pbar { display: inline-block; width: 64px; height: 7px; border-radius: 4px;
          background: #8883; vertical-align: middle; margin-right: 4px; overflow: hidden; }
  .pbar i { display: block; height: 100%; background: var(--s1); border-radius: 4px; }
  .bar { display: flex; gap: 8px; flex-wrap: wrap; align-items: center; margin-bottom: 14px; }
  input, select, button { font: inherit; padding: 7px 10px; border: 1px solid #8886;
           border-radius: 6px; background: Canvas; color: CanvasText; }
  input[type=search] { flex: 1; min-width: 240px; }
  .toolbtn { cursor: pointer; }
  .note { padding: 8px 12px; border-radius: 6px; margin: 10px 0;
          background: #f5a62322; border: 1px solid #f5a62366; font-size: 14px; }
  .p { border: 1px solid #8883; border-radius: 8px; padding: 10px 12px; margin: 10px 0; }
  .p .meta { font-size: 12px; opacity: .85; margin-bottom: 5px;
             display: flex; gap: 8px; flex-wrap: wrap; align-items: baseline; }
  .p .title { font-weight: 600; }
  .p .score { margin-left: auto; font-variant-numeric: tabular-nums; }
  .p .text { white-space: pre-wrap; }
  .p .src { font-size: 11px; opacity: .65; margin-top: 6px; word-break: break-all; }
  .rel { font-size: 15px; }
  .rel .arrow { font-weight: 600; }
  .meta-link { font-size: 12px; margin-top: 5px; }
  .meta-link.warn { color: #d9534f; }
  ol.steps { margin: 6px 0 0; padding-left: 22px; }
  .focus { border-left: 3px solid #4a90d9; padding: 6px 12px; margin: 4px 0 12px; }
  .focus .focus-l { font-size: 18px; font-weight: 600; }
  .focus .focus-s { opacity: .8; margin-top: 2px; }
  .crit { margin: 6px 0; display: flex; flex-direction: column; gap: 4px; }
  .crit .crit-row { font-size: 13px; }
  .crit .crit-row b { font-size: 11px; text-transform: uppercase; opacity: .6; margin-right: 6px; }
  .struct { margin-top: 14px; border-top: 1px solid #8883; padding-top: 8px; }
  .struct .struct-h { font-size: 11px; text-transform: uppercase; opacity: .55; margin-bottom: 6px; }
  .struct .struct-sec { margin: 5px 0; display: flex; gap: 8px; align-items: baseline; flex-wrap: wrap; }
  .struct .struct-role { font-size: 12px; opacity: .7; min-width: 130px; font-weight: 600; }
  .struct .struct-ent { font-size: 13px; border: 1px solid #8883; border-radius: 6px; padding: 1px 8px; }
  table { border-collapse: collapse; width: 100%; font-size: 13px; }
  th, td { text-align: left; padding: 5px 8px; border-bottom: 1px solid #8883; }
  th { opacity: .7; font-weight: 600; }
  .empty { opacity: .6; padding: 20px 0; }
  /* per-tab help (help.json via /help) + the Import formats table (live probes) */
  .tabhelp { margin: 0 0 12px; font-size: 13px; }
  .tabhelp > summary { cursor: pointer; opacity: .6; list-style: none; }
  .tabhelp > summary::-webkit-details-marker { display: none; }
  .tabhelp > div { opacity: .85; border-left: 2px solid #8886; padding: 8px 12px;
                   margin-top: 6px; max-width: 860px; }
  .fmt-ok { color: #22aa66; font-weight: 600; }
  .fmt-no { color: #d9534f; font-weight: 600; }
  #fmtbox { margin: 10px 0 14px; }
  #fmtbox .fmt-h { font-size: 12px; text-transform: uppercase; opacity: .55; margin-bottom: 4px; }
</style>
</head>
<body>
<header>
  <div class="hwrap">
    <h1>Knowledge Host — viewer</h1>
    <div id="stats">loading…</div>
    <div id="live"></div>
    <div class="tabs" id="tabs"></div>
  </div>
</header>
<main>
  <div class="subtabs" id="subtabs"></div>
  <div class="bar" id="bar"></div>
  <div id="tabhelp"></div>
  <div id="banner"></div>
  <div id="results" class="empty">…</div>
</main>
<div id="cktip"></div>
<script>
const $ = s => document.querySelector(s);
const esc = t => { const d = document.createElement('div'); d.textContent = t == null ? '' : t; return d.innerHTML; };
const badge = t => `<span class="badge">${esc(t)}</span>`;

// ── two-level nav: 6 groups over the (unchanged) leaf panels ─────────────────
// Leaf keys are load-bearing — they key help.json, the loaders and go()'s
// dispatch — so consolidation happens HERE, at the nav level, only.
// Distilled runs in pipeline order: provenance → raw text → concepts →
// relations → cards (each layer is distilled from the one before it).
const GROUPS = [
  ['ask', 'Ask', []],            // one query box; the MODE select replaces sub-tabs
  ['distilled', 'Distilled', [['overview', 'Overview'],
                              ['sources', 'Sources'], ['raw', 'Raw'], ['nodes', 'Concepts'],
                              ['edges', 'Relations'], ['cards', 'Cards']]],
  ['curation', 'Curation', [['adjudication', 'Adjudication'], ['gaps', 'Gaps']]],
  ['ops', 'Operations', []],
  ['serving', 'Serving', []],
  ['stats', 'Stats', []],
  ['settings', 'Settings', [['settings', 'General'], ['bundles', 'Bundles'],
                            ['library', 'Library'], ['autopilot', 'Prioritizer']]],
];
const PARENT = {};                    // leaf key -> its group key
GROUPS.forEach(([g, _l, kids]) => { PARENT[g] = g; kids.forEach(([k]) => PARENT[k] = g); });
const lastLeaf = {};                  // group key -> the sub-view left open there
let active = 'ask';

function buildTabs() {
  $('#tabs').innerHTML = GROUPS.map(([k, lbl]) =>
    `<button data-k="${k}" onclick="go('${k}')">${lbl}</button>`).join('');
}

// pills carry the live class counts (from the same poll the header uses)
const SUBCOUNT = { sources: 'sources', raw: 'chunks', nodes: 'nodes', edges: 'edges',
                   cards: 'cards', adjudication: 'adjudicate', gaps: 'gaps' };
let LASTCOUNTS = {};

function renderSubtabs(leaf) {
  const grp = GROUPS.find(g => g[0] === PARENT[leaf]);
  const kids = grp ? grp[2] : [];
  $('#subtabs').innerHTML = kids.length < 2 ? '' : kids.map(([k, lbl]) => {
    const n = LASTCOUNTS[SUBCOUNT[k]];
    const cnt = n != null ? ` <span style="opacity:.55">${fmtCompact(n)}</span>` : '';
    return `<button data-k="${k}" class="${k === leaf ? 'active' : ''}" onclick="go('${k}',1)">${lbl}${cnt}</button>`;
  }).join('');
}
// ── Help: tab intros from help.json + live import-format probes (/help) ─────
let HELP = { help: {}, formats: [] };
async function loadHelp() {
  try { HELP = await (await fetch('/help')).json(); } catch (e) { /* keep empty */ }
  renderHelp(active);
}
function renderHelp(k) {
  const box = $('#tabhelp');
  const txt = (HELP.help || {})[k];
  let h = txt ? `<details class="tabhelp"><summary>ⓘ about this tab</summary><div>${esc(txt)}</div></details>` : '';
  if (k === 'ops' && (HELP.formats || []).length) {
    const rows = HELP.formats.map(f =>
      `<tr><td>${esc(f.format)}</td><td><code>${esc(f.matches)}</code></td>
       <td class="${f.ready ? 'fmt-ok' : 'fmt-no'}">${f.ready ? '✓ ready' : '✗ not installed'}</td>
       <td style="opacity:.75">${esc(f.how)}</td></tr>`).join('');
    h += `<details id="fmtbox"><summary style="cursor:pointer;opacity:.6;font-size:13px">📄 Import formats — what this host can ingest (click to expand)</summary>
      <div style="opacity:.85;font-size:13px;margin:8px 0;max-width:860px">${esc((HELP.help || {}).import || '')}</div>
      <table><tr><th>format</th><th>files</th><th>status</th><th>enable / notes</th></tr>${rows}</table></details>`;
  }
  if (k === 'ops' && (HELP.datasets || []).length) {
    const rows = HELP.datasets.map(d => {
      const st = d.present ? '<span class="fmt-ok">✓ file present</span>'
        : d.path ? '<span class="fmt-no">✗ file not found</span>'
        : `<span style="opacity:.6">path not set — <code>${esc(d.config_key)}</code> in Settings</span>`;
      const inkb = d.imported === true ? '<span class="fmt-ok">✓ imported</span>'
        : d.imported === false ? '<span style="opacity:.5">—</span>'
        : '<span style="opacity:.4">?</span>';
      return `<tr><td>${esc(d.name)}</td><td><code>${esc(d.verb)}</code></td>
        <td>${st}</td><td>${inkb}</td><td style="opacity:.6;word-break:break-all">${esc(d.path || '')}</td>
        <td style="opacity:.75">${esc(d.note)}</td></tr>`;
    }).join('');
    h += `<details id="dsbox"><summary style="cursor:pointer;opacity:.6;font-size:13px">🧩 External datasets — bulk commonsense/causal imports (click to expand)</summary>
      <div style="opacity:.85;font-size:13px;margin:8px 0;max-width:860px">Each imports as its own
      low-trust source under the epistemic firewall (they can never override your distilled or
      empirical knowledge). Drop the file into <code>external/</code> (the default path), run the
      verb below, then <code>embed-nodes</code> to backfill vectors. To retune thresholds, run
      <code>unimport</code> with the dataset name — it removes exactly what that import
      contributed (shared/fused nodes survive) so you can re-import with new settings.</div>
      <table><tr><th>dataset</th><th>run</th><th>file</th><th>in KB</th><th>path</th><th>what it is</th></tr>${rows}</table></details>`;
  }
  box.innerHTML = h;
}

function go(k, leaf) {
  // A group button reopens the sub-view last used there (or its first).
  // A group and its default leaf may share a key ('ask', 'settings'), so
  // sub-tab pills pass leaf=1 to say "this exact panel" — without it the
  // General/Answer pills would group-redirect straight back to wherever
  // you just were.  The redirect assigns once and falls through.
  const grp = leaf ? null : GROUPS.find(g => g[0] === k);
  if (grp && grp[2].length) k = lastLeaf[k] || grp[2][0][0];
  // legacy leaf keys from the flat-tab era map onto Ask's modes
  if (k === 'search' || k === 'libsearch') {
    ASK_MODE = k === 'search' ? 'passages' : 'library';
    k = 'ask';
  }
  active = k;
  lastLeaf[PARENT[k]] = k;
  if (opsTimer) { clearInterval(opsTimer); opsTimer = null; }   // stop polling when leaving Ops
  document.querySelectorAll('#tabs button').forEach(b =>
    b.classList.toggle('active', b.dataset.k === PARENT[k]));
  renderSubtabs(k);
  renderBar(k);
  renderHelp(k);
  if (k === 'ask') { askEmptyState(); if ($('#aq')) $('#aq').focus(); }
  else if (k === 'ops') { loadOps(); opsTimer = setInterval(() => { if (active === 'ops') pollOps(); }, 2500); }
  else if (k === 'serving') { loadServing(); opsTimer = setInterval(() => { if (active === 'serving') pollServing(); }, 2500); }
  else if (k === 'stats') { loadStatsTab(); opsTimer = setInterval(() => { if (active === 'stats') loadStatsTab(true); }, 10000); }
  else if (k === 'overview') { loadOverview(); }
  else if (k === 'settings') { loadSettings(); }
  else if (k === 'autopilot') { loadAutopilot(); }
  else if (k === 'bundles') { loadBundles(); }
  else if (k === 'library') { loadLibrary(); }
  else load(k);
}

function renderBar(k) {
  $('#banner').innerHTML = '';
  if (k === 'ask') {
    $('#bar').innerHTML =
      `<select id="amode" onchange="askSetMode(this.value)" title="what to query">
         <option value="answer"${ASK_MODE === 'answer' ? ' selected' : ''}>Answer — structured KB</option>
         <option value="passages"${ASK_MODE === 'passages' ? ' selected' : ''}>Passages — kb search</option>
         <option value="library"${ASK_MODE === 'library' ? ' selected' : ''}>Library — documents</option>
       </select>
       <input id="aq" type="search" placeholder="Ask a what / how / why question…" autofocus
              value="${esc(ASK_Q)}" onkeydown="if(event.key==='Enter')doAskGo()">
       <span id="askx">${askExtras()}</span>
       <button class="toolbtn" onclick="doAskGo()">Go</button>`;
  } else if (k === 'overview') {
    $('#bar').innerHTML = tokInput()
      + ` <button class="toolbtn" onclick="loadOverview()">Reload</button>
        <span style="opacity:.6;font-size:13px">token enables the Upkeep actions below</span>`;
  } else if (k === 'adjudication') {
    $('#bar').innerHTML = tokInput()
      + ` <button class="toolbtn" onclick="upkeepRun('adjudicate')" title="clear the ambiguous node-merge queue — one job, watch it under Operations">Run adjudicate now</button>`
      + ` <button class="toolbtn" onclick="upkeepQueue('adjudicate')" title="append it to the Prioritizer plan instead of running immediately">Queue as step</button>`
      + ` <button class="toolbtn" onclick="load('adjudication')">Reload</button>`;
  } else if (k === 'gaps') {
    $('#bar').innerHTML = tokInput()
      + ` <button class="toolbtn" onclick="load('gaps')">Reload</button>
        <span style="opacity:.6;font-size:13px">open gaps flow to Vinkona automatically via the research handshake — dismiss the ones not worth researching</span>`;
  } else if (k === 'raw') {
    $('#bar').innerHTML =
      `<select id="srcfilter" title="source"></select>
       <button class="toolbtn" onclick="load('raw')">Reload sample</button>`;
    fillSources();
  } else if (k === 'serving') {
    $('#bar').innerHTML = tokInput()
      + ` <button class="toolbtn" onclick="pollServing()">Refresh</button>`;
  } else if (k === 'stats') {
    $('#bar').innerHTML = tokInput()
      + ` <select id="strange" onchange="loadStatsTab()" title="time range">`
      + STAT_RANGES.map(([l, m]) =>
          `<option value="${m}"${m === STAT_MINS ? ' selected' : ''}>last ${l}</option>`).join('')
      + `</select>
       <button class="toolbtn" onclick="loadStatsTab()">Refresh</button>
       <button class="toolbtn" onclick="dropMark()" title="label an A/B boundary: what did you just change?">⚑ Drop mark</button>
       <span style="opacity:.6;font-size:13px">auto-refreshes every 10 s while open</span>`;
  } else if (k === 'ops') {
    $('#bar').innerHTML = tokInput()
      + ` <select id="opcmd" onchange="onCmdChange()" title="maintenance command"></select>`
      + ` <span id="opopts"></span>`
      + ` <button class="toolbtn" onclick="runOp()">Run</button>`
      + ` <button class="toolbtn" onclick="stopOp()">Stop</button>`
      + ` <button class="toolbtn" onclick="reloadKB()" title="re-warm caches after a job">Reload KB</button>`;
  } else if (k === 'bundles') {
    $('#bar').innerHTML = tokInput()
      + ` <select id="scensel" title="scenario"></select>`
      + ` <button class="toolbtn" onclick="applyScenario()" title="reassemble + hot-swap the live session">Apply scenario</button>`
      + ` <button class="toolbtn" onclick="loadBundles()">Refresh</button>`;
  } else if (k === 'autopilot') {
    $('#bar').innerHTML = tokInput()
      + ` <button class="toolbtn" onclick="addAutopilotStep()">+ Add step</button>`
      + ` <button class="toolbtn" onclick="saveAutopilot()">Save plan</button>`
      + ` <button class="toolbtn" onclick="loadAutopilot()">Refresh</button>`;
  } else if (k === 'settings') {
    $('#bar').innerHTML = tokInput()
      + ` <button class="toolbtn" onclick="saveSettings()">Save</button>`
      + ` <span style="opacity:.6;font-size:13px">edits config.toml — restart or Reload KB to apply</span>`;
  } else if (k === 'library') {
    $('#bar').innerHTML = tokInput()
      + ` <button class="toolbtn" onclick="saveLibrarySelection()">Save selection</button>`
      + ` <button class="toolbtn" onclick="saveLibrarySelection(true)">Save + index now</button>`
      + ` <button class="toolbtn" onclick="loadLibrary()">Refresh</button>`;
  } else {
    $('#bar').innerHTML = `<button class="toolbtn" onclick="load('${k}')">Reload</button>
       <span style="opacity:.6;font-size:13px">newest / random first</span>`;
  }
}

// ── live counters: poll every few seconds, show per-minute rates ──────────────
const REFRESH_MS = 3000, RATE_WINDOW_MS = 60000;
const fmt = n => (n || 0).toLocaleString();
let statHist = [];          // rolling [{t, c:{metric:value}}] over the last minute

function computeRates(counts) {
  const now = Date.now();
  statHist.push({ t: now, c: counts });
  while (statHist.length > 2 && now - statHist[0].t > RATE_WINDOW_MS) statHist.shift();
  const base = statHist[0], dt = (now - base.t) / 1000;   // seconds spanned
  const rates = {};
  for (const k in counts) rates[k] = dt >= 1 ? (counts[k] - (base.c[k] || 0)) / dt * 60 : null;
  return { dt, rates };
}
function rateSpan(r) {
  if (r == null) return '';
  const v = Math.abs(r) >= 10 ? Math.round(r) : Math.round(r * 10) / 10;
  return `<span class="rate ${r > 0.049 ? 'up' : 'zero'}">${v > 0 ? '+' : ''}${v}/min</span>`;
}
function statEl(lbl, val, r) {
  return `<span class="stat" title="${lbl}"><span class="lbl">${lbl}</span>`
    + `<span class="val">${fmt(val)}</span>${rateSpan(r)}</span>`;
}

async function refreshStats() {
  let s = {}, kb = {};
  try { s = await (await fetch('stats')).json(); } catch (e) {}
  try { kb = (await (await fetch('kb')).json()).counts || {}; } catch (e) {}
  const counts = {
    chunks: s.chunks || 0, nodes: kb.nodes || 0, edges: kb.edges || 0, cards: kb.cards || 0,
    distilled: kb.distilled_chunks || 0, adjudicate: kb.merge_candidates || 0, gaps: kb.gaps || 0,
  };
  LASTCOUNTS = Object.assign({ sources: kb.sources }, counts);
  const { dt, rates } = computeRates(counts);
  const by = Object.entries(s.by_source || {}).map(([k, v]) => badge(`${k}: ${fmt(v)}`)).join('');
  $('#stats').innerHTML = badge('backend: ' + (s.backend || '?'))
    + badge('dense: ' + (s.dense ? 'yes' : 'no')) + badge('v' + (s.version || '1')) + (by ? ' ' + by : '');
  const order = [['chunks', 'chunks'], ['nodes', 'nodes'], ['edges', 'edges'], ['cards', 'cards'],
    ['distilled', 'distilled'], ['adjudicate', 'adjudicate'], ['gaps', 'gaps']];
  $('#live').innerHTML = order.map(([k, lbl]) => statEl(lbl, counts[k], rates[k])).join('')
    + `<span class="stat upd">⟳ ${dt >= 1 ? '/min over ' + Math.min(60, Math.round(dt)) + 's' : 'measuring…'}</span>`;
}

async function fillSources() {
  try {
    const s = await (await fetch('stats')).json();
    const sel = $('#srcfilter');
    if (sel) sel.innerHTML = '<option value="">all sources</option>' +
      Object.keys(s.by_source || {}).map(k => `<option value="${esc(k)}">${esc(k)}</option>`).join('');
  } catch (e) {}
}

// ── Distilled → Overview: what the pipeline has produced, at a glance ───────
function fmtCompact(v) {
  if (v == null) return '—';
  if (v >= 1e6) return (Math.round(v / 1e5) / 10) + 'M';
  if (v >= 1e4) return (Math.round(v / 100) / 10) + 'K';
  return Math.round(v).toLocaleString();
}

async function loadOverview() {
  $('#banner').innerHTML = '';
  $('#results').className = 'empty';
  $('#results').textContent = 'loading…';
  let s = {}, kb = {};
  try { s = await (await fetch('stats')).json(); } catch (e) {}
  try { kb = (await (await fetch('kb')).json()).counts || {}; } catch (e) {}
  const chunks = s.chunks || 0, dist = kb.distilled_chunks || 0;
  const tiles = [
    ['sources', kb.sources, 'registered documents'],
    ['chunks', chunks, 'ingested passages'],
    ['distilled', dist, chunks ? Math.round(dist / chunks * 100) + '% of chunks' : ''],
    ['concepts', kb.nodes, 'distilled nodes'],
    ['relations', kb.edges, 'typed edges'],
    ['cards', kb.cards, 'procedures & criteria'],
  ];
  const r1 = kb.nodes ? kb.edges / kb.nodes : null;
  const r2 = dist ? kb.cards / dist * 100 : null;
  const ratios = [
    ['relations per concept', r1 == null ? null : Math.round(r1 * 100) / 100,
     'graph connectivity — grows with the link pass'],
    ['cards per 100 distilled chunks', r2 == null ? null : Math.round(r2 * 10) / 10,
     'how card-rich the corpus is (how-to / criteria density)'],
    ['open merge candidates', kb.merge_candidates, 'ambiguous node pairs → Curation › Adjudication'],
    ['open knowledge gaps', kb.gaps, 'unanswered queries → Curation › Gaps'],
  ];
  const bysrc = Object.entries(s.by_source || {})
    .map(([k2, v]) => badge(`${k2}: ${fmtCompact(v)}`)).join(' ');
  setRows(
    `<div style="font-size:13px;opacity:.7;margin-bottom:10px">The pipeline's layers, in order:
       <b>Sources → Raw → Concepts → Relations → Cards</b> — each distilled from the one before.
       Click a pill above to browse a layer.</div>`
    + '<div class="tiles">' + tiles.map(([l, v, sub]) =>
        `<div class="tile"><div class="tl">${esc(l)}</div><div class="tvv">${fmtCompact(v)}</div>
         <div class="ts">${esc(sub)}</div></div>`).join('') + '</div>'
    + '<table><tr><th>ratio / queue</th><th>value</th><th>what it tells you</th></tr>'
    + ratios.map(([l, v, note]) =>
        `<tr><td>${esc(l)}</td><td>${v == null ? '—' : esc(String(v.toLocaleString ? v.toLocaleString() : v))}</td>
         <td style="opacity:.7">${esc(note)}</td></tr>`).join('') + '</table>'
    + (bysrc ? `<div style="margin-top:12px;font-size:13px"><span style="opacity:.6">chunks by source type:</span> ${bysrc}</div>` : '')
    + renderUpkeep(kb, r1));
}

// ── Upkeep: the tidy-up verbs, with WHEN and WHY — run now or queue a step ──
function renderUpkeep(kb, relPerNode) {
  const merge = kb.merge_candidates || 0;
  const acts = [
    ['link',
     'Finds typed relations between EXISTING concepts. Run after a big ingest/distill '
     + 'wave — it is what raises relations-per-concept. Fast-LM eligible.',
     relPerNode != null ? 'now ' + (Math.round(relPerNode * 100) / 100) + ' rel/concept' : ''],
    ['adjudicate',
     'Clears the node-merge queue (pairs too similar to keep apart, too different to '
     + 'auto-merge). Duplicates dilute retrieval — run when the queue climbs.',
     merge ? merge + ' queued' + (merge > 50 ? ' — worth clearing' : '') : 'queue empty'],
    ['refine',
     'Source-grounded, in-place rewrite of weak or stale cards. Run occasionally — '
     + 'after corrections land or when new sources touch old topics. Big-LM work.',
     ''],
    ['embed-nodes',
     'Backfills vectors for nodes created without embeddings (the bulk import-* verbs '
     + 'skip them). Run once after any import; a no-op when nothing is missing.',
     ''],
  ];
  return `<h3 style="margin:18px 0 6px;font-size:14px">Upkeep</h3>
    <div style="font-size:12px;opacity:.6;margin-bottom:8px">One job at a time (they share the GPU
    and the KB) — <b>Run now</b> launches it like the Operations tab; <b>Queue as step</b> appends it
    to the Prioritizer plan (Settings → Prioritizer) to run when the box is idle.  Every run shows up
    as an event line on Stats.</div>
    <table><tr><th>verb</th><th>when &amp; why</th><th>signal</th><th></th></tr>`
    + acts.map(([cmd, why, sig]) =>
      `<tr><td><code>${esc(cmd)}</code></td><td style="opacity:.8">${esc(why)}</td>
       <td style="white-space:nowrap">${esc(sig)}</td>
       <td style="white-space:nowrap">
         <button class="toolbtn" style="font-size:12px;padding:3px 9px" onclick="upkeepRun('${cmd}')">Run now</button>
         <button class="toolbtn" style="font-size:12px;padding:3px 9px" onclick="upkeepQueue('${cmd}')">Queue as step</button>
       </td></tr>`).join('') + '</table>';
}

async function upkeepRun(cmd) {
  const r = await postJSON('/ops/run', { command: cmd, args: {} })
    .catch(e => ({ ok: false, error: '' + e }));
  $('#banner').innerHTML = `<div class="note">${r.ok
    ? '▶ ' + esc(cmd) + ' started — follow it under Operations; it will appear as an event line on Stats'
    : '✗ ' + esc(r.error || 'failed — auth token?')}</div>`;
}

async function upkeepQueue(cmd) {
  let r0;
  try { r0 = await (await authFetch('/ops/autopilot')).json(); }
  catch (e) { r0 = { ok: false, error: '' + e }; }
  if (!r0.ok) {
    $('#banner').innerHTML = `<div class="note">✗ ${esc(r0.error || 'failed — auth token?')}</div>`;
    return;
  }
  const plan = r0.plan || {};
  (plan.steps = plan.steps || []).push({ command: cmd, args: {}, enabled: true });
  const r = await postJSON('/ops/autopilot', { plan }).catch(e => ({ ok: false, error: '' + e }));
  $('#banner').innerHTML = `<div class="note">${r.ok
    ? '✓ ' + esc(cmd) + ' queued as a Prioritizer step — reorder or edit it under Settings → Prioritizer'
    : '✗ ' + esc(r.error || 'failed')}</div>`;
}

// ── renderers ────────────────────────────────────────────────────────────────
function setRows(html, emptyMsg, grid) {
  const r = $('#results');
  if (!html) { r.className = 'empty'; r.textContent = emptyMsg; return; }
  r.className = grid ? 'grid2' : '';   // grid2 = two columns on >=1500px screens
  r.innerHTML = html;
}

function renderPassages(ps, withScore) {
  setRows(ps && ps.length ? ps.map(p => `
    <div class="p">
      <div class="meta"><span class="title">${esc(p.title) || '(untitled)'}</span>
        ${p.section ? '<span>› ' + esc(p.section) + '</span>' : ''}
        ${badge(p.source_type || '?')}
        ${withScore && p.score != null ? '<span class="score">score ' + Number(p.score).toFixed(3) + '</span>' : ''}</div>
      <div class="text">${esc(p.text)}</div>
      ${p.path_or_url ? '<div class="src">' + esc(p.path_or_url) + '</div>' : ''}
    </div>`).join('') : '', 'No passages.', true);
}

function renderNodes(ns) {
  setRows(ns && ns.length ? ns.map(n => `
    <div class="p">
      <div class="meta"><span class="title">${esc(n.label)}</span>${badge(n.kind || 'concept')}
        ${(n.aliases && n.aliases.length) ? '<span>aka ' + esc(n.aliases.join(', ')) + '</span>' : ''}</div>
      <div class="text">${esc(n.summary)}</div>
      ${(n.sources && n.sources.length) ? '<div class="src">distilled from: ' + esc(n.sources.join(' · ')) + '</div>' : ''}
    </div>`).join('') : '', 'No distilled concepts yet — run:  python -m knowledgehost distill', true);
}

function renderEdges(es) {
  setRows(es && es.length ? es.map(e => {
    const meta = (e.meta || []).map(m => {
      const warn = m.type === 'disagrees_with';
      return `<div class="meta-link ${warn ? 'warn' : ''}">${warn ? '⚠ ' : '↔ '}${esc(m.type)} — ${esc(m.other)}</div>`;
    }).join('');
    return `<div class="p">
      <div class="rel"><span>${esc(e.src)}</span> <span class="arrow">—${esc(e.type)}→</span> <span>${esc(e.dst)}</span></div>
      <div class="meta">${badge(e.family)}${badge(e.regime)}${e.polarity ? badge(e.polarity) : ''}
        ${e.mechanism ? '<span>via ' + esc(e.mechanism) + '</span>' : ''}
        ${e.conditions ? '<span>if ' + esc(e.conditions) + '</span>' : ''}</div>
      ${(e.support && e.support.length) ? '<div class="src">support: ' + esc(e.support.join(' · ')) + '</div>' : ''}
      ${meta}
    </div>`;
  }).join('') : '', 'No relations yet — distil some sources.', true);
}

// Typed cards (criteria/staging/requirements/decision/playbook/case…) keep their
// content as a JSON payload in `criteria` — render it generically so every shape
// (and any future card type) displays: strings as lines, string-lists as bullets,
// object-lists as nested blocks, dicts as labelled sub-sections.
function renderPayload(v) {
  if (v == null || v === '') return '';
  if (Array.isArray(v)) {
    if (!v.length) return '';
    return '<ul class="steps">' + v.map(x =>
      '<li>' + (typeof x === 'object' ? renderPayload(x) : esc(x)) + '</li>').join('') + '</ul>';
  }
  if (typeof v === 'object') {
    return Object.entries(v).map(([k, x]) => {
      const body = renderPayload(x);
      return body ? `<div class="text"><b>${esc(k.replace(/_/g, ' '))}:</b> ${body}</div>` : '';
    }).join('');
  }
  return esc(v);
}

function renderCards(cs) {
  setRows(cs && cs.length ? cs.map(c => `
    <div class="p">
      <div class="meta"><span class="title">${esc(c.title)}</span>${badge(c.card_type || 'procedure')}${badge(c.regime)}
        ${c.node ? '<span>node: ' + esc(c.node) + '</span>' : ''}</div>
      ${c.goal ? '<div class="text">Goal: ' + esc(c.goal) + '</div>' : ''}
      ${(c.steps && c.steps.length) ? '<ol class="steps">' + c.steps.map(s => '<li>' + esc(s) + '</li>').join('') + '</ol>' : ''}
      ${c.criteria ? '<div class="text">' + renderPayload(c.criteria) + '</div>' : ''}
      ${(c.support && c.support.length) ? '<div class="src">support: ' + esc(c.support.join(' · ')) + '</div>' : ''}
    </div>`).join('') : '', 'No cards yet — distil some sources.', true);
}

function renderTable(rows, cols, emptyMsg) {
  if (!rows || !rows.length) return setRows('', emptyMsg);
  const head = '<tr>' + cols.map(c => `<th>${esc(c[1])}</th>`).join('') + '</tr>';
  const body = rows.map(r => '<tr>' + cols.map(c => {
    let v = r[c[0]];
    if (typeof v === 'number') v = (c[0] === 'trust_weight' || c[0] === 'similarity') ? v : v;
    return `<td>${esc(v)}</td>`;
  }).join('') + '</tr>').join('');
  setRows('<table>' + head + body + '</table>');
}

async function load(kind) {
  $('#banner').innerHTML = ''; $('#results').className = 'empty'; $('#results').textContent = 'loading…';
  try {
    if (kind === 'raw') {
      const src = $('#srcfilter') ? $('#srcfilter').value : '';
      const res = await (await fetch(`sample?n=25${src ? '&source_type=' + encodeURIComponent(src) : ''}`)).json();
      return renderPassages(res.passages, false);
    }
    const res = await (await fetch(`browse?kind=${kind}&n=100`)).json();
    const rows = res.rows || [];
    if (kind === 'nodes') return renderNodes(rows);
    if (kind === 'edges') return renderEdges(rows);
    if (kind === 'cards') return renderCards(rows);
    if (kind === 'sources') return renderSources(rows);
    if (kind === 'adjudication') return renderTable(rows,
      [['node_a', 'node A'], ['node_b', 'node B'], ['similarity', 'sim'],
       ['reason', 'reason'], ['status', 'status']], 'Adjudication queue empty.');
    if (kind === 'gaps') return renderGaps(rows);
  } catch (e) { $('#results').textContent = 'request failed: ' + e; }
}

// sources get distillation-progress + the source file's own date
function renderSources(rows) {
  if (!rows || !rows.length) return setRows('', 'No sources registered yet.');
  let tot = 0, dist = 0;
  rows.forEach(r => { tot += r.chunks || 0; dist += r.distilled || 0; });
  const summary = tot
    ? `<div style="font-size:13px;opacity:.7;margin-bottom:8px">listed ${rows.length} source(s) ·
       ${fmtCompact(dist)} of ${fmtCompact(tot)} chunks distilled
       (${Math.round(dist / tot * 100)}%)</div>` : '';
  const head = '<tr><th>doc</th><th>title</th><th>type</th><th>trust</th><th>regime</th>'
    + '<th>status</th><th>chunks</th><th>distilled</th><th>file date</th></tr>';
  const body = rows.map(r => {
    const pcell = r.pct == null ? '—'
      : `<span class="pbar" title="${esc(r.distilled)} / ${esc(r.chunks)} chunks">`
        + `<i style="width:${r.pct}%"></i></span> ${r.pct}%`;
    return `<tr><td>${esc(r.doc_id)}</td><td>${esc(r.title)}</td><td>${esc(r.source_type)}</td>
      <td>${esc(r.trust_weight)}</td><td>${esc(r.regime)}</td><td>${esc(r.status)}</td>
      <td>${r.chunks != null ? fmtCompact(r.chunks) : '—'}</td>
      <td style="white-space:nowrap">${pcell}</td>
      <td style="white-space:nowrap;opacity:.75">${esc(r.file_time || '—')}</td></tr>`;
  }).join('');
  setRows(summary + '<table>' + head + body + '</table>');
}

// gaps get a per-row dismiss; rows are index-keyed (query text is untrusted —
// it must never be interpolated into an onclick attribute)
let GAPROWS = [];

function renderGaps(rows) {
  GAPROWS = rows || [];
  if (!GAPROWS.length) return setRows('', 'No knowledge gaps logged.');
  const head = '<tr><th>query</th><th>intent</th><th>count</th><th>status</th><th></th></tr>';
  const body = GAPROWS.map((r, i) =>
    `<tr><td>${esc(r.query_text)}</td><td>${esc(r.intent)}</td><td>${esc(r.count)}</td>
     <td>${esc(r.status)}</td><td>${r.status === 'open'
       ? `<button class="toolbtn" style="font-size:11px;padding:2px 8px" onclick="dismissGap(${i})">dismiss</button>`
       : ''}</td></tr>`).join('');
  setRows('<table>' + head + body + '</table>');
}

async function dismissGap(i) {
  const q = (GAPROWS[i] || {}).query_text;
  if (!q) return;
  const r = await postJSON('/gaps/close', { query: q, status: 'dismissed' })
    .catch(e => ({ ok: false, error: '' + e }));
  $('#banner').innerHTML = `<div class="note">${r.ok
    ? '✓ dismissed ' + (r.closed || 0) + ' gap(s)'
    : '✗ ' + esc(r.error || 'failed — auth token?')}</div>`;
  if (r.ok) load('gaps');
}

const BAND_COLOR = { high: '#22aa66', medium: '#e0a800', low: '#f5a623', contra: '#d9534f', none: '#888' };

function fchip(d, color) {
  return `<span class="badge" style="background:${color}22;border-color:${color}66">${esc(d.feature)}: ${esc(d.value)}</span>`;
}
function renderCriteria(c) {
  if (!c) return '';
  let h = '<div class="crit">';
  if (c.required && c.required.length) h += `<div class="crit-row"><b>must have</b>${c.required.map(d => fchip(d, '#22aa66')).join(' ')}</div>`;
  if (c.supportive && c.supportive.length) h += `<div class="crit-row"><b>may have</b>${c.supportive.map(d => fchip(d, '#4a90d9')).join(' ')}</div>`;
  if (c.exclusion && c.exclusion.length) h += `<div class="crit-row"><b>must NOT have</b>${c.exclusion.map(d => fchip(d, '#d9534f')).join(' ')}</div>`;
  if (c.threshold) h += `<div class="crit-row"><b>rule</b>${esc(c.threshold)}</div>`;
  if (c.gold_standard) h += `<div class="crit-row"><b>confirm</b>${esc(c.gold_standard)}</div>`;
  if (c.differentials && c.differentials.length) h += `<div class="crit-row"><b>differential</b>${c.differentials.map(d => esc(d.condition) + (d.discriminator ? ' <span style="opacity:.6">(' + esc(d.discriminator) + ')</span>' : '')).join(' · ')}</div>`;
  if (c.levels && c.levels.length) h += `<div class="crit-row"><b>stages</b>${c.levels.map(l => '<b style="opacity:1">' + esc(l.level) + '</b> ' + esc(l.label || '')).join(' · ')}</div>`;
  // typed-card payloads (requirements/decision/playbook/case…) carry OTHER keys — render
  // whatever the rows above didn't handle generically, so no card ever draws empty.
  const known = ['required', 'supportive', 'exclusion', 'threshold', 'gold_standard',
                 'differentials', 'levels'];
  const rest = Object.fromEntries(Object.entries(c).filter(([k, v]) =>
    !known.includes(k) && v != null && v !== '' && (!Array.isArray(v) || v.length)));
  if (Object.keys(rest).length) h += renderPayload(rest);
  return h + '</div>';
}
function renderGrade(g) {
  if (!g) return '';
  return `<div class="src">recommendation: <b>${esc(g.strength || '?')}</b>${g.evidence_quality ? ' · evidence ' + esc(g.evidence_quality) : ''}${g.population ? ' · ' + esc(g.population) : ''}</div>`;
}
function renderFinding(f) {
  if (!f) return '';
  const p = ['study_design', 'direction', 'effect_size', 'n', 'certainty']
    .filter(k => f[k]).map(k => esc((k === 'n' ? 'n=' : '') + f[k]));
  return `<div class="src">finding: ${p.join(' · ')}</div>`;
}
function renderItem(it) {
  const contra = (it.contradictions || []).map(x =>
    `<div class="meta-link warn">⚠ disagrees: ${esc(x.claim)}${x.support && x.support.length ? ' [' + esc(x.support.join(', ')) + ']' : ''}</div>`).join('');
  const steps = (it.steps && it.steps.length) ? '<ol class="steps">' + it.steps.map(s => '<li>' + esc(s) + '</li>').join('') + '</ol>' : '';
  const rf = (it.red_flags && it.red_flags.length) ? `<div class="src" style="color:#d9534f">⚠ red flags: ${it.red_flags.map(esc).join(' · ')}</div>` : '';
  return `<div class="p">
    <div class="meta">${badge(it.kind)}${it.card_type ? badge(it.card_type) : ''}${badge(it.regime || 'empirical')}
      ${it.provenance ? badge(it.provenance) : ''}
      ${it.strength != null ? badge('strength ' + Number(it.strength).toFixed(3)) : ''}
      ${it.score != null ? '<span class="score">score ' + Number(it.score).toFixed(3) + '</span>' : ''}</div>
    <div class="text">${esc(it.text) || esc(it.label)}</div>
    ${renderCriteria(it.criteria)}${steps}${rf}${renderGrade(it.grade)}${renderFinding(it.finding)}
    ${(it.support && it.support.length) ? '<div class="src">support: ' + esc(it.support.join(' · ')) + '</div>' : ''}
    ${contra}
  </div>`;
}
function renderStructure(structure) {
  if (!structure || !structure.length) return '';
  return '<div class="struct"><div class="struct-h">structure — how this connects</div>' + structure.map(s =>
    `<div class="struct-sec"><span class="struct-role">${esc(s.label)}</span>`
    + s.entries.map(e => `<span class="struct-ent" title="${esc(e.text || '')}">`
      + (e.relation ? '<span style="opacity:.55">' + esc(e.relation) + '</span> ' : '')
      + esc(e.label) + (e.has_card ? ' <span style="opacity:.5">[card]</span>' : '') + '</span>').join('')
    + '</div>').join('') + '</div>';
}
function renderBundle(b) {
  const c = BAND_COLOR[b.grounding] || '#888';
  $('#banner').innerHTML =
    `<div class="note" style="background:${c}22;border-color:${c}66">`
    + (b.speech_act ? `read as <b>${esc(b.speech_act)}</b>${b.broaden ? ' <span style="opacity:.7">(ambiguous — showing the map)</span>' : ''} · ` : '')
    + `intent <b>${esc(b.intent)}</b> · rigor <b>${esc(b.rigor)}</b> · confidence ${Number(b.confidence).toFixed(3)} · `
    + `grounding <b>${esc(b.grounding)}</b>${b.abstain ? ' · <b>ABSTAIN</b>' : ''}${b.rights && b.rights.redistributable === false ? ' · <b>not redistributable</b>' : ''}<br>${esc(b.note)}</div>`;
  const items = b.items || [];
  const focus = b.focus ? `<div class="focus"><span class="focus-l">${esc(b.focus.label || '')}</span>`
    + (b.focus.summary ? '<div class="focus-s">' + esc(b.focus.summary) + '</div>' : '') + '</div>' : '';
  const html = focus + items.map(renderItem).join('') + renderStructure(b.structure);
  setRows(items.length || b.focus ? html : '', b.abstain ? 'No grounded answer — logged as a gap.' : 'No items.');
}

async function doAsk() {
  const q = $('#aq').value.trim(); if (!q) return;
  const rigor = $('#rigor') ? $('#rigor').value : '';
  $('#banner').innerHTML = ''; $('#results').className = 'empty'; $('#results').textContent = 'asking…';
  try {
    const res = await (await fetch(`ask?q=${encodeURIComponent(q)}${rigor ? '&rigor=' + rigor : ''}`)).json();
    if (!res.ok) { $('#results').textContent = 'error: ' + (res.error || 'unknown'); return; }
    renderBundle(res);
  } catch (e) { $('#results').textContent = 'request failed: ' + e; }
}

async function doSearch() {
  const q = $('#aq').value.trim(); if (!q) return;
  const k = ($('#ak') && $('#ak').value) || 8;
  $('#banner').innerHTML = ''; $('#results').className = 'empty'; $('#results').textContent = 'searching…';
  try {
    const res = await (await fetch(`search?q=${encodeURIComponent(q)}&k=${k}`)).json();
    if (!res.ok) { $('#results').textContent = 'error: ' + (res.error || 'unknown'); return; }
    $('#banner').innerHTML =
      `<div class="note" style="background:${res.low_confidence ? '#f5a62322' : '#22aa6622'};border-color:${res.low_confidence ? '#f5a62366' : '#22aa6666'}">`
      + `confidence ${Number(res.confidence).toFixed(3)} · ${res.passages.length} passage(s) · dense ${res.dense_used ? 'on' : 'off'}`
      + (res.low_confidence ? ' · <b>low confidence</b>' : '') + '</div>';
    renderPassages(res.passages, true);
  } catch (e) { $('#results').textContent = 'request failed: ' + e; }
}

// ── the unified Ask surface: one query box, three modes ─────────────────────
let ASK_MODE = 'answer', ASK_Q = '';

function askExtras() {
  if (ASK_MODE === 'passages') {
    return `<input id="ak" type="number" value="8" min="1" max="50" style="width:64px" title="passages">`;
  }
  if (ASK_MODE === 'library') {
    const opts = ((typeof LIBCFG !== 'undefined' && LIBCFG && LIBCFG.subdirs) || [])
      .map(s => `<option value="${esc(s.name)}">`).join('');
    return `<input id="acoll" list="acolls" placeholder="collection (all)" style="width:150px"
              title="restrict to a library collection"><datalist id="acolls">${opts}</datalist>
            <input id="ak" type="number" value="8" min="1" max="50" style="width:64px" title="results">`;
  }
  return `<select id="rigor" title="rigor"><option value="">auto rigor</option>
     <option value="low">low</option><option value="high">high (stakes)</option></select>`;
}

function askEmptyState() {
  $('#results').className = 'empty';
  $('#results').textContent =
    ASK_MODE === 'passages' ? 'Ranked passages from the ingested corpus (dense + sparse + rerank, with scores).'
    : ASK_MODE === 'library' ? 'Lexical search over the indexed document library (configure it under Settings → Library).'
    : 'Ask the structured KB a what / how / why question.';
}

function askSetMode(m) {
  if ($('#aq')) ASK_Q = $('#aq').value;   // the query survives a mode switch
  ASK_MODE = m;
  renderBar('ask');
  askEmptyState();
  $('#banner').innerHTML = '';
  if ($('#aq')) $('#aq').focus();
}

function doAskGo() {
  ASK_Q = $('#aq').value;
  if (ASK_MODE === 'passages') return doSearch();
  if (ASK_MODE === 'library') return doLibraryAsk();
  return doAsk();
}

async function doLibraryAsk() {
  const q = $('#aq').value.trim(); if (!q) return;
  const k = ($('#ak') && $('#ak').value) || 8;
  const coll = ($('#acoll') && $('#acoll').value.trim()) || '';
  $('#banner').innerHTML = ''; $('#results').className = 'empty'; $('#results').textContent = 'searching the library…';
  let url = `library?q=${encodeURIComponent(q)}&k=${k}`;
  if (coll) url += `&collection=${encodeURIComponent(coll)}`;
  const t0 = performance.now();
  let res;
  try { res = await (await fetch(url)).json(); }
  catch (e) { $('#results').textContent = 'request failed: ' + e; return; }
  const ms = Math.round(performance.now() - t0);
  if (!res.ok) {
    $('#results').textContent = 'error: ' + (res.error || 'library not configured — Settings → Library');
    return;
  }
  const ps = res.passages || [];
  $('#banner').innerHTML =
    `<div class="note" style="background:${res.low_confidence ? '#f5a62322' : '#22aa6622'};border-color:${res.low_confidence ? '#f5a62366' : '#22aa6666'}">`
    + `confidence ${Number(res.confidence).toFixed(3)} · ${ps.length} passage(s) · ${ms} ms`
    + (res.low_confidence ? ' · <b>no match</b>' : '') + '</div>';
  // the collection rides the source_type badge slot — same card, no new renderer
  renderPassages(ps.map(p => Object.assign({}, p, { source_type: p.collection || 'library' })), true);
}

// ── control panel: Operations + Settings (auth-gated) ────────────────────────
const TOKKEY = 'kh_token';
const tok = () => localStorage.getItem(TOKKEY) || '';
const setTok = v => localStorage.setItem(TOKKEY, v || '');
function authFetch(path, opts = {}) {
  const h = Object.assign({}, opts.headers || {});
  const t = tok(); if (t) h['Authorization'] = 'Bearer ' + t;
  return fetch(path, Object.assign({}, opts, { headers: h }));
}
function postJSON(path, body) {
  return authFetch(path, { method: 'POST', headers: { 'Content-Type': 'application/json' },
                           body: JSON.stringify(body || {}) }).then(r => r.json());
}
const tokInput = () => `<input id="tok" type="password" placeholder="auth token" value="${esc(tok())}"
  title="auth token — saved in this browser only" style="width:130px" onchange="setTok(this.value)">`;

let OPSPEC = {}, OPHELP = {}, opsTimer = null, SETVALS = {};

function renderOpOptions(cmd) {
  const spec = OPSPEC[cmd] || {}, help = OPHELP[cmd] || {};
  const fields = Object.entries(spec).map(([opt, t]) => {
    const id = 'op_' + opt, h = esc(help[opt] || '');
    if (t === 'bool') return `<label class="op" style="font-size:13px" title="${h}"><input type="checkbox" id="${id}"> ${opt}</label>`;
    if (t === 'int') return `<label class="op" style="font-size:13px" title="${h}">${opt} <input type="number" id="${id}" style="width:64px" title="${h}"></label>`;
    if (t === 'float') return `<label class="op" style="font-size:13px" title="${h}">${opt} <input type="number" step="any" id="${id}" style="width:76px" title="${h}"></label>`;
    if (t === 'str') return `<label class="op" style="font-size:13px" title="${h}">${opt} <input id="${id}" style="width:120px" title="${h}"></label>`;
    if (t === 'list') return `<label class="op" style="font-size:13px" title="${h}">${opt} <input id="${id}" style="width:220px" placeholder="comma-separated" title="${h}"></label>`;
    if (t === 'path') return `<label class="op" style="font-size:13px" title="${h}">${opt} <input id="${id}" style="width:260px" placeholder="(defaults to the path in Settings)" title="${h}"></label>`;
    if (t.startsWith('choice:')) return `<label class="op" style="font-size:13px" title="${h}">${opt} <select id="${id}" title="${h}">`
      + t.split(':')[1].split(',').map(v => `<option>${v}</option>`).join('') + `</select></label>`;
    return '';
  }).join(' ');
  const summary = (help._ || '');
  return (summary ? `<span style="opacity:.6;font-size:12px">${esc(summary)} — </span>` : '') + fields;
}
function onCmdChange() { const c = $('#opcmd'); if (c) $('#opopts').innerHTML = renderOpOptions(c.value); }
function gatherArgs() {
  const cmd = $('#opcmd').value, spec = OPSPEC[cmd] || {}, args = {};
  for (const [opt, t] of Object.entries(spec)) {
    const el = $('#op_' + opt); if (!el) continue;
    if (t === 'bool') { if (el.checked) args[opt] = true; }
    else if (t === 'int') { if (el.value !== '') args[opt] = parseInt(el.value, 10); }
    else if (el.value !== '') args[opt] = el.value;   // blank text/float/path = omit
  }
  return { command: cmd, args };
}
function healthStrip(h) {
  if (!h) return '';
  const busy = h.lease_fast || h.lease_big;
  const c = busy ? '#e0a800' : '#22aa66';
  return `<span class="badge" style="background:${c}33;border-color:${c}66">GPU `
    + (busy ? 'busy — Vinkona' + (h.lease_fast ? ' · fast' : '') + (h.lease_big ? ' · big' : '') : 'free') + '</span>';
}
async function pollOps() {
  let r; try { r = await (await authFetch('/ops/log?tail=400')).json(); } catch (e) { return; }
  if (!r.ok) { $('#opstatus').textContent = (r.error === 'unauthorized')
    ? 'enter the auth token above to use Operations' : ('error: ' + r.error); return; }
  const s = r.status || {};
  const run = s.running ? `▶ ${esc(s.command)} ${esc((s.argv || []).join(' '))} — ${s.elapsed_s}s`
    : (s.command ? `■ ${esc(s.command)} finished (exit ${s.exit_code})` : 'idle — pick a command and Run');
  $('#opstatus').innerHTML = run + ' &nbsp; ' + healthStrip(r.health);
  const log = $('#opslog'); if (log) { const atBottom = log.scrollTop + log.clientHeight >= log.scrollHeight - 20;
    log.textContent = r.log || '(no output yet)'; if (atBottom) log.scrollTop = log.scrollHeight; }
}
async function loadOps() {
  $('#results').className = '';
  $('#results').innerHTML = `<div id="opstatus" style="margin:4px 0;font-size:13px">…</div>
    <pre id="opslog" style="background:#0c0c0c;color:#d8d8d8;padding:10px;border-radius:6px;
      height:46vh;overflow:auto;font-size:12px;white-space:pre-wrap;margin:0"></pre>`;
  let r; try { r = await (await authFetch('/ops/status')).json(); } catch (e) { $('#opstatus').textContent = 'request failed: ' + e; return; }
  if (!r.ok) { $('#opstatus').textContent = 'enter the auth token above to use Operations'; return; }
  OPSPEC = r.commands || {}; OPHELP = r.help || {};
  const sel = $('#opcmd');
  if (sel) { sel.innerHTML = Object.keys(OPSPEC).sort().map(c => `<option>${c}</option>`).join(''); onCmdChange(); }
  pollOps();
}
async function runOp() {
  const body = gatherArgs();
  $('#opstatus').textContent = 'launching ' + body.command + '…';
  const r = await postJSON('/ops/run', body).catch(e => ({ ok: false, error: '' + e }));
  if (!r.ok) { $('#opstatus').textContent = '✗ ' + (r.error || 'failed'); return; }
  pollOps();
}
async function stopOp() { await postJSON('/ops/stop').catch(() => {}); pollOps(); }
async function reloadKB() {
  $('#opstatus').textContent = 'reloading KB caches…';
  const r = await postJSON('/ops/reload').catch(e => ({ ok: false, error: '' + e }));
  $('#opstatus').textContent = r.ok ? '✓ KB reloaded' : ('✗ ' + (r.error || 'failed'));
}

// ── Serving: which models THIS box hosts, weights-on-disk, swap control ─────
function svChip(txt, c) {
  return `<span class="badge" style="background:${c}33;border-color:${c}66">${esc(txt)}</span>`;
}
function svState(s) {
  if (s === 'up') return svChip('up', '#22aa66');
  if (s === 'standby') return svChip('standby', '#888888');
  if (s === 'failed') return svChip('FAILED', '#cc4444');
  if (s === 'dead') return svChip('dead', '#cc4444');
  if (s === 'supervisor-down') return svChip('supervisor down', '#e0a800');
  return svChip(s || '?', '#888888');
}
function svWeights(w) {
  if (!w) return '';
  const c = w.status === 'ready' ? '#22aa66' : w.status === 'incomplete' ? '#e0a800' : '#cc4444';
  const t = 'weights: ' + w.status + (w.size_gb ? ` · ${w.size_gb} GB` : '');
  const tip = (w.path || '') + (w.detail ? ' — ' + w.detail : '');
  return `<span class="badge" title="${esc(tip)}" style="background:${c}33;border-color:${c}66">${esc(t)}</span>`;
}
async function doSwap(name) {
  $('#banner').innerHTML = `swapping to <b>${esc(name)}</b> — weights load; this can take minutes…`;
  const r = await postJSON('/serving/swap', { name }).catch(e => ({ ok: false, error: '' + e }));
  if (!r.ok) $('#banner').innerHTML = '✗ ' + esc(r.error || 'swap request failed');
  pollServing();
}
async function pollServing() {
  let r; try { r = await (await authFetch('/serving/status')).json(); } catch (e) { return; }
  if (!r.ok) { $('#results').className = 'empty';
    $('#results').textContent = (r.error === 'unauthorized')
      ? 'enter the auth token above to view Serving' : ('error: ' + r.error); return; }
  $('#results').className = '';
  if (!r.hosting) {
    $('#results').innerHTML = `<p style="opacity:.7">This box hosts <b>no models</b> — the
      <code>[serving]</code> table in config.toml is empty, so the knowledge host only answers
      queries and borrows LMs from the endpoints in Settings (distill/extract/verify URLs).
      To serve models here, declare them in <code>[serving]</code> and start with
      <code>./vinur.sh</code> — see <code>serving/README.md</code>.</p>`;
    return;
  }
  const sup = r.supervisor || {};
  const sw = r.swap || {};
  let banner = sup.running ? '' :
    `⚠ the process supervisor is not running — states below are from disk only. Start with <code>./vinur.sh start</code>.`;
  if (sw.status === 'swapping') banner = `⏳ swap in progress → <b>${esc(sw.request || '')}</b> (weights loading)`;
  else if (sw.status === 'error') banner = `✗ last swap failed: ${esc(sw.error || '')}`;
  $('#banner').innerHTML = banner;
  const rows = (r.llms || []).map(m => {
    const role = m.exclusive ? (m.default ? 'exclusive · boots' : 'exclusive') : 'resident';
    const canSwap = sup.running && m.exclusive && m.service === 'standby' && sw.status !== 'swapping';
    const act = canSwap ? `<button class="toolbtn" onclick="doSwap('${esc(m.name)}')">Swap in</button>` : '';
    const note = (m.hint ? '💡 ' + m.hint + ' · ' : '')
      + (m.reason || m.last_log || (m.weights && m.weights.detail) || '');
    return `<tr><td><b>${esc(m.name)}</b></td>
      <td>${esc(m.model)}<br><span style="opacity:.6">${esc(m.engine)} · :${m.port} · ${role}</span></td>
      <td>${svState(m.service)}</td><td>${svWeights(m.weights)}</td>
      <td style="max-width:340px;font-size:12px;opacity:.8">${esc(note)}</td><td>${act}</td></tr>`;
  }).join('');
  const aux = [];
  if (r.embed && r.embed.enabled) aux.push({ name: 'embed', port: r.embed.port, s: r.embed });
  if (r.reranker && r.reranker.enabled) aux.push({ name: 'reranker', port: '', s: r.reranker });
  const auxRows = aux.map(a => `<tr><td>${esc(a.name)}</td>
      <td><span style="opacity:.6">${a.port ? ':' + a.port : ''}</span></td>
      <td>${svState(a.s.service)}</td><td>${svWeights(a.s.weights)}</td>
      <td style="font-size:12px;opacity:.8">${esc(a.s.last_log || (a.s.weights && a.s.weights.detail) || '')}</td><td></td></tr>`).join('');
  $('#results').innerHTML =
    `<p style="margin:4px 0 10px">This box <b>hosts models</b> for the knowledge host`
    + (sup.running ? ` — supervisor pid ${sup.pid}.` : '.')
    + ` Weight chips show the on-disk state: <i>incomplete</i> during a download <b>and</b> after a
       failed one (the note column carries the service's last log line — a crash there plus
       incomplete weights usually means the fetch died: gated repo token, disk, network).</p>
     <table><tr><th>model</th><th>what</th><th>service</th><th>weights</th><th>note</th><th></th></tr>
     ${rows}${auxRows}</table>`;
}
async function loadServing() {
  $('#banner').innerHTML = ''; $('#results').className = ''; $('#results').textContent = 'loading serving state…';
  pollServing();
}

// ── document library: toggle which subfolders of the trusted root get indexed ──
let LIBCFG = {};
async function loadLibrary() {
  $('#banner').innerHTML = ''; $('#results').className = ''; $('#results').textContent = 'loading library…';
  let r; try { r = await (await authFetch('/library/config')).json(); }
  catch (e) { $('#results').textContent = 'request failed: ' + e; return; }
  if (!r.ok) { $('#results').className = 'empty'; $('#results').textContent = 'enter the auth token above to manage the Library'; return; }
  LIBCFG = r;
  const rootForm = (msg) => `<div style="padding:12px;line-height:1.6">${msg}<br>
      <div style="display:flex;gap:6px;flex-wrap:wrap;margin-top:8px;max-width:560px">
        <input id="libroot" type="text" value="${esc(r.root || '')}"
               placeholder="/absolute/path/on/the/server (e.g. /data/library)"
               style="flex:1;min-width:260px" onkeydown="if(event.key==='Enter')saveLibraryRoot()">
        <button class="toolbtn" onclick="saveLibraryRoot()">Set root</button>
      </div>
      <span style="opacity:.6;font-size:12px">An existing folder on the machine running Vinur.
      Its immediate subfolders become tickable search collections.</span></div>`;
  if (!r.root) {
    $('#results').className = 'empty';
    $('#results').innerHTML = rootForm('No library root is set yet — point one here:');
    return;
  }
  if (!r.root_exists) {
    $('#results').className = 'empty';
    $('#results').innerHTML = rootForm(
      `library_root <code>${esc(r.root)}</code> is not a directory on the server — fix it:`);
    return;
  }
  const rows = (r.subdirs || []).map(s =>
    `<tr><td><label style="cursor:pointer"><input type="checkbox" class="libchk" value="${esc(s.name)}" ${s.active ? 'checked' : ''}> <code>${esc(s.name)}</code></label></td>`
    + `<td style="opacity:.55">${s.active ? 'indexed' : ''}</td></tr>`).join('')
    || '<tr><td colspan=2 style="opacity:.5">no subfolders under the root</td></tr>';
  $('#results').innerHTML =
    `<div style="margin:6px 0 10px;font-size:13px">Trusted root <code>${esc(r.root)}</code> `
    + `<a style="cursor:pointer;text-decoration:underline;font-size:12px" `
    + `onclick="this.parentNode.insertAdjacentHTML('afterend', rootChangeForm()); this.remove()">change…</a> `
    + `<span style="opacity:.6">— tick the subfolders (each becomes a search <b>collection</b>), then <b>Save + index now</b>.</span></div>`
    + `<table><tr><th>subfolder / collection</th><th></th></tr>${rows}</table>`
    + `<div class="hint" style="margin-top:14px;font-size:13px;opacity:.7">Search the indexed library from `
    + `<a style="cursor:pointer;text-decoration:underline" onclick="go('libsearch')">Ask → Library</a> `
    + `— the exact ranked path Vinkona's research loop calls (BM25${r && r.dense ? ' + dense' : ''} → rerank), with latency in the banner.</div>`;
}
function rootChangeForm() {
  return `<div style="display:flex;gap:6px;flex-wrap:wrap;margin:6px 0 10px;max-width:560px">
    <input id="libroot" type="text" value="${esc((LIBCFG && LIBCFG.root) || '')}"
           style="flex:1;min-width:260px" onkeydown="if(event.key==='Enter')saveLibraryRoot()">
    <button class="toolbtn" onclick="saveLibraryRoot()">Set root</button></div>`;
}
async function saveLibraryRoot() {
  const root = ($('#libroot') || {}).value || '';
  $('#banner').innerHTML = 'saving root…';
  const r = await postJSON('/library/root', { root: root.trim() })
    .catch(e => ({ ok: false, error: '' + e }));
  if (!r.ok) { $('#banner').innerHTML = `<span style="color:#c00">✗ ${esc(r.error || 'failed')}</span>`; return; }
  $('#banner').innerHTML = '<span style="color:#0a0">✓ root saved — tick subfolders, then Save + index now</span>';
  loadLibrary();
}
async function saveLibrarySelection(index) {
  const active = Array.from(document.querySelectorAll('.libchk:checked')).map(c => c.value);
  $('#banner').innerHTML = 'saving…';
  const r = await postJSON('/library/config', { active }).catch(e => ({ ok: false, error: '' + e }));
  if (!r.ok) { $('#banner').innerHTML = `<span style="color:#c00">✗ ${esc(r.error || 'failed')}</span>`; return; }
  if (index) {
    const j = await postJSON('/ops/run', { command: 'ingest-library', args: {} })
      .catch(e => ({ ok: false, error: '' + e }));
    $('#banner').innerHTML = j.ok
      ? '<span style="color:#0a0">✓ saved — indexing started (watch it in Operations)</span>'
      : `<span style="color:#c00">saved, but indexing did not start: ${esc(j.error || 'failed')}</span>`;
  } else {
    $('#banner').innerHTML = '<span style="color:#0a0">✓ saved — “Save + index now” (or Operations → ingest-library) to (re)index</span>';
  }
  loadLibrary();
}

// ── modular knowledge: Bundles (groups, scenarios, source rename/regroup) ────
let BUNDLES = {};
async function loadBundles() {
  $('#banner').innerHTML = ''; $('#results').className = ''; $('#results').textContent = 'loading bundles…';
  let r; try { r = await (await authFetch('/bundles')).json(); } catch (e) { $('#results').textContent = 'request failed: ' + e; return; }
  if (!r.ok) { $('#results').className = 'empty'; $('#results').textContent = 'enter the auth token above to manage Bundles'; return; }
  BUNDLES = r;
  const scen = $('#scensel');
  if (scen) {
    const names = Array.from(new Set(['all', ...Object.keys(r.scenarios || {})]));
    scen.innerHTML = names.map(n => `<option ${n === r.active ? 'selected' : ''}>${esc(n)}</option>`).join('');
  }
  const enc = new Set(r.encrypted_bundles || []);
  const off = new Set(r.unloaded || []);
  const grp = (r.bundles || []).map(b => {
    const loaded = !off.has(b.bundle);
    return `<tr><td><code>${esc(b.bundle)}</code>${enc.has(b.bundle) ? ' 🔒' : ''}</td>`
      + `<td>${b.sources}</td>`
      + `<td>${loaded ? '<span style="color:#0a0">loaded</span>' : '<span style="opacity:.5">unloaded</span>'}</td>`
      + `<td><button class="toolbtn" onclick="brainToggle('${esc(b.bundle)}', ${loaded ? 'false' : 'true'})">`
      + `${loaded ? 'unload' : 'load'}</button> `
      + `<button class="toolbtn" title="export to .kdb, then permanently remove from the master (shared rows survive)" `
      + `onclick="ejectBundle('${esc(b.bundle)}')">eject…</button></td></tr>`;
  }).join('')
    || '<tr><td colspan=4 style="opacity:.5">no sources yet</td></tr>';
  const scenRows = Object.entries(r.scenarios || {}).map(([n, d]) =>
    `<tr><td><code>${esc(n)}</code>${n === r.active ? ' <span class="badge">active</span>' : ''}</td>`
    + `<td style="opacity:.7">${d.include ? 'include ' + esc((d.include || []).join(', ')) : ''}`
    + `${d.exclude ? ' exclude ' + esc((d.exclude || []).join(', ')) : ''}</td></tr>`).join('')
    || '<tr><td colspan=2 style="opacity:.5">none defined — edit config.toml [scenarios.NAME]</td></tr>';
  const bopts = (r.bundles || []).map(b => esc(b.bundle));
  const SPDX = ['CC0-1.0','public-domain','CC-BY-4.0','CC-BY-SA-4.0','CC-BY-NC-4.0',
    'CC-BY-NC-SA-4.0','CC-BY-ND-4.0','CC-BY-NC-ND-4.0','MIT','Apache-2.0','GPL-3.0-only',
    'proprietary','all-rights-reserved','unknown'];
  const srcRows = (r.sources || []).map(s => {
    const b = s.bundle || 'base';
    return `<tr><td style="max-width:190px;overflow:hidden;text-overflow:ellipsis" title="${esc(s.doc_id)}"><code>${esc(s.doc_id)}</code></td>`
      + `<td><input data-doc="${esc(s.doc_id)}" data-f="title" value="${esc(s.title || '')}" style="width:150px"></td>`
      + `<td><input data-doc="${esc(s.doc_id)}" data-f="bundle" value="${esc(b)}" list="blist" style="width:96px"></td>`
      + `<td><input data-doc="${esc(s.doc_id)}" data-f="license" value="${esc(s.license || '')}" list="spdxlist" placeholder="unknown" style="width:130px"></td>`
      + `<td><input data-doc="${esc(s.doc_id)}" data-f="license_holder" value="${esc(s.license_holder || '')}" placeholder="licensor" style="width:150px"></td>`
      + `<td><button class="toolbtn" onclick="saveSource('${esc(s.doc_id)}')">save</button></td></tr>`;
  }).join('') || '<tr><td colspan=6 style="opacity:.5">no sources</td></tr>';
  $('#results').innerHTML =
    `<datalist id="blist">${bopts.map(b => `<option value="${b}">`).join('')}</datalist>`
    + `<datalist id="spdxlist">${SPDX.map(b => `<option value="${b}">`).join('')}</datalist>`
    + `<div style="opacity:.6;margin-bottom:8px">serving <b>${esc(r.active)}</b>`
    + (r.modular ? ` — working DB <code>${esc((r.working || '').split('/').pop())}</code>` : ' — master (no scenario)')
    + `</div>`
    + `<div style="display:flex;gap:24px;flex-wrap:wrap;align-items:flex-start">`
    + `<div><h4 style="margin:4px 0">Brains (bundles)</h4>`
    + `<table><tr><th>bundle</th><th>sources</th><th>state</th><th></th></tr>${grp}</table>`
    + `<div style="opacity:.6;font-size:.85em;max-width:420px;margin-top:4px">load/unload is instant and`
    + ` non-destructive (reassembles the working DB); eject exports the bundle to its .kdb first,`
    + ` then removes it from the master — re-import the file to undo.</div>`
    + `<div style="margin-top:8px"><input id="brainfile" placeholder="/path/to/brain.kdb" style="width:220px">`
    + ` <input id="brainname" placeholder="name (optional)" style="width:110px">`
    + ` <label title="cap the brain's trust to 'low' (recommended for shipped files); keep = trust its own values">`
    + `<input type="checkbox" id="braintrust"> keep trust</label>`
    + ` <button class="toolbtn" onclick="importBrain()">import brain…</button></div></div>`
    + `<div><h4 style="margin:4px 0">Scenarios</h4><table><tr><th>name</th><th>rule</th></tr>${scenRows}</table></div>`
    + `</div>`
    + `<h4 style="margin:14px 0 4px">Sources — rename, assign a bundle, set the licence &amp; licensor (writes to master)</h4>`
    + `<table><tr><th>doc_id</th><th>title</th><th>bundle</th><th>licence</th><th>licensor (to whom)</th><th></th></tr>${srcRows}</table>`;
}
async function saveSource(doc) {
  const g = f => document.querySelector(`[data-doc="${CSS.escape(doc)}"][data-f="${f}"]`).value;
  const r = await postJSON('/source', { doc_id: doc, title: g('title'), bundle: g('bundle'),
    license: g('license'), license_holder: g('license_holder') }).catch(e => ({ ok: false, error: '' + e }));
  $('#banner').innerHTML = `<div class="note">${r.ok ? '✓ saved ' + esc(doc) + ' — ' + esc(r.note || '') : '✗ ' + esc(r.error || 'failed')}</div>`;
  if (r.ok) loadBundles();
}
async function brainToggle(name, load) {
  $('#banner').innerHTML = `<div class="note">${load ? 'loading' : 'unloading'} '${esc(name)}' — reassembling…</div>`;
  const r = await postJSON('/brain', { action: load ? 'load' : 'unload', brain: name })
    .catch(e => ({ ok: false, error: '' + e }));
  $('#banner').innerHTML = `<div class="note">${r.ok ? '✓ ' + esc(r.note || 'done')
    + (r.persisted === false ? ' — <b>not persisted</b> (no config file)' : '')
    : '✗ ' + esc(r.error || 'failed')}</div>`;
  if (r.ok) { loadBundles(); refreshStats(); }
}
async function importBrain() {
  const path = $('#brainfile').value.trim();
  if (!path) { $('#banner').innerHTML = '<div class="note">✗ enter the .kdb path (on the host box)</div>'; return; }
  const args = { path };
  const name = $('#brainname').value.trim();
  if (name) args.name = name;
  if ($('#braintrust').checked) args.trust = 'keep';
  $('#banner').innerHTML = '<div class="note">importing — watch Operations for progress…</div>';
  const r = await postJSON('/ops/run', { command: 'import-bundle', args })
    .catch(e => ({ ok: false, error: '' + e }));
  $('#banner').innerHTML = `<div class="note">${r.ok
    ? '✓ import started — see Operations; when it finishes, the brain appears here (load it to serve it)'
    : '✗ ' + esc(r.error || 'failed')}</div>`;
}
async function ejectBundle(name) {
  if (!confirm(`Eject '${name}' from the master?\n\nIts closure is exported to ${name}.kdb first, `
    + `so re-importing that file undoes this. Shared rows survive with the ejected provenance `
    + `stripped.\n\n(Want counts first? Cancel, then run eject-bundle with dry_run in Operations.)`))
    return;
  const r = await postJSON('/ops/run', { command: 'eject-bundle', args: { bundle: name } })
    .catch(e => ({ ok: false, error: '' + e }));
  $('#banner').innerHTML = `<div class="note">${r.ok
    ? '✓ eject started — see Operations; reload this tab when it finishes'
    : '✗ ' + esc(r.error || 'failed')}</div>`;
}
async function applyScenario() {
  const name = $('#scensel') ? $('#scensel').value : '';
  $('#banner').innerHTML = `<div class="note">reassembling scenario '${esc(name)}' — this may take a moment…</div>`;
  const r = await postJSON('/scenario', { scenario: name }).catch(e => ({ ok: false, error: '' + e }));
  $('#banner').innerHTML = `<div class="note">${r.ok
    ? "✓ now serving '" + esc(r.scenario) + "' — " + (r.counts ? JSON.stringify(r.counts) : '')
    : '✗ ' + esc(r.error || 'failed')}</div>`;
  if (r.ok) { loadBundles(); refreshStats(); }
}

// ── Prioritizer: the autopilot plan (ordered auto-run of maintenance verbs) ──
let APLAN = null, APSPEC = {}, APHELP = {}, ABUNDLES = [], APSTATE = {},
    AMODELS = [], AAUTO = [];
async function loadAutopilot() {
  $('#banner').innerHTML = ''; $('#results').className = ''; $('#results').textContent = 'loading plan…';
  let r; try { r = await (await authFetch('/ops/autopilot')).json(); } catch (e) { $('#results').textContent = 'request failed: ' + e; return; }
  if (!r.ok) { $('#results').className = 'empty'; $('#results').textContent = 'enter the auth token above to use the Prioritizer'; return; }
  APLAN = r.plan; APSPEC = r.commands || {}; APHELP = r.help || {}; ABUNDLES = r.bundles || [];
  AMODELS = r.serving_models || []; AAUTO = r.auto_models || [];
  renderAutopilot(r.state || {});
}
function apArgsFields(cmd, args) {
  // Typed, documented inputs for this command's options — hover any field for
  // what it means.  This replaces the old raw-JSON args box.
  const spec = APSPEC[cmd] || {}, help = APHELP[cmd] || {};
  const fields = Object.keys(spec).map(k => {
    const t = spec[k], v = (args || {})[k], h = esc(help[k] || '');
    let inp;
    if (t === 'bool')
      inp = `<input type="checkbox" data-a="${k}" ${v ? 'checked' : ''} title="${h}">`;
    else if (t === 'int' || t === 'float')
      inp = `<input type="number" ${t === 'float' ? 'step="any"' : ''} data-a="${k}"
                    value="${v === undefined || v === null ? '' : v}" style="width:64px" title="${h}">`;
    else if (t.startsWith('choice:'))
      inp = `<select data-a="${k}" title="${h}"><option value=""></option>`
        + t.split(':')[1].split(',').map(c => `<option ${v === c ? 'selected' : ''}>${c}</option>`).join('')
        + `</select>`;
    else                                       // str | path | list
      inp = `<input data-a="${k}" value="${esc(String(v === undefined || v === null ? '' : v))}"
                    ${k === 'bundle' ? 'list="apblist"' : ''} style="width:100px" title="${h}"
                    ${t === 'list' ? 'placeholder="a,b,c"' : ''}>`;
    return `<label title="${h}" style="display:inline-flex;align-items:center;gap:3px;
                   margin:1px 8px 1px 0;font-size:12px;white-space:nowrap">${k} ${inp}</label>`;
  }).join('');
  return fields || '<span style="opacity:.5;font-size:12px">(no options)</span>';
}
function renderAutopilot(state) {
  APSTATE = state || APSTATE || {};
  const p = APLAN;
  const st = APSTATE.enabled
    ? `<b style="color:#2e7d32">ON</b> — ${esc(APSTATE.running_step || APSTATE.last_reason || 'idle')}`
    : `<b style="color:#999">off</b>`;
  const showModel = AMODELS.length > 0 || (p.steps || []).some(s => s.model);
  const rows = (p.steps || []).map((s, i) => {
    const opts = Object.keys(APSPEC).sort().map(c =>
      `<option ${c === s.command ? 'selected' : ''} title="${esc((APHELP[c] || {})._ || '')}">${c}</option>`).join('');
    return `<tr data-i="${i}">
      <td style="white-space:nowrap">
        <button class="toolbtn" onclick="moveStep(${i},-1)" title="higher priority" ${i === 0 ? 'disabled' : ''}>▲</button>
        <button class="toolbtn" onclick="moveStep(${i},1)" title="lower priority" ${i === p.steps.length - 1 ? 'disabled' : ''}>▼</button>
      </td>
      <td><input type="checkbox" data-f="enabled" ${s.enabled ? 'checked' : ''}></td>
      <td><select data-f="command" onchange="apCmdChanged()" title="${esc((APHELP[s.command] || {})._ || '')}">${opts}</select></td>
      <td style="max-width:380px">${apArgsFields(s.command, s.args)}</td>
      <td><input data-f="min_interval_s" type="number" min="0" value="${s.min_interval_s || 0}" style="width:90px"></td>
      ${showModel ? `<td><input data-f="model" value="${esc(s.model || '')}" list="apmlist"
          style="width:100px" placeholder="${esc(AAUTO[i] ? 'auto: ' + AAUTO[i] : '—')}"
          title="Exclusive [serving] model swapped in before this step. Empty = automatic (the model the verb's LM lane points at — shown greyed); type a name to pin one."></td>` : ''}
      <td><input data-f="label" value="${esc(s.label || '')}" style="width:200px"></td>
      <td><button class="toolbtn" onclick="delStep(${i})">✕</button></td></tr>`;
  }).join('');
  $('#results').innerHTML =
    `<datalist id="apblist">${ABUNDLES.map(b => `<option value="${esc(b)}">`).join('')}</datalist>
     <div style="margin:6px 0 12px;font-size:13px">
       <label><input type="checkbox" id="apEnabled" ${p.enabled ? 'checked' : ''}> <b>Autopilot enabled</b></label>
       &nbsp;·&nbsp; status: ${st}
       <div style="opacity:.65;margin-top:6px">Steps run top-to-bottom by priority; after each, the list is
         re-checked from the top, so a higher step that just gained work (e.g. fresh Vinkona drops) preempts
         the backlog below it. A step that finds <b>no work</b> (or fails) stands aside for one idle re-check
         interval, so the steps below it get their turn — two distill steps with different bundles both run.
         “Min interval” additionally throttles a step; hover any field for what it does.</div>
     </div>
     <label style="font-size:13px"><input type="checkbox" id="apLeases" ${p.respect_leases ? 'checked' : ''}>
       Yield to the assistant (pause while it's using the LMs)</label>
     &nbsp;·&nbsp; <label style="font-size:13px">idle re-check
       <input id="apInterval" type="number" min="5" value="${p.idle_interval_s || 60}" style="width:70px">s</label>
     ${showModel ? `&nbsp;·&nbsp; <label style="font-size:13px" title="Exclusive [serving] models can't co-reside;
       with this on, each step automatically swaps in the model its verb's LM lane points at
       (shown greyed in the model column) before running. Pin a model on a step to override.">
       <input type="checkbox" id="apAutoModels" ${p.auto_models !== false ? 'checked' : ''}>
       Automatic model swapping</label>` : ''}
     <datalist id="apmlist">${AMODELS.map(m => `<option value="${esc(m)}">`).join('')}</datalist>
     <table style="margin-top:10px"><tr><th>order</th><th>on</th><th>command</th><th>arguments</th>
       <th>min interval s</th>${showModel ? '<th>model</th>' : ''}<th>label</th><th></th></tr>${rows}</table>`;
}
function apCmdChanged() { APLAN = _readAutopilotForm(); renderAutopilot(APSTATE); }
function _readAutopilotForm() {
  const steps = [];
  document.querySelectorAll('#results tr[data-i]').forEach(tr => {
    const g = f => tr.querySelector(`[data-f="${f}"]`);
    const cmd = g('command').value, spec = APSPEC[cmd] || {}, args = {};
    tr.querySelectorAll('[data-a]').forEach(inp => {
      const k = inp.dataset.a, t = spec[k] || 'str';
      if (t === 'bool') { if (inp.checked) args[k] = true; }
      else if (t === 'int') { if (inp.value.trim() !== '') args[k] = parseInt(inp.value, 10); }
      else if (t === 'float') { if (inp.value.trim() !== '') args[k] = parseFloat(inp.value); }
      else { if (inp.value.trim() !== '') args[k] = inp.value.trim(); }
    });
    steps.push({ command: cmd, enabled: g('enabled').checked,
                 args, min_interval_s: parseInt(g('min_interval_s').value || '0', 10),
                 model: g('model') ? g('model').value.trim() : '',
                 label: g('label').value });
  });
  const am = $('#apAutoModels');
  return { enabled: $('#apEnabled').checked, respect_leases: $('#apLeases').checked,
           idle_interval_s: parseInt($('#apInterval').value || '60', 10),
           auto_models: am ? am.checked : (APLAN.auto_models !== false), steps };
}
function moveStep(i, d) { const s = APLAN.steps; const j = i + d;
  if (j < 0 || j >= s.length) return; APLAN = _readAutopilotForm();
  [APLAN.steps[i], APLAN.steps[j]] = [APLAN.steps[j], APLAN.steps[i]]; renderAutopilot(APSTATE); }
function delStep(i) { APLAN = _readAutopilotForm(); APLAN.steps.splice(i, 1); renderAutopilot(APSTATE); }
function addAutopilotStep() { APLAN = _readAutopilotForm();
  APLAN.steps.push({ command: 'distill', args: {}, enabled: true, min_interval_s: 0, label: 'new step' });
  renderAutopilot(APSTATE); }
async function saveAutopilot() {
  const plan = _readAutopilotForm();
  const r = await postJSON('/ops/autopilot', { plan }).catch(e => ({ ok: false, error: '' + e }));
  $('#banner').innerHTML = `<div class="note">${r.ok ? '✓ plan saved' : '✗ ' + esc(r.error || 'failed')}</div>`;
  if (r.ok) loadAutopilot();
}

async function loadSettings() {
  $('#banner').innerHTML = ''; $('#results').className = ''; $('#results').textContent = 'loading config…';
  let r; try { r = await (await authFetch('/config')).json(); } catch (e) { $('#results').textContent = 'request failed: ' + e; return; }
  if (!r.ok) { $('#results').className = 'empty'; $('#results').textContent = 'enter the auth token above to edit Settings'; return; }
  const sch = r.schema || {}; SETVALS = r.values || {};
  const rows = Object.keys(sch).sort().map(k => {
    const t = sch[k].type, v = SETVALS[k];
    const inp = t === 'bool'
      ? `<input type="checkbox" data-k="${k}" ${v ? 'checked' : ''}>`
      : `<input data-k="${k}" value="${esc(v)}" style="width:170px">`;
    return `<tr><td><code>${esc(k)}</code></td><td style="opacity:.6">${t}</td><td>${inp}</td>`
      + `<td style="opacity:.45">${esc(sch[k].default)}</td></tr>`;
  }).join('');
  $('#results').innerHTML = `<div style="opacity:.6;margin-bottom:6px">writing to ${esc(r.config_path || '(server started without -c — read-only)')}</div>`
    + `<table><tr><th>key</th><th>type</th><th>value</th><th>default</th></tr>${rows}</table>`;
}
async function saveSettings() {
  const updates = {};
  document.querySelectorAll('#results [data-k]').forEach(el => {
    const k = el.dataset.k;
    const v = el.type === 'checkbox' ? el.checked : el.value;
    const changed = el.type === 'checkbox' ? (v !== !!SETVALS[k]) : (String(v) !== String(SETVALS[k] == null ? '' : SETVALS[k]));
    if (changed) updates[k] = v;
  });
  if (!Object.keys(updates).length) { $('#banner').innerHTML = '<div class="note">no changes</div>'; return; }
  const r = await postJSON('/config', { updates }).catch(e => ({ ok: false, error: '' + e }));
  $('#banner').innerHTML = `<div class="note">${r.ok
    ? '✓ saved ' + Object.keys(r.applied || {}).length + ' setting(s) — ' + esc(r.note || '')
    : '✗ ' + esc(r.error || 'failed')}</div>`;
  if (r.ok) loadSettings();
}

// ── Stats tab: banked telemetry → small-multiple SVG time-series ────────────
// The dataviz contract this follows: ONE axis per panel (util and temp are
// separate panels, never dual-axis), 2px lines, hairline grid, a legend for
// ≥2 series (one series: the title names it), crosshair + tooltip that lists
// EVERY series at the hovered time, and a per-chart data table — the relief
// channel for the two light-mode hues that sit below 3:1 on a light surface.
// Palette = the first four validated categorical slots (light/dark stepped
// via the CSS vars above); text always wears text tokens, never series color.
const STAT_RANGES = [['15 m', 15], ['1 h', 60], ['6 h', 360], ['24 h', 1440], ['7 d', 10080]];
const SLOTS = ['var(--s1)', 'var(--s2)', 'var(--s3)', 'var(--s4)'];
let STAT_MINS = 60, STATS_DATA = null, CHARTREG = {};

async function loadStatsTab(silent) {
  const sel = $('#strange');
  if (sel) STAT_MINS = +sel.value || 60;
  if (!silent) { $('#results').className = 'empty'; $('#results').textContent = 'loading…'; }
  try { STATS_DATA = await (await fetch(`metrics/history?mins=${STAT_MINS}`)).json(); }
  catch (e) { if (!silent) $('#results').textContent = 'request failed: ' + e; return; }
  renderStatsCharts();
}

function deriveRate(pts, perSec) {
  // cumulative counter → rate; a negative delta (restart/reset) becomes a gap
  const out = [];
  for (let i = 1; i < pts.length; i++) {
    const dt = pts[i][0] - pts[i - 1][0], dv = pts[i][1] - pts[i - 1][1];
    if (dt <= 0) continue;
    out.push([pts[i][0], dv < 0 ? null : dv / dt * (perSec ? 1 : 60)]);
  }
  return out;
}

function statPanels(S) {
  const P = [];
  const pick = re => Object.keys(S).filter(k => re.test(k)).sort();
  const gpuPanel = (suffix, title, unit, opts = {}) => {
    const keys = pick(new RegExp('^gpu[0-9]+[.]' + suffix + '$'));
    if (!keys.length) return;
    P.push({ title, unit, ymax: opts.ymax, series: keys.map((k, i) => ({
      name: k.split('.')[0], color: SLOTS[i % 4],
      pts: opts.scale ? S[k].map(pt => [pt[0], pt[1] / opts.scale]) : S[k] })) });
  };
  gpuPanel('util', 'GPU utilisation', '%', { ymax: 100 });
  // VRAM charts against the card's TOTAL, so headroom (the
  // gpu_memory_utilization question) reads at a glance
  let vtot = 0;
  pick(/^gpu[0-9]+[.]vram_total_mb$/).forEach(k =>
    S[k].forEach(pt => { if (pt[1] > vtot) vtot = pt[1]; }));
  gpuPanel('vram_mb', 'VRAM used', 'GB', { scale: 1024, ymax: vtot ? vtot / 1024 : undefined });
  gpuPanel('power_w', 'Power draw', 'W');
  gpuPanel('temp_c', 'Temperature', '°C');
  const q = pick(/^vllm[.].+[.](running|waiting)$/);
  if (q.length) {
    const entries = [...new Set(q.map(k => k.split('.')[1]))].sort();
    P.push({ title: 'vLLM queue', unit: 'requests', series: q.map(k => {
      const parts = k.split('.'), entry = parts[1], which = parts[2];
      return { name: entries.length > 1 ? entry + ' ' + which : which,
               color: SLOTS[(entries.indexOf(entry) * 2 + (which === 'waiting' ? 1 : 0)) % 4],
               pts: S[k] };
    }) });
  }
  const kv = pick(/^vllm[.].+[.]kv_pct$/);
  if (kv.length) P.push({ title: 'KV-cache usage', unit: '%', ymax: 100,
    series: kv.map((k, i) => ({ name: kv.length > 1 ? k.split('.')[1] : 'KV cache',
                                color: SLOTS[i % 4], pts: S[k] })) });
  const tok = pick(/^vllm[.].+[.](gen|prompt)_toks$/);
  if (tok.length) {
    const entries = [...new Set(tok.map(k => k.split('.')[1]))].sort();
    const ser = tok.map(k => {
      const parts = k.split('.'), entry = parts[1], which = parts[2] === 'gen_toks' ? 'generated' : 'prompt';
      return { name: entries.length > 1 ? entry + ' ' + which : which,
               color: SLOTS[(entries.indexOf(entry) * 2 + (which === 'prompt' ? 1 : 0)) % 4],
               pts: deriveRate(S[k], true) };
    }).filter(s => s.pts.length);
    if (ser.length) P.push({ title: 'Token throughput', unit: 'tok/s', series: ser });
  }
  // fixed spec order → fixed slot per entity, even when one series is absent
  const rateSpec = [['kb.distilled', 'chunks distilled'], ['kb.nodes', 'concepts'],
                    ['kb.edges', 'relations'], ['kb.cards', 'cards']];
  const rs = [];
  rateSpec.forEach(([k, n], i) => {
    if (S[k] && S[k].length > 1) rs.push({ name: n, color: SLOTS[i], pts: deriveRate(S[k]) });
  });
  if (rs.length) P.push({ title: 'Distillation throughput', unit: 'per min', series: rs });
  const backSpec = [['kb.merge_q', 'merge queue'], ['kb.gaps', 'open gaps']];
  const bs = [];
  backSpec.forEach(([k, n], i) => {
    if (S[k] && S[k].length) bs.push({ name: n, color: SLOTS[i], pts: S[k] });
  });
  if (bs.length) P.push({ title: 'Curation backlog', unit: 'items', series: bs });
  return P;
}

function niceMax(v) {
  if (!(v > 0)) return 1;
  const p = Math.pow(10, Math.floor(Math.log10(v)));
  for (const m of [1, 2, 2.5, 5, 10]) if (m * p >= v) return m * p;
  return 10 * p;
}
function fmtVal(v) {
  if (v == null) return '—';
  return Math.abs(v) >= 100 ? Math.round(v).toLocaleString()
    : Math.round(v * 10) / 10;
}
function fmtClock(t) {
  const d = new Date(t * 1000), pad = n => String(n).padStart(2, '0');
  const hm = pad(d.getHours()) + ':' + pad(d.getMinutes());
  if (STAT_MINS <= 60) return hm + ':' + pad(d.getSeconds());
  if (STAT_MINS > 2880) return (d.getMonth() + 1) + '/' + d.getDate() + ' ' + hm;
  return hm;
}

function renderStatsCharts() {
  if (active !== 'stats' || !STATS_DATA) return;
  const panels = statPanels(STATS_DATA.series || {});
  const evs = (STATS_DATA.events || []);
  CHARTREG = {};
  if (!panels.length) {
    setRows('', 'No telemetry in this window yet.  The sampler starts with the '
      + 'server (stats_interval_s in Settings; 0 disables it) — if this box '
      + 'predates the Stats feature, pull and restart the kb service.');
    return;
  }
  const shown = evs.slice(-20);
  const chips = evs.length ? '<div class="evchips">'
    + (evs.length > shown.length ? `<span class="badge">+${evs.length - shown.length} earlier</span>` : '')
    + shown.map(e => `<span class="badge" title="${esc(e.kind)}">${esc(fmtClock(e.ts))} ${esc(
        e.kind === 'mark' ? '⚑ ' + e.label
        : (e.kind === 'op_start' ? '▶ ' : '■ ') + e.label)}</span>`).join('')
    + '</div>' : '';
  setRows(chips + '<div class="charts">' + panels.map((p, i) => {
    const legend = p.series.length > 1 ? '<div class="clegend">' + p.series.map(s =>
      `<span><i style="border-color:${s.color}"></i>${esc(s.name)}</span>`).join('') + '</div>' : '';
    return `<div class="chartcard"><div class="ct">${esc(p.title)} <span class="cu">${esc(p.unit)}</span></div>
      ${legend}<svg id="cv${i}" height="150" tabindex="0" role="img" aria-label="${esc(p.title)}"></svg>
      <details class="ctable"><summary>data table</summary><div></div></details></div>`;
  }).join('') + '</div>'
  + renderCompare(STATS_DATA.series || {}, evs, STATS_DATA.now - STAT_MINS * 60, STATS_DATA.now));
  panels.forEach((p, i) => fillChart(p, i, evs));
}

function fillChart(p, i, evs) {
  const svg = $('#cv' + i);
  if (!svg) return;
  const W = Math.max(280, Math.round(svg.clientWidth) || 360), H = 150,
        padl = 46, padr = 10, padt = 8, padb = 20,
        xw = W - padl - padr, yh = H - padt - padb;
  const now = STATS_DATA.now, span = STAT_MINS * 60, t0 = now - span;
  let vmax = 0;
  p.series.forEach(s => s.pts.forEach(pt => {
    if (pt[0] >= t0 && pt[1] != null && pt[1] > vmax) vmax = pt[1]; }));
  const ymax = p.ymax || niceMax(vmax * 1.08);
  const X = t => padl + (t - t0) / span * xw;
  const Y = v => padt + yh - Math.min(v, ymax) / ymax * yh;
  let g = '';
  [0, ymax / 2, ymax].forEach(v => {
    const y = Y(v);
    g += `<line x1="${padl}" y1="${y}" x2="${W - padr}" y2="${y}" style="stroke:var(--cgrid);stroke-width:1"/>`
      + `<text x="${padl - 6}" y="${y + 3.5}" text-anchor="end" style="fill:var(--cmuted);font-size:10px">${fmtVal(v)}</text>`;
  });
  [[t0, 'start'], [t0 + span / 2, 'middle'], [now, 'end']].forEach(([t, anchor]) => {
    g += `<text x="${X(t)}" y="${H - 6}" text-anchor="${anchor}" style="fill:var(--cmuted);font-size:10px">${esc(fmtClock(t))}</text>`;
  });
  evs.forEach(e => {
    if (e.ts >= t0) g += `<line x1="${X(e.ts).toFixed(1)}" y1="${padt}" x2="${X(e.ts).toFixed(1)}" y2="${padt + yh}" style="stroke:var(--caxis);stroke-width:1"/>`;
  });
  p.series.forEach(s => {
    let d = '', pen = false;
    s.pts.forEach(pt => {
      if (pt[0] < t0 || pt[1] == null) { pen = false; return; }
      d += (pen ? 'L' : 'M') + X(pt[0]).toFixed(1) + ' ' + Y(pt[1]).toFixed(1) + ' ';
      pen = true;
    });
    if (d) g += `<path d="${d.trim()}" fill="none" style="stroke:${s.color};stroke-width:2;stroke-linejoin:round;stroke-linecap:round"/>`;
  });
  g += `<line id="cx${i}" y1="${padt}" y2="${padt + yh}" style="stroke:var(--cmuted);stroke-width:1;display:none"/>`;
  svg.setAttribute('viewBox', `0 0 ${W} ${H}`);
  svg.setAttribute('width', W);
  svg.setAttribute('height', H);
  svg.innerHTML = g;
  CHARTREG['cv' + i] = { p, t0, span, padl, xw, W, cursor: null, evs };
  svg.onpointermove = chartMove;
  svg.onpointerleave = chartLeave;
  svg.onfocus = chartFocus;
  svg.onblur = chartLeave;
  svg.onkeydown = chartKey;
  const det = svg.parentElement.querySelector('details.ctable');
  det.ontoggle = () => { if (det.open) buildChartTable(det, p); };
}

function nearestVal(pts, t, tol) {
  let best = null, bd = tol;
  for (const pt of pts) {
    const d = Math.abs(pt[0] - t);
    if (pt[1] != null && d <= bd) { bd = d; best = pt; }
  }
  return best;
}

function showTip(reg, t, cx, cy) {
  // built with textContent — mark labels are user input, never innerHTML
  const tip = $('#cktip');
  tip.textContent = '';
  const head = document.createElement('div');
  head.className = 'tt';
  head.textContent = fmtClock(t);
  tip.appendChild(head);
  const tol = Math.max((STATS_DATA.bucket || 5) * 1.5, reg.span / 120);
  let rows = 0;
  reg.p.series.forEach(s => {
    const pt = nearestVal(s.pts, t, tol);
    if (!pt) return;
    const row = document.createElement('div');
    const key = document.createElement('i');
    key.style.borderTopColor = s.color;
    const v = document.createElement('span');
    v.className = 'tv';
    v.textContent = String(fmtVal(pt[1]));
    const n = document.createElement('span');
    n.className = 'tn';
    n.textContent = s.name;
    row.append(key, ' ', v, n);
    tip.appendChild(row);
    rows++;
  });
  reg.evs.forEach(e => {
    if (e.ts < reg.t0 || Math.abs(e.ts - t) > tol) return;
    const row = document.createElement('div');
    row.className = 'tt';
    row.textContent = (e.kind === 'mark' ? '⚑ ' : e.kind === 'op_start' ? '▶ ' : '■ ') + e.label;
    tip.appendChild(row);
    rows++;
  });
  if (!rows) { tip.style.display = 'none'; return; }
  tip.style.display = 'block';
  tip.style.left = Math.max(4, Math.min(cx + 14, innerWidth - tip.offsetWidth - 8)) + 'px';
  tip.style.top = Math.max(4, Math.min(cy + 14, innerHeight - tip.offsetHeight - 8)) + 'px';
}

function drawCursor(i, reg, t) {
  const ln = $('#cx' + i);
  if (!ln) return;
  const x = (reg.padl + (t - reg.t0) / reg.span * reg.xw).toFixed(1);
  ln.setAttribute('x1', x);
  ln.setAttribute('x2', x);
  ln.style.display = '';
}
function chartMove(ev) {
  const svg = ev.currentTarget, reg = CHARTREG[svg.id];
  if (!reg) return;
  const r = svg.getBoundingClientRect();
  const vx = (ev.clientX - r.left) * (reg.W / r.width);      // client px → viewBox px
  let t = reg.t0 + (vx - reg.padl) / reg.xw * reg.span;
  t = Math.max(reg.t0, Math.min(reg.t0 + reg.span, t));
  reg.cursor = t;
  drawCursor(svg.id.slice(2), reg, t);
  showTip(reg, t, ev.clientX, ev.clientY);
}
function chartLeave(ev) {
  const ln = $('#cx' + ev.currentTarget.id.slice(2));
  if (ln) ln.style.display = 'none';
  $('#cktip').style.display = 'none';
}
function chartFocus(ev) {
  const svg = ev.currentTarget, reg = CHARTREG[svg.id];
  if (!reg) return;
  const t = reg.cursor == null ? reg.t0 + reg.span : reg.cursor;
  reg.cursor = t;
  drawCursor(svg.id.slice(2), reg, t);
  const r = svg.getBoundingClientRect();
  showTip(reg, t, r.left + 60, r.top + 30);
}
function chartKey(ev) {
  if (ev.key !== 'ArrowLeft' && ev.key !== 'ArrowRight') return;
  ev.preventDefault();
  const svg = ev.currentTarget, reg = CHARTREG[svg.id];
  if (!reg) return;
  const step = (STATS_DATA.bucket || 5) * (ev.key === 'ArrowLeft' ? -1 : 1);
  let t = (reg.cursor == null ? reg.t0 + reg.span : reg.cursor) + step;
  t = Math.max(reg.t0, Math.min(reg.t0 + reg.span, t));
  reg.cursor = t;
  drawCursor(svg.id.slice(2), reg, t);
  const r = svg.getBoundingClientRect();
  showTip(reg, t, r.left + 60, r.top + 30);
}

function buildChartTable(det, p) {
  const tol = (STATS_DATA.bucket || 5) / 2 + 0.01;
  let ts = [...new Set(p.series.flatMap(s =>
    s.pts.filter(pt => pt[1] != null).map(pt => pt[0])))].sort((a, b) => a - b);
  if (ts.length > 400) ts = ts.slice(-400);
  const head = '<tr><th>time</th>' + p.series.map(s => `<th>${esc(s.name)}</th>`).join('') + '</tr>';
  const body = ts.map(t => '<tr><td>' + esc(fmtClock(t)) + '</td>' + p.series.map(s => {
    const pt = nearestVal(s.pts, t, tol);
    return '<td>' + (pt ? fmtVal(pt[1]) : '—') + '</td>';
  }).join('') + '</tr>').join('');
  det.querySelector('div').innerHTML =
    `<table>${head}${body}</table>` + (ts.length === 400 ? '<div style="opacity:.5">latest 400 rows</div>' : '');
}

// ── A/B compare: the window partitioned at every mark / job event ───────────
// Workflow: ⚑ mark "A: <what you changed>" → run → ⚑ mark "B: <the change>" →
// run → tick the two intervals and read the Δ row.  Aggregates are computed
// client-side from the same history the charts draw, so the table and the
// charts can never disagree.
let AB_SEL = new Set();            // interval identity = rounded start ts

async function dropMark() {
  const label = prompt('Mark label — what did you just change?  (e.g. distill_parallel=8)');
  if (!label || !label.trim()) return;
  const r = await postJSON('/metrics/mark', { label: label.trim() })
    .catch(e => ({ ok: false, error: '' + e }));
  $('#banner').innerHTML = `<div class="note">${r.ok ? '⚑ marked: ' + esc(r.label)
    : '✗ ' + esc(r.error || 'failed — auth token?')}</div>`;
  if (r.ok) loadStatsTab(true);
}

function abIntervals(evs, t0, now) {
  const inWin = evs.filter(e => e.ts > t0 && e.ts < now);
  const bounds = [t0, ...inWin.map(e => e.ts), now];
  const evAt = {};
  inWin.forEach(e => { evAt[e.ts] = e; });
  const out = [];
  for (let i = 0; i < bounds.length - 1; i++) {
    const a = bounds[i], b = bounds[i + 1], ev = evAt[a];
    if (b - a < 20) continue;                     // slivers carry no signal
    const label = !ev ? 'window start'
      : ev.kind === 'mark' ? '⚑ ' + ev.label
      : ev.kind === 'op_start' ? '▶ ' + ev.label
      : 'after ' + ev.label;
    out.push({ a, b, label });
  }
  return out.slice(-12);
}

function valsIn(pts, a, b) {
  return (pts || []).filter(p => p[0] >= a && p[0] < b && p[1] != null).map(p => p[1]);
}
function meanIn(pts, a, b) {
  const v = valsIn(pts, a, b);
  return v.length ? v.reduce((x, y) => x + y, 0) / v.length : null;
}
function counterRate(pts, a, b, perMin) {
  const w = (pts || []).filter(p => p[0] >= a && p[0] < b && p[1] != null);
  if (w.length < 2) return null;
  const d = w[w.length - 1][1] - w[0][1], dt = w[w.length - 1][0] - w[0][0];
  return (dt <= 0 || d < 0) ? null : d / dt * (perMin ? 60 : 1);
}
function sumSeries(S, re, f) {                     // per-entry values, summed
  let tot = null;
  Object.keys(S).filter(k => re.test(k)).forEach(k => {
    const v = f(S[k]);
    if (v != null) tot = (tot || 0) + v;
  });
  return tot;
}
function meanAcross(S, re, a, b) {                 // mean of per-GPU means
  const ms = Object.keys(S).filter(k => re.test(k))
    .map(k => meanIn(S[k], a, b)).filter(v => v != null);
  return ms.length ? ms.reduce((x, y) => x + y, 0) / ms.length : null;
}

const AB_COLS = [
  ['GPU % avg', (S, a, b) => meanAcross(S, /^gpu[0-9]+[.]util$/, a, b)],
  ['GPU % max', (S, a, b) => {
    const v = Object.keys(S).filter(k => /^gpu[0-9]+[.]util$/.test(k))
      .flatMap(k => valsIn(S[k], a, b));
    return v.length ? Math.max(...v) : null;
  }],
  ['VRAM GB', (S, a, b) => {
    const m = meanAcross(S, /^gpu[0-9]+[.]vram_mb$/, a, b);
    return m == null ? null : m / 1024;
  }],
  ['run', (S, a, b) => sumSeries(S, /^vllm[.].+[.]running$/, pts => meanIn(pts, a, b))],
  ['wait', (S, a, b) => sumSeries(S, /^vllm[.].+[.]waiting$/, pts => meanIn(pts, a, b))],
  ['tok/s', (S, a, b) => sumSeries(S, /^vllm[.].+[.]gen_toks$/, pts => counterRate(pts, a, b))],
  ['dist/min', (S, a, b) => counterRate(S['kb.distilled'], a, b, true)],
  ['cards/min', (S, a, b) => counterRate(S['kb.cards'], a, b, true)],
];

function abKey(iv) { return Math.round(iv.a); }
function fmtDur(s) {
  return s >= 5940 ? Math.round(s / 360) / 10 + ' h'
    : s >= 99 ? Math.round(s / 60) + ' m' : Math.round(s) + ' s';
}

function renderCompare(S, evs, t0, now) {
  const ivs = abIntervals(evs, t0, now);
  if (ivs.length < 2) return '';
  const live = new Set(ivs.map(abKey));
  [...AB_SEL].forEach(k => { if (!live.has(k)) AB_SEL.delete(k); });
  const cells = ivs.map(iv => AB_COLS.map(c => c[1](S, iv.a, iv.b)));
  const rows = ivs.map((iv, i) => {
    const k = abKey(iv);
    return `<tr><td><input type="checkbox"${AB_SEL.has(k) ? ' checked' : ''}
        onchange="abToggle(${k})" style="width:auto" title="tick two rows to diff"></td>
      <td>${esc(iv.label)}</td><td>${esc(fmtClock(iv.a))}</td><td>${fmtDur(iv.b - iv.a)}</td>`
      + cells[i].map(v => `<td>${fmtVal(v)}</td>`).join('') + '</tr>';
  }).join('');
  let delta = '';
  if (AB_SEL.size === 2) {
    const [ka, kb] = [...AB_SEL].sort((x, y) => x - y);   // A = earlier interval
    const ia = ivs.findIndex(iv => abKey(iv) === ka);
    const ib = ivs.findIndex(iv => abKey(iv) === kb);
    delta = '<tr class="abdelta"><td></td><td>Δ B − A</td><td></td><td></td>'
      + AB_COLS.map((c, ci) => {
        const va = cells[ia][ci], vb = cells[ib][ci];
        if (va == null || vb == null) return '<td>—</td>';
        const d = vb - va;
        const pct = va > 0 ? ` (${d >= 0 ? '+' : ''}${Math.round(d / va * 100)}%)` : '';
        return `<td>${d >= 0 ? '+' : ''}${fmtVal(d)}${esc(pct)}</td>`;
      }).join('') + '</tr>';
  }
  return `<div class="chartcard" style="margin-top:14px">
    <div class="ct">A/B compare <span class="cu">the window, split at every ⚑ mark and job event — tick two rows to diff</span></div>
    <div style="overflow-x:auto"><table><tr><th></th><th>interval</th><th>start</th><th>dur</th>`
    + AB_COLS.map(c => `<th>${esc(c[0])}</th>`).join('')
    + `</tr>${rows}${delta}</table></div></div>`;
}

function abToggle(k) {
  if (AB_SEL.has(k)) AB_SEL.delete(k);
  else {
    if (AB_SEL.size >= 2) AB_SEL.delete([...AB_SEL][0]);
    AB_SEL.add(k);
  }
  renderStatsCharts();
}

addEventListener('resize', () => { if (active === 'stats' && STATS_DATA) renderStatsCharts(); });

buildTabs(); refreshStats(); setInterval(refreshStats, REFRESH_MS); loadHelp(); go('ask');
</script>
</body>
</html>
"""
