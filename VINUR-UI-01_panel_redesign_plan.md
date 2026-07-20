# VINUR-UI-01 — panel redesign, staged plan

Dan's brief (2026-07-20): the viewer is left-justified and wastes >50% of a
large screen; 15 tabs in a row is too many.  Consolidate: Ask/Search/Library-
query → **Ask**; Raw/Concepts/Relations/Cards/Sources → **Distilled** (class
toggle inside, hierarchical order, basic statistics incl. ratios, tidy-up
actions like scheduling refine/link with when/why notes); Adjudication+Gaps →
one **Curation** tab managing the clean-up queue with scheduling;
Bundles/Library/Prioritizer become subsections of **Settings**.  Staged,
because a big-bang rewrite of a 1000-line single-file panel breeds bugs.

## Target information architecture

15 flat tabs → 6 top-level tabs, two-level nav (group row + sub-tab pills):

| Tab | Sub-views (leaf keys unchanged) | Notes |
|---|---|---|
| **Ask** | Answer (`ask`) · Passages (`search`) · Library (`libsearch`) | Stage 2 unifies into ONE query box with a mode toggle |
| **Distilled** | Overview · Sources (`sources`) · Raw (`raw`) · Concepts (`nodes`) · Relations (`edges`) · Cards (`cards`) | pipeline order = the hierarchy: provenance → raw text → concepts → edges → cards |
| **Curation** | Adjudication (`adjudication`) · Gaps (`gaps`) | the clean-up queue + scheduling |
| **Operations** | (unchanged) | job runner, import formats, datasets |
| **Serving** | (unchanged) | models, weights, swap |
| **Settings** | General (`settings`) · Bundles (`bundles`) · Library (`library`) · Prioritizer (`autopilot`) | config-ish things live together |
| **Stats** | (single view, Stages 6–8) | graphed GPU + pipeline performance for tuning & A/B |

Design invariants that keep the stages safe:
- **Leaf keys never change.**  They key `help.json`, the loaders, and the
  `go()` dispatch — regrouping is nav-only until a stage explicitly merges
  panel logic.
- One stage = one commit = independently shippable + revertable.
- Every stage ends with: extracted-JS `node --check`, `py_compile`,
  standalone_test + swap_test green, and a manual click-through by Dan.

## Stages

### Stage 1 — layout + two-level nav  ✅ (this commit)
Centre the page (`main` + header content in a shared max-width column,
~1240px, wider at ≥1500px), two-column card grid for browse lists on big
screens.  `GROUPS` structure + sub-tab pill row; `go()` resolves a group key
to its remembered (or first) leaf, then dispatches exactly as before — zero
loader changes.  Group and leaf can share a key (`ask`, `settings`): the
group match redirects once, no recursion.

### Stage 2 — Ask unification  ✅
One query input kept across modes; mode toggle Answer / Passages / Library
with per-mode extras (rigor · k · collection).  Library *search* moves here
from the Library panel (the panel under Settings keeps root/selection
config).  Renderers unchanged.

### Stage 3 — Distilled overview + statistics  ✅
New default sub-view "Overview": count tiles (sources, chunks, % distilled,
concepts, relations, cards) + the ratios that say something (relations per
concept, cards per 100 distilled chunks, merge-queue depth, open gaps) from
the existing `/stats` + `/kb` counts; per-class counts as badges on the
sub-tab pills.  Server change only if a count is missing and cheap.

### Stage 4 — upkeep + curation actions  ✅
Overview gains an **Upkeep** box: refine / link / adjudicate / embed-nodes,
each with a when/why note, a "Run now" (existing authed `POST /ops/run`) and
a "Queue as Prioritizer step" (existing `/ops/autopilot`).  Signal chips
drive attention (merge-queue > N → adjudicate; gaps > N → research/export).
Curation tab gets the same treatment for its queue (close-gap action,
schedule-adjudicate) — new server routes only where an action has none.

### Stage 5 — polish
help.json regrouped (group-level intros; the leaf fallback already works),
README panel section, docstring, dead-code sweep, memory notes.

## The Stats page (Dan, 2026-07-20): graphed GPU + performance for tuning & A/B

Design principles: **dependency-free** (inline SVG charts, no libs — the
panel's standing rule); **collector ≠ UI** (a sampler that is so cheap it can
always be on, banking history whether or not anyone is looking); telemetry
lives in its own `var/metrics.db`, never inside kb.db; everything degrades
gracefully (no nvidia-smi → no GPU series; no vLLM → no queue series; the
page shows what it has and says what it lacks).

### Stage 6 — metrics collector + history API (server only, no UI)  ✅
`knowledgehost/metrics.py`: a sampler thread in the kb server (config
`stats_interval_s`, default 5, 0 = off; `stats_keep_days`, default 14 —
both on the Settings allowlist).  Each tick collects:
- **nvidia-smi** (if present): per-GPU utilisation %, VRAM used/total,
  power draw, temperature (one `--query-gpu … --format=csv` subprocess).
- **vLLM /metrics** from every `[[serving.llms]]` entry with engine
  vllm/container that answers: requests **running**, requests **waiting**
  (the queue Dan wants to watch), KV-cache usage %, token throughput
  counters where the version exposes them (parse defensively — metric
  names drift between vLLM releases).  llama.cpp `/metrics` when enabled.
- **KB counts** (cached `kb.counts()`): chunks, distilled, nodes, edges,
  cards, merge-queue, gaps — server-side history means rates over ANY
  window, not the browser's 60-second rolling view.
- **Ops-runner transitions** → an events table: `op_start`/`op_end` with
  exit code + the parsed OPS_RESULT stats.  The Prioritizer's steps write
  the same events (SQLite WAL is fine cross-process).
Schema: `samples(ts, series, value)` + `events(ts, kind, label, data)`,
indexed (series, ts); hourly retention prune.  Routes: `GET
/metrics/history?mins=&step=` (server-side bucket downsample, ≤ ~600
points/series) and authed `POST /metrics/mark {label}`.
Tests: stubbed nvidia-smi on PATH, stubbed /metrics HTTP endpoint,
downsample maths, retention, mark-route auth.

### Stage 7 — Stats tab (UI)  ✅
7th top-level tab (leafless, like Operations).  Inline-SVG time-series
panels: GPU util + temp · VRAM · power · vLLM running/waiting · KV-cache %
· derived throughput (chunks/cards/edges per minute from count deltas).
Range picker (15m/1h/6h/24h/7d), auto-refresh while the tab is active
(same pattern as Ops polling), event marks drawn as labelled vertical
lines on every chart.

### Stage 8 — A/B compare  ✅
"Drop mark" button (label = what you just changed, e.g.
`distill_parallel=8`).  A compare table lists the intervals bounded by
marks/op events with per-interval aggregates — duration, avg/max GPU
util, avg VRAM, avg queue depth, cards/min, chunks/min, tokens/s where
available — and any two intervals selected show a delta column.  The A/B
workflow is then: mark A → run under config X → mark B → run under
config Y → read the table.

### Recommended order
**Stage 6 first, before Stages 2–5.**  History only exists from the moment
the collector ships — every tuning day before that is unlogged — and two
A/B questions are already queued (the distill fan-out just landed;
FlashInfer-vs-Triton is pending).  Then 7 → 8, then back to 2–5.  The two
tracks touch different code (metrics.py + server routes vs viewer panels),
so nothing blocks on this choice.

## Rollback
Each stage is a single commit touching viewer.py (+ help.json/server.py only
where a stage says so).  `git revert <stage-commit>` restores the previous
panel wholesale — no data-format or config migrations anywhere in this plan.
