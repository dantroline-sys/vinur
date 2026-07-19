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

### Stage 2 — Ask unification
One query input kept across modes; mode toggle Answer / Passages / Library
with per-mode extras (rigor · k · collection).  Library *search* moves here
from the Library panel (the panel under Settings keeps root/selection
config).  Renderers unchanged.

### Stage 3 — Distilled overview + statistics
New default sub-view "Overview": count tiles (sources, chunks, % distilled,
concepts, relations, cards) + the ratios that say something (relations per
concept, cards per 100 distilled chunks, merge-queue depth, open gaps) from
the existing `/stats` + `/kb` counts; per-class counts as badges on the
sub-tab pills.  Server change only if a count is missing and cheap.

### Stage 4 — upkeep + curation actions
Overview gains an **Upkeep** box: refine / link / adjudicate / embed-nodes,
each with a when/why note, a "Run now" (existing authed `POST /ops/run`) and
a "Queue as Prioritizer step" (existing `/ops/autopilot`).  Signal chips
drive attention (merge-queue > N → adjudicate; gaps > N → research/export).
Curation tab gets the same treatment for its queue (close-gap action,
schedule-adjudicate) — new server routes only where an action has none.

### Stage 5 — polish
help.json regrouped (group-level intros; the leaf fallback already works),
README panel section, docstring, dead-code sweep, memory notes.

## Rollback
Each stage is a single commit touching viewer.py (+ help.json/server.py only
where a stage says so).  `git revert <stage-commit>` restores the previous
panel wholesale — no data-format or config migrations anywhere in this plan.
