# VIN-LIT-01 — Literary Distillation Layer Contract

**Status:** Adopted 2026-09-01 with §0 adaptations (was: draft for commissioning)
**Applies to:** Vinur (knowledgehost) ingestion pipeline
**Convention:** Follows VIN-MEM-01 / VIN-TOOL-01 contract style
**Prime directive:** LLM proposes; deterministic layers dispose. No pass may require whole-work context.

---

## 0. Adaptations (adopted 2026-09-01)

Agreed in review; these amend the sections they name.

1. **Gate 0b — track-assignment / reveal-leak fixture (amends §7).** The two real
   spoiler channels in downstream consumers are (a) a retrospective fact mislabeled as
   presented, which hands the drafter the spoiler *legitimately*, and (b) the drafter's
   own pretraining volunteering a true-but-unrevealed fact about a famous work into
   permitted fuzz. The draft-side reveal-leak detector catches (b) only if (a) is sound.
   So: a **thin** golden-twists fixture (a handful of works with known delayed reveals —
   the Magwitch reveal is the canonical case) that the `TrackClaim` assigner must label
   correctly before any consumer gate (retell/invent reveal-leak) may block on its
   output. One fixture file, not a phase.
2. **P3d — thread reconciliation (amends §5).** P2 emits thread *guesses*; P4's braid
   statistics need canonical threads. New deterministic sub-pass after P3c: merge thread
   guesses on shared member events/characters, LLM confirmation from a closed candidate
   list only. P4 consumes canonical thread ids exclusively.
3. **Slot-structured claims (amends §5/P2, for VIN-STORY-01 §3.4's diff).** Segment-card
   claims are emitted with typed slots (who / where / when / state) so the grounding
   diff's supported/absent/conflicting classification is deterministic after alignment.
   Conflict detection operates on typed slots only — free-text NLI is out of contract.
4. **Scratch graph class (amends §2.6 / §3).** A third graph class beside staging and
   promoted: `scratch` — the ephemeral ingestion target for draft re-ingestion (the
   grounding diff), never promotable, deleted after the consuming mission step. A new
   kind of knowledge, separable by construction. Vinur exposes an
   ingest-to-scratch-and-return-claims lane for it; implementation note: this lane is a
   serving consumer, not a maintenance job — it must not be refused by the ops
   endpoint-mode gate (dedicated endpoint, not an OpsRunner verb).
5. **Pinned initial contract values (amends §7.0 / §9).** Gate zero band: median-of-k
   (k = 5) per-segment score σ ≤ 1.0 on the 0–10 anchored scale AND run-to-run segment
   ordering stability Kendall τ ≥ 0.7 on the pinned fixture set. Segment-card budgets
   per form: prose chapter/scene 200 tokens, tale 250, stanza 80; work card 500.
   Revisable by version bump like any contract value.
6. **Licence note (records the judgement behind §8.2).** The `abstract`-class
   "unrestricted regardless of source licence" position rests on the idea–expression
   dichotomy; it is a deliberate judgement of the project owner, recorded as such. The
   export-profile machinery is designed to be correct under either reading.

---

## 1. Purpose & Scope

Add a second, *interpretive* ontology to Vinur's graph for works of fiction and poetry: narrative roles, beats, themes, and literary devices, expressed in **generic concept language** (protagonist, benefactor-figure, dark moment) rather than proper names, with every abstract claim anchored by evidence edges into the existing concrete story graph.

**Scope guard (per Vinur's frozen-scope rule):** this is an *annotation layer over ingested text*, not a scope expansion. Vinur remains a knowledge graph whose purpose is making Vinkona smarter and more grounded. The distillation layer exists so Vinur can answer craft-of-story questions ("story smithing") for a human or AI user with grounded, evidence-backed answers.

**Out of scope (non-goals):**
- No literary-quality judgements or rankings of works.
- No claims of author intent stated as fact.
- No free-prose essays written into the graph — structured records only.
- No changes to the semantics of the existing concrete extraction layer (P1 below is consumed as-is).

---

## 2. Design Principles

1. **Two layers, one graph.** Concrete layer: entities, places, events with proper names (existing Vinur behaviour, unchanged). Abstract layer: roles, beats, themes, devices in generic language. Abstract nodes point *down* via `evidence` edges; an abstract node with zero evidence edges is invalid and must be rejected at write time.
2. **Closed vocabularies only.** Every classification the LLM performs is a choice from an enumerated vocabulary pack (§4), schema-constrained (GBNF via the existing llama.cpp schema converter). No open-ended abstraction at any pass.
3. **Extract concrete, abstract late.** Names are never genericised during extraction. Role assignment is a separate resolution pass after the whole work is ingested (late binding), because early segments cannot know late-revealed truths. Genericisation is then a deterministic substitution from the resolved mapping table.
4. **Position is the backbone.** Every event and segment carries position (segment index + fraction-through-work). Macro structure (tension curve, beat placement, pacing) is *computed* from position-tagged local scores, never asked of an LLM.
5. **Interpretive layer is marked interpretive.** Every abstract node carries provenance (model id, pass id, prompt/pack version) and confidence. Competing annotations may coexist; the layer never forces a single truth.
6. **Staging before promotion.** Distillation writes go to a staging literary graph; promotion to the queryable graph occurs only after the validation harness (§7) passes for that work.
7. **Small-model friendly.** All passes must run on the 3090 / llama.cpp path with bounded context. Slow is acceptable; overflow is not.

---

## 3. Data Model

### 3.1 New node types

| Node | Purpose | Key fields |
|---|---|---|
| `Work` | The ingested text | title, form (novel/epic/tale-collection/serial/poem), tradition tags, licence class (§8), segment-unit |
| `Segment` | Chunk unit: chapter, scene, tale, stanza (parameterised per form) | index, char/token span, fraction-of-work |
| `NarrativeEvent` | A story event as narrated | position, thread refs, tension score, stakes score, valence |
| `Thread` | A plot line | id, label, member events |
| `RoleAssignment` | Actant/role binding | character ref, role (from pack), **thread scope**, **segment-range scope**, confidence |
| `Beat` | Structural beat instance | beat id (from pack), position (fraction), evidence |
| `Theme` / `Motif` | Bottom-up cluster label | label (from pack or proposed-for-pack), instance edges |
| `DeviceInstance` | A matched literary device | device id (from pack), matched subgraph refs, evidence |
| `Frame` | Narrative frame / teller-tale relation | teller ref, contained-work ref (frames may nest) |
| `TrackClaim` | Presented-vs-retrospective fact pair | claim, track ∈ {presented, retrospective}, divergence span |

### 3.2 Edge types

`evidence` (abstract → concrete, mandatory ≥1 per abstract node), `instantiates` (character → RoleAssignment), `precedes`, `in-thread`, `supports-theme`, `frames`, `diverges-from` (presented TrackClaim → retrospective TrackClaim).

### 3.3 Mandatory metadata on every abstract node

`provenance {model, pass, pack_version, prompt_version}`, `confidence [0–1]`, `created_at`. Deterministically computed nodes (shape, device matches) carry `provenance.model = "deterministic"`.

---

## 4. Vocabulary Packs (data, not code)

Packs are versioned YAML/JSON data files. Adding a tradition later (e.g. an Icelandic saga pack) is **authoring, not engineering**.

### 4.1 Pack schema

```yaml
pack: <id>            # e.g. core, oral-formulaic, romance-beats
version: <semver>
devices:
  - id: ring-composition
    name: Ring composition
    definition: <one paragraph, generic language>
    structural_signature: <subgraph pattern, machine-matchable, or null if classifier-only>
    classifier_hint: <prompt fragment for LLM confirmation>
    exemplars: [<work id + span>, ...]   # populated as corpus grows
vocab:
  roles: [...]        # closed lists used by classification passes
  beats: [...]
  themes: [...]
scales:
  tension: {min: 0, max: 10, anchors: {0: "...", 5: "...", 10: "..."}}
```

### 4.2 Required packs at v1

- **core** — Greimas actants (subject, object, sender, receiver, helper, opponent), universal beat set (inciting incident, midpoint reversal, dark moment, climax, resolution), tension/stakes scales with worded anchors, base theme list.
- **oral-formulaic** (Homer) — epithet, type-scene (arming, feasting, supplication, catalogue), ring composition, in medias res.
- **frame-narrative** (Chaucer) — frame, teller-tale relation, per-tale genre tag, genre parody.
- **interlacement** (Tolkien) — thread braiding, cliffhanger-at-switch, information asymmetry between threads, eucatastrophe.
- **comic** (Adams) — bathos, digression-as-device, setup→subversion pairs with measured token distance.
- **romance-beats** — meet-cute, forced proximity, dark moment, grand gesture, HEA/HFN; used doubly as the **calibration pack** (§7).
- **prosody** (poems) — metre, rhyme scheme, stanza form, volta; segment unit = line/stanza; scansion and rhyme detection are deterministic modules, LLM used only for volta/imagery labelling.

---

## 5. Pipeline Passes

Map-reduce over segments; the graph is the working memory between passes. Fixed token budgets per artefact; all LLM outputs schema-validated (reject-and-retry on violation, max N retries, then flag segment for review — never silently drop).

**P0 — Segmentation (deterministic).** Split by declared segment unit (chapters, tales, stanzas); heuristic scene detection as fallback. Emit `Segment` nodes with positions.

**P1 — Concrete extraction (existing, unchanged).** Standard Vinur ingestion per segment: entities, places, events, relations, with proper names.

**P2 — Segment card (LLM, bounded).** Per segment, single-context pass emitting a ≤ ~200-token structured card: narrative events (with thread guesses), tension/stakes/valence scores against the pack's anchored scales, motif sightings (from closed list), and **role micro-claims** ("<char>: opponent-evidence", "<char>: sender-evidence") — claims, not conclusions. Contradictory claims across segments are *retained*: they are the character-development signal.

**P3 — Aggregation (deterministic + small LLM).** Roll segment cards → act cards → work card, each roll-up sized to fit one context. **`TrackClaim` (presented/retrospective) assignment happens here and at P4, never at P2 — a segment pass cannot know what is retrospectively true.** Sub-passes:
- **P3a Role resolution:** aggregate all micro-claims per character per thread (a few thousand tokens total); LLM assigns time-scoped, thread-scoped `RoleAssignment`s from the closed role list. This is where late binding happens.
- **P3b Genericisation:** deterministic substitution using the resolved mapping (Pip → protagonist, the forge → home-of-origin). Concrete layer untouched; abstract layer speaks only generic language.
- **P3c Theme clustering:** cluster motif sightings; LLM labels clusters from the theme vocabulary; `supports-theme` evidence edges to every instance.
- **P3d Thread reconciliation (§0.2):** deterministic merge of P2 thread guesses (shared member events/characters), LLM confirmation from a closed candidate list; emits canonical `Thread` nodes — the only thread ids P4 may consume.

**P4 — Shape computation (fully deterministic).** From position-tagged scores: tension curve, beat candidate positions, pacing profile, thread-braid pattern, presented/retrospective divergence spans (delayed-reveal detection). No LLM.

**P5 — Device matching (deterministic first).** Match pack `structural_signature`s as subgraph patterns over the assembled graph (ring composition = palindromic motif sequence over position; interlacement = alternating thread membership with cliffhanger edges at switches; frame = work-node containing work-nodes). LLM used only to confirm/label ambiguous matches, choosing from the pack's device list.

**P6 — Work card emission.** A few-hundred-token card, entirely generic language: actant table, beat list with positions, theme list, tension curve reference, device inventory, frame structure. Every line traceable through evidence edges to passages. This card is the unit a small model can hold many of for cross-corpus reasoning.

---

## 6. Cross-Corpus Capability (the payoff)

- **Technique = recurring subgraph pattern + evidence set.** Query surface must support: "all instances of device X across corpus, ranked by confidence, with passages."
- **Positional statistics:** with volume (romance shelf, web-fiction archives), beat placement becomes distributional — "midpoint reversal typically lands 48–55%" — computed, stored per corpus slice, and queryable.
- **Pre-labelled corpora** (e.g. AO3-style tag folksonomies) are ingested as *external annotations* with their own provenance class, used for validation and comparison — never blindly trusted as ground truth.

---

## 7. Validation Harness (gate to promotion)

0. **Gate zero — P2 scorer reliability (build FIRST, before any downstream work).** The entire shape layer rests on tension/stakes scores being consistent across segments scored in isolation. Fixture: a pinned segment set scored k times each; measure run-to-run variance and inter-segment ordering stability. Pipeline work on P3–P6 may not begin until median-of-k sampling plus anchored scale exemplars brings variance inside a declared band (initial values in §0.5). If a 9B cannot reach the band, the P2 scorer escalates to the larger model — a cost decision, not a design change.
0b. **Gate 0b — TrackClaim assigner reliability (§0.1).** The golden-twists fixture: the assigner must label the pinned delayed-reveal set correctly before any consumer gate may block on reveal-leak.
1. **Golden works (beat-level resolution only):** one public-domain work per pack, annotated at *beat and device-instance level* — not exhaustive scholarly annotation. Bootstrap from published analyses (Propp indices, romance beat sheets, published Homeric type-scene catalogues) and large-model drafts, then human spot-check. A thin gate that exists beats a rigorous one that doesn't. Per-pack pass thresholds defined in the pack file.
2. **Calibration test:** on the romance calibration set, the dark moment must be detected within the expected positional window (~70–80%) for ≥ agreed fraction of works. Failure blocks promotion of *pipeline versions*, not individual works.
3. **Schema conformance:** zero abstract nodes without evidence edges; zero out-of-vocabulary classifications.
4. **Regression:** re-running a golden work on a new pipeline/pack version must not degrade agreed metrics.

### 7.1 Consumer version pinning

Work cards, segment cards, and packs are versioned interfaces. Downstream consumers (VIN-STORY family) must pin the pack versions and card-schema version they were launched against, recorded in mission provenance; the executor performs a compatibility check at mission launch and refuses on mismatch. A pack bump never silently changes the meaning of an in-flight or replayed mission.

---

## 8. Provenance & Licensing

### 8.1 Work licence classes

Every `Work` carries a licence class: `public-domain` (promotable to shippable starter graphs), `personal-use` (queryable locally — e.g. in-copyright novels), `externally-annotated` (carries third-party tag provenance).

### 8.2 Artefact classes (the unit of licence filtering)

Licence restriction operates **per artefact class**, not by blanket reachability from a `Work` node:

| Artefact class | Contents | Distributability |
|---|---|---|
| `abstract` | Work cards, beat lists, role/actant tables, theme labels, device inventories — generic concept language, no proper names, no expression | **Unrestricted, regardless of source licence.** These are allegorical structure — ideas, not expression — and sit outside copyright protection (idea–expression dichotomy). |
| `pooled-stats` | Cross-corpus positional statistics and device frequencies | **Unrestricted**, even where in-copyright works contributed to the pool: no individual expression survives aggregation. |
| `concrete` | Entity/event graphs with proper names, evidence spans, quoted passages | Distributable only for `public-domain` works. |
| `retelling` | Generated retellings and rendered audio | Inherits the source `Work`'s licence class (derivative). `personal-use` sources → `personal-use` artefact. |

### 8.3 Export profiles

Export is a deterministic traversal governed by named profiles (data-defined, like packs):

- **`distributable`** — `abstract` + `pooled-stats` from all sources; `concrete` from `public-domain` works only; no `retelling` artefacts from non-`public-domain` sources.
- **`personal`** — everything, flagged non-distributable in the export manifest.

Packaging/appliance tooling must hard-refuse any export not produced under the `distributable` profile.

### 8.4 Personal-use attestation gate (retellings)

Executing a retelling/narration mission on a non-`public-domain` source requires a **`personal-use` attestation flag** at mission launch (a use-case tick-box in the UI), enforced executor-side per the VIN-DEL-01 warrant pattern — the LLM is never party to the decision. The attestation is logged into the artefact's provenance, and rendered audio files carry the licence class in their file metadata so downstream tooling can read it without graph access. Missing attestation → mission refused deterministically.

---

## 9. Compute & Budget Constraints

- All passes runnable on the 3090 llama.cpp path; multi-day whole-corpus runs acceptable.
- Per-pass context ceilings declared in config; a pass that would overflow must recursively split, never truncate silently.
- Segment card budget and work card budget are contract values (initial values per form in §0.5); changes require a pack/pipeline version bump.

---

## 10. Deliverables for the coding agent

1. Schema migration: new node/edge types (§3) in the staging literary graph, plus the `scratch` graph class (§0.4) with its ingest-to-scratch claims lane.
2. Pack loader + pack schema validator; the seven v1 packs (§4.2) authored as data.
3. Passes P0–P6 (incl. P3d) with schema-constrained LLM calls and reject-retry-flag handling.
4. Deterministic modules: shape computation, subgraph device matcher, scansion/rhyme (prosody pack).
5. Validation harness + golden-work fixtures + gate 0b golden-twists fixture + promotion gate.
6. Query endpoints: device-instance retrieval, positional statistics, work-card fetch, theme-evidence traversal.
7. Licence enforcement per §8: artefact-class tagging, export profiles (`distributable`/`personal`) as data-defined traversal rules, executor-side attestation gate for retelling missions, licence metadata embedding in rendered audio.
