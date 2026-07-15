# VINUR-OPS-01 — Op-Annotation Surface (external-oracle candidate annotation)

**Status:** Draft 1 · **Companion to:** RUST-02 (external annotation-layer contract; first consumer programme) and RUST-03 (learned layer) · **Consumer:** implementing agent · **Partner:** the oracle runtime (first instance: the Rust coding layer's rust-analyzer client)

The key words MUST, MUST NOT, REQUIRED, SHALL, SHALL NOT, SHOULD, SHOULD NOT, MAY are per RFC 2119.

---

## 1. Scope

### 1.1 Purpose
This contract is the **vinur half** of the RUST-02 programme: the surface vinur exposes to a runtime whose candidates come from an **external legality oracle** (rust-analyzer in the first programme), plus the storage and separation conventions that let that knowledge live in the shared master kb without contaminating the conversational surfaces — and without the conversational data leaking into coding responses.

vinur's role is fixed by RUST-02 §2 and restated here as this contract's cardinal invariant:

> **vinur annotates ids it is handed. It has no mechanism to originate, remove, or reorder candidates.**

### 1.2 In scope
- The `ops_annotate` tool (batch id-join annotation).
- Region storage conventions: id region, bundles, facet tagging.
- Persistence of the RUST-02 §7.4 reserved learned fields.
- The two-way separation firewall and its config mechanism.
- CI gates and fixtures.

### 1.3 Out of scope (MUST NOT be built under this contract)
- Anything oracle-facing: LSP clients, typed-context extraction, rustc probes, trace capture (RUST-02, oracle-runtime side).
- Computation of rank, counts, or anti-pattern status (RUST-03). This surface **relays** those fields from storage; it never computes them.
- Any change to how the oracle runtime orders or truncates candidates.
- Code generation, LM inference, assistant UX.

### 1.4 Region-agnosticism (binding)
Nothing in the engine may be specific to Rust. Region prefixes, facet values, and exclusion lists are **configuration shipped by the consumer pack** (the established overlay pattern: the mechanism is core, the vocabulary is a config fragment). An engine source file that hard-codes a region value outside fixtures is defective (§6 gate 6).

---

## 2. The invariant, by construction

For a request carrying candidate id set `R`:

```
keys(response.annotations) ≡ R
```

- The response MUST contain an entry for **every** id in `R`. An id the graph knows nothing about returns `{"annotated": false}` — presence is the contract (fail-open on knowledge, RUST-02 §2).
- The response MUST NOT contain any key not in `R` (no origination).
- The response is a **map keyed by op id**, not a list. Ordering is therefore structurally not vinur's to express; the oracle runtime preserves oracle order (RUST-02 §6.4).
- Ids MUST be treated as opaque keys. Fuzzy or normalised matching on ids is prohibited — *identity may not be fuzzy* (RUST-02 §4).

This is stronger than a gated promise: the surface has no representation in which a containment or suppression violation can be expressed.

---

## 3. The tool: `ops_annotate`

### 3.1 Request
```json
{
  "ops": ["rust:op:std::vec::Vec::push#inherent", "..."],
  "context_features": {"receiver_type": "Vec<T>", "ownership": "ref_mut"}
}
```
- `ops`: REQUIRED, 1–500 ids per call. Duplicates MAY be deduplicated (the response is a map).
- `context_features`: OPTIONAL, `{feature: value}`. Under this contract it is accepted and MAY be ignored. Post-RUST-03 it selects the conditional-rank entry. It MUST NOT affect the response key set under any contract.

### 3.2 Response
```json
{
  "annotations": {
    "rust:op:std::vec::Vec::push#inherent": {
      "annotated": true,
      "display": "Vec::push",
      "caveats": [{"card_id": "rust:diag:card:E0502", "severity": "error", "title": "..."}],
      "rank": null,
      "rank_specificity": null,
      "anti_pattern_of": [],
      "provenance": {"source": "error_index", "toolchain": "1.xx.x"}
    },
    "rust:op:unknown::thing#inherent": {"annotated": false}
  },
  "graph_version": "sha256:...",
  "requested": 2,
  "joined": 1
}
```
- `caveats` are drawn from hazard cards attached to the op-node (`attaches_to` / card-graph edges) **within the same region** (§5.2). `severity` is relayed from the card.
- `rank`, `rank_specificity`, `anti_pattern_of` are relayed **verbatim from storage** (null/`[]` until RUST-03 populates them). When populated, `anti_pattern_of` carries hazard-card ids; demotion semantics remain the runtime's job (RUST-03 §4.3: demote, never remove).
- `joined`/`requested` implement RUST-02 §9.3: join coverage is **reported, never gated** — gating it would incentivise suppression.
- `graph_version` MUST identify the exact knowledge state (content digest of the loaded region bundles), so the runtime can satisfy RUST-02 §6's purity claim end-to-end.

### 3.3 Purity and determinism
- `ops_annotate` MUST be read-only: no graph writes, no gap logging, no facet derivation, no cache mutation observable in the store (RUST-02 §9.6). Coverage signals come from the runtime's trace store, which already records `candidates_offered`.
- Identical `(request, graph_version)` MUST yield byte-identical responses (canonical key ordering in serialisation).

### 3.4 Advertisement
The tool MUST appear in the `/call` catalogue only when `ops_regions` (§5.2) is non-empty — a purely conversational deployment never advertises a coding surface.

---

## 4. Storage conventions

### 4.1 Id region
Every node, card, and edge of an oracle programme is minted under its region prefix (first consumer: `rust:`). The region prefix is a **minting authority** and immutable (main spec §4.9). Region membership is derivable from the id alone — this is what makes the leak gates (§6 gate 5) mechanical.

### 4.2 Bundles
Region knowledge ships and loads as bundles named `<region>-base` (op-nodes, hazard cards) and `<region>-learned` (RUST-03 aggregates). All existing brains machinery applies unchanged: content-hash idempotent import, foreign-import trust caps, load/unload, export-first eject.

### 4.3 Facets
Every region row carries the facet `domain: <region-tag>` (first consumer: `domain: rust-coding`). `facetize` MUST derive this facet from the id-region prefix, so backfill after import is automatic and cannot drift from the id.

### 4.4 Reserved learned fields (RUST-02 §7.4)
`observed_count`, `validated_count`, `last_observed`, `conditional_rank`, `anti_pattern_of` MUST be:
- persisted on op-nodes and hazard cards at import, defaulted `0 / 0 / null / null / []`;
- preserved **byte-identically** through bundle export → import round-trips;
- relayed verbatim by `ops_annotate`.

This contract does not compute them. Omitting them is a contract violation — they are cheap to reserve now and impossible to backfill.

---

## 5. Separation (the two-way firewall)

### 5.1 Conversational paths exclude the region by default
A new generic config key `ask_exclude_facets` (list of `axis:value`, default `[]`) MUST be honoured by the conversational read paths — `kb_ask`, `kb_search`, and everything built on them (guidance). Rows matching an excluded facet MUST NOT enter the candidate pool unless the request **explicitly** opts in (facets parameter naming the axis). The consumer pack ships the value (`ask_exclude_facets = ["domain:rust-coding"]`); the engine ships only the mechanism.

### 5.2 Coding responses stay in-region
A new config key `ops_regions` (list of region prefixes, default `[]`) declares which regions `ops_annotate` may serve. Every id in an `ops_annotate` response — keys, card ids, `anti_pattern_of` entries — MUST belong to a listed region. A general-region or personal id in an `ops_annotate` response is a violation.

### 5.3 General responses stay out of the region
A `kb_ask`/`kb_search`/guidance answer containing a region-prefixed id, when the request did not opt in, is a violation.

---

## 6. Regression gates (hard — CI MUST fail on violation)

| # | Gate | Threshold |
|---|---|---|
| 1 | **Key-set identity**: `keys(annotations) ≡ request.ops` on every fixture (containment + non-suppression in one property) | 0 violations |
| 2 | **Determinism**: identical `(fixtures, graph_version)` → identical response digests across two runs | 0 |
| 3 | **Purity**: store content digest unchanged across an `ops_annotate` fixture batch | 0 |
| 4 | **Reserved-field round-trip**: export `<region>-base` → import into a fresh master → §4.4 fields byte-identical | 0 |
| 5 | **Region leak, both directions**: §5.2 and §5.3 fixtures | 0 |
| 6 | **Neutrality**: engine sources contain no hard-coded region values outside fixtures and config examples (grep gate) | 0 |
| 7 | **Feature inertness**: a request with and without `context_features` yields identical key sets | 0 |

---

## 7. Golden fixtures (minimum)

1. Op id with an attached hazard card → `annotated: true`, caveat present with `severity: "error"`.
2. Unknown id → `annotated: false`, **present** in the response.
3. Mixed batch (known + unknown + duplicate) → key-set identity holds.
4. Request with `context_features` → identical key set to the same request without (gate 7).
5. General `kb_ask` fixture with the region bundle loaded → no region ids in the answer; the same query with explicit opt-in → region items permitted.
6. Region bundle round-trip preserving §4.4 fields.
7. `ops_regions` empty → tool absent from the catalogue.

---

## 8. Deliverables

1. `ops_annotate` tool + server route + catalogue gating (§3).
2. `ask_exclude_facets` mechanism on the conversational read paths (§5.1).
3. `ops_regions` config + region-derived facet rule in `facetize` (§4.3, §5.2).
4. Reserved-field persistence + bundle round-trip (§4.4).
5. CI gates (§6) and fixtures (§7).
6. An example consumer config fragment (documentation, not defaults).

## 9. Definition of done

All §6 gates pass on the §7 fixtures; a fixture region imports as a bundle, annotates end-to-end, and ejects cleanly; the engine remains region-value-free; and the RUST-02 runtime can consume this surface with **no further vinur changes** for -02 scope — RUST-03 then needs only to populate the reserved fields and the conditional-rank lookup behind the already-specified response fields.
