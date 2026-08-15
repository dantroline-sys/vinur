"""Verifier tier (the 'is this reasonable?' gate).

A fast, smaller LM does the bulk extraction; this module asks the big, slower LM to
VET each submission and return a compact per-item verdict — accept (correct &
supported), reject (unsupported/hallucinated/trivial/mis-framed), or adjust (mostly
right, fix a field).  The verdict patch is applied here, so the writer downstream sees
a cleaned extraction.

Output is deliberately small (verdicts + minimal patches, not a regenerated
structure), so the slow LM stays fast and the speed win of the fast extractor is
preserved.  Items the verifier omits default to accept — silence is assent.
"""
from __future__ import annotations

import logging

from . import sanitize

log = logging.getLogger("knowledgehost.verify")

_VERDICT = {"type": "string", "enum": ["accept", "reject", "adjust"]}
_FV = {"type": "array", "items": {"type": "object",
       "properties": {"feature": {"type": "string"}, "value": {"type": "string"}},
       "required": ["feature", "value"]}}

# Compact judge output: per item, an index `i` into the submitted list, a verdict, and
# (for adjust) only the fields to overwrite.  Unmentioned items are accepted as-is.
VERIFY_SCHEMA = {
    "type": "object",
    "properties": {
        "concepts": {"type": "array", "items": {"type": "object", "properties": {
            "i": {"type": "integer"}, "verdict": _VERDICT,
            "summary": {"type": "string"}, "kind": {"type": "string"}},
            "required": ["i", "verdict"]}},
        "relations": {"type": "array", "items": {"type": "object", "properties": {
            "i": {"type": "integer"}, "verdict": _VERDICT,
            "type": {"type": "string"}, "mechanism": {"type": "string"},
            "mechanism_basis": {"type": "string", "enum": ["stated", "inferred", ""]},
            "polarity": {"type": "string", "enum": ["positive", "negative", ""]},
            "conditions": {"type": "string"}, "discriminators": _FV,
            "regime": {"type": "string", "enum": ["empirical", "conventional",
                       "fictional", "interpretive", "historical", ""]}},
            "required": ["i", "verdict"]}},
        "procedures": {"type": "array", "items": {"type": "object", "properties": {
            "i": {"type": "integer"}, "verdict": _VERDICT,
            "title": {"type": "string"}, "goal": {"type": "string"},
            "steps": {"type": "array", "items": {"type": "string"}}},
            "required": ["i", "verdict"]}},
        "criteria": {"type": "array", "items": {"type": "object", "properties": {
            "i": {"type": "integer"}, "verdict": _VERDICT,
            "title": {"type": "string"}, "threshold": {"type": "string"},
            "required": _FV, "supportive": _FV, "exclusion": _FV},
            "required": ["i", "verdict"]}},
    },
    "required": [],
}

VERIFY_SYSTEM = (
    "You are a meticulous senior reviewer. A faster, smaller model extracted the items "
    "below from the SOURCE passage. For EACH item decide:\n"
    "- accept: correct, general, and genuinely supported by the source;\n"
    "- reject: unsupported / hallucinated / trivial / mis-framed / not actually general;\n"
    "- adjust: mostly right but a field is wrong — return ONLY the corrected fields.\n"
    "Be strict about: claims the source does not support; a `mechanism` that merely "
    "restates 'X causes Y' instead of explaining the chain (reject or fix it, and set "
    "mechanism_basis); a wrong `regime` (FIREWALL: a fictional/opinion claim tagged "
    "empirical must be fixed); discriminators that don't actually distinguish this "
    "cause; and concepts too tied to one specific instance to reuse.\n"
    "CRITERIA are the highest stakes: reject a diagnostic/classification card whose "
    "features or `threshold` the source does not state (an invented cutoff is worse "
    "than none), and fix modality placement — a may-have listed under `required`, or "
    "a rule-out feature outside `exclusion`, silently corrupts every later match.\n"
    "Refer to items by their index `i`. Omit an item to accept it. Output JSON only."
)


def _with_chunk_index(arr_schema):
    """The single-item verdict schema, plus a chunk index `c` for batched review."""
    it = arr_schema["items"]
    return {"type": "array", "items": {
        "type": "object",
        "properties": {"c": {"type": "integer"}, **it["properties"]},
        "required": ["c"] + it["required"]}}


# Batched: verdicts across SEVERAL chunks at once, each tagged with its chunk index `c`,
# so the big LM's fixed per-call cost (system-prompt prefill, sampling/HTTP setup) is
# amortised over the batch instead of paid once per chunk.
BATCH_VERIFY_SCHEMA = {
    "type": "object",
    "properties": {k: _with_chunk_index(v) for k, v in VERIFY_SCHEMA["properties"].items()},
    "required": [],
}

BATCH_VERIFY_SYSTEM = VERIFY_SYSTEM.replace(
    "Refer to items by their index `i`. Omit an item to accept it. Output JSON only.",
    "You are reviewing SEVERAL passages at once — each is a CHUNK with index `c`. "
    "Refer to every item by its (chunk `c`, index `i`). Omit an item to accept it. "
    "Output JSON only.")


def _fmt_items(concepts, relations, procedures, criteria=None) -> str:
    lines = ["CONCEPTS:"]
    for i, c in enumerate(concepts):
        lines.append(f"[{i}] {c.get('label','')} ({c.get('kind','')}): {c.get('summary','')}")
    lines.append("RELATIONS:")
    for i, r in enumerate(relations):
        disc = ", ".join(f"{d.get('feature')}={d.get('value')}"
                         for d in (r.get('discriminators') or []) if isinstance(d, dict))
        lines.append(f"[{i}] {r.get('src','')} -[{r.get('type','')}/{r.get('regime','')}]-> "
                     f"{r.get('dst','')}; mech={r.get('mechanism','')}"
                     + (f"; features={disc}" if disc else ""))
    lines.append("PROCEDURES:")
    for i, p in enumerate(procedures):
        lines.append(f"[{i}] {p.get('title','')}: {len(p.get('steps') or [])} steps")
    if criteria:                    # section only when present — most chunks carry none
        def _fv(feats):
            return ", ".join(f"{d.get('feature')}={d.get('value')}"
                             for d in (feats or []) if isinstance(d, dict))
        lines.append("CRITERIA:")
        for i, c in enumerate(criteria):
            parts = [f"[{i}] {c.get('title','')} ({c.get('concept','')})"]
            for mod in ("required", "supportive", "exclusion"):
                fv = _fv(c.get(mod))
                if fv:
                    parts.append(f"{mod}: {fv}")
            if c.get("threshold"):
                parts.append(f"threshold: {c.get('threshold')}")
            if c.get("levels"):
                parts.append(f"levels: {len(c['levels'])} (staging)")
            lines.append("; ".join(parts))
    return "\n".join(lines)


# fields the verifier may overwrite on an `adjust`, per item kind.
_ADJUST_FIELDS = {
    "concepts": ("summary", "kind"),
    "relations": ("type", "mechanism", "mechanism_basis", "polarity", "conditions",
                  "discriminators", "regime"),
    "procedures": ("title", "goal", "steps"),
    # criteria: the scalar rule + the modality arrays (a may-have filed under
    # required is the classic corruption); levels/differentials are reject-or-keep.
    "criteria": ("title", "threshold", "required", "supportive", "exclusion"),
}


def _apply(items, verdicts, kind):
    """Return (kept_items, n_reject, n_adjust) after applying the verdict list."""
    by_i = {}
    for v in (verdicts or []):
        if isinstance(v, dict) and isinstance(v.get("i"), int):
            by_i[v["i"]] = v
    kept, n_rej, n_adj = [], 0, 0
    for i, it in enumerate(items):
        v = by_i.get(i)
        if not v:
            kept.append(it)                       # silence = accept
            continue
        verdict = v.get("verdict")
        if verdict == "reject":
            n_rej += 1
            continue
        if verdict == "adjust":
            patched = dict(it)
            for f in _ADJUST_FIELDS[kind]:
                if f in v and v[f] not in (None, "", []):
                    patched[f] = v[f]
            kept.append(patched)
            n_adj += 1
        else:
            kept.append(it)
    return kept, n_rej, n_adj


def _by_chunk(arr):
    g: dict = {}
    for v in (arr or []):
        if isinstance(v, dict) and isinstance(v.get("c"), int):
            g.setdefault(v["c"], []).append(v)
    return g


def verify_batch(big_lm, drafts, cfg):
    """Vet SEVERAL fast-LM extractions in ONE big-LM call.  `drafts` is a list of
    {chunk, concepts, relations, procedures[, criteria]}; returns a list of
    (concepts', relations', procedures', criteria', stats) aligned to it.  One
    transport failure raises BackendUnavailable for the whole batch (the caller
    requeues it); a parse failure is fail-OPEN per draft (keep it unchanged).
    Sources are truncated harder than the single path so a batch still fits the
    big LM's context window.  Criteria joined the vetted set with July #14 —
    they were the one item family that sailed past the judge entirely."""
    if not drafts:
        return []
    src_chars = int(cfg.get("verify_source_chars", 800))
    blocks = []
    for c, d in enumerate(drafts):
        src = sanitize.clean(d["chunk"].get("text") or "", src_chars)
        blocks.append(f"### CHUNK {c}\nSOURCE: {src}\n"
                      + _fmt_items(d["concepts"], d["relations"], d["procedures"],
                                   d.get("criteria")))
    user = ("Review each CHUNK's items; return verdicts by (c, i). Omit to accept.\n\n"
            + "\n\n".join(blocks))
    max_tokens = min(4096, max(int(cfg.get("verify_max_tokens", 1024)), 256 * len(drafts)))
    out = big_lm.chat_json(BATCH_VERIFY_SYSTEM, user, BATCH_VERIFY_SCHEMA, max_tokens=max_tokens)
    if not out:                                   # unparseable -> fail open for all
        log.debug("verifier returned nothing — keeping %d draft(s) unchanged", len(drafts))
        return [(d["concepts"], d["relations"], d["procedures"],
                 d.get("criteria") or [],
                 {"rejected": 0, "adjusted": 0, "failed": 1}) for d in drafts]
    gc, gr, gp, gcr = (_by_chunk(out.get("concepts")), _by_chunk(out.get("relations")),
                       _by_chunk(out.get("procedures")), _by_chunk(out.get("criteria")))
    results = []
    for c, d in enumerate(drafts):
        co, rc, ac = _apply(d["concepts"], gc.get(c), "concepts")
        re_, rr, ar = _apply(d["relations"], gr.get(c), "relations")
        pr, rp, ap = _apply(d["procedures"], gp.get(c), "procedures")
        cr, rcr, acr = _apply(d.get("criteria") or [], gcr.get(c), "criteria")
        results.append((co, re_, pr, cr,
                        {"rejected": rc + rr + rp + rcr,
                         "adjusted": ac + ar + ap + acr, "failed": 0}))
    return results


def verify_extraction(big_lm, chunk, concepts, relations, procedures, cfg,
                      criteria=None):
    """Single-chunk convenience wrapper over verify_batch (kept for callers/tests)."""
    return verify_batch(big_lm, [{"chunk": chunk, "concepts": concepts,
                                  "relations": relations, "procedures": procedures,
                                  "criteria": criteria or []}], cfg)[0]
