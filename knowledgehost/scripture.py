"""Parallel scripture reading — the study surface over a canonical-text KB.

Once several editions (KJV, Douay-Rheims, …) are ingested, every one lands its verses on
the SAME canonical keys (bible:John.3.16), so a reference can be read across all of them
at once: the verse in each translation, its cross-references (the citation graph), and
the commentary notes attached to it (the annotates graph).  That is what makes the KB a
scripture-study tool rather than a pile of text.

Pure reader — no LM, no writes.  Resolves a human reference ("John 3:16", "1 Cor 13:4-7")
to canonical keys, then gathers the aligned material for each verse.
"""
from __future__ import annotations

import re

from . import structure

_KEY_RE = re.compile(r"bible:([^.]+)\.(\d+)\.(\d+)(?:-(\d+))?$")


def resolve_reference(ref_text: str, maps=None) -> list:
    """A human reference → the individual canonical verse keys it names, expanding a
    verse range.  'John 3:16-18' → [bible:John.3.16, .17, .18]."""
    out: list = []
    for r in structure.parse_citations(ref_text, {"kind": "scripture"}, maps=maps):
        m = _KEY_RE.match(r.key)
        if not m:
            out.append(r.key)
            continue
        book, chap, v1, v2 = m.group(1), m.group(2), int(m.group(3)), int(m.group(4) or m.group(3))
        for v in range(v1, min(v2, v1 + 175) + 1):     # bound a pathological range
            key = f"bible:{book}.{chap}.{v}"
            if key not in out:
                out.append(key)
    return out


def _node_id(kb, label: str, kind: str = "passage"):
    r = kb.db.execute("SELECT id FROM nodes WHERE label=? AND kind=?", (label, kind)).fetchone()
    return r["id"] if r else None


def alias_views(store) -> tuple:
    """The versification-alias lens over the raw store, built once per reading:
    (inverse, per_doc) where inverse = {canonical_key: [(stored_key, path), …]} — which
    document's chunk at which stored key RENDERS this canonical verse — and per_doc =
    {path: key_aliases} — each document's own alias map.  A Vulgate-numbered edition's
    verses are STORED under its printed keys (the store is never rewritten); its doc_meta
    key_aliases (the Psalm reconciliation) say where each one belongs.  Both directions are
    PER DOCUMENT: another edition's chunk sitting at the same printed key must not be
    dragged along, and a chunk whose key is aliased away no longer answers for it."""
    inv: dict[str, list] = {}
    per_doc: dict[str, dict] = {}
    try:
        metas = store.all_doc_meta()
    except Exception:
        return inv, per_doc
    for path, meta in metas.items():
        ka = ((meta or {}).get("reference_map") or {}).get("key_aliases") or {}
        if not ka:
            continue
        per_doc[path] = ka
        for src, dst in ka.items():
            inv.setdefault(dst, []).append((src, path))
    return inv, per_doc


def verse_view(store, kb, key: str, *, aliases: tuple | None = None) -> dict:
    """The aligned material for ONE canonical verse: every translation's text, the verses
    it cross-references, and the commentary notes attached to it.  `aliases` (from
    alias_views) folds versification-aliased editions in: their rendering of this verse is
    pulled from its stored key, and their chunk at THIS key is skipped when it belongs
    elsewhere."""
    inv, per_doc = aliases or ({}, {})
    editions = []
    for c in store.chunks_for_section(key):
        if c.get("source_type") not in ("scripture", "legal"):
            continue
        if per_doc.get(c.get("path_or_url"), {}).get(key):
            continue                                     # this doc's verse here is aliased away
        editions.append({"translation": c["title"] or "?", "text": c["text"]})
    for src, path in inv.get(key, []):
        for c in store.chunks_for_section(src):
            if c.get("path_or_url") == path and c.get("source_type") in ("scripture", "legal"):
                editions.append({"translation": c["title"] or "?", "text": c["text"]})
    cross: list = []
    commentary: list = []
    nid = _node_id(kb, key)
    if nid:
        for r in kb.db.execute(
                "SELECT dst.label AS lab FROM edges e JOIN nodes dst ON dst.id=e.dst_id "
                "WHERE e.src_id=? AND e.family='citation' AND e.status='active' ORDER BY dst.label",
                (nid,)):
            disp = structure.display_for_key(r["lab"])
            if disp not in cross:
                cross.append(disp)
        for r in kb.db.execute(
                "SELECT src.summary AS note FROM edges e JOIN nodes src ON src.id=e.src_id "
                "WHERE e.dst_id=? AND e.family='commentary' AND e.status='active'",
                (nid,)):
            if r["note"]:
                commentary.append(r["note"])
    return {"key": key, "display": structure.display_for_key(key),
            "editions": editions, "cross_references": cross, "commentary": commentary}


def parallel_reading(store, kb, cfg, ref_text: str) -> dict:
    """Read a reference across every ingested edition + its graph.  Returns
    {reference, verses: [verse_view, …]}."""
    maps = structure.load_reference_maps(cfg.get("reference_maps") or [])
    keys = resolve_reference(ref_text, maps)
    aliases = alias_views(store)
    return {"reference": ref_text,
            "verses": [verse_view(store, kb, k, aliases=aliases) for k in keys]}
