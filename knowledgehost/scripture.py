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


def verse_view(store, kb, key: str) -> dict:
    """The aligned material for ONE canonical verse: every translation's text, the verses
    it cross-references, and the commentary notes attached to it."""
    editions = [{"translation": c["title"] or "?", "text": c["text"]}
                for c in store.chunks_for_section(key)
                if c.get("source_type") in ("scripture", "legal")]
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
    return {"reference": ref_text, "verses": [verse_view(store, kb, k) for k in keys]}
