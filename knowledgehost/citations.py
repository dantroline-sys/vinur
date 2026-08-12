"""VINUR — deterministic citation-edge graph for structured (scripture/legal) corpora.

After a structured ingest (ingest._ingest_structured_doc) the raw store holds one chunk
per canonical unit — a verse / a section — keyed by its canonical citation
(bible:John.3.16, usc:17/106), plus (when the confirm step opted in) one chunk per
interleaved commentary note, keyed by the verse it annotates.  This pass turns that into
a GRAPH, with NO language model:

  * one KB node per unit — label = the canonical key, so units from DIFFERENT documents
    that name the same passage converge on ONE node (id = hash(label, kind));
  * a 'citation' edge from each unit to every canonical reference its text makes;
  * one node per commentary note, an 'annotates' edge onto the verse it explains, and
    'citation' edges for the references the note itself makes.

'Romans 9:16 … see Exodus 33:19' becomes an edge; a Challoner note on Genesis 1:6 that
cites Job 26:7 becomes a note node annotating Gen.1.6 and citing Job.26.7.  Deterministic
and idempotent (content-derived ids), so it is safe to re-run — new documents add edges.
"""
from __future__ import annotations

import hashlib
import logging

from . import structure

log = logging.getLogger("knowledgehost.citations")

_UNIT_TYPES = ("scripture", "legal")
_NODE_KIND = {"scripture": "passage", "legal": "provision"}
_SUMMARY_CAP = 600


def _collect(store) -> dict:
    """Group the raw store's structured + commentary chunks by document, in order:
    {path: {"title", "units": [(key, text)], "notes": [(anchor_key, text)]}}."""
    docs: dict = {}
    for ch in store.iter_chunks():
        st = ch.get("source_type")
        if st not in _UNIT_TYPES and st != "commentary":
            continue
        path = ch.get("path_or_url") or ""
        d = docs.setdefault(path, {"title": ch.get("title") or "", "units": [], "notes": []})
        d["notes" if st == "commentary" else "units"].append(
            (ch.get("section") or "", ch.get("text") or ""))
    return docs


def build(store, kb, cfg, *, log=log) -> dict:
    """Build the citation + commentary graph over every structured document in the raw
    store.  Returns {docs, units, references, edges, notes, annotations}."""
    docs = _collect(store)
    ref_maps_cfg = cfg.get("reference_maps") or []
    n_units = n_refs = n_edges = n_notes = n_annot = 0
    for path, d in docs.items():
        meta = store.get_doc_meta(path) or {}
        kind = meta.get("kind") or (
            "legal" if any(s.startswith("usc:") for s, _ in d["units"]) else "scripture")
        node_kind = _NODE_KIND.get(kind, "passage")
        maps = structure.load_reference_maps(ref_maps_cfg, extra=meta.get("reference_map"))
        profile = {"kind": kind, "work": meta.get("work")}
        title = d["title"] or path.rsplit("/", 1)[-1]
        try:                                        # provenance for the nodes/edges
            kb.register_source(path, title, source_type=kind)
        except Exception as e:
            log.warning("citations: could not register %s: %s", path, e)

        # ── units → citation edges ───────────────────────────────────────────
        for key, text in d["units"]:
            if not key:
                continue
            unit = kb._new_node(key, node_kind, (text or "")[:_SUMMARY_CAP], None, [])
            kb.add_node_support(unit, path, summary=(text or "")[:_SUMMARY_CAP])
            n_units += 1
            book = structure.book_of_key(key)
            for r in structure.parse_citations(text, profile, book=book, maps=maps):
                if r.key == key:
                    continue
                n_refs += 1
                tgt = kb._new_node(r.key, _NODE_KIND.get(r.kind, node_kind), "", None, [])
                _eid, how = kb.add_edge(unit, tgt, family="citation", type="cites",
                                        doc_id=path, evidence=key)
                if how == "insert":
                    n_edges += 1

        # ── commentary → 'annotates' its verse + its own citations ───────────
        for anchor_key, note in d["notes"]:
            if not anchor_key or not note:
                continue
            verse = kb._new_node(anchor_key, node_kind, "", None, [])     # ensure the anchor exists
            nid_label = "note:%s:%s" % (anchor_key,
                                        hashlib.sha1(note.encode("utf-8")).hexdigest()[:10])
            cnode = kb._new_node(nid_label, "commentary", note[:_SUMMARY_CAP], None, [])
            kb.add_node_support(cnode, path, summary=note[:_SUMMARY_CAP])
            n_notes += 1
            _eid, how = kb.add_edge(cnode, verse, family="commentary", type="annotates",
                                    doc_id=path, evidence=anchor_key)
            if how == "insert":
                n_annot += 1
            book = structure.book_of_key(anchor_key)
            for r in structure.parse_citations(note, profile, book=book, maps=maps):
                if r.key == anchor_key:
                    continue                        # the note referencing its own verse
                n_refs += 1
                tgt = kb._new_node(r.key, _NODE_KIND.get(r.kind, node_kind), "", None, [])
                _eid, how = kb.add_edge(cnode, tgt, family="citation", type="cites",
                                        doc_id=path, evidence=nid_label)
                if how == "insert":
                    n_edges += 1
        log.info("citations: %s (%s) — %d unit(s), %d note(s)", title, kind,
                 len(d["units"]), len(d["notes"]))

    kb.db.commit()
    out = {"docs": len(docs), "units": n_units, "references": n_refs, "edges": n_edges,
           "notes": n_notes, "annotations": n_annot}
    log.info("citations: %s", out)
    return out
