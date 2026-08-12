"""VINUR — deterministic citation-edge graph for structured (scripture/legal) corpora.

After a structured ingest (ingest._ingest_structured_doc) the raw store holds one chunk
per canonical unit — a verse / a section — keyed by its canonical citation
(bible:John.3.16, usc:17/106).  This pass turns that into a GRAPH, with NO language model:

  * one KB node per unit — label = the canonical key, so units from DIFFERENT documents
    that name the same passage converge on ONE node (node id = hash(label, kind));
  * a 'citation' edge from each unit to every canonical reference its text makes
    (structure.parse_citations, resolved with the document's reference map).

That is the cross-reference payoff: 'Romans 9:16 … see Exodus 33:19' becomes an edge
bible:Rom.9.16 --cites--> bible:Exod.33.19, and a commentary that later cites the same
verse attaches to the same node.  Deterministic and idempotent (content-derived node ids
and edge hashes), so it is safe to re-run — new documents just add their edges.
"""
from __future__ import annotations

import logging

from . import structure

log = logging.getLogger("knowledgehost.citations")

_NODE_KIND = {"scripture": "passage", "legal": "provision"}
_SUMMARY_CAP = 600


def _structured_docs(store) -> dict:
    """Group the raw store's structured chunks by document, in document order:
    {path: (title, [(section_key, text), …])}."""
    docs: dict = {}
    for ch in store.iter_chunks():
        if ch.get("source_type") in ("scripture", "legal"):
            path = ch.get("path_or_url") or ""
            entry = docs.setdefault(path, [ch.get("title") or "", []])
            entry[1].append((ch.get("section") or "", ch.get("text") or ""))
    return docs


def build(store, kb, cfg, *, log=log) -> dict:
    """Build the citation graph over every structured document in the raw store.
    Returns {docs, units, references, edges} (edges = citation edges newly inserted)."""
    docs = _structured_docs(store)
    ref_maps_cfg = cfg.get("reference_maps") or []
    n_units = n_refs = n_edges = 0
    for path, (title, units) in docs.items():
        meta = store.get_doc_meta(path) or {}
        kind = meta.get("kind") or ("legal" if any(s.startswith("usc:") for s, _ in units)
                                    else "scripture")
        node_kind = _NODE_KIND.get(kind, "passage")
        maps = structure.load_reference_maps(ref_maps_cfg, extra=meta.get("reference_map"))
        profile = {"kind": kind, "work": meta.get("work")}
        title = title or path.rsplit("/", 1)[-1]
        try:                                        # provenance for the nodes/edges
            kb.register_source(path, title, source_type=kind)
        except Exception as e:                      # never fail the graph over a registry hiccup
            log.warning("citations: could not register %s: %s", path, e)

        for key, text in units:
            if not key:
                continue
            unit = kb._new_node(key, node_kind, (text or "")[:_SUMMARY_CAP], None, [])
            kb.add_node_support(unit, path, summary=(text or "")[:_SUMMARY_CAP])
            n_units += 1
            book = structure.book_of_key(key)       # for bare 'C:V' within this book
            for r in structure.parse_citations(text, profile, book=book, maps=maps):
                if r.key == key:
                    continue                        # a unit citing itself is not an edge
                n_refs += 1
                tgt = kb._new_node(r.key, _NODE_KIND.get(r.kind, node_kind), "", None, [])
                _eid, how = kb.add_edge(unit, tgt, family="citation", type="cites",
                                        doc_id=path, evidence=key)
                if how == "insert":
                    n_edges += 1
        log.info("citations: %s (%s) — %d unit(s)", title, kind, len(units))

    kb.db.commit()
    out = {"docs": len(docs), "units": n_units, "references": n_refs, "edges": n_edges}
    log.info("citations: %s", out)
    return out
