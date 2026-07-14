"""Provenance-aware removal: undo a bulk dataset import, or eject a bundle.

Why this exists: **threshold tuning** first (import → inspect → adjust →
re-import makes the middle step cheap instead of "rebuild the KB"), and now
also **brain eject** — permanently removing a shipped bundle's contribution
(bundles.eject_bundle passes the bundle's whole doc_id set through the same
machinery).

Why it can be done safely: every imported/distilled row carries its sources in
its ``support`` JSON, and the writers only ever ``INSERT OR IGNORE`` — they
never modify a row that already existed.  So provenance tells the whole story:

  * a row whose support entries are ALL in the removed set was created by that
    provenance → delete;
  * a row with MIXED support was enriched later (link/refine merging) → keep
    it, strip the removed support entries out;
  * a provenance-created node that other edges/cards still lean on must
    SURVIVE even when solely provenance-supported — fusion works both ways
    (CauseNet's insert of "cancer" was ignored because ConceptNet created it
    first, so the shared node's support names only ConceptNet).  A node is
    deleted only when, after the edge and card passes, nothing references it.

Known asymmetry (accepted): if a later distillation asserted the same
edge_hash as an imported edge, the INSERT OR IGNORE swallowed the distilled
copy — the row still looks purely dataset-owned and is removed here.
Re-running distill restores it; the tuning loop re-imports anyway.

After a removal, `build-ann` refreshes the dense node index (deleted nodes
otherwise linger in the ANN file until the next build; lookups of missing rows
are skipped, so it degrades gracefully, not wrongly).
"""
from __future__ import annotations

import json
import logging
import time

log = logging.getLogger("knowledgehost.unimport")

# dataset name (CLI/ops) -> the source doc_id its importer registers
DATASETS = {
    "conceptnet": "conceptnet:5.7",
    "atomic":     "atomic:v4",
    "glucose":    "glucose",
    "causenet":   "causenet:precision",
}


def _ownership(support_json: str, doc_ids: set):
    """-> ('all'|'some'|'none', stripped_support_json).  'all' = every support
    entry is in the removed set; 'some' = mixed (stripped JSON keeps the others)."""
    try:
        entries = json.loads(support_json or "[]")
    except ValueError:
        return "none", support_json
    if not isinstance(entries, list) or not entries:
        return "none", support_json
    keep = [e for e in entries
            if not (isinstance(e, dict) and e.get("doc_id") in doc_ids)]
    if len(keep) == len(entries):
        return "none", support_json
    if not keep:
        return "all", support_json
    return "some", json.dumps(keep)


def _scan(db, table: str, doc_ids: set):
    """One pass over `table` → (delete_ids, [(stripped_support, id)]).  The LIKE
    prefilter is only a cheap narrowing when removing a single doc; for a whole
    bundle the JSON decides on a full scan (support is small, this is fine)."""
    del_ids, upd = [], []
    if len(doc_ids) == 1:
        like = f"%{next(iter(doc_ids))}%"
        rows = db.execute(f"SELECT id, support FROM {table} WHERE support LIKE ?",
                          (like,)).fetchall()
    else:
        rows = db.execute(f"SELECT id, support FROM {table}").fetchall()
    for rid, support in rows:
        own, stripped = _ownership(support, doc_ids)
        if own == "all":
            del_ids.append((rid,))
        elif own == "some":
            upd.append((stripped, rid))
    return del_ids, upd


def remove_docs(kb, doc_ids, *, batch: int = 20_000, dry_run: bool = False) -> dict:
    """Remove everything the ``doc_ids`` set contributed to the KB.  Idempotent.
    dry_run scans and counts but writes nothing (node counts are then upper
    bounds — the still-referenced check needs the edge/card deletes applied)."""
    db = kb.db
    doc_ids = {d for d in doc_ids if d}
    t0 = time.time()
    st = {"docs": len(doc_ids), "edges_deleted": 0, "edges_stripped": 0,
          "cards_deleted": 0, "cards_stripped": 0,
          "nodes_deleted": 0, "nodes_kept_referenced": 0, "nodes_stripped": 0,
          "refs_cleaned": 0}

    e_del, e_upd = _scan(db, "edges", doc_ids)
    c_del, c_upd = _scan(db, "procedure_cards", doc_ids)
    n_cand, n_upd = _scan(db, "nodes", doc_ids)
    st["edges_deleted"], st["edges_stripped"] = len(e_del), len(e_upd)
    st["cards_deleted"], st["cards_stripped"] = len(c_del), len(c_upd)
    st["nodes_stripped"] = len(n_upd)

    if dry_run:
        st["nodes_deleted"] = len(n_cand)          # upper bound (see docstring)
        st["dry_run"] = True
        st["elapsed_s"] = round(time.time() - t0, 1)
        return st

    db.executescript("""
        CREATE TEMP TABLE IF NOT EXISTS _uni_gone(id TEXT PRIMARY KEY);
        CREATE TEMP TABLE IF NOT EXISTS _uni_cand(id TEXT PRIMARY KEY);
        DELETE FROM _uni_gone; DELETE FROM _uni_cand;
    """)

    # ── edges + cards: delete owned, strip mixed ─────────────────────────────
    for table, del_ids, upd in (("edges", e_del, e_upd),
                                ("procedure_cards", c_del, c_upd)):
        for i in range(0, len(del_ids), batch):
            chunk = del_ids[i:i + batch]
            db.executemany("INSERT OR IGNORE INTO _uni_gone(id) VALUES(?)", chunk)
            db.executemany(f"DELETE FROM {table} WHERE id=?", chunk)
        db.executemany(f"UPDATE {table} SET support=? WHERE id=?", upd)

    # ── nodes: candidates are owned; only unreferenced ones may go ───────────
    db.executemany("UPDATE nodes SET support=? WHERE id=?", n_upd)
    for i in range(0, len(n_cand), batch):
        db.executemany("INSERT OR IGNORE INTO _uni_cand(id) VALUES(?)",
                       n_cand[i:i + batch])
    # Anything still referenced — an edge endpoint or a card's subject — stays,
    # whoever created it: deleting shared structure would orphan other sources.
    db.execute("""
        INSERT OR IGNORE INTO _uni_gone(id)
        SELECT c.id FROM _uni_cand c
        WHERE NOT EXISTS (SELECT 1 FROM edges e WHERE e.src_id = c.id)
          AND NOT EXISTS (SELECT 1 FROM edges e WHERE e.dst_id = c.id)
          AND NOT EXISTS (SELECT 1 FROM procedure_cards p WHERE p.node_id = c.id)
    """)
    gone_nodes = db.execute(
        "SELECT COUNT(*) FROM _uni_gone WHERE id IN (SELECT id FROM _uni_cand)").fetchall()[0][0]
    db.execute("DELETE FROM nodes WHERE id IN (SELECT id FROM _uni_gone)")
    st["nodes_deleted"] = gone_nodes
    st["nodes_kept_referenced"] = len(n_cand) - gone_nodes

    # ── referencing side tables: facets, surfaces, merge queue ──────────────
    # target ids are content hashes, so matching by id alone is unambiguous.
    n = 0
    for sql in (
        "DELETE FROM facets WHERE target_id IN (SELECT id FROM _uni_gone)",
        "DELETE FROM surface_questions WHERE target_id IN (SELECT id FROM _uni_gone)",
        "DELETE FROM surface_propositions WHERE target_id IN (SELECT id FROM _uni_gone)",
        "DELETE FROM node_merge_candidates WHERE node_a IN (SELECT id FROM _uni_gone)"
        " OR node_b IN (SELECT id FROM _uni_gone)",
    ):
        n += db.execute(sql).rowcount
    st["refs_cleaned"] = n

    db.executemany("DELETE FROM source_registry WHERE doc_id=?",
                   [(d,) for d in doc_ids])
    db.executescript("DELETE FROM _uni_gone; DELETE FROM _uni_cand;")
    db.commit()

    # the resident dense-search caches now hold deleted nodes — drop them the
    # same way the importers do, so the next query reloads from disk.
    kb._nodes_loaded = False
    kb._node_ids, kb._node_vecs, kb._node_mat = [], [], None

    st["elapsed_s"] = round(time.time() - t0, 1)
    return st


def unimport(kb, doc_id: str, *, batch: int = 20_000) -> dict:
    """Remove everything `doc_id` contributed to the KB.  Idempotent.  (The
    original dataset-undo entry point — now a one-doc remove_docs.)"""
    st = remove_docs(kb, {doc_id}, batch=batch)
    st = {"doc_id": doc_id, **st}
    log.info("unimport %s: %s", doc_id, st)
    return st
