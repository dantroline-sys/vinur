"""Yield audit — find documents whose distillation produced far less than their
size predicts, and put them back on the queue.

A distil pass can stamp every chunk of a document "done" while landing almost
nothing: an output budget that 400s each request, an LM that answers with
unparseable JSON on the single-endpoint path (a parse-fail counts as done), a
claim poisoned by a copy that never landed.  The checkpoint is honest about
*coverage* but says nothing about *yield*, so such a document looks finished
in every view.  This module asks the question the checkpoint can't: for the
chunks that actually went through an extractor, how many nodes / edges / cards
cite this document — and is that abnormally low against the corpus?

  audit(store, kb)   → the documents flagged, with expected vs actual
  reset_docs(…)      → un-stamp them (checkpoint, recard stamp, dupe rows, text
                       claims) so the next distil pass picks them up again.
                       Existing items stay — ids are content hashes, so a
                       re-distil merges into them rather than duplicating.
                       Furniture-zone skips are kept: they were deliberate.

"Abnormally low" is relative, not absolute: the corpus median yield per landed
chunk (over healthy documents) sets the expectation, and a document is flagged
when its yield is at or below `ratio` of that expectation — so a corpus of
terse legal sections and one of chatty how-tos each judge their own.
"""
from __future__ import annotations

import json
import logging
import sqlite3
import statistics

log = logging.getLogger("knowledgehost.yield_audit")

ITEM_TABLES = ("nodes", "edges", "procedure_cards")
DEFAULT_MIN_CHUNKS = 3          # below this a document is too small to judge
DEFAULT_RATIO = 0.1             # flagged at ≤ 10% of the expected yield
_BATCH = 500


# ── per-document yield ───────────────────────────────────────────────────────

def doc_yield(db: sqlite3.Connection) -> dict:
    """{doc_id: {nodes, edges, cards}} — how many active items cite each
    document in their support.  One grouped json_each query per table; a
    build without JSON1 falls back to a Python scan (support is small)."""
    out: dict = {}
    col = {"nodes": "nodes", "edges": "edges", "procedure_cards": "cards"}
    for table in ITEM_TABLES:
        key = col[table]
        try:
            rows = db.execute(
                f"SELECT json_extract(j.value, '$.doc_id') AS d, COUNT(DISTINCT t.id) "
                f"FROM {table} t, json_each(t.support) j "
                f"WHERE t.status='active' AND json_valid(t.support) "
                f"GROUP BY d").fetchall()
            for d, n in rows:
                if d:
                    out.setdefault(str(d), {"nodes": 0, "edges": 0, "cards": 0})[key] += int(n)
        except sqlite3.OperationalError:          # no JSON1 → scan
            for rid, support in db.execute(
                    f"SELECT id, support FROM {table} WHERE status='active'"):
                try:
                    entries = json.loads(support or "[]")
                except ValueError:
                    continue
                seen = set()
                for e in entries if isinstance(entries, list) else []:
                    d = e.get("doc_id") if isinstance(e, dict) else None
                    if d and d not in seen:
                        seen.add(d)
                        out.setdefault(str(d), {"nodes": 0, "edges": 0, "cards": 0})[key] += 1
    return out


def doc_progress_all(store_db_path: str, kb_path: str) -> dict:
    """{doc_id: {title, chunks, distilled, dupes, zoned}} for EVERY ingested
    document — the per-document progress the Sources view computes for the
    listed rows, asked once for the whole store (kb.db ATTACHed)."""
    con = sqlite3.connect(store_db_path, timeout=10)
    try:
        con.execute("PRAGMA busy_timeout=10000")
        con.execute("ATTACH ? AS kbdb", (str(kb_path),))
        out = {}
        for doc, title, tot, dist, dupe, zoned in con.execute(
                "SELECT path_or_url, MAX(title), COUNT(*), "
                "SUM(EXISTS(SELECT 1 FROM kbdb.distilled_chunks d WHERE d.chunk_id=chunks.id)), "
                "SUM(EXISTS(SELECT 1 FROM kbdb.chunk_dupes cd WHERE cd.chunk_id=chunks.id)), "
                "SUM(EXISTS(SELECT 1 FROM kbdb.zone_skips z WHERE z.chunk_id=chunks.id)) "
                "FROM chunks GROUP BY path_or_url"):
            out[doc] = {"title": title or doc, "chunks": int(tot),
                        "distilled": int(dist or 0), "dupes": int(dupe or 0),
                        "zoned": int(zoned or 0)}
        return out
    finally:
        con.close()


# ── the audit ────────────────────────────────────────────────────────────────

def audit(store, kb, *, min_chunks: int = DEFAULT_MIN_CHUNKS,
          ratio: float = DEFAULT_RATIO, limit: int = 200) -> dict:
    """Flag documents whose yield per landed chunk is at or below `ratio` × the
    corpus median.  `landed` = chunks stamped distilled that were NOT dupe
    stamps — the ones an extractor actually saw.  Documents with fewer than
    `min_chunks` landed are not judged (too small to call).

    Returns {median_per_chunk, judged, healthy, flagged:[…], truncated}; each
    flagged row carries doc_id, title, chunks, landed, items (+ nodes/edges/
    cards), expected, ratio and a plain-language reason."""
    db_path = getattr(store, "cfg", {}).get("db_path") if store is not None else None
    if not db_path:
        return {"ok": False, "error": "no sqlite chunk store on this box",
                "flagged": [], "judged": 0, "healthy": 0, "median_per_chunk": None}
    prog = doc_progress_all(db_path, kb.path)
    yields = doc_yield(kb.db)
    judged, per_chunk = [], []
    for doc, p in prog.items():
        landed = p["distilled"] - p["dupes"]
        if landed < max(1, int(min_chunks)):
            continue
        y = yields.get(doc) or {"nodes": 0, "edges": 0, "cards": 0}
        items = y["nodes"] + y["edges"] + y["cards"]
        judged.append((doc, p, landed, y, items))
        if items > 0:
            per_chunk.append(items / landed)
    median = statistics.median(per_chunk) if per_chunk else None
    flagged = []
    for doc, p, landed, y, items in judged:
        expected = (median * landed) if median else None
        if expected is None:
            # no healthy document to calibrate against: only outright zeros
            bad = items == 0
            r = 0.0 if items == 0 else None
        else:
            r = items / expected if expected else None
            bad = r is not None and r <= float(ratio)
        if not bad:
            continue
        if items == 0:
            reason = f"{landed} chunk(s) went through the extractor and NOTHING landed"
        else:
            reason = (f"{items} item(s) from {landed} chunk(s) — about "
                      f"{round(expected)} expected at this corpus's rate")
        flagged.append({"doc_id": doc, "title": p["title"], "chunks": p["chunks"],
                        "distilled": p["distilled"], "dupes": p["dupes"],
                        "zoned": p["zoned"], "landed": landed, "items": items,
                        "nodes": y["nodes"], "edges": y["edges"], "cards": y["cards"],
                        "expected": round(expected, 1) if expected is not None else None,
                        "ratio": round(r, 3) if r is not None else None,
                        "reason": reason})
    flagged.sort(key=lambda f: (f["ratio"] if f["ratio"] is not None else 0.0, -f["landed"]))
    return {"ok": True, "median_per_chunk": round(median, 2) if median else None,
            "judged": len(judged), "healthy": len(per_chunk),
            "min_chunks": int(min_chunks), "ratio": float(ratio),
            "flagged": flagged[: int(limit)], "flagged_total": len(flagged),
            "truncated": len(flagged) > int(limit)}


# ── the reset ────────────────────────────────────────────────────────────────

def chunk_ids_for(store_db_path: str, doc_ids) -> dict:
    """{doc_id: [chunk ids]} straight from the chunk store."""
    con = sqlite3.connect(store_db_path, timeout=10)
    try:
        con.execute("PRAGMA busy_timeout=10000")
        out = {}
        for d in doc_ids:
            out[d] = [r[0] for r in con.execute(
                "SELECT id FROM chunks WHERE path_or_url=?", (str(d),))]
        return out
    finally:
        con.close()


def unstamp_chunks(kb, chunk_ids) -> int:
    """Put chunks back on the distil queue: drop their distilled/recard stamps
    and dupe rows, and release any text claims they hold (so a copy elsewhere
    can win the text, or they reclaim it themselves next pass).  zone_skips
    are left alone — furniture was skipped on purpose.  Items already in the
    graph stay; content-hash ids mean a re-distil merges, not duplicates."""
    ids = [str(c) for c in chunk_ids if c]
    n = 0
    for i in range(0, len(ids), _BATCH):
        batch = ids[i:i + _BATCH]
        marks = ",".join("?" * len(batch))
        cur = kb.db.execute(f"DELETE FROM distilled_chunks WHERE chunk_id IN ({marks})", batch)
        n += cur.rowcount or 0
        for table in ("recarded_chunks", "chunk_dupes", "chunk_texts"):
            try:
                kb.db.execute(f"DELETE FROM {table} WHERE chunk_id IN ({marks})", batch)
            except sqlite3.OperationalError:          # an older kb.db
                pass
    kb.db.commit()
    return n


def reset_docs(store, kb, doc_ids) -> dict:
    """Re-queue whole documents for distillation.  Returns {docs, chunks,
    per_doc:{doc: unstamped}}; a doc the store doesn't know counts 0."""
    db_path = getattr(store, "cfg", {}).get("db_path") if store is not None else None
    if not db_path:
        return {"ok": False, "error": "no sqlite chunk store on this box",
                "docs": 0, "chunks": 0, "per_doc": {}}
    wanted = [str(d) for d in (doc_ids or []) if str(d).strip()]
    by_doc = chunk_ids_for(db_path, wanted)
    per_doc, total = {}, 0
    for doc, cids in by_doc.items():
        n = unstamp_chunks(kb, cids) if cids else 0
        per_doc[doc] = n
        total += n
    log.info("redistil: re-queued %d chunk(s) across %d document(s)", total, len(wanted))
    return {"ok": True, "docs": len(wanted), "chunks": total, "per_doc": per_doc}
