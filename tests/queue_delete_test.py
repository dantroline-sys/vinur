#!/usr/bin/env python
"""queue delete — purge_source() removes a queued document and everything ingest wrote
for it, leaving other documents untouched.

  * chunks for the doc are gone; a sibling doc's chunks remain
  * vectors AND the FTS index are cleaned via the chunks AFTER-DELETE trigger
  * doc_meta and the manifest entry are cleared (so re-ingest re-adds it rather than
    skipping it as 'unchanged')

    python tests/queue_delete_test.py
"""

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from knowledgehost import store as S    # noqa: E402

PASS = FAIL = 0


def check(name, cond):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  ok  {name}")
    else:
        FAIL += 1
        print(f"FAIL  {name}")


def _rec(cid, path, text, vec):
    return {"id": cid, "path_or_url": path, "text": text, "title": "T", "section": "",
            "source_type": "pdf", "tokens": len(text.split()), "vector": vec}


def _count(db, sql, *p):
    return db.execute(sql, p).fetchone()[0]


def main():
    tmp = tempfile.mkdtemp()
    store = S.make_store({"backend": "sqlite", "db_path": str(Path(tmp) / "index.db")})
    A, B = "/docA.pdf", "/docB.pdf"
    store.add_chunks([_rec("a1", A, "alpha zebra content", [0.1, 0.2]),
                      _rec("a2", A, "more alpha material", [0.3, 0.4])])
    store.add_chunks([_rec("b1", B, "beta gamma delta", [0.5, 0.6])])
    store.set_doc_meta(A, {"k": "v"})
    store.manifest.set(A, "hash123", 1.0, 1, "active")
    db = store.db

    check("setup: A has 2 chunks, B has 1", _count(db, "SELECT COUNT(*) FROM chunks WHERE path_or_url=?", A) == 2
          and _count(db, "SELECT COUNT(*) FROM chunks WHERE path_or_url=?", B) == 1)
    check("setup: 3 vectors, A doc_meta + manifest present, FTS finds 'zebra'",
          _count(db, "SELECT COUNT(*) FROM vectors") == 3
          and store.get_doc_meta(A) == {"k": "v"} and store.manifest.get(A) is not None
          and _count(db, "SELECT COUNT(*) FROM chunks_fts WHERE chunks_fts MATCH 'zebra'") == 1)

    res = store.purge_source(A)
    check("purge_source reports the chunk count removed", res == {"chunks": 2})

    check("A's chunks are gone; B untouched",
          _count(db, "SELECT COUNT(*) FROM chunks WHERE path_or_url=?", A) == 0
          and _count(db, "SELECT COUNT(*) FROM chunks WHERE path_or_url=?", B) == 1)
    check("vectors cascaded via the delete trigger (only B's remains)",
          _count(db, "SELECT COUNT(*) FROM vectors") == 1
          and _count(db, "SELECT COUNT(*) FROM vectors WHERE id='b1'") == 1)
    check("FTS index no longer matches A's text, still matches B's",
          _count(db, "SELECT COUNT(*) FROM chunks_fts WHERE chunks_fts MATCH 'zebra'") == 0
          and _count(db, "SELECT COUNT(*) FROM chunks_fts WHERE chunks_fts MATCH 'gamma'") == 1)
    check("A's doc_meta cleared; manifest entry cleared (re-ingest will re-add)",
          store.get_doc_meta(A) is None and store.manifest.get(A) is None)

    # idempotent: purging an unknown / already-purged path removes nothing, doesn't raise
    check("purging an already-gone doc is a no-op", store.purge_source(A) == {"chunks": 0})

    print()
    if FAIL:
        print(f"{FAIL} FAILURE(S)")
        return 1
    print(f"ALL OK ({PASS} checks)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
