#!/usr/bin/env python
"""clear-queue — bulk-revert the distillation queue and quarantine the source files.

Covers store.clear_queue (the DB side) and ingest.clear_ingest_queue /
quarantine_docs (the file-move side):

  * the QUEUE = docs with nothing distilled (not in kb.db source_registry);
    default clears exactly those (chunks → vectors + FTS via trigger, doc_meta,
    manifest), leaving distilled + partially-distilled docs untouched;
  * include_partial ALSO trims a partial doc's still-pending chunks, keeping its
    distilled chunk (and the cards in kb.db);
  * quarantine MOVES a queued file under a source root into quarantine_dir with
    its tree preserved; URL/virtual docs and files outside any root are skipped;
    the intended layout is a `quarantined/` folder INSIDE a source root, which the
    crawl walk prunes so reverted files never re-ingest; a quarantine dir that IS
    or CONTAINS a source root is refused, and an unset one suggests <root>/quarantined;
  * dry_run counts and writes/moves nothing.

    python tests/clear_queue_test.py
"""

import os
import sqlite3
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from knowledgehost import store as S       # noqa: E402
from knowledgehost import ingest as I      # noqa: E402

PASS = FAIL = 0


def check(name, cond):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  ok  {name}")
    else:
        FAIL += 1
        print(f"FAIL  {name}")


def _rec(cid, path, text):
    return {"id": cid, "path_or_url": path, "text": text, "title": "T", "section": "",
            "source_type": "pdf", "tokens": len(text.split()), "vector": [0.1, 0.2]}


def _make_kb(kb_path, registry, distilled):
    con = sqlite3.connect(kb_path)
    con.executescript(
        "CREATE TABLE source_registry(doc_id TEXT PRIMARY KEY);"
        "CREATE TABLE distilled_chunks(chunk_id TEXT PRIMARY KEY);"
        "CREATE TABLE zone_skips(chunk_id TEXT PRIMARY KEY);"
        "CREATE TABLE chunk_dupes(chunk_id TEXT PRIMARY KEY);")
    con.executemany("INSERT INTO source_registry(doc_id) VALUES(?)", [(d,) for d in registry])
    con.executemany("INSERT INTO distilled_chunks(chunk_id) VALUES(?)", [(c,) for c in distilled])
    con.commit()
    con.close()


def _cnt(store, path):
    return store.db.execute("SELECT COUNT(*) FROM chunks WHERE path_or_url=?", (path,)).fetchone()[0]


def _build():
    """A fresh tmp world: store + kb.db + real files.  Returns handles/paths."""
    tmp = Path(tempfile.mkdtemp())
    root = tmp / "src"
    (root / "a").mkdir(parents=True)
    afile = root / "a" / "a.pdf"
    afile.write_text("alpha zebra content")           # A: queued, a real file under a root
    (tmp / "other").mkdir()
    efile = tmp / "other" / "e.pdf"
    efile.write_text("e outside content")             # E: queued, file OUTSIDE any root
    A, E, D = str(afile), str(efile), "http://example.com/x"   # D: queued URL (virtual)
    B, C = "/docB.pdf", "/docC.pdf"                    # B partial, C complete (in registry)

    store = S.make_store({"backend": "sqlite", "db_path": str(tmp / "index.db")})
    store.add_chunks([_rec("a1", A, "alpha one"), _rec("a2", A, "alpha two")])
    store.add_chunks([_rec("b1", B, "beta one"), _rec("b2", B, "beta two")])
    store.add_chunks([_rec("c1", C, "cee one"), _rec("c2", C, "cee two")])
    store.add_chunks([_rec("d1", D, "dee one")])
    store.add_chunks([_rec("e1", E, "eee one")])
    store.set_doc_meta(A, {"k": "v"})
    store.manifest.set(A, "h", 1.0, 1, "active")

    kb_path = str(tmp / "kb.db")
    _make_kb(kb_path, registry=[B, C], distilled=["b1", "c1", "c2"])
    cfg = {"sources": [str(root)], "quarantine_dir": str(tmp / "quarantined")}
    return tmp, store, kb_path, cfg, {"A": A, "B": B, "C": C, "D": D, "E": E,
                                      "root": root, "afile": afile, "efile": efile}


def main():
    # ── DB side: dry-run counts ──────────────────────────────────────────────
    tmp, store, kb_path, cfg, p = _build()
    dry = store.clear_queue(kb_path, dry_run=True)
    check("dry-run: 3 untouched docs (A,D,E), 4 queued chunks",
          dry["queued_docs"] == 3 and dry["queued_chunks"] == 4)
    check("dry-run: 1 partial doc (B), 1 pending chunk (b2)",
          dry["partial_docs"] == 1 and dry["partial_chunks"] == 1)
    check("dry-run: queued_doc_ids are exactly A,D,E",
          set(dry["queued_doc_ids"]) == {p["A"], p["D"], p["E"]})
    check("dry-run: removes nothing", dry["chunks_removed"] == 0 and _cnt(store, p["A"]) == 2)

    # ── DB side: default (untouched only) ────────────────────────────────────
    res = store.clear_queue(kb_path, dry_run=False)
    check("default: removed all 4 untouched chunks", res["chunks_removed"] == 4)
    check("default: A/D/E chunks gone",
          _cnt(store, p["A"]) == 0 and _cnt(store, p["D"]) == 0 and _cnt(store, p["E"]) == 0)
    check("default: partial B untouched (still 2 chunks), complete C untouched (2)",
          _cnt(store, p["B"]) == 2 and _cnt(store, p["C"]) == 2)
    check("default: A's doc_meta + manifest cleared",
          store.get_doc_meta(p["A"]) is None and store.manifest.get(p["A"]) is None)
    check("default: vectors cascaded (b1,b2,c1,c2 remain = 4)",
          store.db.execute("SELECT COUNT(*) FROM vectors").fetchone()[0] == 4)

    # ── DB side: include_partial trims B's pending chunk only ─────────────────
    tmp2, store2, kb2, cfg2, p2 = _build()
    res2 = store2.clear_queue(kb2, include_partial=True, dry_run=False)
    check("partial: removed 4 untouched + 1 pending (b2) = 5", res2["chunks_removed"] == 5)
    check("partial: B keeps only its distilled chunk b1",
          _cnt(store2, p2["B"]) == 1
          and store2.db.execute("SELECT id FROM chunks WHERE path_or_url=?",
                                (p2["B"],)).fetchone()[0] == "b1")
    check("partial: complete C still fully intact (2 chunks)", _cnt(store2, p2["C"]) == 2)

    # ── file side: quarantine dry-run classifies without moving ──────────────
    tmp3, store3, kb3, cfg3, p3 = _build()
    prev = I.clear_ingest_queue(cfg3, store3, kb3, quarantine=True, dry_run=True)
    q = prev["quarantine"]
    check("quarantine dry-run: 1 movable file (A), 1 URL, 1 outside-root",
          q["moved"] == 1 and q["skipped_nonfile"] == 1 and q["skipped_outside"] == 1)
    check("quarantine dry-run: A's file still in place, chunks still present",
          p3["afile"].exists() and _cnt(store3, p3["A"]) == 2)

    # ── file side: execute moves the file, tree preserved, chunks dropped ────
    res3 = I.clear_ingest_queue(cfg3, store3, kb3, quarantine=True, dry_run=False)
    dest = Path(cfg3["quarantine_dir"]) / "a" / "a.pdf"     # single root => no basename prefix
    check("execute: ok + 4 chunks removed", res3["ok"] and res3["chunks_removed"] == 4)
    check("execute: A's file MOVED out of the ingest tree", not p3["afile"].exists())
    check("execute: A's file lands in quarantine with its tree preserved", dest.is_file())
    check("execute: the outside-root file E was left in place", p3["efile"].exists())
    check("execute: quarantine reported 1 moved, 0 errors",
          res3["quarantine"]["moved"] == 1 and res3["quarantine"]["errors"] == 0)

    # ── the intended layout: a quarantine dir INSIDE a source root ────────────
    # (~/Documents/quarantined).  The file moves in with its tree preserved, and
    # the crawl skips the quarantine dir so what was reverted never re-ingests.
    tmp4, store4, kb4, cfg4, p4 = _build()
    qin = p4["root"] / "quarantined"
    cfg4["quarantine_dir"] = str(qin)                     # INSIDE the source root
    ok4 = I.clear_ingest_queue(cfg4, store4, kb4, quarantine=True, dry_run=False)
    check("inside-root: accepted, file moved to <root>/quarantined/a/a.pdf, source gone",
          ok4["ok"] and (qin / "a" / "a.pdf").is_file() and not p4["afile"].exists())
    walked = [dp for dp, _d, _f in I._walk_pruned(str(p4["root"]),
                                                  I._skip_dirs({"quarantine_dir": str(qin)}))]
    check("inside-root: the crawl walk never descends into the quarantine dir",
          all(not os.path.realpath(dp).startswith(os.path.realpath(str(qin)) + os.sep)
              and os.path.realpath(dp) != os.path.realpath(str(qin)) for dp in walked))

    # ── degenerate: a quarantine dir that IS or CONTAINS a source root is refused ─
    tmp6, store6, kb6, cfg6, p6 = _build()
    eq = I.clear_ingest_queue({**cfg6, "quarantine_dir": str(p6["root"])},
                              store6, kb6, quarantine=True, dry_run=True)
    check("degenerate: quarantine == a source root is refused (nothing removed)",
          eq["ok"] is False and "equals or contains" in (eq.get("error") or "")
          and _cnt(store6, p6["A"]) == 2)
    contains = I.clear_ingest_queue({**cfg6, "quarantine_dir": str(tmp6)},
                                    store6, kb6, quarantine=True, dry_run=True)
    check("degenerate: quarantine CONTAINING a source root is refused",
          contains["ok"] is False and "equals or contains" in (contains.get("error") or ""))

    # ── explicit-dir requirement: unset quarantine_dir refuses (no DB changes) ─
    tmp5, store5, kb5, cfg5, p5 = _build()
    cfg5["quarantine_dir"] = ""                            # user hasn't chosen one
    need = I.clear_ingest_queue(cfg5, store5, kb5, quarantine=True, dry_run=True)
    check("unset quarantine_dir: refused with needs_quarantine_dir, nothing removed",
          need["ok"] is False and need.get("needs_quarantine_dir") is True
          and _cnt(store5, p5["A"]) == 2)
    check("unset quarantine_dir: suggests <root>/quarantined as the default",
          need.get("suggested_quarantine_dir")
          == os.path.join(os.path.realpath(str(p5["root"])), "quarantined"))
    check("unset quarantine_dir: --no-quarantine still clears DB rows (files left in place)",
          I.clear_ingest_queue(cfg5, store5, kb5, quarantine=False,
                               dry_run=False)["chunks_removed"] == 4
          and p5["afile"].exists())

    print()
    if FAIL:
        print(f"{FAIL} FAILURE(S)")
        return 1
    print(f"ALL OK ({PASS} checks)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
