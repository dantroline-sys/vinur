"""ANN battery — the dense-index lane for EXTERNAL standalone KBs (collect
collections) and the collect-time automation around it.

Covers: build_for_file over a bare .kdb (read-only, no KB class), the sidecar
contract (<file>.ann.usearch + .ids.json — where a consumer KB looks), query
correctness, the min_nodes floor, pack._index_collection's rebuild-if-exists /
threshold-if-absent rule, the scratch pre-link build's cache reset, and the
CLI --target branch.  Needs usearch + numpy (both in the recommended extras);
exits SKIPPED loudly when absent rather than passing silently.
"""
import logging
import os
import sqlite3
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from knowledgehost import ann as ann_mod  # noqa: E402

if not ann_mod.available():
    print("SKIPPED: usearch/numpy not installed — the ANN battery needs them "
          "(uv sync / pip install usearch numpy)")
    raise SystemExit(1)

import numpy as np  # noqa: E402

from knowledgehost import pack as pack_mod  # noqa: E402
from knowledgehost.__main__ import _run_build_ann  # noqa: E402
from knowledgehost.kb import KB  # noqa: E402

PASS = FAIL = 0


def check(name, cond):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  ok  {name}")
    else:
        FAIL += 1
        print(f"FAIL  {name}")


DIM = 8

_NODES_DDL = ("CREATE TABLE nodes(id TEXT PRIMARY KEY, label TEXT, kind TEXT, "
              "summary TEXT, aliases TEXT, support TEXT, "
              "status TEXT DEFAULT 'active', embedding BLOB)")


def _vec(i):
    v = np.zeros(DIM, dtype="f4")
    v[i % DIM] = 1.0                                   # one-hot ⇒ exact nearest is itself
    return v


def _mk_kb(path, n):
    con = sqlite3.connect(path)
    con.execute(_NODES_DDL)
    for i in range(n):
        con.execute("INSERT INTO nodes(id, label, embedding) VALUES(?,?,?)",
                    (f"n{i}", f"node {i}", _vec(i).tobytes()))
    con.commit()
    con.close()


_KNOBS = {"ann_connectivity": 16, "ann_expansion_add": 64,
          "ann_expansion_search": 64, "ann_dtype": "f16", "ann_min_nodes": 50}


def test_build_for_file():
    with tempfile.TemporaryDirectory() as td:
        kdb = os.path.join(td, "col.kdb")
        _mk_kb(kdb, 6)
        st = ann_mod.build_for_file(kdb, kdb + ".ann", min_nodes=0)
        check("external build reports built", st.get("built") is True and st["nodes"] == 6)
        check("sidecars land next to the .kdb (the consumer KB contract)",
              os.path.exists(kdb + ".ann.usearch") and os.path.exists(kdb + ".ann.ids.json"))
        idx = ann_mod.AnnIndex.load(kdb + ".ann")
        hits = idx.query(_vec(2), 3)
        check("query returns the exact node first",
              hits and hits[0][0] == "n2" and hits[0][1] > 0.99)
        # threshold floor: a fresh build below min_nodes is refused with the reason named
        kdb2 = os.path.join(td, "small.kdb")
        _mk_kb(kdb2, 3)
        st2 = ann_mod.build_for_file(kdb2, kdb2 + ".ann", min_nodes=50)
        check("below the floor: not built, reason names ann_min_nodes",
              st2.get("built") is False and "ann_min_nodes" in str(st2.get("reason")))
        check("below the floor: no sidecars appear",
              not os.path.exists(kdb2 + ".ann.usearch"))


def test_build_for_file_refuses_junk():
    with tempfile.TemporaryDirectory() as td:
        try:
            ann_mod.build_for_file(os.path.join(td, "absent.kdb"), "x.ann")
            check("missing file raises ValueError", False)
        except ValueError:
            check("missing file raises ValueError", True)
        junk = os.path.join(td, "junk.db")
        sqlite3.connect(junk).execute("CREATE TABLE t(x)")
        try:
            ann_mod.build_for_file(junk, junk + ".ann")
            check("a non-KB sqlite raises ValueError", False)
        except ValueError:
            check("a non-KB sqlite raises ValueError", True)


def test_index_collection_rules():
    """No sidecar + below threshold → skip; an EXISTING sidecar → rebuild even below
    threshold (after a merge it is stale by definition)."""
    with tempfile.TemporaryDirectory() as td:
        kdb = os.path.join(td, "col.kdb")
        _mk_kb(kdb, 6)                                 # 6 nodes < ann_min_nodes 50
        said = []
        st = pack_mod._index_collection(kdb, dict(_KNOBS), said.append)
        check("collection below threshold, no sidecar: skipped",
              st is not None and st.get("built") is False
              and not os.path.exists(kdb + ".ann.usearch"))
        ann_mod.build_for_file(kdb, kdb + ".ann", min_nodes=0)   # user built one (--target)
        st = pack_mod._index_collection(kdb, dict(_KNOBS), said.append)
        check("existing sidecar: rebuilt even below threshold",
              st is not None and st.get("built") is True)
        check("the refresh is narrated", any("refreshed" in s for s in said))


def test_scratch_index_resets_kb_cache():
    """_index_scratch must make the OPEN scratch KB see the fresh index — link's
    _get_ann() had already cached 'no index' by the time distill finished."""
    with tempfile.TemporaryDirectory() as td:
        cfg = {"kb_path": os.path.join(td, "kb.db"), **_KNOBS,
               "ann_path": os.path.join(td, "kb.db.ann"), "ann_min_nodes": 1}
        kb = KB(cfg)
        try:
            for i in range(6):
                kb.db.execute("INSERT INTO nodes(id, label, status, embedding) "
                              "VALUES(?,?,'active',?)", (f"n{i}", f"node {i}",
                                                         _vec(i).tobytes()))
            kb.db.commit()
            check("before the build, _get_ann caches None", kb._get_ann() is None)
            said = []
            pack_mod._index_scratch(kb, cfg, said.append, "collect")
            ann = kb._get_ann()
            check("after _index_scratch, the SAME kb serves the index",
                  ann is not None and len(ann) == 6)
            check("the build is narrated for the collect log",
                  any("dense index built" in s for s in said))
            nid, sim = ann.query(_vec(4), 1)[0]
            check("linked-stage neighbour query is exact", nid == "n4" and sim > 0.99)
        finally:
            kb.close()


def test_cli_target_branch():
    log = logging.getLogger("ann_test")
    with tempfile.TemporaryDirectory() as td:
        kdb = os.path.join(td, "col.kdb")
        _mk_kb(kdb, 6)
        rc = _run_build_ann(dict(_KNOBS), log, target=kdb)
        check("--target builds an external KB's sidecars (min_nodes floor waived)",
              rc == 0 and os.path.exists(kdb + ".ann.usearch"))
        rc = _run_build_ann(dict(_KNOBS), log, target=os.path.join(td, "nope.kdb"))
        check("--target on a missing file fails loudly", rc == 1)


def main():
    test_build_for_file()
    test_build_for_file_refuses_junk()
    test_index_collection_rules()
    test_scratch_index_resets_kb_cache()
    test_cli_target_branch()
    print(f"\n{PASS} passed, {FAIL} failed")
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
