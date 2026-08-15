"""Link battery — the graph-linkage stage's fan-out, rotation and resume semantics.

A real KB (temp file) + a fake LM: nodes with hand-built embeddings so the
neighbour sets are exact and known, canned judgements per pair.  Covers: edges
written with the right direction/family, the pair checkpoint (re-runs spend no
LM), anchor ROTATION (a `limit` is a moving window — the autopilot's small pass
used to freeze on the first N anchors forever), fan-out concurrency (clones of
one batching endpoint), and the resumable abort (consumed pairs committed,
unfinished anchors unstamped).  Needs numpy (the node matrix).
"""
import os
import sys
import tempfile
import threading
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

try:
    import numpy as np
except Exception:
    print("SKIPPED: numpy not installed — the link battery needs it")
    raise SystemExit(1)

from knowledgehost import link as link_mod  # noqa: E402
from knowledgehost.distill import BackendUnavailable  # noqa: E402
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


def _unit(v):
    v = np.asarray(v, dtype="f4")
    return v / np.linalg.norm(v)


def _mk_kb(td, spec):
    """spec: {node_id: (vector, has_card)}.  Support stays empty → every node is
    anchor-like (_is_anchor treats no-support as a real node)."""
    kb = KB({"kb_path": os.path.join(td, "kb.db")})
    for nid, (vec, has_card) in spec.items():
        kb.db.execute("INSERT INTO nodes(id, label, status, embedding) "
                      "VALUES(?,?, 'active', ?)",
                      (nid, f"label {nid}", _unit(vec).tobytes()))
        if has_card:
            kb.db.execute("INSERT INTO procedure_cards(id, node_id, title, goal, status) "
                          "VALUES(?,?,?,?, 'active')",
                          (f"card-{nid}", nid, f"card {nid}", "a goal"))
    kb.db.commit()
    return kb


class FakeLM:
    """Judges pairs from a canned {frozenset(labels): relation} table.  Mutable
    state (counters, locks) is SHARED across copy.copy clones — exactly what the
    fan-out relies on for one batching server counted as several slots."""

    def __init__(self, verdicts, *, delay=0.0, fail_after=None):
        self.verdicts = verdicts
        self.delay = delay
        self.calls = []                       # shared across clones
        self._lock = threading.Lock()
        self._live = [0]                      # current in-flight
        self.peak = [0]                       # max in-flight ever seen
        self.fail_after = fail_after

    def _content(self, system, user, schema, mtok):
        with self._lock:
            if self.fail_after is not None and len(self.calls) >= self.fail_after:
                raise BackendUnavailable("stub endpoint down")
            self.calls.append(user)
            self._live[0] += 1
            self.peak[0] = max(self.peak[0], self._live[0])
        try:
            if self.delay:
                time.sleep(self.delay)
            pair = frozenset(n for n in ("n0", "n1", "n2", "n3", "n4", "n5", "n6", "n7")
                             if f"label {n}" in user)
            rel = self.verdicts.get(pair, "none")
            return ('{"relation": "%s", "confidence": 0.9, "rationale": "stub"}' % rel)
        finally:
            with self._lock:
                self._live[0] -= 1


def _edges(kb):
    return [(r["src_id"], r["dst_id"], r["family"], r["type"]) for r in
            kb.db.execute("SELECT src_id, dst_id, family, type FROM edges "
                          "WHERE status='active'")]


def _stamped(kb):
    return {r["node_id"] for r in kb.db.execute("SELECT node_id FROM link_anchors")}


# n0 is the anchor; n1 (sim .8) and n2 (sim .6) are its neighbours; n3 is orthogonal
# (below the .5 floor) and must never be judged.
_BASIC = {
    "n0": ([1, 0, 0, 0, 0, 0, 0, 0], True),
    "n1": ([0.8, 0.6, 0, 0, 0, 0, 0, 0], False),
    "n2": ([0.6, 0, 0.8, 0, 0, 0, 0, 0], False),
    "n3": ([0, 0, 0, 1, 0, 0, 0, 0], False),
}


def test_edges_and_checkpoint():
    with tempfile.TemporaryDirectory() as td:
        kb = _mk_kb(td, _BASIC)
        try:
            lm = FakeLM({frozenset(("n0", "n1")): "a_is_a_b",
                         frozenset(("n0", "n2")): "alternative"})
            st = link_mod.link_concepts(kb, lm, {"link_top_k": 4})
            check("both neighbours judged, the sub-floor node never offered",
                  st["judged"] == 2 and len(lm.calls) == 2
                  and not any("label n3" in c for c in lm.calls))
            es = _edges(kb)
            check("is_a points specific → general (a_is_a_b: n0 → n1)",
                  ("n0", "n1", "taxonomic", "is_a") in es)
            check("alternative writes both directions",
                  ("n0", "n2", "functional", "alternative_to") in es
                  and ("n2", "n0", "functional", "alternative_to") in es)
            check("the anchor is rotation-stamped", "n0" in _stamped(kb))
            st2 = link_mod.link_concepts(kb, lm, {"link_top_k": 4})
            check("re-run spends no LM (pair checkpoint)",
                  st2["skipped_checkpoint"] == 2 and len(lm.calls) == 2
                  and st2["judged"] == 0)
        finally:
            kb.close()


def test_limit_is_a_moving_window():
    """Four anchors, limit=2: the second run must take the OTHER two — the old
    anchors[:limit] took the same first two forever."""
    spec = {}
    for i in range(4):                       # 4 card-bearing anchors, mutually similar
        v = [0.0] * DIM
        v[0], v[i + 1] = 1.0, 0.9
        spec[f"n{i}"] = (v, True)
    with tempfile.TemporaryDirectory() as td:
        kb = _mk_kb(td, spec)
        try:
            lm = FakeLM({})                  # every verdict 'none' — checkpoints still land
            link_mod.link_concepts(kb, lm, {"link_top_k": 2}, limit=2)
            first = _stamped(kb)
            check("run 1 visits exactly `limit` anchors", len(first) == 2)
            link_mod.link_concepts(kb, lm, {"link_top_k": 2}, limit=2)
            second = _stamped(kb) - first
            check("run 2 moves on to the OTHER anchors", len(second) == 2
                  and not (second & first))
            st3 = link_mod.link_concepts(kb, lm, {"link_top_k": 2}, limit=2)
            check("run 3 wraps to the oldest stamps and re-offers nothing new",
                  st3["judged"] == 0 and st3["anchor_pool"] == 4)
        finally:
            kb.close()


def test_fanout_concurrency():
    """distill_parallel=3 → three clones of the one endpoint, real overlap."""
    spec = dict(_BASIC)
    for i in (4, 5, 6, 7):                   # more anchors → enough pairs in flight
        v = [0.0] * DIM
        v[0], v[i % DIM] = 1.0, 0.9
        spec[f"n{i}"] = (v, True)
    with tempfile.TemporaryDirectory() as td:
        kb = _mk_kb(td, spec)
        try:
            lm = FakeLM({}, delay=0.08)
            st = link_mod.link_concepts(kb, lm, {"link_top_k": 3, "distill_parallel": 3})
            check("fan-out reports its slot count", st["parallel"] == 3)
            check("judgements actually overlapped", lm.peak[0] >= 2)
            check("every candidate was judged exactly once",
                  st["judged"] == st["candidates"] == len(lm.calls))
        finally:
            kb.close()


def test_abort_is_resumable():
    """A transport failure aborts the run but keeps what was consumed: judged pairs
    stay checkpointed, the unfinished anchor stays UNstamped and goes first next run."""
    with tempfile.TemporaryDirectory() as td:
        kb = _mk_kb(td, _BASIC)
        try:
            lm = FakeLM({frozenset(("n0", "n1")): "a_requires_b"}, fail_after=1)
            try:
                link_mod.link_concepts(kb, lm, {"link_top_k": 4})
                check("BackendUnavailable propagates (resumable abort)", False)
            except BackendUnavailable:
                check("BackendUnavailable propagates (resumable abort)", True)
            done = kb.db.execute("SELECT COUNT(*) c FROM link_pairs").fetchone()["c"]
            check("the consumed pair is committed", done == 1)
            check("the unfinished anchor is NOT stamped", "n0" not in _stamped(kb))
            lm.fail_after = None             # endpoint recovers
            st = link_mod.link_concepts(kb, lm, {"link_top_k": 4})
            check("resume judges only the missing pair",
                  st["judged"] == 1 and st["skipped_checkpoint"] == 1)
            check("requires edge points dependent → prerequisite (n0 → n1)",
                  ("n0", "n1", "functional", "requires") in _edges(kb))
        finally:
            kb.close()


def main():
    test_edges_and_checkpoint()
    test_limit_is_a_moving_window()
    test_fanout_concurrency()
    test_abort_is_resumable()
    print(f"\n{PASS} passed, {FAIL} failed")
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
