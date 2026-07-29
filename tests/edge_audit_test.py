#!/usr/bin/env python
"""edge-audit — the LM-free graph-hygiene pass (knowledgehost/edge_audit.py).

Builds a tiny KB with hand-placed node embeddings and checks:
  * SOUND-ALIKE fires on look-alike labels that are semantically distant
    (complement↔compliment), and NOT on look-alike labels that ARE related
    (type-1 ↔ type-2 diabetes: high orth but high semantic — the AND guard).
  * UNGROUNDED fires on distant + uncited + never-co-occurs (banana↔tractor), and
    NOT when the edge is cited (support present), and NOT when the two concepts DO
    co-occur in the corpus.
  * apply=True soft-retracts (status='retracted' → gone from edges_from), logs to
    edge_audit_log, and restore() brings them back.

    python tests/edge_audit_test.py
"""

import struct
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from knowledgehost.kb import KB          # noqa: E402
from knowledgehost import edge_audit as EA   # noqa: E402

PASS = FAIL = 0


def check(name, cond):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  ok  {name}")
    else:
        FAIL += 1
        print(f"FAIL  {name}")


E1 = [1.0, 0.0, 0.0, 0.0]         # unit vectors: E1·E2 = 0 (distant), E1·E1 = 1 (same)
E2 = [0.0, 1.0, 0.0, 0.0]
CFG = {"edge_audit": {"orth_high": 0.82, "sem_low": 0.35, "sem_vlow": 0.15,
                      "min_label_len": 3, "sample": 25}}


def _node(kb, nid, label, vec):
    kb.db.execute("INSERT INTO nodes(id,label,kind,summary,embedding,aliases,support,status) "
                  "VALUES (?,?,?,?,?,?,?, 'active')",
                  (nid, label, "concept", "", struct.pack(f"<{len(vec)}f", *vec), "[]", "[]"))


def _build():
    tmp = tempfile.mkdtemp()
    kb = KB({"kb_path": str(Path(tmp) / "kb.db")})
    nodes = [("n_comp", "complement", E1), ("n_cpl", "compliment", E2),   # sound-alike
             ("n_dt1", "type 1 diabetes", E1), ("n_dt2", "type 2 diabetes", E1),  # related
             ("n_ins", "insulin", E1), ("n_glu", "glucose", E1),          # related, co-occurs
             ("n_ban", "banana", E1), ("n_tra", "tractor", E2),           # ungrounded
             ("n_a", "alpha", E1), ("n_b", "beta", E2)]                   # distant but CITED
    for nid, label, vec in nodes:
        _node(kb, nid, label, vec)
    kb.db.commit()
    e = lambda s, d, doc=None: kb.add_edge(s, d, family="causal", type="causes", doc_id=doc)
    e("n_comp", "n_cpl")            # sound-alike
    e("n_dt1", "n_dt2")             # look-alike but related  → keep
    e("n_ins", "n_glu")            # related + co-occurs      → keep
    e("n_ban", "n_tra")            # ungrounded over-link
    e("n_a", "n_b", doc="d1")      # distant but cited        → keep
    # co-occurrence table present + populated (only insulin↔glucose actually co-occurs)
    kb.db.execute("CREATE TABLE concept_cooccurrence (concept_a TEXT, concept_b TEXT, "
                  "ppmi REAL, cooc_count INTEGER, PRIMARY KEY (concept_a, concept_b))")
    lo, hi = sorted(["n_ins", "n_glu"])
    kb.db.execute("INSERT INTO concept_cooccurrence VALUES (?,?,?,?)", (lo, hi, 1.5, 3))
    kb.db.commit()
    return kb


def _verdicts(sample):
    return {(f["src"], f["dst"]): f["verdict"] for f in sample}


def main():
    kb = _build()
    try:
        rep = EA.audit_edges(kb, CFG, apply=False)
        v = _verdicts(rep["sample"])
        check("scanned all 5 active edges", rep["scanned"] == 5)
        check("sound-alike: complement↔compliment flagged",
              v.get(("complement", "compliment")) == "sound_alike")
        check("AND-guard: look-alike BUT related (type-1↔type-2 diabetes) NOT flagged",
              ("type 1 diabetes", "type 2 diabetes") not in v)
        check("ungrounded: banana↔tractor flagged",
              v.get(("banana", "tractor")) == "ungrounded")
        check("support guard: distant BUT cited (alpha↔beta) NOT flagged",
              ("alpha", "beta") not in v)
        check("co-occurrence guard: distant-looking but co-occurring pair not ungrounded "
              "(insulin↔glucose kept)", ("insulin", "glucose") not in v)
        check("exactly two flagged, report-only (nothing applied)",
              rep["flagged"] == 2 and rep["applied"] == 0)
        check("no spurious cooc-mismatch warning (a real pair matched)",
              rep["cooc_available"] is True and rep["cooc_note"] is None)

        # report-only must not have touched anything
        active = kb.db.execute("SELECT COUNT(*) FROM edges WHERE status='active'").fetchone()[0]
        check("report-only leaves every edge active", active == 5)

        rep2 = EA.audit_edges(kb, CFG, apply=True)
        check("apply retracts exactly the two flagged", rep2["applied"] == 2)
        active2 = kb.db.execute("SELECT COUNT(*) FROM edges WHERE status='active'").fetchone()[0]
        check("two edges now retracted (gone from the active graph)", active2 == 3)
        logged = kb.db.execute("SELECT COUNT(*) FROM edge_audit_log").fetchone()[0]
        check("both retractions recorded in edge_audit_log", logged == 2)
        # the retracted sound-alike no longer walks out of its source node
        outs = [it for it in kb.edges_from("n_comp")]
        check("edges_from skips the retracted edge (retrieval improved)", outs == [])

        # re-running apply is idempotent (already retracted → nothing new)
        rep3 = EA.audit_edges(kb, CFG, apply=True)
        check("re-run finds nothing (idempotent)", rep3["flagged"] == 0 and rep3["applied"] == 0)

        n = EA.restore(kb)
        active3 = kb.db.execute("SELECT COUNT(*) FROM edges WHERE status='active'").fetchone()[0]
        check("restore() reactivates the audited edges", n == 2 and active3 == 5)
    finally:
        kb.close()

    print()
    if FAIL:
        print(f"{FAIL} FAILURE(S)")
        return 1
    print(f"ALL OK ({PASS} checks)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
