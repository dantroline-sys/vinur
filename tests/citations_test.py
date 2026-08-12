"""Deterministic citation-edge graph (knowledgehost/citations.build): after a
structured ingest, one KB node per canonical unit and a 'citation' edge per reference —
the cross-reference payoff.  Covers scripture + legal edges, cross-document convergence
on one node, and idempotency.  Real sqlite store + KB, no LM, no embed server.

    python tests/citations_test.py
"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from knowledgehost.config import load_config          # noqa: E402
from knowledgehost.store import make_store             # noqa: E402
from knowledgehost.kb import KB                         # noqa: E402
from knowledgehost import ingest as ingest_mod, citations as cite_mod, structure as S  # noqa: E402

PASS = FAIL = 0


def check(name, cond):
    global PASS, FAIL
    if cond:
        PASS += 1; print(f"  ok  {name}")
    else:
        FAIL += 1; print(f"FAIL  {name}")


ROMANS = ("Romans 9:16 So then it is not of him that willeth; but see Exodus 33:19.\n"
          "Romans 5:8 God commendeth his love toward us; compare John 3:16.\n")
JOHN = "John 3:16 For God so loved the world, that he gave his only begotten Son.\n"
STATUTE = ("§ 106. Exclusive rights in copyrighted works\n"
           "The owner of copyright has the exclusive rights.\n"
           "§ 107. Limitations: Fair use\n"
           "Notwithstanding the provisions of § 106, fair use is not an infringement.\n")


def _write(d, name, text):
    p = os.path.join(d, name)
    with open(p, "w", encoding="utf-8") as f:
        f.write(text)
    return p


def main():
    tmp = tempfile.mkdtemp(prefix="kb-cite-")
    cfg = load_config(None)
    cfg["db_path"] = os.path.join(tmp, "index.db")
    cfg["kb_path"] = os.path.join(tmp, "kb.db")
    cfg["backend"] = "sqlite"

    store = make_store(cfg)
    kb = KB(cfg)

    def ingest(name, text, answers):
        p = _write(tmp, name, text)
        prof = S.apply_answers(S.analyze(text), answers)
        ingest_mod.ingest_file(store, None, cfg, p, profile=prof)

    ingest("romans.txt", ROMANS, {"kind": "structured"})
    ingest("statute.txt", STATUTE, {"kind": "structured", "work_title": "17"})

    stats = cite_mod.build(store, kb, cfg)
    check("build reports units + citation edges", stats["units"] >= 3 and stats["edges"] >= 3)

    def node(label, kind):
        r = kb.db.execute("SELECT id, summary FROM nodes WHERE label=? AND kind=?",
                          (label, kind)).fetchone()
        return r

    def cites(src_label, src_kind, dst_label, dst_kind):
        s, d = node(src_label, src_kind), node(dst_label, dst_kind)
        if not s or not d:
            return False
        return bool(kb.db.execute(
            "SELECT 1 FROM edges WHERE src_id=? AND dst_id=? AND family='citation' "
            "AND status='active'", (s["id"], d["id"])).fetchone())

    # ── scripture: verse → verse ─────────────────────────────────────────────
    check("a verse node is created for each unit (Romans 9:16)", node("bible:Rom.9.16", "passage"))
    check("Romans 9:16 --cites--> Exodus 33:19 (inline cross-ref)",
          cites("bible:Rom.9.16", "passage", "bible:Exod.33.19", "passage"))
    check("the cited-but-not-ingested target exists as a node (stub to be enriched later)",
          node("bible:Exod.33.19", "passage") is not None)

    # ── legal: section → section, self-reference not an edge ─────────────────
    check("§ 107 --cites--> § 106 (local section ref, resolved under title 17)",
          cites("usc:17/107", "provision", "usc:17/106", "provision"))
    check("a section does not cite itself",
          not cites("usc:17/106", "provision", "usc:17/106", "provision"))

    # ── cross-document convergence: the payoff ───────────────────────────────
    # Romans 5:8 cited John 3:16 as a STUB (no text).  Ingest the actual Gospel and
    # rebuild: the same node id is enriched with the verse text, and the earlier edge
    # already points to it — two documents married on one node.
    stub = node("bible:John.3.16", "passage")
    check("before the Gospel is ingested, John 3:16 is an empty stub",
          stub is not None and not (stub["summary"] or ""))
    check("Romans 5:8 --cites--> John 3:16 (the stub)",
          cites("bible:Rom.5.8", "passage", "bible:John.3.16", "passage"))
    ingest("john.txt", JOHN, {"kind": "structured"})
    cite_mod.build(store, kb, cfg)
    enriched = node("bible:John.3.16", "passage")
    check("after ingesting John, the SAME node now carries the verse text (converged)",
          enriched["id"] == stub["id"] and enriched["summary"].startswith("For God so loved"))
    check("the earlier citation edge still points at that one node",
          cites("bible:Rom.5.8", "passage", "bible:John.3.16", "passage"))

    # ── idempotent ───────────────────────────────────────────────────────────
    again = cite_mod.build(store, kb, cfg)
    check("re-running adds no new edges (deterministic ids)", again["edges"] == 0)

    kb.close()
    store.close()
    print()
    if FAIL:
        print(f"{FAIL} FAILURE(S)")
        return 1
    print(f"citations_test: ALL OK ({PASS} checks)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
