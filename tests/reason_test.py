"""Deterministic graph reasoning (knowledgehost/reason.py): the kb_reason ops
(compare/paths/about/effects/siblings/contradictions), the conservative|permissive mode
gate, and the derive pass — taxonomic inheritance + causal sign composition written as a
QUARANTINED family='derived' status='proposed' layer, plus contradiction/sibling-gap
mining.  Real KB (sqlite), no LM, no embed server.

    python tests/reason_test.py
"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from knowledgehost.config import load_config          # noqa: E402
from knowledgehost.kb import KB                        # noqa: E402
from knowledgehost import reason as R                  # noqa: E402

PASS = FAIL = 0


def check(name, cond):
    global PASS, FAIL
    if cond:
        PASS += 1; print(f"  ok  {name}")
    else:
        FAIL += 1; print(f"FAIL  {name}")


def main():
    tmp = tempfile.mkdtemp(prefix="kb-reason-")
    cfg = load_config(None)
    cfg["kb_path"] = os.path.join(tmp, "kb.db")
    cfg["db_path"] = os.path.join(tmp, "index.db")
    cfg["derive_sibling_frac"] = 0.6   # 2 of 3 siblings sharing a relation flags the third
    kb = KB(cfg)

    def node(label, kind="concept"):
        return kb._new_node(label, kind, f"about {label}", None, [])

    # a small pharmacology-flavoured world:
    #   aspirin/ibuprofen/naproxen is_a NSAID; NSAID causes(+) stomach irritation
    #   aspirin causes(-) platelet aggregation; platelet aggregation causes(+) clotting
    #   warfarin causes(-) clotting; clotting causes(+) stroke risk
    #   aspirin treats fever; ibuprofen treats fever; naproxen (no fever edge → sibling gap)
    #   contradiction: coffee causes(+/-) alertness from two sources
    nsaid = node("NSAID", "class")
    asp, ibu, nap = node("aspirin"), node("ibuprofen"), node("naproxen")
    sto = node("stomach irritation")
    plt = node("platelet aggregation")
    clo = node("clotting")
    stk = node("stroke risk")
    fev = node("fever")
    war = node("warfarin")
    cof, ale = node("coffee"), node("alertness")
    for x in (asp, ibu, nap):
        kb.add_edge(x, nsaid, family="taxonomic", type="is_a", doc_id="doc1")
    kb.add_edge(nsaid, sto, family="causal", type="causes", polarity="positive", doc_id="doc1")
    kb.add_edge(asp, plt, family="causal", type="causes", polarity="negative", doc_id="doc2")
    kb.add_edge(plt, clo, family="causal", type="causes", polarity="positive", doc_id="doc2")
    kb.add_edge(war, clo, family="causal", type="causes", polarity="negative", doc_id="doc3")
    kb.add_edge(clo, stk, family="causal", type="causes", polarity="positive", doc_id="doc3")
    kb.add_edge(asp, fev, family="functional", type="treats", doc_id="doc2")
    kb.add_edge(ibu, fev, family="functional", type="treats", doc_id="doc4")
    kb.add_edge(cof, ale, family="causal", type="causes", polarity="positive", doc_id="doc5")
    kb.add_edge(cof, ale, family="causal", type="causes", polarity="negative", doc_id="doc6")
    kb.db.commit()

    # ── resolution + about ────────────────────────────────────────────────────
    res = R.query(kb, cfg, {"op": "about", "a": "Aspirin"})
    check("resolve is case-insensitive; about returns the relation picture",
          res["ok"] and res["node"] == "aspirin" and "taxonomic:is_a" in res["relations"])
    check("unknown label errors cleanly",
          not R.query(kb, cfg, {"op": "about", "a": "zzz-nope"})["ok"])
    check("unknown op lists the vocabulary",
          "compare" in R.query(kb, cfg, {"op": "??", "a": "x"}).get("ops", []))

    # ── compare ───────────────────────────────────────────────────────────────
    res = R.query(kb, cfg, {"op": "compare", "a": "aspirin", "b": "warfarin"})
    meet_nodes = {m["node"] for m in res["meeting_nodes"]}
    check("compare finds the 2-hop connection (both reach clotting)",
          res["ok"] and any(p for p in res["paths"]))
    contrast = {c["relation"]: c for c in res["contrast"]}
    check("contrast: aspirin treats fever, warfarin does not",
          any("fever" in c.get("only_a", []) for c in contrast.values()))
    res2 = R.query(kb, cfg, {"op": "compare", "a": "aspirin", "b": "ibuprofen"})
    m2 = {m["node"] for m in res2["meeting_nodes"]}
    check("compare siblings: NSAID and fever are meeting nodes",
          {"NSAID", "fever"} <= m2)

    # ── paths ─────────────────────────────────────────────────────────────────
    res = R.query(kb, cfg, {"op": "paths", "a": "aspirin", "b": "stroke risk"})
    check("paths reaches stroke risk within 3 hops, every step typed",
          res["ok"] and res["paths"]
          and all("relation" in s for s in res["paths"][0]))

    # ── effects: signed reach ─────────────────────────────────────────────────
    res = R.query(kb, cfg, {"op": "effects", "a": "aspirin"})
    eff = {r["node"]: r["net"] for r in res["reach"]}
    check("sign algebra: aspirin ⊣ platelets ⊕ clotting ⇒ DECREASES clotting",
          eff.get("clotting") == "decreases")
    check("…and decreases stroke risk at 3 hops", eff.get("stroke risk") == "decreases")
    res = R.query(kb, cfg, {"op": "effects", "a": "clotting", "direction": "up"})
    upn = {r["node"] for r in res["reach"]}
    check("upstream effects finds both causes (aspirin chain + warfarin)",
          {"warfarin", "platelet aggregation"} <= upn)

    # ── contradictions ────────────────────────────────────────────────────────
    res = R.query(kb, cfg, {"op": "contradictions", "a": "coffee"})
    check("opposite-polarity assertion is surfaced (coffee → alertness both ways)",
          res["ok"] and any(c["to"] == "alertness" for c in res["contradictions"]))

    # ── derive: the quarantined layer ────────────────────────────────────────
    stats = R.derive(kb, cfg)
    check("derive produced inheritance + sign compositions",
          stats["inherited"] >= 3 and stats["sign_composed"] >= 1)
    row = kb.db.execute("SELECT COUNT(*) FROM edges WHERE family='derived' "
                        "AND status='proposed'").fetchone()[0]
    active_derived = kb.db.execute("SELECT COUNT(*) FROM edges WHERE family='derived' "
                                   "AND status='active'").fetchone()[0]
    check("every derived edge is QUARANTINED (proposed, never active)",
          row == stats["derived_edges"] and active_derived == 0)
    check("contradiction mined into surface_questions",
          kb.db.execute("SELECT COUNT(*) FROM surface_questions WHERE text LIKE "
                        "'%Contradiction%alertness%'").fetchone()[0] >= 1)
    check("sibling-completion gap logged (naproxen lacks the treats edge its siblings have)",
          kb.db.execute("SELECT COUNT(*) FROM knowledge_gaps WHERE query_text LIKE "
                        "'naproxen%treats%'").fetchone()[0] == 1)
    st2 = R.derive(kb, cfg)
    row2 = kb.db.execute("SELECT COUNT(*) FROM edges WHERE family='derived'").fetchone()[0]
    check("derive is wipe-and-rebuild idempotent (same layer, no growth)",
          row2 == row and st2["derived_edges"] == stats["derived_edges"])

    # ── the mode gate ─────────────────────────────────────────────────────────
    cons = R.query(kb, cfg, {"op": "about", "a": "ibuprofen", "mode": "conservative"})
    perm = R.query(kb, cfg, {"op": "about", "a": "ibuprofen", "mode": "permissive"})
    cons_rel = set(cons["relations"])
    perm_rel = set(perm["relations"])
    check("conservative mode never shows derived relations",
          not any(k.startswith("derived:") for k in cons_rel))
    check("permissive mode shows the inherited relation, MARKED inferred",
          any(k.startswith("derived:inherited_causes") for k in perm_rel)
          and all(x.get("inferred") for x in
                  perm["relations"].get("derived:inherited_causes", [{}])))
    check("permissive answers carry the caveat", "caveat" in perm and "caveat" not in cons)
    # effects through the derived layer: ibuprofen inherits NSAID's stomach irritation
    pe = R.query(kb, cfg, {"op": "effects", "a": "ibuprofen", "mode": "permissive"})
    ce = R.query(kb, cfg, {"op": "effects", "a": "ibuprofen", "mode": "conservative"})
    pn = {r["node"]: r for r in pe["reach"]}
    check("permissive effects reach stomach irritation via inheritance, marked inferred",
          pn.get("stomach irritation", {}).get("inferred") is True)
    check("conservative effects do NOT use the derived edge",
          "stomach irritation" not in {r["node"] for r in ce["reach"]})

    # derivation evidence: the parent chain is stored on the edge
    ev = kb.db.execute("SELECT support FROM edges WHERE family='derived' "
                       "AND type='composed_causes' LIMIT 1").fetchone()
    check("a composed edge carries its derivation chain as evidence",
          ev and "derivation" in (ev[0] or ""))
    # wipe = one delete reverses the whole layer
    kb.db.execute("DELETE FROM edges WHERE family='derived'")
    kb.db.commit()
    R._CACHE.clear()
    perm3 = R.query(kb, cfg, {"op": "about", "a": "ibuprofen", "mode": "permissive"})
    check("one DELETE reverses the layer entirely (permissive falls back to observed)",
          not any(k.startswith("derived:") for k in perm3["relations"]))

    kb.close()
    print()
    if FAIL:
        print(f"{FAIL} FAILURE(S)")
        return 1
    print(f"reason_test: ALL OK ({PASS} checks)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
