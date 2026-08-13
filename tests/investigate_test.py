#!/usr/bin/env python
"""Iterative agentic graph reasoning (knowledgehost/investigate.py): anchor → the
LM-navigated walk → evidence subgraph → grounded synthesis → claim verification, and
every degrade rung (no navigator → greedy; no synthesiser → evidence-only; no anchors
or a dead end → a logged gap).  Real KB (sqlite), scripted LM stubs, no server.

    python tests/investigate_test.py
"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from knowledgehost.config import load_config          # noqa: E402
from knowledgehost.kb import KB                        # noqa: E402
from knowledgehost import investigate as I             # noqa: E402
from knowledgehost import reason as R                  # noqa: E402

PASS = FAIL = 0


def check(name, cond):
    global PASS, FAIL
    if cond:
        PASS += 1; print(f"  ok  {name}")
    else:
        FAIL += 1; print(f"FAIL  {name}")


def build_world():
    """The reason_test pharmacology world: chains, siblings, one contested edge."""
    tmp = tempfile.mkdtemp(prefix="kb-investigate-")
    cfg = load_config(None)
    cfg["kb_path"] = os.path.join(tmp, "kb.db")
    cfg["db_path"] = os.path.join(tmp, "index.db")
    kb = KB(cfg)

    def node(label, kind="concept"):
        return kb._new_node(label, kind, f"about {label}", None, [])

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
    return kb, cfg


def chain_navigator(*route):
    """A scripted navigator: each hop it picks the FIRST candidate whose rendered line
    contains the next term on the route, and reports done when the route is spent.
    Exercises the real numbered-candidate rendering, exactly as an LM would see it."""
    remaining = list(route)

    def nav(system, user, schema, max_tokens):
        assert "Candidate edges:" in user and "Question:" in user
        lines = user.split("Candidate edges:\n", 1)[1].split("\n\nJSON:")[0].splitlines()
        if not remaining:
            return {"picks": [], "done": True, "why": "route complete"}
        want = remaining[0]
        for ln in lines:
            k, _, text = ln.partition(". ")
            if want in text:
                remaining.pop(0)
                return {"picks": [int(k)], "done": not remaining,
                        "why": f"following {want}"}
        return {"picks": [1], "done": False, "why": "route term not on offer"}

    return nav


def main():
    kb, cfg = build_world()

    # ── anchoring ────────────────────────────────────────────────────────────
    g = R.view(kb, cfg)
    seeds = I._anchor(g, "How could Aspirin change my stroke risk?")
    check("anchors on exact labels, case-insensitive, multi-word spans included",
          {g.labels[i] for i in seeds} == {"aspirin", "stroke risk"})
    seeds = I._anchor(g, "does stomach irritation matter?")
    check("the 2-word span wins over its 1-word parts",
          [g.labels[i] for i in seeds] == ["stomach irritation"])

    # ── the LM-navigated walk: aspirin → platelets → clotting → stroke ───────
    nav = chain_navigator("platelet aggregation", "clotting", "stroke risk")
    synth_seen = {}

    def synth(system, user, schema, max_tokens):
        synth_seen["user"] = user
        return {"answer": "Aspirin lowers platelet aggregation, which lowers clotting, "
                          "which lowers stroke risk [1][2][3].",
                "confident": True,
                "claims": [{"a": "aspirin", "b": "platelet aggregation",
                            "relation": "causes", "polarity": "negative"}]}

    res = I.investigate(kb, cfg, "How could aspirin change my stroke risk?",
                        navigator=nav, synthesizer=synth)
    check("the walk follows the causal chain the navigator picked",
          [ev["to"] for ev in res["evidence"]][:3]
          == ["platelet aggregation", "clotting", "stroke risk"])
    check("an answer is produced and grounded",
          res["answer"] and not res["abstain"] and res["confidence"] == "grounded")
    check("the verified claim counts as supported",
          res["verified"]["supported"] == 1 and res["verified"]["contradicted"] == 0)
    check("the trail records each hop with by='lm' and the navigator's why",
          [t["by"] for t in res["trail"]][:3] == ["lm", "lm", "lm"]
          and "following" in res["trail"][0]["why"])
    check("the synthesiser saw ONLY numbered evidence (grounded prompt)",
          "1. " in synth_seen["user"] and "Evidence:" in synth_seen["user"])
    check("evidence rows carry the auditable edge fields",
          all(ev.get("relation") and ev.get("from") and ev.get("to")
              for ev in res["evidence"]))
    check("the speech act steered the walk (a 'how could' is hypothetical)",
          res["speech_act"] in ("hypothetical", "factual", "feasibility"))

    # ── done=true stops early ────────────────────────────────────────────────
    res = I.investigate(kb, cfg, "How could aspirin change my stroke risk?",
                        navigator=chain_navigator("platelet aggregation"),
                        synthesizer=synth)
    check("the navigator's done=true ends the walk early",
          res["hops"] <= 2 and any(t.get("done") for t in res["trail"]))

    # ── degrade rung 1: navigator up, synthesiser down → evidence-only ───────
    res = I.investigate(kb, cfg, "How could aspirin change my stroke risk?",
                        navigator=chain_navigator("platelet aggregation", "clotting"),
                        synthesizer=None)
    check("no synthesiser → honest abstain WITH the evidence subgraph",
          res["abstain"] and res["answer"] is None
          and res["confidence"] == "evidence_only" and len(res["evidence"]) >= 2)

    # ── degrade rung 2: no LM at all → the engine's greedy walk ──────────────
    res = I.investigate(kb, cfg, "aspirin and stroke risk", navigator=None,
                        synthesizer=None)
    check("LM-free greedy walk still gathers evidence (kb always answers)",
          len(res["evidence"]) >= 2 and all(t["by"] == "greedy" for t in res["trail"]))
    check("what ran is declared", res["lm"] == {"navigator": False, "synthesizer": False,
                                                "navigator_failures": 0})

    # ── degrade rung 3: navigator returns garbage → that hop goes greedy ─────
    res = I.investigate(kb, cfg, "aspirin and stroke risk",
                        navigator=lambda *a: None, synthesizer=None)
    check("a failed navigator call falls back to greedy and is counted",
          res["lm"]["navigator_failures"] >= 1
          and all(t["by"] == "greedy" for t in res["trail"]) and res["evidence"])

    # ── a navigator that picks nothing (not done) doesn't spin ───────────────
    res = I.investigate(kb, cfg, "aspirin and stroke risk",
                        navigator=lambda *a: {"picks": [99], "done": False},
                        synthesizer=None)
    check("empty/out-of-range picks fall back to the engine's shortlist",
          res["evidence"] and any(t["by"] == "lm+greedy" for t in res["trail"]))

    # ── no anchors → gap logged, honest abstain ──────────────────────────────
    res = I.investigate(kb, cfg, "what about quantum chromodynamics?",
                        navigator=None, synthesizer=None)
    gaps = [r["query_text"] for r in kb.list_gaps()]
    check("an unanchorable question abstains with confidence 'none'",
          res["abstain"] and res["confidence"] == "none" and res.get("gap_logged"))
    check("…and the research loop got the gap",
          "what about quantum chromodynamics?" in gaps)

    # ── contested ground: the graph disagrees with itself ────────────────────
    res = I.investigate(kb, cfg, "does coffee increase alertness?",
                        navigator=chain_navigator("alertness"),
                        synthesizer=lambda s, u, sc, m: {
                            "answer": "Coffee increases alertness [1].", "confident": True,
                            "claims": [{"a": "coffee", "b": "alertness",
                                        "relation": "causes", "polarity": "positive"}]})
    check("walking onto a both-signs edge surfaces the contradiction",
          res["contradictions"]
          and "BOTH polarities" in res["contradictions"][0]["note"])
    check("a contested answer is banded 'contested', never 'grounded'",
          res["confidence"] == "contested" and "contested" in res["note"])

    # ── claim verification catches an unsupported claim ──────────────────────
    res = I.investigate(kb, cfg, "aspirin and fever",
                        navigator=chain_navigator("fever"),
                        synthesizer=lambda s, u, sc, m: {
                            "answer": "Aspirin treats fever [1].", "confident": True,
                            "claims": [{"a": "aspirin", "b": "fever",
                                        "relation": "treats"},
                                       {"a": "warfarin", "b": "fever",
                                        "relation": "treats"}]})
    check("a claim the graph is SILENT on counts unsupported → banded 'partial'",
          res["verified"]["supported"] == 1 and res["verified"]["unsupported"] == 1
          and res["confidence"] == "partial")

    # ── modes: derived edges join only the permissive walk ───────────────────
    R.derive(kb, cfg)
    g2 = R.view(kb, cfg)
    qtok = {"aspirin", "stroke", "risk"}
    seeds = I._anchor(g2, "aspirin")
    cons = I._candidates(g2, seeds, set(), set(seeds), "conservative", qtok, (), 50)
    perm = I._candidates(g2, seeds, set(), set(seeds), "permissive", qtok, (), 50)
    check("conservative candidates never include derived edges",
          all(not g2.eattr[c["e"]][3] for c in cons))
    check("permissive candidates DO include them (marked inferred in the line)",
          any(g2.eattr[c["e"]][3] for c in perm)
          and any("(inferred)" in I._edge_line(g2, c) for c in perm
                  if g2.eattr[c["e"]][3]))
    res = I.investigate(kb, cfg, "aspirin and clotting", mode="permissive",
                        navigator=None, synthesizer=None)
    check("a permissive result carries the derived-material caveat",
          "inferred" in res.get("caveat", ""))

    # ── bounds: no edge is walked twice; hops are capped ─────────────────────
    res = I.investigate(kb, cfg, "aspirin ibuprofen naproxen fever clotting",
                        navigator=None, synthesizer=None, max_hops=6)
    texts = [ev["text"] for ev in res["evidence"]]
    check("no edge enters the evidence twice", len(texts) == len(set(texts)))
    check("the walk respects the hop cap", res["hops"] <= 6)

    kb.close()
    print()
    if FAIL:
        print(f"{FAIL} FAILURE(S)")
        return 1
    print(f"investigate: ALL OK ({PASS} checks)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
