"""Typed-card hints (brains): research drops that declare their answer's shape.

A solved drop may carry front-matter ``card_type`` (requirements | decision |
playbook | case) + ``context_features`` (one-line JSON).  The parser passes them
through, chunks the shaped ## Answer first, and the distiller runs the matching
typed extractor on THAT chunk only — one card per drop, discriminators seeded
from the drop's own features, payload in the `criteria` column, gap closed.

Run:  python tests/typed_cards_test.py     (stdlib only; LM + embedder stubbed)
"""
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from knowledgehost import distill as D
from knowledgehost import research
from knowledgehost.kb import KB


def check(label, cond):
    print(("  ok  " if cond else "  FAIL ") + label)
    if not cond:
        check.failed += 1
check.failed = 0


DROP = """---
provenance: vinkona
kind: research
trust: low
question: "How should I handle Dan going quiet mid-conversation?"
kb_query: handle user silence
card_type: playbook
context_features: {"situation": "user goes quiet", "channel": "voice"}
---
# Question

How should I handle Dan going quiet mid-conversation?

## Answer

When Dan goes quiet mid-conversation, first wait a beat. If it lasts, offer a
gentle check-in; never repeat the last question verbatim.

## Sources

### advice column

Long silences usually mean thinking, not disengagement. Interrupting restarts
the thought; a soft check-in after ~20s respects it.
"""


def main():
    td = tempfile.mkdtemp(prefix="kb-typed-")

    # ── parse: hints + answer-first chunking ─────────────────────────────────
    p = os.path.join(td, "drop.md")
    open(p, "w").write(DROP)
    q, blocks, meta = research.parse_research_doc(p)
    check("hints parsed", meta["card_type"] == "playbook"
          and meta["context_features"] == {"situation": "user goes quiet",
                                           "channel": "voice"})
    check("hinted drop chunks the Answer first",
          blocks[0][0] == "Answer" and len(blocks) == 2)
    # malformed features never break the drop
    bad = DROP.replace('{"situation": "user goes quiet", "channel": "voice"}', "not json")
    open(p, "w").write(bad)
    _, _, meta2 = research.parse_research_doc(p)
    check("malformed features -> None (drop still parses)",
          meta2["context_features"] is None and meta2["card_type"] == "playbook")
    # no hint → old behaviour: sources only
    plain = DROP.replace("card_type: playbook\n", "").replace(
        'context_features: {"situation": "user goes quiet", "channel": "voice"}\n', "")
    open(p, "w").write(plain)
    _, blocks3, meta3 = research.parse_research_doc(p)
    check("unhinted drop keeps sources-only blocks",
          meta3["card_type"] is None and len(blocks3) == 1
          and "advice column" in blocks3[0][0])

    # ── payload cleaners: shape gates ────────────────────────────────────────
    t, pay, disc, concept, ev = D._clean_typed_payload("playbook", {
        "title": "Handling Dan going quiet", "concept": "user silence",
        "state": "user quiet mid-conversation",
        "continuations": [{"move": "wait a beat", "when": "first seconds",
                           "why": "silence is thinking", "prerequisites": []},
                          {"move": "gentle check-in", "when": "after ~20s"}],
        "discriminators": [{"feature": "channel", "value": "voice"}],
        "evidence": "a soft check-in after ~20s respects it"})
    check("playbook payload cleans", t and pay["state"] and len(pay["continuations"]) == 2)
    t2, *_ = D._clean_typed_payload("playbook", {"title": "x", "state": "", "continuations": []})
    check("unsupported shape -> empty title", t2 == "")
    t3, pay3, *_ = D._clean_typed_payload("requirements", {
        "title": "Done means done", "target": "a finished task",
        "must": ["tests pass", "docs updated"], "verify": ["run suite"]})
    check("requirements payload cleans", t3 and pay3["must"] == ["tests pass", "docs updated"])
    t4, pay4, *_ = D._clean_typed_payload("decision", {
        "title": "Ask vs act", "decision": "ask first or act-then-announce",
        "options": [{"option": "ask first", "favors_when": ["destructive"]},
                    {"option": "act then announce", "tradeoffs": "may annoy"}],
        "default": "ask first"})
    check("decision payload cleans", t4 and len(pay4["options"]) == 2
          and pay4["default"] == "ask first")
    t5, pay5, *_ = D._clean_typed_payload("case", {
        "title": "The interrupted thought", "situation": "Dan paused to think",
        "action": "I repeated the question", "outcome": "he lost the thread",
        "lesson": "never repeat the question into a silence"})
    check("case payload cleans", t5 and pay5["lesson"].startswith("never repeat"))

    # ── full distill_chunk with stub LM/embedder against a real KB ──────────
    class StubLM:
        def extract(self, chunk, regime=None):
            return ([{"label": "user silence", "kind": "concept",
                      "summary": "a quiet spell in conversation", "evidence": "silences",
                      "questions": ["what does silence mean?"]}], [], [], [])
        def extract_typed(self, chunk, card_type):
            assert card_type == "playbook"
            return {"title": "Handling Dan going quiet", "concept": "user silence",
                    "state": "user quiet mid-conversation",
                    "continuations": [{"move": "wait a beat", "when": "first seconds",
                                       "why": "silence is thinking"},
                                      {"move": "gentle check-in", "when": "after ~20s"}],
                    "discriminators": [{"feature": "tone", "value": "gentle"}],
                    "evidence": "a soft check-in after ~20s"}

    class StubEmbedder:
        def embed_many(self, texts, kind):
            return [[1.0] + [0.0] * 7 for _ in texts]

    kb = KB({"kb_path": os.path.join(td, "kb.db")})
    kb.db.execute("INSERT INTO knowledge_gaps(query_text, intent, first_seen) "
                  "VALUES('handle user silence','how',1)")
    kb.db.commit()
    chunk = {"id": "c1", "path_or_url": p, "title": "How should I handle Dan going quiet?",
             "section": "Answer", "text": "When Dan goes quiet, wait a beat...",
             "source_type": "vinkona", "provenance": "vinkona", "trust": 0.25,
             "question": "How should I handle Dan going quiet mid-conversation?",
             "kb_query": "handle user silence", "card_type": "playbook",
             "context_features": {"situation": "user goes quiet", "channel": "voice"}}
    nc, nr, ncard = D.distill_chunk(kb, StubLM(), StubEmbedder(), chunk)
    check("distill made the typed card", ncard >= 1)
    row = kb.db.execute("SELECT title, card_type, criteria, discriminators FROM "
                        "procedure_cards WHERE card_type='playbook'").fetchall()
    check("card stored as playbook", len(row) == 1
          and row[0][0] == "Handling Dan going quiet")
    pay = json.loads(row[0][2])
    check("payload in criteria column", pay["state"] == "user quiet mid-conversation"
          and len(pay["continuations"]) == 2)
    disc = json.loads(row[0][3])
    feats = {(d["feature"], d["value"]) for d in disc}
    check("drop features seeded into discriminators",
          ("situation", "user goes quiet") in feats and ("tone", "gentle") in feats)
    sq = kb.db.execute("SELECT text FROM surface_questions WHERE target_kind='card' "
                       "AND text LIKE '%going quiet%'").fetchall()
    check("drop question is the card's retrieval surface", len(sq) >= 1)
    gap = kb.db.execute("SELECT status FROM knowledge_gaps "
                        "WHERE query_text='handle user silence'").fetchall()[0][0]
    check("knowledge gap closed", gap != "open")

    # a SOURCE chunk of the same hinted drop must NOT re-run the typed pass
    chunk2 = {**chunk, "id": "c2", "section": "advice column",
              "text": "Long silences usually mean thinking."}
    D.distill_chunk(kb, StubLM(), StubEmbedder(), chunk2)
    n = kb.db.execute("SELECT COUNT(*) FROM procedure_cards "
                      "WHERE card_type='playbook'").fetchall()[0][0]
    check("typed pass runs on the Answer chunk only", n == 1)

    # idempotent: re-distilling the Answer chunk dedups on content hash
    D.distill_chunk(kb, StubLM(), StubEmbedder(), chunk)
    n = kb.db.execute("SELECT COUNT(*) FROM procedure_cards "
                      "WHERE card_type='playbook'").fetchall()[0][0]
    check("re-distill dedups the typed card", n == 1)
    kb.close()

    # ── cards_fts stays live with writes (July #15) ──────────────────────────
    kb2 = KB({"kb_path": os.path.join(td, "kb2.db")})
    cid, verdict = kb2.add_card(None, title="Flush the zorble valve",
                                goal="stop the leak",
                                steps=["close the intake", "vent the line"])
    hits = kb2.search_cards_bm25("zorble", 5)
    check("insert-on-write: a fresh card is BM25-findable with NO reindex",
          verdict == "insert" and len(hits) == 1)
    kb2.refresh_card(cid, {"title": "Flush the quibbex valve"})
    check("refine refreshes the FTS row: new term found, old term gone",
          len(kb2.search_cards_bm25("quibbex", 5)) == 1
          and kb2.search_cards_bm25("zorble", 5) == [])
    # upgrade edge: a pre-insert-on-write store (cards indexed by NOTHING) plus
    # one fresh write must still surface the OLD body — partial index rebuilds.
    kb2.db.execute("DELETE FROM cards_fts")
    kb2.db.commit()
    kb2.add_card(None, title="Grease the sprocket", goal="quiet the chain",
                 steps=["wipe", "grease"])
    check("partial index (old un-indexed body + one live row) rebuilds on search",
          len(kb2.search_cards_bm25("quibbex", 5)) == 1
          and len(kb2.search_cards_bm25("sprocket", 5)) == 1)
    kb2.close()

    # ── the judge vets criteria too (July #14) ───────────────────────────────
    from knowledgehost import verify as V

    class Judge:
        def __init__(self, out):
            self.out, self.user, self.schema = out, None, None
        def chat_json(self, system, user, schema, max_tokens=0):
            self.user, self.schema = user, schema
            return self.out

    crit = [
        {"title": "Zorble syndrome", "concept": "zorble",
         "required": [{"feature": "sign", "value": "spots"}],
         "threshold": ">=3 of 5"},
        {"title": "Invented scale", "concept": "quibbex",
         "threshold": "2 major + 1 minor"},
    ]
    drafts = [{"chunk": {"text": "spots, sometimes stripes"}, "criteria": list(crit),
               "concepts": [{"label": "z", "summary": "s"}],
               "relations": [], "procedures": []}]
    j = Judge({"criteria": [{"c": 0, "i": 1, "verdict": "reject"},
                            {"c": 0, "i": 0, "verdict": "adjust",
                             "threshold": ">=2 of 5",
                             "supportive": [{"feature": "sign", "value": "stripes"}]}]})
    (co, rl, pr, cr, vs), = V.verify_batch(j, drafts, {})
    check("criteria ride the judge's prompt", "CRITERIA:" in j.user and "Zorble" in j.user)
    check("the verdict schema offers criteria", "criteria" in j.schema["properties"])
    check("a rejected criteria card dies, and is counted",
          len(cr) == 1 and cr[0]["title"] == "Zorble syndrome" and vs["rejected"] == 1)
    check("adjust patches threshold + modality arrays",
          cr[0]["threshold"] == ">=2 of 5"
          and cr[0]["supportive"] == [{"feature": "sign", "value": "stripes"}]
          and vs["adjusted"] == 1)
    (co2, rl2, pr2, cr2, vs2), = V.verify_batch(Judge(None), drafts, {})
    check("fail-open keeps criteria unchanged", cr2 == crit and vs2["failed"] == 1)
    check("the judge is told criteria are the highest stakes",
          "CRITERIA are the highest stakes" in V.VERIFY_SYSTEM)
    check("a chunk with no criteria adds no CRITERIA section (token thrift)",
          "CRITERIA:" not in V._fmt_items([{"label": "a"}], [], []))

    # ── speech-act role-pulls: requirements / criteria / cards (July #17) ────
    from knowledgehost import query as Q
    from knowledgehost import understand as U

    NODE = {"kind": "node", "id": "n1", "label": "zorble", "text": "a bench device",
            "score": 0.9, "support": []}
    REQ_EDGE = {"kind": "edge", "id": "e1", "type": "requires", "family": "functional",
                "src": "zorble", "dst": "calibration", "label": "calibration",
                "text": "a zorble requires calibration before use", "support": []}
    CRIT_CARD = {"kind": "card", "id": "c-crit", "node_id": "n1",
                 "label": "Zorble syndrome", "text": "spots + stripes",
                 "card_type": "criteria", "support": []}
    PROC_CARD = {"kind": "card", "id": "c-proc", "node_id": "n1",
                 "label": "Run the zorble", "text": "steps", "card_type": "procedure",
                 "support": []}

    class RoleKB:
        def search(self, qvec, n, empirical_only=False):
            return [dict(NODE)]
        def edges_from(self, nid, families=None, direction=None, empirical_only=False):
            return [dict(REQ_EDGE)] if families == ["functional"] else []
        def cards_for(self, nid, limit=10):
            return [dict(CRIT_CARD), dict(PROC_CARD)]
        def alternatives(self, nid, limit=8):
            return []
        def search_cards_bm25(self, q, pool):
            return []
        def neighbours(self, nid, limit=8):
            return []
        def contra_pressure(self, it):
            return 0.0
        def log_gap(self, q, intent):
            pass

    class Emb:
        def embed_one(self, text, kind):
            return [0.1] * 8

    def sections(bundle):
        return {s["role"]: s for s in bundle.get("structure") or []}

    b = Q.answer(RoleKB(), Emb(), "can we run the zorble without calibration?")
    s = sections(b)
    check("feasibility pulls `requires` edges into their own section",
          b["speech_act"] == "feasibility" and "requires" in s
          and s["requires"]["entries"][0]["relation"] == "requires")

    b = Q.answer(RoleKB(), Emb(), "why is the zorble wilting?")
    s = sections(b)
    check("diagnostic surfaces the RECOGNITION card (criteria), not the how-to",
          b["speech_act"] == "diagnostic" and "criteria" in s
          and [e["id"] for e in s["criteria"]["entries"]] == ["c-crit"])

    b = Q.answer(RoleKB(), Emb(), "the zorble developed spots and stripes")
    s = sections(b)
    check("observation pulls criteria AND the other card families",
          b["speech_act"] == "observation" and "criteria" in s
          and any(e["id"] == "c-proc" for e in s.get("managed_by", {}).get("entries", [])))

    check("mid_procedure flag is gone — in_progress is a plain `how` plan",
          U.TRAVERSAL_PLANS["in_progress"] == {"intent": "how"})
    check("every remaining plan flag is consumed by query.answer",
          all(f in ("intent", "safety", "requirements", "criteria", "cards",
                    "counterfactual", "alternatives", "broaden")
              for plan in U.TRAVERSAL_PLANS.values() for f in plan))

    print()
    if check.failed:
        print(f"{check.failed} FAILURE(S)")
        return 1
    print("ALL OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
