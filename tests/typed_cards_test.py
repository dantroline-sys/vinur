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

    print()
    if check.failed:
        print(f"{check.failed} FAILURE(S)")
        return 1
    print("ALL OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
