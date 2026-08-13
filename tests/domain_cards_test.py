"""Domain card lenses (Slice 3): a structured scripture/legal unit yields its domain
card shapes — scripture: theme / parallel; legal: definition / obligation / exception.
Payload cleaners (shape gates) + an end-to-end distill_chunk with a stub LM/embedder
against a real KB, asserting the cards land with the right type and are located at the
canonical unit.  Stdlib only; LM + embedder stubbed.

    python tests/domain_cards_test.py
"""
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from knowledgehost import distill as D                 # noqa: E402
from knowledgehost.kb import KB                         # noqa: E402

PASS = FAIL = 0


def check(name, cond):
    global PASS, FAIL
    if cond:
        PASS += 1; print(f"  ok  {name}")
    else:
        FAIL += 1; print(f"FAIL  {name}")


def main():
    # ── payload cleaners: shape gates for each new type ──────────────────────
    t, pay, *_ = D._clean_typed_payload("definition", {
        "title": "Copies defined", "term": "copies",
        "definition": "material objects in which a work is fixed",
        "scope": "this title", "applies_to": ["106"]})
    check("definition cleans", t and pay["term"] == "copies" and pay["applies_to"] == ["106"])
    check("definition needs term + definition",
          D._clean_typed_payload("definition", {"title": "x", "term": "", "definition": ""})[0] == "")
    t, pay, *_ = D._clean_typed_payload("obligation", {
        "title": "Owner rights", "subject": "owner of copyright", "modality": "may",
        "action": "reproduce the work", "conditions": ["subject to 107-122"],
        "exceptions": ["fair use"]})
    check("obligation cleans", t and pay["subject"].startswith("owner") and pay["modality"] == "may")
    t, pay, *_ = D._clean_typed_payload("exception", {
        "title": "Fair use", "rule": "section 106", "condition": "the use is fair",
        "effect": "not an infringement"})
    check("exception cleans", t and pay["effect"].startswith("not an"))
    check("exception needs condition + effect",
          D._clean_typed_payload("exception", {"title": "x", "condition": "", "effect": ""})[0] == "")
    t, pay, *_ = D._clean_typed_payload("theme", {
        "title": "God's love", "theme": "divine love",
        "statement": "God loves the world sacrificially", "support": "God so loved the world"})
    check("theme cleans", t and pay["theme"] == "divine love")
    t, pay, *_ = D._clean_typed_payload("parallel", {
        "title": "Creation echo", "relationship": "echoes",
        "parallels": ["Genesis 1:1", "John 1:1"]})
    check("parallel cleans", t and pay["parallels"] == ["Genesis 1:1", "John 1:1"])
    check("parallel needs at least one parallel",
          D._clean_typed_payload("parallel", {"title": "x", "parallels": []})[0] == "")

    check("domain map: scripture→theme/parallel, legal→definition/obligation/exception",
          D.DOMAIN_CARD_TYPES["scripture"] == ("theme", "parallel")
          and D.DOMAIN_CARD_TYPES["legal"] == ("definition", "obligation", "exception"))

    # ── end to end: distill_chunk over structured units ──────────────────────
    _PAYLOADS = {
        "theme": {"title": "God's love", "concept": "divine love", "theme": "divine love",
                  "statement": "God loves the world sacrificially",
                  "support": "God so loved the world", "evidence": "God so loved the world"},
        "parallel": {"title": "Love shown", "concept": "John 3:16", "relationship": "echoes",
                     "parallels": ["1 John 4:9", "Romans 5:8"], "evidence": "his only begotten Son"},
        "definition": {"title": "Copies", "concept": "copies", "term": "copies",
                       "definition": "material objects in which a work is fixed",
                       "scope": "this title"},
        "obligation": {"title": "Exclusive rights", "concept": "reproduction right",
                       "subject": "owner of copyright", "modality": "may",
                       "action": "reproduce the copyrighted work in copies",
                       "conditions": ["subject to sections 107 through 122"]},
        "exception": {"title": "Fair use", "concept": "fair use", "rule": "section 106",
                      "condition": "the use is fair", "effect": "is not an infringement"},
    }

    class StubLM:
        def extract(self, chunk, regime=None):
            return ([], [], [], [])
        def extract_typed(self, chunk, card_type):
            return _PAYLOADS.get(card_type, {})

    class StubEmbedder:
        def embed_many(self, texts, kind):
            return [[1.0] + [0.0] * 7 for _ in texts]

    td = tempfile.mkdtemp(prefix="kb-domain-")
    kb = KB({"kb_path": os.path.join(td, "kb.db")})

    verse = {"id": "v1", "path_or_url": "/x/john.txt", "title": "john",
             "section": "bible:John.3.16", "source_type": "scripture",
             "text": "For God so loved the world, that he gave his only begotten Son."}
    D.distill_chunk(kb, StubLM(), StubEmbedder(), verse)
    ctypes = {r[0] for r in kb.db.execute("SELECT card_type FROM procedure_cards")}
    check("scripture unit → theme + parallel cards", {"theme", "parallel"} <= ctypes)
    sup = kb.db.execute("SELECT support FROM procedure_cards WHERE card_type='theme'").fetchone()[0]
    check("the theme card is LOCATED at its verse (bible:John.3.16 in support locator)",
          "bible:John.3.16" in (sup or ""))
    pay = json.loads(kb.db.execute(
        "SELECT criteria FROM procedure_cards WHERE card_type='parallel'").fetchone()[0])
    check("parallel card carries its related passages",
          "1 John 4:9" in pay.get("parallels", []))

    sec = {"id": "s1", "path_or_url": "/x/t17.txt", "title": "t17",
           "section": "usc:17/106", "source_type": "legal",
           "text": ("Subject to sections 107 through 122, the owner of copyright has the "
                    "exclusive rights to reproduce the copyrighted work in copies.")}
    D.distill_chunk(kb, StubLM(), StubEmbedder(), sec)
    ctypes = {r[0] for r in kb.db.execute("SELECT DISTINCT card_type FROM procedure_cards")}
    check("legal unit → definition + obligation + exception cards",
          {"definition", "obligation", "exception"} <= ctypes)

    # an ordinary (non-structured) chunk gets NO domain cards
    before = kb.db.execute("SELECT COUNT(*) FROM procedure_cards").fetchone()[0]
    D.distill_chunk(kb, StubLM(), StubEmbedder(),
                    {"id": "p1", "path_or_url": "/x/essay.txt", "title": "essay",
                     "section": "Intro", "source_type": "pdf",
                     "text": "An ordinary paragraph of prose."})
    after = kb.db.execute("SELECT COUNT(*) FROM procedure_cards").fetchone()[0]
    check("an ordinary chunk gets no domain cards (lenses are structured-only)", after == before)

    # a surface question is attached so the card is retrievable
    sq = kb.db.execute("SELECT COUNT(*) FROM surface_questions WHERE target_kind='card'").fetchone()[0]
    check("domain cards get a retrieval surface question", sq >= 5)

    # ── worker-side prefetch: the lenses leave the writer thread ─────────────
    # (_prefetch_domain runs in the parallel path's workers so the single writer
    # no longer pays 2 serial LM calls per scripture window — that lane, not
    # extraction, gated the whole pass.)
    pre = D._prefetch_domain(StubLM(), verse)
    check("prefetch runs every scripture lens in the worker",
          set(pre) == {"theme", "parallel"} and pre["theme"]["title"] == "God's love")
    check("an unstructured chunk prefetches nothing (no wasted LM calls)",
          D._prefetch_domain(StubLM(), {"source_type": "pdf", "text": "x"}) == {})
    check("no LM → no prefetch (writer skips, exactly as before)",
          D._prefetch_domain(None, verse) == {})

    class FlakyLM(StubLM):
        def extract_typed(self, chunk, card_type):
            if card_type == "parallel":
                raise ValueError("shape drift")
            return _PAYLOADS[card_type]
    pre_flaky = D._prefetch_domain(FlakyLM(), verse)
    check("a lens that fails in the worker is marked None (skipped, not fatal)",
          pre_flaky["theme"] and pre_flaky["parallel"] is None)

    class DeadLM(StubLM):
        def extract_typed(self, chunk, card_type):
            raise D.BackendUnavailable("down")
    try:
        D._prefetch_domain(DeadLM(), verse)
        ba_raised = False
    except D.BackendUnavailable:
        ba_raised = True
    check("BackendUnavailable propagates (the whole job retries on another slot)",
          ba_raised)

    def _fresh(tag):
        """Distinct payloads per landing — identical cards would dedupe, hiding the count."""
        return {"theme": {**_PAYLOADS["theme"], "title": f"Theme {tag}",
                          "concept": f"theme {tag}", "theme": f"motif {tag}",
                          "statement": f"A statement about {tag}"},
                "parallel": {**_PAYLOADS["parallel"], "title": f"Parallel {tag}",
                             "concept": f"passage {tag}", "parallels": [f"Ref {tag}"]}}

    _EMPTY_GEN = ([], [], [], [])                     # the writer always passes `extraction`

    class MustNotCall(StubLM):
        def extract_typed(self, chunk, card_type):
            raise AssertionError("writer re-ran a lens the worker already paid for")
    verse2 = dict(verse, id="v2", section="bible:John.3.17",
                  text="For God sent not his Son into the world to condemn the world.")
    before = kb.db.execute("SELECT COUNT(*) FROM procedure_cards").fetchone()[0]
    D.distill_chunk(kb, MustNotCall(), StubEmbedder(), verse2,
                    extraction=_EMPTY_GEN, domain_typed=_fresh("v2"))
    after = kb.db.execute("SELECT COUNT(*) FROM procedure_cards").fetchone()[0]
    check("prefetched lenses land through distill_chunk with NO writer LM call",
          after == before + 2)

    verse3 = dict(verse, id="v3", section="bible:John.3.18",
                  text="He that believeth on him is not condemned.")
    before = after
    D.distill_chunk(kb, None, StubEmbedder(), verse3,
                    extraction=_EMPTY_GEN, domain_typed=_fresh("v3"))
    after = kb.db.execute("SELECT COUNT(*) FROM procedure_cards").fetchone()[0]
    check("a downed write-side LM no longer costs the prefetched domain cards",
          after == before + 2)

    flaky_v4 = {**_fresh("v4"), "parallel": None}     # that lens failed in the worker
    verse4 = dict(verse, id="v4", section="bible:John.3.19",
                  text="And this is the condemnation, that light is come into the world.")
    before = after
    D.distill_chunk(kb, None, StubEmbedder(), verse4,
                    extraction=_EMPTY_GEN, domain_typed=flaky_v4)
    after = kb.db.execute("SELECT COUNT(*) FROM procedure_cards").fetchone()[0]
    check("a worker-failed lens is skipped at landing (theme lands, parallel doesn't)",
          after == before + 1)

    # prefetched embeds mirror the writer's text formats EXACTLY — the writer
    # serves every vector from the cache and never embeds live
    class NoLiveEmbeds:
        def embed_many(self, texts, kind="document"):
            raise AssertionError(f"writer embedded live — prefetch text drifted: {texts!r}")
        def embed_one(self, text, kind="document"):
            raise AssertionError(f"writer embedded live — prefetch text drifted: {text!r}")
    cache = D._precompute_domain_embeds(StubEmbedder(), pre)
    cached_emb = D._CacheEmbedder(NoLiveEmbeds(), cache)
    verse5 = dict(verse, id="v5", section="bible:John.3.20",
                  text="For every one that doeth evil hateth the light.")
    made = D._distil_domain(kb, None, cached_emb, verse5,
                            D.DOMAIN_CARD_TYPES["scripture"], {}, "/x/john.txt",
                            lambda item: "canonical", lambda reg: None,
                            prefetched=pre)
    check("prefetched embeds hit the cache — the writer never embeds live", made == 2)

    kb.close()
    print()
    if FAIL:
        print(f"{FAIL} FAILURE(S)")
        return 1
    print(f"domain_cards_test: ALL OK ({PASS} checks)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
