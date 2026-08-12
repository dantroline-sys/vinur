"""Deferred-ingest inbox (knowledgehost/pending.py) + the bulk-crawl confirm gate
(knowledgehost/ingest.crawl): a structured/ambiguous plain-text doc is set aside for
confirmation instead of ingested on a guess, grouped once per profile; answering it
lets the next crawl ingest it unit-by-unit.  Real sqlite store, no embed server.

    python tests/pending_test.py
"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from knowledgehost import pending as PEND                 # noqa: E402
from knowledgehost import structure as S                  # noqa: E402
from knowledgehost import ingest as ingest_mod            # noqa: E402
from knowledgehost.config import load_config              # noqa: E402
from knowledgehost.store import make_store                # noqa: E402

PASS = FAIL = 0


def check(name, cond):
    global PASS, FAIL
    if cond:
        PASS += 1; print(f"  ok  {name}")
    else:
        FAIL += 1; print(f"FAIL  {name}")


KJV = ("John 3:16 For God so loved the world, that he gave his only begotten Son.\n"
       "John 3:17 For God sent not his Son into the world to condemn the world.\n"
       "Genesis 1:1 In the beginning God created the heaven and the earth.\n")
PROSE = ("This is an ordinary note. It has sentences and paragraphs and discusses "
         "several ideas, but nothing that looks like a verse or a statute section.\n")


def _write(d, name, text):
    p = os.path.join(d, name)
    with open(p, "w", encoding="utf-8") as f:
        f.write(text)
    return p


def store_side():
    # ── the store, in isolation ───────────────────────────────────────────────
    with tempfile.TemporaryDirectory() as td:
        p = PEND.Pending(os.path.join(td, "pending.db"))
        prof = S.analyze(KJV)
        qs = S.questions_for(prof)
        sig = S.profile_signature(prof)
        rid, is_new = p.defer(sig, "scripture", prof, qs, "/x/kjv.txt")
        check("first defer creates a request", is_new and rid)
        rid2, is_new2 = p.defer(sig, "scripture", prof, qs, "/x/kjv2.txt")
        check("second doc of the SAME profile joins that one request (once per profile)",
              rid2 == rid and not is_new2)
        req = p.get(rid)
        check("request carries both docs + its questions", req["doc_count"] == 2 and req["questions"])
        check("pending: no confirmed profile yet", p.confirmed_profile(sig) is None
              and p.answer_for_path("/x/kjv.txt") is None)
        check("pending_count == 1 request", p.pending_count() == 1)

        confirmed = S.apply_answers(prof, {"kind": "structured"})
        ok = p.answer(rid, {"kind": "structured"}, confirmed)
        check("answering flips the request to 'answered'", ok
              and p.confirmed_profile(sig)["ingest_as"] == "structured")
        check("answer reaches every doc filed under the request",
              (p.answer_for_path("/x/kjv2.txt") or {}).get("ingest_as") == "structured")
        check("no longer pending", p.pending_count() == 0 and p.list("pending") == []
              and len(p.list("answered")) == 1)
        p.close()


def gate_side():
    # ── the bulk crawl gate, end to end ───────────────────────────────────────
    with tempfile.TemporaryDirectory() as td:
        src = os.path.join(td, "src"); os.makedirs(src)
        _write(src, "kjv.txt", KJV)
        _write(src, "notes.txt", PROSE)
        cfg = load_config(None)
        cfg["db_path"] = os.path.join(td, "index.db")
        cfg["kb_path"] = os.path.join(td, "kb.db")
        cfg["backend"] = "sqlite"
        cfg["sources"] = [src]
        cfg["pending_db"] = os.path.join(td, "pending.db")

        store = make_store(cfg)
        r1 = ingest_mod.crawl(store, None, cfg)
        check("crawl ingests the ordinary note", r1["chunks"] > 0)
        check("crawl DEFERS the scripture doc (needs_confirm surfaced)",
              r1.get("needs_confirm") == 1)
        kjv = os.path.join(src, "kjv.txt")
        check("the scripture doc was NOT ingested yet",
              not store.chunks_for_path(kjv))

        pend = PEND.open_pending(cfg)
        reqs = pend.list("pending")
        check("the inbox holds one pending scripture request with questions",
              len(reqs) == 1 and reqs[0]["kind"] == "scripture"
              and any(q["id"] == "kind" for q in reqs[0]["questions"]))
        # answer it exactly as the server would
        req = reqs[0]
        confirmed = S.apply_answers(req["profile"], {"kind": "structured"})
        pend.answer(req["id"], {"kind": "structured"}, confirmed)
        pend.close()

        r2 = ingest_mod.crawl(store, None, cfg)
        secs = sorted(c["section"] for c in store.chunks_for_path(kjv))
        check("after answering, the next crawl ingests the scripture unit-by-unit",
              secs == ["bible:Gen.1.1", "bible:John.3.16", "bible:John.3.17"])
        check("the ordinary note is untouched on re-crawl (unchanged skip)",
              not r2.get("needs_confirm"))
        check("the crawl AUTO-builds the cross-reference graph (confirm default on)",
              r2.get("citations") and r2["citations"]["units"] == 3)
        store.close()


def main():
    store_side()
    gate_side()
    print()
    if FAIL:
        print(f"{FAIL} FAILURE(S)")
        return 1
    print(f"pending_test: ALL OK ({PASS} checks)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
