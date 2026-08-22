#!/usr/bin/env python
"""dedupe.py — the janitor: duplicate chunk text, exactly and near-exactly, with
no LM in the loop.  Plus the distill-time claim that keeps a duplicate from ever
reaching an extractor."""
import sqlite3
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from knowledgehost import dedupe as dd          # noqa: E402
from knowledgehost.kb import KB                 # noqa: E402

OK = 0


def ok(name):
    global OK
    OK += 1
    print(f"  ok {OK:2d}  {name}")


def _kb():
    d = tempfile.mkdtemp()
    return KB({"kb_path": str(Path(d) / "kb.db"), "embed_dim": 8})


# ── normalisation + exact hash ───────────────────────────────────────────────
A = "The frost warning covers the whole south coast tonight."
assert dd.text_hash(A) == dd.text_hash(A.upper()), "case must not matter"
assert dd.text_hash(A) == dd.text_hash("  The   frost warning covers the whole\n"
                                       "south coast tonight.  ")
ok("normalisation: case and whitespace differences hash the same")

# the differences a PDF extractor or a word processor introduces on its own
assert dd.text_hash("it's a “quote” — really") == dd.text_hash("it's a \"quote\" - really")
assert dd.text_hash("hy­phen") == dd.text_hash("hyphen")
ok("normalisation: smart quotes, dash width and soft hyphens fold away")

assert dd.text_hash(A) != dd.text_hash(A.replace("south", "north"))
ok("a real wording change is a different hash")

# ── near-duplicates ──────────────────────────────────────────────────────────
long_a = ("Copper sprays slow tomato blight but do not cure it. Remove affected "
          "leaves promptly, water at the base rather than overhead, and give the "
          "plants room so the foliage dries quickly after rain. Resistant "
          "varieties are the only durable answer in a wet summer.")
# the same answer, written again — an assistant researching the same question twice
long_b = ("Copper sprays slow tomato blight but they do not cure it. Remove "
          "affected leaves promptly, water at the base rather than overhead, and "
          "give the plants room so the foliage dries quickly after rain. "
          "Resistant varieties are the only durable answer in a wet summer.")
unrelated = ("Sourdough starters stall when the kitchen is cold or the flour is "
             "bleached. Feed at a steady temperature, use unbleached flour, and "
             "give it a week before deciding anything is wrong with the culture.")

pairs = list(dd.near_pairs([("a", long_a), ("b", long_b), ("c", unrelated)],
                           threshold=0.8))
assert [(p[0], p[1]) for p in pairs] == [("a", "b")], pairs
assert pairs[0][2] >= 0.8
ok("near_pairs finds the same answer written twice, and only that pair")

assert list(dd.near_pairs([("a", long_a), ("c", unrelated)], threshold=0.8)) == []
ok("different content is not a near-duplicate")

# a strict threshold must not fire on a genuine revision that adds substance
revised = long_a + (" In a bad year, strip the lower half of the plant entirely "
                    "and accept a smaller crop; blight travels upward from soil "
                    "splash and a bare stem buys you weeks of harvest.")
strict = list(dd.near_pairs([("a", long_a), ("r", revised)], threshold=0.95))
assert strict == [], strict
ok("a revision that adds substance survives a strict threshold")

# the first-seen chunk is the one kept
p = list(dd.near_pairs([("first", long_a), ("second", long_b)], threshold=0.8))
assert p[0][0] == "first", p
ok("the earlier chunk is reported as the one to keep")

# ── the KB side: claiming text, and what loses the claim ─────────────────────
kb = _kb()
h = dd.text_hash(A)
assert kb.claim_text(h, "chunk1") == "chunk1"
ok("the first chunk to claim a text owns it")

assert kb.claim_text(h, "chunk2") == "chunk1"
ok("a later chunk with the same text is told who owns it")

kb.record_dupe("chunk2", "chunk1", h, kind="exact", similarity=1.0)
kb.mark_distilled("chunk2")
d = kb.dupe_of("chunk2")
assert d and d["of_chunk_id"] == "chunk1" and d["kind"] == "exact"
ok("the duplicate records WHERE the text already lives (provenance isn't lost)")

assert kb.is_distilled("chunk2") and not kb.is_distilled("chunk1")
ok("the duplicate is marked done without being distilled; the owner still needs doing")

kb.record_dupe("chunk3", "chunk1", "", kind="near", similarity=0.93)
assert kb.dupe_stats() == {"exact": 1, "near": 1, "total": 2}
ok("dupe_stats separates exact from near")

kb.record_dupe("chunk3", "chunk1", "", kind="exact", similarity=1.0)
assert kb.dupe_stats()["total"] == 2, "re-recording must update, not duplicate"
ok("re-recording a duplicate updates it in place")
kb.close()

# ── the distill-time gate ────────────────────────────────────────────────────
from knowledgehost import distill as D           # noqa: E402


class _Store:
    def __init__(self, chunks): self.chunks = chunks
    def iter_chunks(self): return iter(self.chunks)


kb2 = _kb()
same = "Copper sprays slow blight. Remove affected leaves and water at the base."
chunks = [
    {"id": "c1", "path_or_url": "/drops/aaa.md", "section": "", "text": same},
    # the same drop re-exported under a new name: different path -> different id
    {"id": "c2", "path_or_url": "/drops/bbb.md", "section": "", "text": same},
    {"id": "c3", "path_or_url": "/docs/other.md", "section": "", "text": "Something else entirely, at length."},
]
counter = [0, 0, 0]
got = [c["id"] for c in D._pending_chunks(_Store(chunks), kb2, counter, cfg={})]
assert got == ["c1", "c3"], got
assert counter[2] == 1, counter
ok("distill skips the re-exported copy: it never reaches an LM")

assert kb2.is_distilled("c2") and kb2.dupe_of("c2")["of_chunk_id"] == "c1"
ok("…and it is checkpointed against the chunk that owns the text")

# turning it off restores the old behaviour exactly
kb3 = _kb()
counter2 = [0, 0, 0]
got2 = [c["id"] for c in D._pending_chunks(_Store(chunks), kb3, counter2,
                                           cfg={"distill_dedupe": False})]
assert got2 == ["c1", "c2", "c3"] and counter2[2] == 0, (got2, counter2)
ok("distill_dedupe = false leaves every chunk in the queue")
kb2.close(); kb3.close()

# ── stale claims: a holder that never landed must not poison later copies ───
# The regression: a pass claimed every chunk's text, then every chunk was
# vetoed (context overflow) and dropped un-stamped.  The claims stayed.  A
# re-ingest gave the same text new ids; each lost the claim, was stamped as a
# "duplicate of the owner", never reached an extractor, fell off the queue —
# and the owner had produced nothing.
kb4 = _kb()
gone = [{"id": "old1", "path_or_url": "/books/a.pdf", "section": "", "text": same}]
counter = [0, 0, 0]
assert [c["id"] for c in D._pending_chunks(_Store(gone), kb4, counter, cfg={})] == ["old1"]
assert not kb4.is_distilled("old1")           # …and then old1 was dropped, never stamped
fresh = [{"id": "new1", "path_or_url": "/books/a.pdf", "section": "", "text": same}]
counter = [0, 0, 0]
got = [c["id"] for c in D._pending_chunks(_Store(fresh), kb4, counter, cfg={})]
assert got == ["new1"] and counter[2] == 0, (got, counter)
assert not kb4.is_distilled("new1") and kb4.dupe_of("new1") is None
ok("a claim whose holder never landed is TAKEN OVER, not honoured: the new copy distils")

# …while a genuine duplicate (owner landed) is still skipped
kb4.mark_distilled("new1")
again = [{"id": "new2", "path_or_url": "/books/a-copy.pdf", "section": "", "text": same}]
counter = [0, 0, 0]
got = [c["id"] for c in D._pending_chunks(_Store(again), kb4, counter, cfg={})]
assert got == [] and counter[2] == 1 and kb4.dupe_of("new2")["of_chunk_id"] == "new1"
ok("…and a copy of text whose owner DID land is still a duplicate (skipped)")

# two copies in ONE pass: the second still dedupes against the in-flight first
kb5 = _kb()
pair = [{"id": "p1", "path_or_url": "/x.md", "section": "", "text": same},
        {"id": "p2", "path_or_url": "/y.md", "section": "", "text": same}]
counter = [0, 0, 0]
got = [c["id"] for c in D._pending_chunks(_Store(pair), kb5, counter, cfg={})]
assert got == ["p1"] and counter[2] == 1, (got, counter)
ok("within one pass the second copy dedupes against the in-flight first")

# healing the poisoned state: stamped-as-dupe-of-a-phantom chunks are re-queued
kb6 = _kb()
h2 = dd.text_hash(same)
kb6.claim_text(h2, "phantom")                  # claimed, never distilled
kb6.record_dupe("victim", "phantom", h2, kind="exact", similarity=1.0)
kb6.mark_distilled("victim")                   # the poisoned stamp
kb6.claim_text(dd.text_hash("other text here"), "real")
kb6.mark_distilled("real")                     # a real owner
kb6.record_dupe("legit", "real", "", kind="exact", similarity=1.0)
kb6.mark_distilled("legit")                    # a legitimate dupe stamp
n = kb6.release_phantom_dupes()
assert n == 1, n
assert not kb6.is_distilled("victim") and kb6.dupe_of("victim") is None
assert kb6.is_distilled("legit") and kb6.dupe_of("legit") is not None
assert kb6.claim_text(h2, "newcomer") == "newcomer"     # the phantom's claim was released
assert kb6.claim_text(dd.text_hash("other text here"), "x") == "real"   # the real one stands
ok("release_phantom_dupes re-queues victims of never-landed owners, keeps real dupes")
kb4.close(); kb5.close(); kb6.close()

print(f"dedupe_test: {OK} checks OK")
