"""Parallel scripture reading (knowledgehost/scripture.parallel_reading): the payoff —
ingest TWO editions (KJV + Douay-Rheims) into ONE kb, and read a reference across both,
lined up verse-for-verse on the same canonical key, with its cross-references and the
commentary attached to it.  Real sqlite store + KB, no LM, no embed server.

    python tests/parallel_reading_test.py
"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from knowledgehost.config import load_config          # noqa: E402
from knowledgehost.store import make_store             # noqa: E402
from knowledgehost.kb import KB                         # noqa: E402
from knowledgehost import ingest as ingest_mod, citations as cite_mod   # noqa: E402
from knowledgehost import structure as S, scripture as scripture_mod    # noqa: E402

PASS = FAIL = 0


def check(name, cond):
    global PASS, FAIL
    if cond:
        PASS += 1; print(f"  ok  {name}")
    else:
        FAIL += 1; print(f"FAIL  {name}")


KJV = ("John 3:16 For God so loved the world, that he gave his only begotten Son; "
       "compare 1 John 4:9.\n"
       "John 3:17 For God sent not his Son into the world to condemn the world.\n")

# a Douay-Rheims excerpt: Vulgate '3:16.' verses under a chapter header, with Challoner
# notes interleaved, and the title line that identifies the edition.
DRB = ("The Holy Bible, Douay-Rheims, translated from the Latin Vulgate by Bishop Challoner\n"
       "John Chapter 3\n"
       "3:16. For God so loved the world, as to give his only begotten Son.\n"
       "A note on love. This verse teaches God's charity toward mankind.\n"
       "It shows the depth of divine love; compare Romans 5:8.\n"
       "And it is the ground of our hope, as the Fathers observe.\n"
       "3:17. For God sent not his Son into the world, to judge the world.\n")


def _write(d, name, text):
    p = os.path.join(d, name)
    with open(p, "w", encoding="utf-8") as f:
        f.write(text)
    return p


def main():
    tmp = tempfile.mkdtemp(prefix="kb-parallel-")
    cfg = load_config(None)
    cfg["db_path"] = os.path.join(tmp, "index.db")
    cfg["kb_path"] = os.path.join(tmp, "kb.db")
    cfg["backend"] = "sqlite"

    store = make_store(cfg)
    kb = KB(cfg)

    def ingest(name, text):
        p = _write(tmp, name, text)
        prof = S.apply_answers(S.analyze(text), {"kind": "structured"})
        ingest_mod.ingest_file(store, None, cfg, p, profile=prof)
        return prof

    kjv_prof = ingest("kjv.txt", KJV)
    drb_prof = ingest("drb.txt", DRB)
    check("KJV ingested with translation label 'KJV' (from the filename)",
          not kjv_prof.get("edition"))
    check("DRB recognised as the Douay-Rheims edition + commentary layered",
          drb_prof.get("edition") == "douay-rheims" and drb_prof.get("layer_commentary"))

    # both editions land John 3:16 on the SAME canonical key — this is the alignment
    aligned = store.chunks_for_section("bible:John.3.16")
    labels = sorted(c["title"] for c in aligned if c["source_type"] == "scripture")
    check("both editions key John 3:16 identically (aligned by canonical key)",
          labels == ["DRB", "KJV"])

    cite_mod.build(store, kb, cfg)

    # ── the study surface: read one reference across everything ───────────────
    view = scripture_mod.parallel_reading(store, kb, cfg, "John 3:16")
    check("reference resolves to the one verse", len(view["verses"]) == 1)
    v = view["verses"][0]
    check("verse displays as 'John 3:16'", v["display"] == "John 3:16")
    trans = {e["translation"]: e["text"] for e in v["editions"]}
    check("both translations line up side by side (KJV + DRB), each its own wording",
          set(trans) == {"KJV", "DRB"}
          and trans["KJV"].startswith("For God so loved the world, that he gave")
          and trans["DRB"].startswith("For God so loved the world, as to give"))
    check("the verse's cross-reference is present (KJV cited 1 John 4:9)",
          "1 John 4:9" in v["cross_references"])
    check("the commentary attached to the verse is surfaced (Challoner note)",
          any("charity" in n or "divine love" in n for n in v["commentary"]))

    # ── reference resolution incl. a verse range ─────────────────────────────
    check("resolve 'John 3:16' → one canonical key",
          scripture_mod.resolve_reference("John 3:16") == ["bible:John.3.16"])
    check("resolve a range 'John 3:16-17' → each verse key",
          scripture_mod.resolve_reference("John 3:16-17")
          == ["bible:John.3.16", "bible:John.3.17"])
    rng = scripture_mod.parallel_reading(store, kb, cfg, "John 3:16-17")
    check("reading a range returns both verses, each with both editions",
          len(rng["verses"]) == 2 and all(len(vv["editions"]) == 2 for vv in rng["verses"]))

    kb.close()
    store.close()
    print()
    if FAIL:
        print(f"{FAIL} FAILURE(S)")
        return 1
    print(f"parallel_reading_test: ALL OK ({PASS} checks)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
