"""Structured ingest (knowledgehost/ingest._ingest_structured_doc): a CONFIRMED
scripture/legal profile makes ingest_file store ONE chunk per canonical unit (verse /
section), each `section` a canonical key — the durable node identity a later
citation-graph pass builds on.  Uses a real sqlite store, no embed server (sparse-only).

    python tests/structured_ingest_test.py
"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from knowledgehost.config import load_config          # noqa: E402
from knowledgehost.store import make_store             # noqa: E402
from knowledgehost import ingest as ingest_mod, structure as S   # noqa: E402

PASS = FAIL = 0


def check(name, cond):
    global PASS, FAIL
    if cond:
        PASS += 1; print(f"  ok  {name}")
    else:
        FAIL += 1; print(f"FAIL  {name}")


KJV = """John 3:16 For God so loved the world, that he gave his only begotten Son.
John 3:17 For God sent not his Son into the world to condemn the world.
Genesis 1:1 In the beginning God created the heaven and the earth.
Romans 9:16 So then it is not of him that willeth; see Exodus 33:19.
"""

STATUTE = """§ 106. Exclusive rights in copyrighted works
Subject to sections 107 through 122, the owner of copyright has the exclusive rights.
(1) to reproduce the copyrighted work in copies;

§ 107. Limitations on exclusive rights: Fair use
Notwithstanding the provisions of § 106, the fair use of a copyrighted work is not an
infringement of copyright. See also 17 U.S.C. § 501.
"""


def _fresh_cfg(tmp):
    cfg = load_config(None)
    cfg["db_path"] = os.path.join(tmp, "index.db")
    cfg["backend"] = "sqlite"
    cfg["sources"] = [tmp]
    return cfg


def _write(tmp, name, text):
    p = os.path.join(tmp, name)
    with open(p, "w", encoding="utf-8") as f:
        f.write(text)
    return p


def main():
    tmp = tempfile.mkdtemp(prefix="kb-struct-")

    # ── scripture: confirmed structured → one chunk per verse, keyed by OSIS ──
    cfg = _fresh_cfg(tmp)
    store = make_store(cfg)
    bib = _write(tmp, "kjv.txt", KJV)
    prof = S.apply_answers(S.analyze(KJV), {"kind": "structured"})
    n = ingest_mod.ingest_file(store, None, cfg, bib, profile=prof)
    check("all four verses ingested as units", n == 4)
    chunks = store.chunks_for_path(bib)
    secs = sorted(c["section"] for c in chunks)
    check("each chunk's section is a canonical OSIS key",
          secs == ["bible:Gen.1.1", "bible:John.3.16", "bible:John.3.17", "bible:Rom.9.16"])
    stypes = {r[0] for r in store.db.execute(
        "SELECT DISTINCT source_type FROM chunks WHERE path_or_url=?", (bib,))}
    check("chunks are tagged source_type='scripture'", stypes == {"scripture"})
    j316 = next(c for c in chunks if c["section"] == "bible:John.3.16")
    check("the unit's text is just its verse (one verse per node, not a heading-chunk)",
          j316["text"].startswith("For God so loved") and "3:17" not in j316["text"])
    meta = store.get_doc_meta(bib) or {}
    check("doc_meta records the structured profile for the later graph pass",
          meta.get("structured") and meta.get("kind") == "scripture")
    check("the stored key reverses to a friendly citation",
          S.display_for_key("bible:John.3.16") == "John 3:16")

    # ── legal: confirmed structured → one chunk per section, keyed under the title ─
    stat = _write(tmp, "title17.txt", STATUTE)
    profl = S.apply_answers(S.analyze(STATUTE), {"kind": "structured", "work_title": "17"})
    nl = ingest_mod.ingest_file(store, None, cfg, stat, profile=profl)
    check("both sections ingested as units", nl == 2)
    lsecs = sorted(c["section"] for c in store.chunks_for_path(stat))
    check("each section chunk is keyed usc:17/…", lsecs == ["usc:17/106", "usc:17/107"])

    # ── plain confirmation → the NORMAL path (heading chunks), structure untouched ─
    prof_plain = S.apply_answers(S.analyze(KJV), {"kind": "plain"})
    bib2 = _write(tmp, "kjv-plain.txt", KJV)
    ingest_mod.ingest_file(store, None, cfg, bib2, profile=prof_plain)
    psecs = [c["section"] for c in store.chunks_for_path(bib2)]
    check("a 'plain' confirmation does NOT produce canonical keys (normal ingest)",
          not any(s.startswith("bible:") for s in psecs))

    # ── mis-confirmation safety: structured profile but no parsable units → not lost ─
    prose = _write(tmp, "essay.txt", "Just an ordinary essay with sentences and no verses.\n")
    forced = {"kind": "scripture", "ingest_as": "structured", "confirmed": True,
              "scheme": "book-chapter-verse-inline", "reference_map": {}}
    nf = ingest_mod.ingest_file(store, None, cfg, prose, profile=forced)
    check("structured profile that parses 0 units falls back to normal ingest (content kept)",
          nf and nf > 0 and store.chunks_for_path(prose))

    store.close()
    print()
    if FAIL:
        print(f"{FAIL} FAILURE(S)")
        return 1
    print(f"structured_ingest_test: ALL OK ({PASS} checks)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
