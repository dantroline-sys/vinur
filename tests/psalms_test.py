"""Vulgate↔Hebrew Psalm reconciliation (knowledgehost/psalms.py) + the paragraph-model
ingest fixes it rides on: a Vulgate-numbered edition (Douay-Rheims) and a Hebrew-numbered
one (KJV) ingest into ONE kb and line up verse-for-verse — the per-psalm verse offset
RECOVERED from the two texts' wording, never guessed.  Covers the whole zoo from the real
Gutenberg files: numbered Latin titles (→ the verse-0 superscription slot), the split
psalm (Vulgate 9 = Hebrew 9+10, printed as a mid-chapter verse RESTART), hard-wrapped
verse continuation lines (which must join the verse and never flip the book context —
the 'wisdom.' bug), interleaved notes through the alias, and the honest refusal on a
psalm with no wording overlap.  Real sqlite store + KB, no LM, no embed server.

    python tests/psalms_test.py
"""
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from knowledgehost.config import load_config                    # noqa: E402
from knowledgehost.store import make_store                       # noqa: E402
from knowledgehost.kb import KB                                   # noqa: E402
from knowledgehost import ingest as ingest_mod                   # noqa: E402
from knowledgehost import citations as cite_mod                  # noqa: E402
from knowledgehost import psalms as P                             # noqa: E402
from knowledgehost import scripture as scripture_mod              # noqa: E402
from knowledgehost import structure as S                          # noqa: E402

PASS = FAIL = 0


def check(name, cond):
    global PASS, FAIL
    if cond:
        PASS += 1; print(f"  ok  {name}")
    else:
        FAIL += 1; print(f"FAIL  {name}")


# A miniature Douay-Rheims Psalter, faithful to the real Gutenberg formatting:
# paragraph-structured, verse numbers with a trailing period, a numbered Latin title on
# Ps 3, the Vulgate combined Ps 9 printed as TWO halves both numbered chapter 9 (verse
# restart after the parenthetical divider), a hard-wrapped verse whose continuation line
# is the bare word 'wisdom.', a Challoner-style note paragraph, and a nonsense Ps 4 that
# must be refused.  Psalm 1 is identity (no divergence at all).
DRB = """THE BOOK OF PSALMS

Psalms Chapter 1

1:1. Blessed is the man who hath not walked in the counsel of the
ungodly nor stood in the way of sinners.

1:2. But his will is in the law of the Lord and on his law he shall
meditate day and night.

1:3. And he shall be like a tree planted near the running waters which
shall bring forth its fruit in due season.

Psalms Chapter 3

3:1. The psalm of David when he fled from the face of his son Absalom.

3:2. Why O Lord are they multiplied that afflict me? many are they who
rise up against me.

3:3. Many say to my soul: There is no salvation for him in his God.

3:4. But thou O Lord art my protector my glory and the lifter up of my
head. He hath filled his people with the spirit of
wisdom.

The prophet was delivered from his enemies. A figure of the passion
and resurrection of Christ.

Psalms Chapter 4

4:1. Zamzummim quixotic brouhaha effervescent perspicacious.

4:2. Gobbledygook sesquipedalian rambunctious perspicuity flummoxed.

Psalms Chapter 9

9:1. Unto the end for the hidden things of the Son. A psalm for David.

9:2. I will give praise to thee O Lord with my whole heart: I will
relate all thy wonders.

9:3. I will be glad and rejoice in thee: I will sing to thy name O thou
most high.

(Psalm Chapter 10 according to the Hebrews.)

9:1. Why O Lord hast thou retired afar off? why dost thou slight us in
our wants in the time of trouble?

9:2. Whilst the wicked man is proud the poor is set on fire.
"""

KJV = """The Book of Psalms

1:1 Blessed is the man that walketh not in the counsel of the ungodly
nor standeth in the way of sinners.

1:2 But his delight is in the law of the LORD and in his law doth he
meditate day and night.

1:3 And he shall be like a tree planted by the rivers of water that
bringeth forth his fruit in his season.

3:1 Lord how are they increased that trouble me many are they that
rise up against me.

3:2 Many there be which say of my soul There is no help for him in
God.

3:3 But thou O LORD art a shield for me my glory and the lifter up of
mine head. He hath filled his people with the spirit of
wisdom.

5:1 Give ear to my words O LORD consider my meditation.

5:2 Hearken unto the voice of my cry my King and my God.

9:1 I will praise thee O LORD with my whole heart I will shew forth
all thy marvellous works.

9:2 I will be glad and rejoice in thee I will sing praise to thy name
O thou most High.

10:1 Why standest thou afar off O LORD why hidest thou thyself in
times of trouble?

10:2 The wicked in his pride doth persecute the poor.
"""


def _write(d, name, text):
    p = os.path.join(d, name)
    with open(p, "w", encoding="utf-8") as f:
        f.write(text)
    return p


def main():
    # ── the fixed psalm-number concordance ────────────────────────────────────
    check("concordance: Vulgate 1-8 identity", P.hebrew_targets(1) == [1] and P.hebrew_targets(8) == [8])
    check("concordance: Vulgate 9 = Hebrew 9+10", P.hebrew_targets(9) == [9, 10])
    check("concordance: Vulgate 10-112 shift by one",
          P.hebrew_targets(10) == [11] and P.hebrew_targets(112) == [113])
    check("concordance: Vulgate 113 = Hebrew 114+115", P.hebrew_targets(113) == [114, 115])
    check("concordance: Vulgate 114+115 = Hebrew 116",
          P.hebrew_targets(114) == [116] and P.hebrew_targets(115) == [116])
    check("concordance: Vulgate 146+147 = Hebrew 147",
          P.hebrew_targets(146) == [147] and P.hebrew_targets(147) == [147])
    check("concordance: 148-150 identity", P.hebrew_targets(150) == [150])

    # ── the mid-chapter verse-restart continuation (_VerseRun) ────────────────
    run = S._VerseRun()
    seq = [run.number("Psalms", 9, v) for v in (1, 2, 3)] \
        + [run.number("Psalms", 9, v) for v in (1, 2)]      # the printed RESTART
    check("verse restart continues the numbering (Vulgate Ps 9 two halves → 1..5)",
          seq == [1, 2, 3, 4, 5])
    check("a new chapter resets the numbering", run.number("Psalms", 10, 1) == 1)

    # ── ingest both editions into one store ───────────────────────────────────
    tmp = tempfile.mkdtemp(prefix="kb-psalms-")
    cfg = load_config(None)
    cfg["db_path"] = os.path.join(tmp, "index.db")
    cfg["kb_path"] = os.path.join(tmp, "kb.db")
    cfg["backend"] = "sqlite"
    store = make_store(cfg)

    def ingest(name, text, edition=None):
        p = _write(tmp, name, text)
        prof = S.apply_answers(S.analyze(text), {"kind": "structured"})
        if edition:
            prof["edition"] = edition
        prof["layer_commentary"] = True
        ingest_mod.ingest_file(store, None, cfg, p, profile=prof)
        return p

    drb_path = ingest("drb.txt", DRB, edition="douay-rheims")
    ingest("kjv.txt", KJV)

    # paragraph model: the verse is its WHOLE paragraph, and the wrapped 'wisdom.' line
    # neither flips the book context nor truncates the verse.
    row = store.db.execute("SELECT text FROM chunks WHERE section='bible:Ps.3.4' "
                           "AND title='DRB'").fetchone()
    check("hard-wrapped verse joined whole (continuation lines included)",
          row and "lifter up of my head" in row[0] and row[0].rstrip().endswith("wisdom."))
    check("a wrapped line ending 'wisdom.' does NOT flip the book to Wisdom",
          store.db.execute("SELECT COUNT(*) FROM chunks WHERE section LIKE 'bible:Wis.%'")
               .fetchone()[0] == 0)
    check("the restart half of Vulgate Ps 9 keyed uniquely (9:4, 9:5 — nothing lost)",
          store.db.execute("SELECT COUNT(*) FROM chunks WHERE section LIKE 'bible:Ps.9.%' "
                           "AND title='DRB' AND source_type='scripture'").fetchone()[0] == 5)
    check("the note paragraph became a commentary chunk anchored to its verse (not verse text)",
          store.db.execute("SELECT COUNT(*) FROM chunks WHERE source_type='commentary' "
                           "AND section='bible:Ps.3.4'").fetchone()[0] == 1)

    # ── reconcile: recover the offsets from the wording ───────────────────────
    res = P.reconcile(store, cfg, edition="douay-rheims")
    al = ((store.get_doc_meta(drb_path) or {}).get("reference_map") or {}).get("key_aliases") or {}
    check("reconcile found the reference edition and applied aliases",
          res["applied"] and res["reference"] and res["aliases"] > 0)
    check("identity psalm (Ps 1) needs no aliases",
          not any(k.startswith("bible:Ps.1.") for k in al))
    check("numbered title → the verse-0 superscription slot (Ps 3:1 → Ps.3.0)",
          al.get("bible:Ps.3.1") == "bible:Ps.3.0")
    check("title offset recovered (Douay 3:2 → Hebrew 3:1)",
          al.get("bible:Ps.3.2") == "bible:Ps.3.1")
    check("split psalm: first half → Hebrew 9 (Douay 9:2 → 9:1)",
          al.get("bible:Ps.9.2") == "bible:Ps.9.1")
    check("split psalm: restart half → Hebrew 10 (Douay 9:4 → 10:1)",
          al.get("bible:Ps.9.4") == "bible:Ps.10.1")
    check("nonsense psalm REFUSED — no aliases, reported low-confidence",
          4 in res["low_confidence"]
          and not any(k.startswith("bible:Ps.4.") for k in al))
    sidecar = os.path.join(tmp, "psalm_aliases.douay-rheims.json")
    check("human-readable sidecar written next to the kb",
          os.path.isfile(sidecar) and "key_aliases" in json.load(open(sidecar)))

    # idempotent: run again, same aliases (no growth, no churn)
    P.reconcile(store, cfg, edition="douay-rheims")
    al2 = ((store.get_doc_meta(drb_path) or {}).get("reference_map") or {}).get("key_aliases") or {}
    check("reconcile is idempotent (re-run leaves the alias map unchanged)", al2 == al)

    # ── the graph + the reading surface converge through the aliases ──────────
    kb = KB(cfg)
    cite_mod.build(store, kb, cfg)

    def read(ref):
        v = scripture_mod.parallel_reading(store, kb, cfg, ref)["verses"]
        return v[0] if v else None

    v = read("Psalms 3:1")
    trans = {e["translation"]: e["text"] for e in (v["editions"] if v else [])}
    check("Psalm 3:1 lines up KJV + DRB (the title is NOT masquerading as the verse)",
          set(trans) == {"KJV", "DRB"}
          and "increased that trouble" in trans["KJV"]
          and "multiplied that afflict" in trans["DRB"])
    v = read("Psalms 10:1")
    trans = {e["translation"]: e["text"] for e in (v["editions"] if v else [])}
    check("Psalm 10:1 (the split): Douay's restart half answers for Hebrew 10",
          set(trans) == {"KJV", "DRB"}
          and "standest thou afar off" in trans["KJV"]
          and "retired afar off" in trans["DRB"])
    v = read("Psalms 3:2")
    labels = [e["translation"] for e in (v["editions"] if v else [])]
    check("a moved-away chunk no longer answers for its printed key (one DRB row, not two)",
          labels.count("DRB") == 1 and labels.count("KJV") == 1)
    v = read("Psalms 3:3")
    check("commentary rides the alias — the note surfaces on the HEBREW verse it annotates",
          v and any("figure of the passion" in n for n in v["commentary"]))
    v = read("Psalms 4:1")
    trans = {e["translation"]: e["text"] for e in (v["editions"] if v else [])}
    check("the refused psalm stays honestly on its own keys (Douay Ps 4 ≠ Hebrew Ps 4)",
          trans.get("DRB", "").startswith("Zamzummim"))

    kb.close()
    store.close()
    print()
    if FAIL:
        print(f"{FAIL} FAILURE(S)")
        return 1
    print(f"psalms_test: ALL OK ({PASS} checks)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
