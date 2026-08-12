"""Structure detection for canonical texts (knowledgehost/structure.py): the
analyze → propose spine that a human confirms before a structured ingest.

Covers book matching (abbrev / numbered / roman), scripture detection in both the
inline 'Book C:V' layout and the 'book header + bare C:V' layout, verse-unit
parsing, deterministic cross-reference extraction (incl. bare 'C:V' resolved within
the current book), legal §/section detection + section-unit parsing + U.S.C.
citation extraction, and honest warnings (unknown books, apocrypha).

    python tests/structure_test.py
"""
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from knowledgehost import structure as S   # noqa: E402

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
Romans 9:16 So then it is not of him that willeth; see Exodus 33:19 and 9:15.
"""

HEADER_STYLE = """The Gospel According to John
John
1:1 In the beginning was the Word, and the Word was with God.
1:2 The same was in the beginning with God.
1:3 All things were made by him.
"""

STATUTE = """§ 106. Exclusive rights in copyrighted works
Subject to sections 107 through 122, the owner of copyright has the exclusive rights
to do and to authorize any of the following:
(1) to reproduce the copyrighted work in copies;

§ 107. Limitations on exclusive rights: Fair use
Notwithstanding the provisions of § 106 and § 106A, the fair use of a copyrighted
work is not an infringement of copyright. See also 17 U.S.C. § 501.
"""


def main():
    # ── book matching ────────────────────────────────────────────────────────
    check("book match: full name", S.match_book("Genesis") == (1, "Genesis"))
    check("book match: abbreviation", S.match_book("Jn") == (43, "John"))
    check("book match: numbered w/ arabic", S.match_book("1 Cor") == (46, "1 Corinthians"))
    check("book match: numbered w/ roman", S.match_book("I Cor") == (46, "1 Corinthians"))
    check("book match: 'Song of Solomon'", S.match_book("Song of Solomon") == (22, "Song of Solomon"))
    check("book match: unknown → None", S.match_book("Nephi") is None)

    # ── deuterocanon (Catholic/Orthodox) is first-class: resolves + canonical keys ─
    check("deuterocanon: 'Tobit'/'Tobias' → the same book (67)",
          S.match_book("Tobit") == (67, "Tobit") and S.match_book("Tobias") == (67, "Tobit"))
    check("deuterocanon: 'Ecclesiasticus' → Sirach (not Ecclesiastes)",
          S.match_book("Ecclesiasticus") == (71, "Sirach")
          and S.match_book("Ecclesiastes") == (21, "Ecclesiastes"))
    check("deuterocanon: numbered '1 Machabees' → 1 Maccabees",
          S.match_book("1 Machabees") == (78, "1 Maccabees"))
    check("deuterocanon: canonical OSIS keys (Tob/Sir/1Macc/Wis/Bar)",
          S.scripture_ref("Tobit", 1, 1).key == "bible:Tob.1.1"
          and S.scripture_ref("Sirach", 2, 3).key == "bible:Sir.2.3"
          and S.scripture_ref("1 Maccabees", 4, 5).key == "bible:1Macc.4.5"
          and S.display_for_key("bible:Sir.2.3") == "Sirach 2:3")

    # ── THE MARRYING PROPERTY: any spelling → one canonical (OSIS) key ────────
    def ckey(s, **kw):
        r = S.parse_citations(s, {"kind": "scripture"}, **kw)
        return r[0].key if r else None
    check("marry: 'Jn 3.16' == 'John 3:16' == 'JOHN 3:16' → bible:John.3.16",
          ckey("see Jn 3.16") == ckey("see John 3:16") == ckey("see JOHN 3:16") == "bible:John.3.16")
    check("marry: numbered-book variants converge",
          ckey("1 Cor 13:4") == ckey("I Corinthians 13.4") == "bible:1Cor.13.4")
    check("marry: OSIS code used for the key, display keeps the human form",
          S.scripture_ref("Song of Solomon", 1, 1).key == "bible:Song.1.1"
          and S.scripture_ref("Song of Solomon", 1, 1).display == "Song of Solomon 1:1")

    # ── the alias hook (versification / renumbering divergences) ─────────────
    r315 = S.scripture_ref("3 John", 1, 15)
    check("alias hook remaps a divergent key when a mapping exists",
          S.apply_alias(r315, {"bible:3John.1.15": "bible:3John.1.14"}).key == "bible:3John.1.14")
    check("alias hook is identity without a mapping (never invents equivalence)",
          S.apply_alias(r315, {}).key == "bible:3John.1.15" and r315.key == "bible:3John.1.15")

    # ── scripture: inline 'Book C:V' ─────────────────────────────────────────
    p = S.analyze(KJV)
    check("scripture detected (inline)", p["kind"] == "scripture" and p["confidence"] > 0)
    check("scheme = inline", p["scheme"] == "book-chapter-verse-inline")
    got = {b["canonical"] for b in p["books"]}
    check("books found: John, Genesis, Romans", {"John", "Genesis", "Romans"} <= got)
    units = list(S.parse_units(KJV, p))
    keys = [r.key for r, _ in units]
    check("units carry canonical OSIS keys (node identity)",
          keys == ["bible:John.3.16", "bible:John.3.17", "bible:Gen.1.1", "bible:Rom.9.16"])
    check("units carry the human display form", units[0][0].display == "John 3:16")
    check("unit text carried", units[0][1].startswith("For God so loved"))

    # ── cross-references → canonical keys (incl. bare C:V within this book) ───
    refs = [r.key for r in S.parse_citations(units[3][1], p, book="Romans")]
    check("cross-refs from Romans 9:16 → bible:Exod.33.19 + bare 9:15 → bible:Rom.9.15",
          "bible:Exod.33.19" in refs and "bible:Rom.9.15" in refs)

    # ── scripture: 'book header + bare C:V' layout ───────────────────────────
    ph = S.analyze(HEADER_STYLE)
    check("scripture detected (header style)", ph["kind"] == "scripture")
    check("scheme = chapter-verse-lines", ph["scheme"] == "chapter-verse-lines")
    hu = [r.key for r, _ in S.parse_units(HEADER_STYLE, ph)]
    check("header-style units resolve the book → same canonical keys",
          hu == ["bible:John.1.1", "bible:John.1.2", "bible:John.1.3"])

    # ── real-world: Douay-Rheims from Project Gutenberg ──────────────────────
    # verses print as '1:1.' (trailing period, Vulgate style); the file's appended
    # Gutenberg licence carries 'Section N.' markers that must NOT flip it to legal.
    DOUAY = """THE BOOK OF GENESIS

Genesis Chapter 1

1:1. In the beginning God created heaven, and earth.
1:2. And the earth was void and empty, and darkness was upon the deep.
1:3. And God said: Be light made. And light was made.

Book of Exodus

Exodus Chapter 2

2:1. After this there went a man of the house of Levi.
2:2. And she conceived, and bore a son.

*** END OF THE PROJECT GUTENBERG EBOOK ***

Section 1. General Terms of Use of this eBook.
Section 2. Information about the Project Gutenberg Literary Archive.
Section 3. Information about Donations to the Foundation.
"""
    pd = S.analyze(DOUAY)
    check("Douay '1:1.' verse format → scripture (NOT legal, despite a 'Section N.' licence tail)",
          pd["kind"] == "scripture")
    check("Douay chapter headers resolve the book (Book of X / X Chapter N)",
          {"Genesis", "Exodus"} <= {b["canonical"] for b in pd["books"]})
    dk = [r.key for r, _ in S.parse_units(DOUAY, pd)]
    check("Douay verses parse to canonical keys under the right book context",
          dk == ["bible:Gen.1.1", "bible:Gen.1.2", "bible:Gen.1.3",
                 "bible:Exod.2.1", "bible:Exod.2.2"])

    # a Vulgate book name we can't resolve → the header DROPS context (verses are not
    # mislabeled to the previous book); the shipped Douay map makes it resolve.
    ISAIAS = ("Isaias Chapter 1\n"
              "1:1. The vision of Isaias the son of Amos.\n"
              "1:2. Hear, O ye heavens, and give ear, O earth.\n")
    check("unresolved Vulgate header drops the book context (verses dropped, never mislabeled)",
          [r.key for r, _ in S.parse_units(ISAIAS, {"kind": "scripture"})] == [])
    dr = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                      "refmaps", "douay-rheims.example.json")
    drmaps = S.load_reference_maps([dr])
    check("shipped Douay-Rheims map resolves 'Isaias' → canonical Isaiah keys",
          [r.key for r, _ in S.parse_units(ISAIAS, {"kind": "scripture"}, maps=drmaps)]
          == ["bible:Isa.1.1", "bible:Isa.1.2"])

    # a Catholic Bible with deuterocanon ingests whole + flags the tradition/versification
    CATH = ("Tobit Chapter 1\n"
            "1:1. The book of the words of Tobit, son of Tobiel.\n"
            "Ecclesiasticus Chapter 1\n"
            "1:1. All wisdom is from the Lord God, and hath been always with him.\n"
            "1 Machabees Chapter 1\n"
            "1:1. Now it came to pass, after that Alexander the Macedonian reigned.\n")
    pc = S.analyze(CATH)
    check("deuterocanon detected as scripture with canonical keys across books",
          pc["kind"] == "scripture"
          and [r.key for r, _ in S.parse_units(CATH, pc)]
          == ["bible:Tob.1.1", "bible:Sir.1.1", "bible:1Macc.1.1"])
    check("deuterocanon presence surfaces the canon/versification warning",
          any("deuterocanonical" in w for w in pc["warnings"]))

    # ── unknown-book warning ─────────────────────────────────────────────────
    pu = S.analyze("Nephi 3:7 And it came to pass that I, Nephi, said unto my father.\n"
                   "Nephi 3:8 And it came to pass that I did go.\n")
    check("unknown book name surfaces a warning",
          any("unrecognised book" in w for w in pu["warnings"]))

    # ── legal: work detection + canonical section keys + cross-refs ──────────
    pl = S.analyze(STATUTE)
    check("legal detected", pl["kind"] == "legal" and pl["confidence"] > 0)
    check("legal work (title) detected from '17 U.S.C.'", (pl.get("work") or {}).get("title") == "17")
    lu = list(S.parse_units(STATUTE, pl))
    lkeys = [r.key for r, _ in lu]
    check("section units get canonical keys under the title", lkeys == ["usc:17/106", "usc:17/107"])
    lrefs = [r.key for r in S.parse_citations(lu[1][1], pl)]
    check("§ 107 cross-refs → usc:17/106, usc:17/106A, usc:17/501 (local + explicit U.S.C.)",
          "usc:17/106" in lrefs and "usc:17/106A" in lrefs and "usc:17/501" in lrefs)
    check("legal: a local '§ 106' and an explicit '17 U.S.C. § 106' would share a key",
          S.legal_ref("17", "106").key == "usc:17/106")
    # the PROSE cross-reference form: the title comes from the phrase, not the document
    prose_refs = [r.key for r in S.parse_citations(
        "as provided in section 230(c)(1) of title 47, and see section 106 of title 17",
        {"kind": "legal", "work": {"title": "99"}})]
    check("legal prose 'section M of title N' → usc:N/M (phrase title beats the doc's)",
          "usc:47/230/c/1" in prose_refs and "usc:17/106" in prose_refs)

    # ── the REFERENCE-MAP LOADER: multilingual books + versification aliases ─
    with tempfile.TemporaryDirectory() as td:
        mp = os.path.join(td, "fr-de.json")
        with open(mp, "w", encoding="utf-8") as f:
            json.dump({
                "_note": "test map",
                "book_aliases": {"John": ["Jean", "Johannes"], "Genesis": ["1. Mose", "Genèse"]},
                "key_aliases": {"_note": "ignore me", "bible:Ps.9.22": "bible:Ps.10.1"},
            }, f)
        maps = S.load_reference_maps([mp, os.path.join(td, "absent.json")])
        check("loader: absent files skipped, entries counted (underscore keys ignored)",
              maps.stats["book_aliases"] == 4 and maps.stats["key_aliases"] == 1)
        check("loader: a French book name resolves to the canonical English book",
              maps.match_book("Jean") == (43, "John") and maps.match_book("1. Mose") == (1, "Genesis"))
        check("loader: built-in canon still resolves through the maps",
              maps.match_book("Jn") == (43, "John"))

        # a French/German commentary marries up onto the SAME canonical keys
        fr = "Jean 3:16 parle de l'amour de Dieu; cf. 1. Mose 1:1 et Genèse 22:2."
        frefs = [r.key for r in S.parse_citations(fr, {"kind": "scripture"}, maps=maps)]
        check("marry (multilingual): 'Jean 3:16' → bible:John.3.16 via the loaded map",
              "bible:John.3.16" in frefs)
        check("marry (multilingual): '1. Mose 1:1' and 'Genèse 22:2' → Genesis keys",
              "bible:Gen.1.1" in frefs and "bible:Gen.22.2" in frefs)

        # a document in a divergent versification is folded to the canonical frame
        pv = {"kind": "scripture", "book_order": ["Psalms"]}
        vkeys = [r.key for r, _ in S.parse_units("Psalms\n9:22 In the LORD put I my trust.\n", pv, maps=maps)]
        check("key alias: Ps 9:22 (this edition) folded to canonical bible:Ps.10.1",
              vkeys == ["bible:Ps.10.1"])

        # analyze no longer flags a mapped foreign book as 'unrecognised'
        pa = S.analyze("Jean 3:16 Car Dieu a tant aimé le monde.\n", maps=maps)
        check("analyze with maps: French book recognised, no unknown-book warning",
              pa["kind"] == "scripture" and not any("unrecognised" in w for w in pa["warnings"]))

    # the shipped example map file parses and loads
    ex = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                      "refmaps", "scripture-multilingual.example.json")
    exmaps = S.load_reference_maps([ex])
    check("shipped example map loads (multilingual books present)",
          exmaps.match_book("Johannes") == (43, "John")
          and exmaps.match_book("Apocalypse") == (66, "Revelation"))

    # ── THE INTERACTIVE CONFIRM LAYER: profile → questions → confirmed profile ─
    pq = S.analyze(KJV)
    qs = S.questions_for(pq)
    qids = [q["id"] for q in qs]
    check("scripture raises a confirm step (should_confirm)", S.should_confirm(pq))
    check("first question is the kind/how-to-ingest choice", qids and qids[0] == "kind")
    check("scripture asks the canon/versification question", "canon" in qids)
    check("every question is well-formed (id, prompt, type)",
          all(q.get("id") and q.get("prompt") and q.get("type") in ("choice", "text") for q in qs))

    # unrecognised book → a per-book question, and answers route it three ways
    pn2 = S.analyze("Nephi 3:7 And it came to pass that I, Nephi, said unto my father.\n"
                    "Nephi 3:8 And it came to pass that I did go.\n")
    nq = [q["id"] for q in S.questions_for(pn2)]
    check("an unrecognised book gets its own question", "book:Nephi" in nq)
    conf_map = S.apply_answers(pn2, {"kind": "structured", "book:Nephi": "Revelation"})
    check("answer 'it's Revelation' → a book_aliases entry that marries Nephi→Revelation",
          conf_map["reference_map"]["book_aliases"].get("Revelation") == ["Nephi"])
    m2 = S.load_reference_maps([])  # built-ins only …
    m2 = S.ReferenceMaps(conf_map["reference_map"]["book_aliases"], {})
    check("that ad-hoc alias actually resolves through a ReferenceMaps",
          m2.match_book("Nephi") == (66, "Revelation"))
    conf_keep = S.apply_answers(pn2, {"kind": "structured", "book:Nephi": "keep"})
    check("answer 'keep' → tracked as an extra-canonical book, not an alias",
          conf_keep.get("extra_books") == ["Nephi"]
          and not conf_keep["reference_map"]["book_aliases"])
    conf_ign = S.apply_answers(pn2, {"kind": "structured", "book:Nephi": ""})
    check("blank answer → ignored (no alias, no extra book)",
          not conf_ign["reference_map"]["book_aliases"] and "extra_books" not in conf_ign)

    # kind override → plain short-circuits everything
    conf_plain = S.apply_answers(pq, {"kind": "plain"})
    check("choosing 'ordinary prose' → ingest_as plain, no structured fields forced",
          conf_plain["ingest_as"] == "plain" and conf_plain["confirmed"])

    # legal: work-title question + folding the answer into a canonical work
    pl2 = S.analyze(STATUTE)
    lq = [q["id"] for q in S.questions_for(pl2)]
    check("legal raises kind + work_title questions", "kind" in lq and "work_title" in lq)
    conf_leg = S.apply_answers(pl2, {"kind": "structured", "work_title": "Title 17, U.S.C."})
    check("work_title answer 'Title 17, U.S.C.' → work.title '17' (digits extracted)",
          (conf_leg.get("work") or {}).get("title") == "17"
          and conf_leg["ingest_as"] == "structured")

    # 'ask once per profile' batching signatures
    check("profile signature groups a Title-17 corpus under one key",
          S.profile_signature(conf_leg) == "legal:usc:17")
    check("scripture and legal get distinct signatures; prose collapses to 'plain'",
          S.profile_signature(pq) == "scripture:bible"
          and S.profile_signature(S.analyze("Just ordinary prose here, nothing structured.")) == "plain")

    # ── unknown / prose ──────────────────────────────────────────────────────
    pn = S.analyze("This is an ordinary paragraph of prose with no structure at all. "
                   "It simply discusses ideas in sentences.")
    check("prose → kind unknown + guidance warning",
          pn["kind"] == "unknown" and pn["warnings"])
    check("prose asks NO confirmation questions (normal workflow undisturbed)",
          not S.should_confirm(pn) and S.questions_for(pn) == [])

    print()
    if FAIL:
        print(f"{FAIL} FAILURE(S)")
        return 1
    print(f"structure_test: ALL OK ({PASS} checks)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
