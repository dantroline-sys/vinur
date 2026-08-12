"""Structure detection for canonical, cross-referential corpora — scripture and
legal text — so they can be ingested one addressable unit at a time (a verse, a
section) with a stable citation, and their cross-references turned into graph edges.

Plain-text editions vary wildly in HOW they address units: a Bible may print
``John 3:16 For God so loved…`` or a book file of bare ``3:16 …`` lines, with book
names spelled out or abbreviated a dozen ways (even Roman-numeral chapters in old
editions); a statute may write ``§ 106``, ``Sec. 106`` or ``Section 106``.  And
different Bibles follow different versification traditions.  So we DO NOT
guess-and-ingest — ``analyze()`` reads a document and PROPOSES a profile (how units
are addressed, which books were found, how cross-references are written, with
samples + warnings) for a human to CONFIRM or correct before anything is ingested.

Pure stdlib, deterministic, no LM.  ``parse_units(text, profile)`` and
``parse_citations(text, profile, book)`` apply a confirmed profile.
"""
from __future__ import annotations

import json
import os
import re

# ── Protestant canon (66) with common plain-text spellings/abbreviations ──────
# (order, canonical, [aliases…]).  Aliases are matched case-insensitively with
# dots/spaces stripped; numbered books also accept roman/word forms (1/I/First).
_BOOKS: list[tuple[int, str, list[str]]] = [
    (1, "Genesis", ["gen", "ge", "gn"]), (2, "Exodus", ["exod", "exo", "ex"]),
    (3, "Leviticus", ["lev", "le", "lv"]), (4, "Numbers", ["num", "nu", "nm", "nb"]),
    (5, "Deuteronomy", ["deut", "de", "dt"]), (6, "Joshua", ["josh", "jos", "jsh"]),
    (7, "Judges", ["judg", "jdg", "jg"]), (8, "Ruth", ["rth", "ru"]),
    (9, "1 Samuel", ["1sam", "1sm", "1sa"]), (10, "2 Samuel", ["2sam", "2sm", "2sa"]),
    (11, "1 Kings", ["1kgs", "1ki", "1ki"]), (12, "2 Kings", ["2kgs", "2ki"]),
    (13, "1 Chronicles", ["1chron", "1chr", "1ch"]), (14, "2 Chronicles", ["2chron", "2chr", "2ch"]),
    (15, "Ezra", ["ezr", "ez"]), (16, "Nehemiah", ["neh", "ne"]),
    (17, "Esther", ["esth", "est", "es"]), (18, "Job", ["jb"]),
    (19, "Psalms", ["psalm", "pslm", "ps", "psa", "psm"]), (20, "Proverbs", ["prov", "pro", "prv", "pr"]),
    (21, "Ecclesiastes", ["eccles", "eccl", "ecc", "qoh"]), (22, "Song of Solomon", ["song", "sos", "canticles", "cant", "sng"]),
    (23, "Isaiah", ["isa", "is"]), (24, "Jeremiah", ["jer", "je", "jr"]),
    (25, "Lamentations", ["lam", "la"]), (26, "Ezekiel", ["ezek", "eze", "ezk"]),
    (27, "Daniel", ["dan", "da", "dn"]), (28, "Hosea", ["hos", "ho"]),
    (29, "Joel", ["jl", "joe"]), (30, "Amos", ["am", "amo"]),
    (31, "Obadiah", ["obad", "ob"]), (32, "Jonah", ["jon", "jnh"]),
    (33, "Micah", ["mic", "mc"]), (34, "Nahum", ["nah", "na"]),
    (35, "Habakkuk", ["hab", "hb"]), (36, "Zephaniah", ["zeph", "zep", "zp"]),
    (37, "Haggai", ["hag", "hg"]), (38, "Zechariah", ["zech", "zec", "zc"]),
    (39, "Malachi", ["mal", "ml"]), (40, "Matthew", ["matt", "mat", "mt"]),
    (41, "Mark", ["mrk", "mar", "mk"]), (42, "Luke", ["luk", "lk"]),
    (43, "John", ["joh", "jhn", "jn"]), (44, "Acts", ["act", "ac"]),
    (45, "Romans", ["rom", "ro", "rm"]), (46, "1 Corinthians", ["1cor", "1co"]),
    (47, "2 Corinthians", ["2cor", "2co"]), (48, "Galatians", ["gal", "ga"]),
    (49, "Ephesians", ["eph", "ephes"]), (50, "Philippians", ["phil", "php", "pp"]),
    (51, "Colossians", ["col", "co"]), (52, "1 Thessalonians", ["1thess", "1thes", "1th"]),
    (53, "2 Thessalonians", ["2thess", "2thes", "2th"]), (54, "1 Timothy", ["1tim", "1ti"]),
    (55, "2 Timothy", ["2tim", "2ti"]), (56, "Titus", ["tit", "ti"]),
    (57, "Philemon", ["philem", "phlm", "pm"]), (58, "Hebrews", ["heb", "hb"]),
    (59, "James", ["jas", "jm"]), (60, "1 Peter", ["1pet", "1pe", "1pt"]),
    (61, "2 Peter", ["2pet", "2pe", "2pt"]), (62, "1 John", ["1jn", "1jo", "1joh"]),
    (63, "2 John", ["2jn", "2jo", "2joh"]), (64, "3 John", ["3jn", "3jo", "3joh"]),
    (65, "Jude", ["jud", "jd"]), (66, "Revelation", ["rev", "re", "apocalypse", "apoc"]),
    # ── Deuterocanonical / apocryphal books (Catholic + Orthodox canons) ──────
    # So a Catholic/Orthodox Bible ingests WHOLE.  Orders 67+ (display only — they sort
    # after the 66, which keeps every protocanonical order number stable).  Names that
    # genuinely DIVERGE between traditions (the Esdras numbering, Vulgate Kings=Samuel)
    # are left to the per-document reference map + confirm step, never auto-mapped here.
    (67, "Tobit", ["tob", "tb", "tobias"]),
    (68, "Judith", ["jdt", "jdth"]),
    (69, "Additions to Esther", ["addesth", "greek esther", "rest of esther"]),
    (70, "Wisdom", ["wis", "wisd", "wisdom of solomon"]),
    (71, "Sirach", ["sir", "ecclus", "ecclesiasticus", "ben sira", "wisdom of sirach"]),
    (72, "Baruch", ["bar"]),
    (73, "Letter of Jeremiah", ["epjer", "epistle of jeremy", "letter of jeremy"]),
    (74, "Prayer of Azariah", ["praz", "prazar", "song of the three holy children",
                               "song of the three", "song of three children"]),
    (75, "Susanna", ["sus"]),
    (76, "Bel and the Dragon", ["bel"]),
    (77, "Prayer of Manasseh", ["prman", "manasses", "prayer of manasses"]),
    (78, "1 Maccabees", ["1macc", "1mac", "1ma", "1 machabees"]),
    (79, "2 Maccabees", ["2macc", "2mac", "2ma", "2 machabees"]),
    (80, "3 Maccabees", ["3macc", "3mac", "3ma", "3 machabees"]),
    (81, "4 Maccabees", ["4macc", "4mac", "4ma", "4 machabees"]),
    (82, "1 Esdras", ["1esd", "1esdr"]),
    (83, "2 Esdras", ["2esd", "2esdr"]),
    (84, "Psalm 151", ["ps151"]),
]

# The deuterocanonical books (Catholic + Orthodox).  Now first-class in the canon, but
# their PRESENCE is still worth surfacing — it pins the tradition (Catholic/Orthodox)
# and warns that the versification (esp. Vulgate Psalm numbering) may diverge.
_DEUTERO = {"Tobit", "Judith", "Additions to Esther", "Wisdom", "Sirach", "Baruch",
            "Letter of Jeremiah", "Prayer of Azariah", "Susanna", "Bel and the Dragon",
            "Prayer of Manasseh", "1 Maccabees", "2 Maccabees", "3 Maccabees",
            "4 Maccabees", "1 Esdras", "2 Esdras", "Psalm 151"}

_ROMAN = {"i": "1", "ii": "2", "iii": "3"}
_WORDNUM = {"first": "1", "second": "2", "third": "3"}


def _norm_book_token(tok: str) -> str:
    """Lowercase, drop dots, fold a leading roman/word ordinal to a digit, and
    collapse ``1 John``/``1John``/``I John`` to ``1john`` for matching."""
    t = tok.strip().lower().replace(".", "")
    t = re.sub(r"\s+", " ", t).strip()
    m = re.match(r"^([ivx]+|first|second|third|1|2|3)\s+(.*)$", t)
    if m:
        lead = _ROMAN.get(m.group(1), _WORDNUM.get(m.group(1), m.group(1)))
        t = lead + " " + m.group(2)
    return t.replace(" ", "")


# canonical-form index: every accepted spelling → (order, canonical)
_BOOK_INDEX: dict[str, tuple[int, str]] = {}
for _order, _canon, _aliases in _BOOKS:
    for _form in [_canon, *_aliases]:
        _BOOK_INDEX[_norm_book_token(_form)] = (_order, _canon)


def match_book(token: str) -> tuple[int, str] | None:
    """Resolve a book token to (order, canonical name), or None if unrecognised."""
    return _BOOK_INDEX.get(_norm_book_token(token))


# ── canonical reference identity: the "standardised indexing language" ────────
# Every surface spelling of a reference resolves to ONE canonical key, so nodes and
# cross-references from DIFFERENT documents that name the same unit converge on the
# same graph node (node id = hash(label,kind), so an identical key auto-merges).
#   scripture → OSIS-style  bible:<OsisBook>.<chap>.<verse>   (John 3:16 → bible:John.3.16)
#   legal     → USC-style   usc:<title>/<section><subpath>    (17 U.S.C. §106(a)(1) → usc:17/106/a/1)
# OSIS book codes (the Bible-software interchange standard).
_OSIS = {
    "Genesis": "Gen", "Exodus": "Exod", "Leviticus": "Lev", "Numbers": "Num",
    "Deuteronomy": "Deut", "Joshua": "Josh", "Judges": "Judg", "Ruth": "Ruth",
    "1 Samuel": "1Sam", "2 Samuel": "2Sam", "1 Kings": "1Kgs", "2 Kings": "2Kgs",
    "1 Chronicles": "1Chr", "2 Chronicles": "2Chr", "Ezra": "Ezra", "Nehemiah": "Neh",
    "Esther": "Esth", "Job": "Job", "Psalms": "Ps", "Proverbs": "Prov",
    "Ecclesiastes": "Eccl", "Song of Solomon": "Song", "Isaiah": "Isa", "Jeremiah": "Jer",
    "Lamentations": "Lam", "Ezekiel": "Ezek", "Daniel": "Dan", "Hosea": "Hos", "Joel": "Joel",
    "Amos": "Amos", "Obadiah": "Obad", "Jonah": "Jonah", "Micah": "Mic", "Nahum": "Nah",
    "Habakkuk": "Hab", "Zephaniah": "Zeph", "Haggai": "Hag", "Zechariah": "Zech",
    "Malachi": "Mal", "Matthew": "Matt", "Mark": "Mark", "Luke": "Luke", "John": "John",
    "Acts": "Acts", "Romans": "Rom", "1 Corinthians": "1Cor", "2 Corinthians": "2Cor",
    "Galatians": "Gal", "Ephesians": "Eph", "Philippians": "Phil", "Colossians": "Col",
    "1 Thessalonians": "1Thess", "2 Thessalonians": "2Thess", "1 Timothy": "1Tim",
    "2 Timothy": "2Tim", "Titus": "Titus", "Philemon": "Phlm", "Hebrews": "Heb",
    "James": "Jas", "1 Peter": "1Pet", "2 Peter": "2Pet", "1 John": "1John",
    "2 John": "2John", "3 John": "3John", "Jude": "Jude", "Revelation": "Rev",
    # deuterocanon (standard OSIS interchange codes)
    "Tobit": "Tob", "Judith": "Jdt", "Additions to Esther": "AddEsth", "Wisdom": "Wis",
    "Sirach": "Sir", "Baruch": "Bar", "Letter of Jeremiah": "EpJer",
    "Prayer of Azariah": "PrAzar", "Susanna": "Sus", "Bel and the Dragon": "Bel",
    "Prayer of Manasseh": "PrMan", "1 Maccabees": "1Macc", "2 Maccabees": "2Macc",
    "3 Maccabees": "3Macc", "4 Maccabees": "4Macc", "1 Esdras": "1Esd",
    "2 Esdras": "2Esd", "Psalm 151": "AddPs",
}

from collections import namedtuple   # noqa: E402
# key = the canonical merge id (also the node label); display = human form;
# work = the source-work namespace ('bible', 'usc:17'); parts = structured tuple.
Ref = namedtuple("Ref", "kind key display work parts")


_OSIS_REV = {v: k for k, v in _OSIS.items()}


def osis_code(canonical_book: str) -> str:
    return _OSIS.get(canonical_book, canonical_book.replace(" ", ""))


def book_of_key(key: str):
    """The canonical book name for a scripture key ('bible:Rom.9.16' → 'Romans'), so a
    bare 'C:V' cross-reference in that verse resolves within its own book.  None for a
    non-scripture key."""
    if key.startswith("bible:"):
        m = re.match(r"([^.]+)\.", key[6:])
        if m:
            return _OSIS_REV.get(m.group(1), m.group(1))
    return None


def display_for_key(key: str) -> str:
    """Reverse a canonical key to its human citation ('bible:John.3.16' → 'John 3:16',
    'usc:17/106/a/1' → '17 U.S.C. § 106(a)(1)').  For friendly rendering of a stored
    unit whose durable identity is the key; returns the key unchanged if unrecognised."""
    if key.startswith("bible:"):
        m = re.match(r"([^.]+)\.(\d+)\.(\d+)(?:-(\d+))?$", key[6:])
        if m:
            book = _OSIS_REV.get(m.group(1), m.group(1))
            rng = f"-{m.group(4)}" if m.group(4) else ""
            return f"{book} {m.group(2)}:{m.group(3)}{rng}"
    if key.startswith("usc:"):
        body = key[4:]
        if body.startswith("s"):
            parts = body[1:].split("/")
            return f"§ {parts[0]}" + "".join(f"({p})" for p in parts[1:])
        parts = body.split("/")
        if len(parts) >= 2:
            return f"{parts[0]} U.S.C. § {parts[1]}" + "".join(f"({p})" for p in parts[2:])
    return key


def scripture_ref(book_canonical: str, chap, verse, verse_end=None) -> Ref:
    osis = osis_code(book_canonical)
    rng = f"-{verse_end}" if verse_end and str(verse_end) != str(verse) else ""
    return Ref("scripture", f"bible:{osis}.{chap}.{verse}{rng}",
               f"{book_canonical} {chap}:{verse}{rng}", "bible",
               (book_canonical, int(chap), int(verse)))


def legal_ref(title, section, subs: str = "") -> Ref:
    subpath = "".join("/" + s for s in re.findall(r"\(([0-9A-Za-z]+)\)", subs or ""))
    t = str(title or "").strip()
    if t:
        return Ref("legal", f"usc:{t}/{section}{subpath}",
                   f"{t} U.S.C. § {section}{subs or ''}", f"usc:{t}", (t, section, subs or ""))
    return Ref("legal", f"usc:s{section}{subpath}", f"§ {section}{subs or ''}",
               "usc", ("", section, subs or ""))


def apply_alias(ref: Ref, alias_map: dict | None) -> Ref:
    """Fold a canonical key onto its equivalent under a domain-supplied alias table
    — the seam for versification / renumbering divergences between compatible
    systems (e.g. an edition where a verse is split differently).  Identity when the
    map is empty or has no entry, so it never invents equivalences."""
    if alias_map and ref.key in alias_map:
        tgt = alias_map[ref.key]
        return ref._replace(key=tgt)
    return ref


# ── reference maps: multilingual book names + versification/renumbering aliases ──
class ReferenceMaps:
    """Loaded equivalence data so DIFFERENT (even multilingual) documents marry up:
      * book_aliases — foreign / variant book names → a canonical book, so a French
        'Jean', German '1. Mose', or Latin 'Ioannes' resolve to the same OSIS book;
      * key_aliases — canonical-key → canonical-key, for versification / renumbering
        divergences (e.g. an LXX/Vulgate Psalm split), applied to every parsed ref.
    Built from the built-in canon plus any loaded maps; unknown/absent entries fall
    back to the built-ins, never inventing an equivalence."""

    def __init__(self, book_aliases: dict | None = None, key_aliases: dict | None = None):
        self.key_aliases = {str(k): str(v) for k, v in (key_aliases or {}).items()}
        self._books: dict[str, tuple[int, str]] = {}
        self.unmapped_targets: list[str] = []          # alias targets that aren't a known book
        for canon, forms in (book_aliases or {}).items():
            b = match_book(canon)                      # target must be a known canonical book
            if not b:
                self.unmapped_targets.append(canon)
                continue
            forms = forms if isinstance(forms, list) else [forms]
            for form in forms:
                self._books[_norm_book_token(str(form))] = b

    def match_book(self, token: str):
        """Loaded aliases first, then the built-in canon."""
        return self._books.get(_norm_book_token(token)) or match_book(token)

    def apply(self, ref: Ref) -> Ref:
        return apply_alias(ref, self.key_aliases)

    @property
    def stats(self) -> dict:
        return {"book_aliases": len(self._books), "key_aliases": len(self.key_aliases),
                "unmapped_targets": list(self.unmapped_targets)}


def _read_map_file(path: str) -> dict:
    if path.lower().endswith(".toml"):
        import tomllib
        with open(path, "rb") as f:
            return tomllib.load(f)
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _merge_map(into_books: dict, into_keys: dict, data: dict) -> None:
    for canon, forms in (data.get("book_aliases") or {}).items():
        if str(canon).startswith("_"):                  # a doc/comment key, not data
            continue
        into_books.setdefault(canon, []).extend(forms if isinstance(forms, list) else [forms])
    into_keys.update({str(k): str(v) for k, v in (data.get("key_aliases") or {}).items()
                      if not str(k).startswith("_")})


def load_reference_maps(paths, extra: dict | None = None) -> ReferenceMaps:
    """Merge one or more reference-map files (.json / .toml) into a ReferenceMaps.
    Each file may carry ``book_aliases`` ({canonical: [variant, …]}) and
    ``key_aliases`` ({from_key: to_key}); later files win on key_aliases and add to
    book_aliases.  Missing / unreadable files are skipped (never fatal) — a domain
    pack ships these, and a half-written map must not break ingest.  `extra` folds in
    one more in-memory map of the SAME shape last (a document's ad-hoc, answer-derived
    aliases), so per-doc confirmations compose with a pack's shipped maps."""
    book_aliases: dict[str, list] = {}
    key_aliases: dict[str, str] = {}
    for p in (paths or []):
        p = os.path.expanduser(str(p))
        if not os.path.isfile(p):
            continue
        try:
            data = _read_map_file(p)
        except (OSError, ValueError):
            continue
        _merge_map(book_aliases, key_aliases, data)
    if extra:
        _merge_map(book_aliases, key_aliases, extra)
    return ReferenceMaps(book_aliases, key_aliases)


def _match(maps, token):
    return maps.match_book(token) if maps else match_book(token)


# ── citation patterns ─────────────────────────────────────────────────────────
# A book token: an optional ordinal ("1", "1.", "I", "First"), then Unicode LETTERS
# (so "Genèse", "Römer", "Ésaïe" match), optionally "<Book> of <Book>".  Over-matching
# is harmless: only tokens that resolve via match_book/the maps become references.
_LT = r"[^\W\d_]"                                    # a letter in any script
_BOOK_TOK = (r"(?:[1-3]\.?|I{1,3}|First|Second|Third)?\s*"
             + _LT + r"{2,}(?:\s+of\s+" + _LT + r"{2,})?")
# "John 3:16", "1 Cor 13:4-18", "Genèse 22:2"  (optional verse-range end captured)
_RE_REF_BOOK = re.compile(rf"\b({_BOOK_TOK})\.?\s+(\d+)[:.](\d+)(?:\s*[-–]\s*(\d+))?")
# a verse line — inline "Book C:V text" or bare "C:V text".  The verse number may
# carry a trailing period (Douay-Rheims / Vulgate editions print "1:1. In the …").
_RE_LINE_BOOKCV = re.compile(rf"^\s*({_BOOK_TOK})\.?\s+(\d+)[:.](\d+)\.?\s+(.*\S)\s*$")
_RE_LINE_CV = re.compile(r"^\s*(\d+)[:.](\d+)\.?\s+(.*\S)\s*$")
_RE_BARE_CV = re.compile(r"\b(\d+)[:.](\d+)\b")
# a chapter header that carries the book context for the bare 'C:V' lines beneath it:
# "Genesis Chapter 1", "Psalms Chapter 3", "Psalm 23"; and a table-of-contents / section
# title "The Book of Genesis", "Book of Exodus".
_RE_CHAP_HEADER = re.compile(r"^\s*(.+?)\s+(?:chapter|chap\.?|psalm)\s+\d+\.?\s*$", re.I)
_RE_BOOK_OF = re.compile(r"^\s*(?:the\s+)?book\s+of\s+(.+?)\s*$", re.I)


def _header_book(ln: str, maps):
    """Resolve a header line to a book: the bare name, a 'Book Chapter N' / 'Psalm N'
    chapter header, or a 'Book of X' title.  None if it isn't a recognisable header."""
    s = ln.strip()
    if not s or len(s.split()) > 8:
        return None
    b = _match(maps, s)
    if b:
        return b
    for rx in (_RE_CHAP_HEADER, _RE_BOOK_OF):
        m = rx.match(s)
        if m:
            b = _match(maps, m.group(1))
            if b:
                return b
    return None


def _looks_like_header(ln: str) -> bool:
    """True for a chapter/book header SHAPE regardless of whether its book resolves —
    so an unrecognised header (a Vulgate name with no map) DROPS the book context
    instead of silently attributing its verses to the previous book."""
    s = ln.strip()
    return bool(_RE_CHAP_HEADER.match(s) or _RE_BOOK_OF.match(s))

# legal: "§ 106", "Sec. 501(a)(1)", "Section 230", "17 U.S.C. § 106", "Article III"
_RE_SECTION = re.compile(r"(?:§+|\bSec(?:tion)?s?\.?)\s*(\d+[A-Za-z]?(?:[.\-]\d+)*)((?:\([0-9a-zA-Z]+\))*)")
_RE_USC = re.compile(r"\b(\d+)\s+([A-Z][A-Za-z.]{1,12})\s*§+\s*(\d+[A-Za-z]?(?:[.\-]\d+)*)((?:\([0-9a-zA-Z]+\))*)")
_RE_ARTICLE = re.compile(r"\bArticle\s+([IVXLC]+|\d+)\b")
# the PROSE cross-reference form: "section 106 of title 17", "§ 230(c)(1) of title 47"
# — a title-qualified citation that resolves to usc:<title>/<section><subpath>.
_RE_SECTION_OF_TITLE = re.compile(
    r"(?:§+|\bSec(?:tion)?s?\.?)\s*(\d+[A-Za-z]?(?:[.\-]\d+)*)((?:\([0-9a-zA-Z]+\))*)"
    r"\s+of\s+title\s+(\d+)", re.I)


def _sample_lines(text: str, cap: int = 6000) -> list[str]:
    lines = [ln.rstrip("\r") for ln in text.split("\n")]
    return lines if len(lines) <= cap else (lines[:cap // 2] + lines[-cap // 2:])


def analyze(text: str, *, kind_hint: str | None = None, maps: ReferenceMaps | None = None) -> dict:
    """Read a document and PROPOSE a structure profile (no ingest, no writes).
    Returns {kind, confidence, scheme, unit, books, book_order, citation_style,
    warnings, samples, stats} — a human confirms/corrects it before ingest.  `maps`
    (loaded ReferenceMaps) lets multilingual/variant book names resolve during
    analysis so they aren't flagged as 'unrecognised'."""
    lines = _sample_lines(text)
    nonblank = [ln for ln in lines if ln.strip()]
    n = max(1, len(nonblank))

    # ── scripture signals ────────────────────────────────────────────────────
    inline = cv_lines = 0
    found: dict[str, int] = {}
    unknown_books: dict[str, int] = {}
    for ln in nonblank:
        m = _RE_LINE_BOOKCV.match(ln)
        if m:
            b = _match(maps, m.group(1))
            if b:
                inline += 1
                found[b[1]] = found.get(b[1], 0) + 1
            else:
                unknown_books[m.group(1).strip()] = unknown_books.get(m.group(1).strip(), 0) + 1
            continue
        if _RE_LINE_CV.match(ln):
            cv_lines += 1
            continue
        hb = _header_book(ln, maps)                      # "Genesis Chapter 1" / "Book of Exodus"
        if hb:
            found.setdefault(hb[1], 0)
    # book-header + "C:V lines" layout: bare C:V lines dominate AND some book headers seen
    header_books = [b for b, c in found.items() if c == 0]
    # a 'Book C:V' shape with an UNRECOGNISED book still signals scripture structure
    # (odd abbreviations / a non-standard canon) — surface it, don't misclassify.
    unknown_cv = sum(unknown_books.values())
    # A run of bare 'C:V text' lines is scripture even when no book resolves (foreign /
    # Vulgate book names we don't know yet) — a lone "16:9" is noise, but a sequence of
    # verse-numbered lines is unmistakable.  Require a real run so a stray ratio/time
    # doesn't tip an ordinary document.
    cv_signal = cv_lines if (found or unknown_books or cv_lines >= 3) else 0
    scripture_hits = inline + unknown_cv + cv_signal

    # ── legal signals ────────────────────────────────────────────────────────
    section_hits = sum(1 for ln in nonblank if _RE_SECTION.search(ln))
    usc_hits = sum(1 for ln in nonblank if _RE_USC.search(ln))
    article_hits = sum(1 for ln in nonblank if _RE_ARTICLE.search(ln))
    legal_hits = section_hits + usc_hits + article_hits

    scr_score = scripture_hits / n
    leg_score = legal_hits / n
    kind = kind_hint or ("scripture" if scr_score >= leg_score and scripture_hits else
                         "legal" if legal_hits else "unknown")

    prof: dict = {"kind": kind, "warnings": [], "samples": [], "stats": {
        "lines": len(nonblank), "scripture_unit_lines": scripture_hits,
        "legal_markers": legal_hits}}

    if kind == "scripture":
        scheme = "book-chapter-verse-inline" if inline >= cv_lines else "chapter-verse-lines"
        prof["scheme"] = scheme
        prof["unit"] = "verse"
        order = sorted(found.keys(), key=lambda b: match_book(b)[0])
        prof["books"] = [{"canonical": b, "order": match_book(b)[0], "verses_seen": found[b]}
                         for b in order]
        prof["book_order"] = order
        prof["citation_style"] = {"form": "Book C:V", "chapter_verse_sep": ":",
                                  "cross_ref": "Book C:V / bare C:V within a book"}
        prof["confidence"] = round(min(1.0, scr_score * 1.6), 2)
        if unknown_books:
            top = sorted(unknown_books.items(), key=lambda kv: -kv[1])[:8]
            prof["warnings"].append("unrecognised book name(s) — map or confirm: "
                                    + ", ".join(f"{k}×{v}" for k, v in top))
        if scheme == "chapter-verse-lines" and not header_books:
            prof["warnings"].append("bare 'C:V' lines with no detected book headers — "
                                    "the book per unit is ambiguous; confirm the book or its header pattern")
        low = [b["canonical"] for b in prof["books"] if b["verses_seen"] and b["verses_seen"] < 3]
        deutero = [b for b in found if b in _DEUTERO]
        if deutero:
            prof["warnings"].append("deuterocanonical book(s) present (" + ", ".join(sorted(deutero)[:6])
                                    + ") — Catholic/Orthodox canon; confirm the versification "
                                    "(e.g. Vulgate Psalm numbering can differ by one for much of the Psalter)")
        prof["work"] = {"scheme": "bible"}
        for ln in nonblank:
            m = _RE_LINE_BOOKCV.match(ln)
            b = _match(maps, m.group(1)) if m else None
            if b:
                r = scripture_ref(b[1], m.group(2), m.group(3))
                r = maps.apply(r) if maps else r
                prof["samples"].append({"key": r.key, "citation": r.display, "text": m.group(4)[:160]})
            if len(prof["samples"]) >= 5:
                break

    elif kind == "legal":
        prof["scheme"] = "section-hierarchy"
        prof["unit"] = "section"
        # the WORK context: which title/code this file IS, so a local "§ M" gets a
        # canonical key (usc:<title>/M) that a citation elsewhere can marry up to.
        tm = re.search(r"\bTitle\s+(\d+)\b", text) or re.search(r"\b(\d+)\s+U\.?\s?S\.?\s?C\.?\b", text)
        work_title = tm.group(1) if tm else ""
        prof["work"] = {"scheme": "usc", "title": work_title}
        if not work_title:
            prof["warnings"].append("could not detect which title/code this file is — a bare "
                                    "'§ M' can't get a cross-document canonical key until you "
                                    "confirm the work (e.g. Title 17, U.S.C.)")
        prof["citation_style"] = {
            "section_marker": "§" if any("§" in ln for ln in nonblank) else "Section/Sec.",
            "subsection": "(a)(1)(A)-style" if any(_RE_SECTION.search(ln) and _RE_SECTION.search(ln).group(2)
                                                   for ln in nonblank) else "none seen",
            "full_citation": "N U.S.C. § M" if usc_hits else "§ M / Section M",
            "articles": bool(article_hits)}
        prof["confidence"] = round(min(1.0, leg_score * 1.6), 2)
        if usc_hits and section_hits < usc_hits:
            prof["warnings"].append("mostly full citations (N U.S.C. § M) and few local "
                                    "'§ M' headers — confirm whether this file is the CODE "
                                    "itself or a document that CITES it")
        for ln in nonblank:
            m = _RE_SECTION.search(ln)
            if m:
                r = legal_ref(work_title, m.group(1), m.group(2))
                prof["samples"].append({"key": r.key, "citation": r.display, "text": ln.strip()[:160]})
            if len(prof["samples"]) >= 5:
                break
    else:
        prof["scheme"] = "unknown"
        prof["unit"] = "block"
        prof["confidence"] = 0.0
        prof["warnings"].append("no scripture (Book C:V) or legal (§/Section) structure "
                                "detected — this may be prose/commentary; ingest normally, "
                                "or specify --kind and a citation pattern")
    return prof


def parse_units(text: str, profile: dict, *, maps: ReferenceMaps | None = None):
    """Yield (Ref, unit_text) for each addressable unit under a CONFIRMED profile —
    the Ref carries the canonical key (node identity) + display form.  Verse-per-line
    for scripture; section-delimited blocks for legal.  `maps` resolves multilingual
    book names and applies key aliases so units land on the canonical frame."""
    def _fix(r):
        return maps.apply(r) if maps else r
    kind = profile.get("kind")
    if kind == "scripture":
        cur_book = (profile.get("book_order") or [None])[0]
        for raw in text.split("\n"):
            ln = raw.rstrip("\r")
            if not ln.strip():
                continue
            m = _RE_LINE_BOOKCV.match(ln)
            b = _match(maps, m.group(1)) if m else None
            if b:
                cur_book = b[1]
                yield _fix(scripture_ref(cur_book, m.group(2), m.group(3))), m.group(4).strip()
                continue
            m = _RE_LINE_CV.match(ln)
            if m and cur_book:
                yield _fix(scripture_ref(cur_book, m.group(1), m.group(2))), m.group(3).strip()
                continue
            hb = _header_book(ln, maps)                  # "Genesis Chapter 1" / "Book of Exodus"
            if hb:
                cur_book = hb[1]                         # switch the book context
            elif _looks_like_header(ln):
                cur_book = None                          # unresolved book header — drop, don't mislabel
    elif kind == "legal":
        title = (profile.get("work") or {}).get("title", "")
        cur, buf = None, []
        for raw in text.split("\n"):
            ln = raw.rstrip("\r")
            m = _RE_SECTION.match(ln.strip())
            if m:
                if cur is not None:
                    yield cur, "\n".join(buf).strip()
                cur = _fix(legal_ref(title, m.group(1), m.group(2)))
                buf = [ln.strip()]
            elif cur is not None:
                buf.append(ln)
        if cur is not None:
            yield cur, "\n".join(buf).strip()


def parse_citations(text: str, profile: dict, *, book: str | None = None,
                    maps: ReferenceMaps | None = None) -> list[Ref]:
    """Extract the cross-references a unit's text makes, each normalised to a
    canonical Ref — the deterministic edge source for the graph.  `book` gives the
    current book so a bare 'C:V' resolves within it; `maps` resolves multilingual
    book names + key aliases.  De-duplicated by canonical key AFTER aliasing, so two
    spellings (or two versification systems) of the same reference collapse to one."""
    def _fix(r):
        return maps.apply(r) if maps else r
    out: list[Ref] = []
    if profile.get("kind") == "scripture":
        for m in _RE_REF_BOOK.finditer(text):
            b = _match(maps, m.group(1))
            if b:
                out.append(_fix(scripture_ref(b[1], m.group(2), m.group(3), m.group(4))))
        if book:                                          # bare C:V within this book
            # only a RESOLVED book-ref shadows a bare 'C:V' — an unrecognised token
            # (e.g. "and 9:15") must not swallow the bare reference that follows it.
            spans = [m.span() for m in _RE_REF_BOOK.finditer(text) if _match(maps, m.group(1))]
            for m in _RE_BARE_CV.finditer(text):
                if not any(a <= m.start() < b2 for a, b2 in spans):
                    out.append(_fix(scripture_ref(book, m.group(1), m.group(2))))
    elif profile.get("kind") == "legal":
        title = (profile.get("work") or {}).get("title", "")
        for m in _RE_USC.finditer(text):
            out.append(_fix(legal_ref(m.group(1), m.group(3), m.group(4))))   # explicit N U.S.C. § M
        # "section M of title N" prose form — the title comes from the phrase, not the doc
        covered = [m.span() for m in _RE_USC.finditer(text)]
        for m in _RE_SECTION_OF_TITLE.finditer(text):
            out.append(_fix(legal_ref(m.group(3), m.group(1), m.group(2))))
            covered.append(m.span())
        # remaining local "§ M" / "section M" resolve against the document's own title
        for m in _RE_SECTION.finditer(text):
            if not any(a <= m.start() < b2 for a, b2 in covered):
                out.append(_fix(legal_ref(title, m.group(1), m.group(2))))
    seen, uniq = set(), []
    for r in out:
        if r.key not in seen:
            seen.add(r.key); uniq.append(r)
    return uniq


# ── interactive confirm: profile → plain-language questions → confirmed profile ──
# A structured/ambiguous document is never ingested on a guess.  analyze() PROPOSES;
# questions_for() turns that proposal into a short list of plain-language questions a
# human answers; apply_answers() folds the replies into a CONFIRMED profile (kind,
# how to ingest, book/title context, and any ad-hoc book / versification aliases).
# Plain prose asks nothing — should_confirm() is False — so the ordinary bulk-ingest
# workflow is undisturbed; only real canonical structure interrupts.

def _scheme_human(profile: dict) -> str:
    return {"book-chapter-verse-inline": "each line is 'Book chapter:verse …'",
            "chapter-verse-lines": "a book header then bare 'chapter:verse' lines",
            "section-hierarchy": "numbered sections (§ / Section)",
            }.get(profile.get("scheme"), profile.get("scheme") or "an unfamiliar layout")


def should_confirm(profile: dict) -> bool:
    """True when a document carries enough canonical structure to be worth confirming
    before ingest — so plain prose (or a stray '§' in an article) never interrupts the
    normal workflow.  Gated on kind + confidence + a floor of real unit markers."""
    if profile.get("kind") not in ("scripture", "legal"):
        return False
    st = profile.get("stats") or {}
    strong = st.get("scripture_unit_lines", 0) >= 3 or st.get("legal_markers", 0) >= 3
    return bool(profile.get("confidence", 0) >= 0.2 or strong)


def _unknown_book_tokens(profile: dict) -> list[str]:
    """The unrecognised book names analyze() surfaced (parsed back out of its warning)."""
    out: list[str] = []
    for w in profile.get("warnings", []):
        if w.startswith("unrecognised book") and ":" in w:
            for tok in re.findall(r"([^,]+?)\s*[×x]\s*\d+", w.split(":", 1)[1]):
                name = tok.strip()
                if name and name not in out:
                    out.append(name)
    return out


def questions_for(profile: dict) -> list[dict]:
    """Plain-language questions a human answers before a structured ingest.  Each is
    {id, prompt, type: choice|text, options?, default, detail?}; returns [] when there
    is nothing worth asking (should_confirm is False).  Feed the replies (id → value)
    to apply_answers()."""
    if not should_confirm(profile):
        return []
    kind = profile.get("kind")
    qs: list[dict] = []
    if kind == "scripture":
        books = ", ".join(b["canonical"] for b in profile.get("books", [])[:8]) or "—"
        qs.append({
            "id": "kind", "type": "choice", "default": "structured",
            "prompt": "This looks like scripture — " + _scheme_human(profile)
                      + ". How should I ingest it?",
            "detail": "Books detected: " + books,
            "options": [
                {"value": "structured", "label": "Structured scripture — one node per verse, with cross-reference links"},
                {"value": "plain", "label": "Ordinary prose — just chunk it normally"}]})
        if profile.get("scheme") == "chapter-verse-lines" and not profile.get("book_order"):
            qs.append({
                "id": "book", "type": "text", "default": "",
                "prompt": "The file uses bare 'chapter:verse' lines with no book header — "
                          "which book is this file? (e.g. 'John')",
                "detail": "Leave blank if the text already heads several books itself."})
        apoc = any("deuterocanonical" in w for w in profile.get("warnings", []))
        qs.append({
            "id": "canon", "type": "choice", "default": "as_printed",
            "prompt": ("Deuterocanonical/apocryphal books are present, so canon and verse "
                       "numbering vary by tradition — " if apoc else "")
                      + "which verse numbering should I treat as canonical?",
            "options": [
                {"value": "as_printed", "label": "As printed — use the chapter:verse numbers in this file"},
                {"value": "diverges", "label": "This edition's numbering diverges — I'll supply a mapping file"}]})
        for name in _unknown_book_tokens(profile):
            qs.append({
                "id": "book:" + name, "type": "text", "default": "",
                "prompt": f"I found references to '{name}', which isn't one of the standard "
                          "66 books. What is it?",
                "detail": "Leave blank to ignore it; type a standard book name (e.g. "
                          "'Revelation') to treat it AS that book; type 'keep' to keep it "
                          "as its own extra-canonical work."})
    elif kind == "legal":
        qs.append({
            "id": "kind", "type": "choice", "default": "structured",
            "prompt": "This looks like legal text — " + _scheme_human(profile)
                      + ". How should I ingest it?",
            "options": [
                {"value": "structured", "label": "Structured law — one node per section, with citation links"},
                {"value": "plain", "label": "Ordinary prose — just chunk it normally"}]})
        title = (profile.get("work") or {}).get("title", "")
        qs.append({
            "id": "work_title", "type": "text", "default": title,
            "prompt": (f"Which title/code is this document? I detected Title {title}, U.S.C."
                       if title else
                       "Which title/code is this document? (e.g. '17' for Title 17, U.S.C.)"),
            "detail": "Needed so a local '§ 106' gets a cross-document key (usc:17/106). "
                      + ("Correct it, or clear it if it isn't a single title."
                         if title else "Leave blank if it isn't a single title.")})
        if any("cites it" in w.lower() for w in profile.get("warnings", [])):
            qs.append({
                "id": "role", "type": "choice", "default": "code",
                "prompt": "Is this file the law itself, or a document that cites the law?",
                "options": [
                    {"value": "code", "label": "The code/statute itself — its sections are the units"},
                    {"value": "commentary", "label": "A commentary that cites the law — attach it to the sections it cites"}]})
    return qs


def apply_answers(profile: dict, answers: dict) -> dict:
    """Fold confirmation answers (id → value) into a CONFIRMED profile.  Returns the
    profile with `confirmed=True`, an `ingest_as` ('structured'|'plain'), any corrected
    kind / book_order / work / role, and a `reference_map` ({book_aliases, key_aliases})
    assembled from the ad-hoc answers (other book names, versification choice) — the
    same shape load_reference_maps() produces, so it merges with a domain pack's maps."""
    p = dict(profile)
    ans = {k: (v.strip() if isinstance(v, str) else v) for k, v in (answers or {}).items()}
    p["confirmed"] = True

    if ans.get("kind") == "plain":
        p["ingest_as"] = "plain"
        p["reference_map"] = {"book_aliases": {}, "key_aliases": {}}
        return p
    p["ingest_as"] = "structured"

    bk = ans.get("book")
    if bk:
        b = match_book(str(bk))
        if b:
            p["book_order"] = [b[1]]

    if "work_title" in ans:
        digits = re.search(r"\d+", str(ans["work_title"] or ""))
        w = dict(p.get("work") or {})
        w["title"] = digits.group(0) if digits else ""
        w.setdefault("scheme", "usc")
        p["work"] = w

    if ans.get("role") == "commentary":
        p["role"] = "commentary"
    if ans.get("canon") == "diverges":
        p["versification"] = "diverges"          # user will attach a key-alias map file

    book_aliases: dict[str, list] = {}
    extra_books: list[str] = []
    for k, v in ans.items():
        if not k.startswith("book:"):
            continue
        token, v = k[5:], str(v or "").strip()
        if not v or v.lower() == "ignore":
            continue
        if v.lower() == "keep":
            extra_books.append(token)
            continue
        b = match_book(v)
        if b:
            book_aliases.setdefault(b[1], []).append(token)
    if extra_books:
        p["extra_books"] = extra_books
    p["reference_map"] = {"book_aliases": book_aliases, "key_aliases": {}}
    return p


def profile_signature(profile: dict) -> str:
    """A stable key for 'ask once per profile' batching in a bulk crawl: documents
    that would raise the SAME confirmation share a signature (a whole Title-17 folder
    → one key; a Bible and a legal code → two).  Plain docs collapse to 'plain'."""
    kind = profile.get("kind", "unknown")
    if kind == "legal":
        w = profile.get("work") or {}
        return f"legal:{w.get('scheme', 'usc')}:{w.get('title', '')}"
    if kind == "scripture":
        return f"scripture:{(profile.get('work') or {}).get('scheme', 'bible')}"
    return "plain"
