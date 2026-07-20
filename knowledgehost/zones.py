"""Deterministic chunk zoning — what KIND of text a chunk is.

The distiller uses this to skip low-value document furniture (references,
contents pages, indexes, copyright boilerplate) and to adapt its lens to
code-dominant passages.  `classify(section, text)` is a pure function computed
at distill time — never stored, so there is no schema, no chunk-id churn, and
an improved heuristic upgrades the whole corpus on the next pass.

Signals are STRUCTURAL (line shapes: citation density, dot leaders, page-number
lists, symbol density), never topical — genealogies, rosters and narrative
prose ("the wives of Henry VIII were …") always stay `body`.  When unsure,
`body` wins: a mis-skipped body chunk loses knowledge, a mis-kept furniture
chunk only costs one LM call.
"""
from __future__ import annotations

import re

ZONES = ("body", "references", "toc", "index", "boilerplate", "code")

# Heading kill-lists — matched against the LAST segment of the heading path,
# lowercased, punctuation/numbering stripped.  High precision: documents use
# these titles literally.
_H_REFERENCES = frozenset((
    "references", "bibliography", "works cited", "literature cited",
    "citations", "notes and references", "further reading", "reference list"))
_H_TOC = frozenset(("contents", "table of contents", "list of contents"))
_H_INDEX = frozenset(("index", "subject index", "author index", "name index"))
_H_BOILER = frozenset((
    "copyright", "imprint", "colophon", "acknowledgements", "acknowledgments",
    "about the author", "about the authors", "list of figures", "list of tables",
    "list of illustrations", "list of abbreviations", "dedication",
    "publisher's note", "publishers note"))

_YEAR = re.compile(r"\(?(?:19|20)\d{2}[a-z]?\)?")
_CITE_MARK = re.compile(
    r"\bet al\.|\bdoi\b|doi:|\bpp?\.\s*\d|\bvol\.\s*\d|\bno\.\s*\d|"
    r"^\s*\[\d{1,3}\]|^\s*\d{1,3}\.\s+[A-Z]", re.I | re.M)
_DOT_LEADER = re.compile(r"\.{3,}\s*\d{1,4}\s*$")
_TRAIL_PAGENO = re.compile(r"\s\d{1,4}\s*$")
_INDEX_LINE = re.compile(r"^\S.{0,60}?,\s*\d{1,4}([,–-]\s*\d{1,4})*\s*$")
_BOILER_MARK = re.compile(
    r"\bisbn\b|all rights reserved|no part of this (?:publication|book)|"
    r"printed in|©\s*(?:19|20)\d{2}|copyright ©", re.I)
_CODE_LINE = re.compile(
    r"[;{}]\s*$|^\s*[}{]|^\s*(?:def|class|import|from|return|if|elif|else|for|"
    r"while|function|const|let|var|public|private|static|void|int|fn|impl|pub|"
    r"package|#include|struct|enum|try|except|catch)\b|=>|->|:=|\|\||&&|</\w+>|"
    r"^\s*[\w.]+\([^)]*\)\s*[;{]?\s*$|^\s*(?:\$|>>>|#\s*!)")
_PROSE_END = re.compile(r"[.!?][\"')\]]?\s*$")


def _last_heading(section: str) -> str:
    seg = (section or "").split(">")[-1]
    seg = re.sub(r"^[\s\d.:—-]+", "", seg)           # "3.2 References" -> "References"
    return re.sub(r"[^a-z' ]", "", seg.strip().lower()).strip()


def classify(section: str, text: str) -> str:
    """The zone for one chunk.  Heading wins (high precision); line-shape
    statistics catch the same zones in outline-less PDFs; `body` on doubt."""
    head = _last_heading(section)
    if head in _H_REFERENCES:
        return "references"
    if head in _H_TOC:
        return "toc"
    if head in _H_INDEX:
        return "index"
    if head in _H_BOILER:
        return "boilerplate"

    lines = [ln for ln in (text or "").splitlines() if ln.strip()]
    n = len(lines)
    if n < 5:
        # Too little structure to judge from shape alone — except unmistakable
        # copyright-page markers, which cluster on short front-matter chunks.
        if text and len(_BOILER_MARK.findall(text)) >= 2:
            return "boilerplate"
        return "body"

    # references: most lines carry a year AND a citation marker
    cite = sum(1 for ln in lines
               if _YEAR.search(ln) and (_CITE_MARK.search(ln) or ln.count(",") >= 2))
    if cite / n >= 0.4 and n >= 6:
        return "references"

    # toc: dot leaders, or short lines that mostly end in a page number
    leaders = sum(1 for ln in lines if _DOT_LEADER.search(ln))
    if leaders / n >= 0.3:
        return "toc"
    trail = sum(1 for ln in lines if _TRAIL_PAGENO.search(ln))
    if n >= 8 and trail / n >= 0.6 and sum(len(ln) for ln in lines) / n < 60:
        return "toc"

    # index: "Term, 12, 34-36" lines dominate
    if n >= 8 and sum(1 for ln in lines if _INDEX_LINE.match(ln)) / n >= 0.5:
        return "index"

    # boilerplate: multiple distinct legal/imprint markers in a short chunk
    if len(_BOILER_MARK.findall(text)) >= 2 and len(text) < 2000:
        return "boilerplate"

    # code: code-shaped lines dominate and prose punctuation does not
    codey = sum(1 for ln in lines if _CODE_LINE.search(ln))
    prose = sum(1 for ln in lines if _PROSE_END.search(ln))
    if codey / n >= 0.4 and codey > prose:
        return "code"

    return "body"
