"""Parse Vinkona's research drops (research_loop_spec §2).

A solved doc is a single markdown file: YAML-ish front-matter + headed sections.
The host reads three things out of it:

  * the ``# Question`` — the frame distillation is conditioned on, so "how do I X"
    + source prose yields a *card answering X*, not a generic concept (§6.2);
  * the ``## Sources`` blocks — the untrusted evidence that actually gets chunked
    and distilled (the ``## Answer`` is Vinkona's synthesis, kept only as a fallback
    when a doc carries no sources);
  * front-matter ``kb_query`` — the verbatim query that opened the gap, so a card
    grounding it can close that ``knowledge_gap`` (§6.2);
  * optional card HINTS — ``card_type`` (requirements | decision | playbook | case |
    procedure) and ``context_features`` (a one-line JSON object of {feature: value}
    pairs saying WHEN the answer applies).  Vinkona writes these when her research
    concluded in an actionable shape; the distiller uses them to run the matching
    typed-card extractor and to seed the card's discriminators.  Hints are a nudge,
    never authority — extraction stays grounded only in the drop's text.

Stdlib only (no PyYAML): we hand-parse the handful of scalar keys we need.
Everything here is treated as untrusted DATA — the host sanitises on ingest and
the filename is never interpolated anywhere.
"""
from __future__ import annotations

import hashlib
import json
import os
import re

_FM = re.compile(r"^﻿?---[ \t]*\r?\n(.*?)\r?\n---[ \t]*\r?\n", re.DOTALL)
_SCALAR = re.compile(r"^([A-Za-z0-9_]+):[ \t]*(.*?)[ \t]*$")


def _unquote(v: str) -> str:
    v = v.strip()
    if len(v) >= 2 and v[0] == v[-1] and v[0] in "\"'":
        return v[1:-1]
    return v


def parse_front_matter(text: str) -> tuple[dict, str]:
    """Return (top-level scalar front-matter, body-after-front-matter).  Nested list
    keys (like ``sources:``) are skipped — the host doesn't need them."""
    m = _FM.match(text)
    if not m:
        return {}, text
    fm: dict = {}
    for line in m.group(1).splitlines():
        if not line.strip() or line.lstrip() != line:   # skip nested/indented (list items)
            continue
        sm = _SCALAR.match(line)
        if sm and sm.group(2) != "":                     # "key:" with no scalar → a block/list
            fm[sm.group(1).lower()] = _unquote(sm.group(2))
    return fm, text[m.end():]


def is_research_doc(path: str) -> bool:
    """Cheap check: a markdown file whose front-matter declares provenance: vinkona."""
    if not path.lower().endswith((".md", ".markdown")):
        return False
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            head = fh.read(2048)
    except OSError:
        return False
    fm, _ = parse_front_matter(head if "\n---" in head else head + "\n---\n")
    return (fm.get("provenance") or "").strip().lower() == "vinkona"


# The exporter names files <sha1[:16]>.md (research_export.question_hash) — the
# HTTP lane accepts exactly that shape, so a request can never traverse out of
# the drops folder or claim an arbitrary filename.
_DROP_NAME = re.compile(r"^[0-9a-f]{16}\.md$")


def drop_inventory(cfg: dict) -> dict:
    """GET /drop — the exporter's handshake.  Says whether this host can store
    drops at all (``accepts``) and what it already holds: ``drops`` maps each
    solved/*.md name to sha256(content)[:16], so a remote Vinkona can skip
    byte-identical drops without shipping their bytes.  (The hash recipe is a
    cross-repo contract with vinkona's research_export._hash16 — change both
    or neither.)"""
    ddir = cfg.get("research_solved_dir")
    if not ddir:
        return {"ok": True, "accepts": False,
                "reason": "research_solved_dir is not set on this host — set it "
                          "in the panel under Settings › Paths (the folder is "
                          "created on save), then drops flow on the next export"}
    drops = {}
    try:
        for fn in sorted(os.listdir(ddir)):
            if not _DROP_NAME.match(fn):
                continue
            try:
                with open(os.path.join(ddir, fn), "rb") as fh:
                    drops[fn] = hashlib.sha256(fh.read()).hexdigest()[:16]
            except OSError:
                continue
    except OSError:
        pass                        # outbox not created yet — accepts, holds nothing
    return {"ok": True, "accepts": True, "count": len(drops), "drops": drops}


def write_drop(cfg: dict, name, content) -> dict:
    """HTTP lane of the research hand-off (POST /drop): validate one solved-drop
    and write it atomically into ``research_solved_dir``, where the crawl mines
    it exactly as if Vinkona had written the file over a shared filesystem.
    Byte-identical re-posts are a no-op (``changed: false``) — the exporter can
    re-send its whole outbox safely."""
    ddir = cfg.get("research_solved_dir")
    if not ddir:
        raise ValueError("research_solved_dir is not set on this host — set it "
                         "in the panel under Settings › Paths")
    if not isinstance(name, str) or not _DROP_NAME.match(name):
        raise ValueError("drop name must be '<16 hex>.md'")
    if not isinstance(content, str) or not content.strip():
        raise ValueError("drop content must be a non-empty string")
    head = content[:2048]
    fm, _ = parse_front_matter(head if "\n---" in head else head + "\n---\n")
    if (fm.get("provenance") or "").strip().lower() != "vinkona":
        raise ValueError("not a research drop (front-matter 'provenance: vinkona' required)")
    os.makedirs(ddir, exist_ok=True)
    path = os.path.join(ddir, name)
    try:
        with open(path, "r", encoding="utf-8") as fh:
            if fh.read() == content:
                return {"ok": True, "changed": False}
    except OSError:
        pass
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        fh.write(content)
    os.replace(tmp, path)
    return {"ok": True, "changed": True}


def _sections(body: str) -> list[tuple[int, str, str]]:
    """Split markdown into (level, heading, text) by ATX headings, in order."""
    out, level, head, buf = [], 0, "", []
    for line in body.splitlines():
        hm = re.match(r"^(#{1,6})[ \t]+(.*?)[ \t]*#*[ \t]*$", line)
        if hm:
            if head or buf:
                out.append((level, head, "\n".join(buf).strip()))
            level, head, buf = len(hm.group(1)), hm.group(2).strip(), []
        else:
            buf.append(line)
    if head or buf:
        out.append((level, head, "\n".join(buf).strip()))
    return out


def parse_research_doc(path: str) -> tuple[str, list, dict]:
    """Parse a solved doc → (question, blocks, meta).

    * ``question``  the ``# Question`` text (the distillation frame / chunk title).
    * ``blocks``    ``[(section, text)]`` for ``chunk_blocks`` — one per ``### source``
                    subsection under ``## Sources``; if the doc has no sources, a single
                    ``## Answer`` block so a card can still be distilled.
    * ``meta``      ``{provenance, kind, kb_query, question, answer, trust}``.
    """
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        raw = fh.read()
    fm, body = parse_front_matter(raw)
    secs = _sections(body)

    question = ""
    answer = ""
    source_blocks: list = []
    i = 0
    while i < len(secs):
        level, head, text = secs[i]
        h = head.lower()
        if level == 1 and h.startswith("question"):
            question = text.strip()
        elif h.startswith("answer"):
            answer = text.strip()
        elif h.startswith("source"):
            # its own prose (if any) plus every deeper subsection until the next
            # same-or-higher heading = the individual sources.
            if text.strip():
                source_blocks.append((head, text.strip()))
            j = i + 1
            while j < len(secs) and secs[j][0] > level:
                slvl, shead, stext = secs[j]
                if stext.strip():
                    source_blocks.append((f"{head} › {shead}", stext.strip()))
                j += 1
            i = j
            continue
        i += 1

    card_type = (fm.get("card_type") or "").strip().lower() or None
    blocks = source_blocks
    if not blocks and answer:                            # no sources → distil the synthesis
        blocks = [("Answer", answer)]
    elif answer and card_type:
        # Hinted drop: the shaped answer IS the typed-card source (the sources are raw
        # evidence; the conclusion lives in the Answer).  Chunk it too, first — the
        # distiller runs the typed extractor on the Answer chunk exactly once.
        blocks = [("Answer", answer)] + blocks

    meta = {
        "provenance": (fm.get("provenance") or "vinkona").lower(),
        "kind": (fm.get("kind") or "").lower() or None,
        "kb_query": fm.get("kb_query") or None,
        "question": question or None,
        "answer": answer or None,
        "trust": fm.get("trust") or None,               # 'low' | number | None (label only)
        "card_type": card_type,
        "context_features": _parse_features(fm.get("context_features")),
    }
    return question, blocks, meta


def _parse_features(raw) -> dict | None:
    """The ``context_features`` hint: a one-line JSON object of {feature: value}
    strings.  Anything else (absent, malformed, wrong shapes) → None — a bad hint
    must never break ingestion of the drop itself."""
    if not raw or not str(raw).strip():
        return None
    try:
        obj = json.loads(str(raw))
    except ValueError:
        return None
    if not isinstance(obj, dict):
        return None
    out = {str(k).strip().lower(): str(v).strip()
           for k, v in obj.items()
           if str(k).strip() and isinstance(v, (str, int, float)) and str(v).strip()}
    return dict(list(out.items())[:8]) or None
