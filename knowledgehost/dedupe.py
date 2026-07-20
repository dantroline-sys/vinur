"""Duplicate chunk detection — the janitor, no LM involved.

A chunk id is ``sha1(path + section + text)``, which makes re-ingesting the SAME
document idempotent but says nothing about the same text arriving by a different
route: a research drop re-exported under a new name, one PDF filed in two
folders, an article that appears in a collection and again on its own.  Those
distil a second time, mint a second set of concepts, and cost LM hours to do it.

Two tiers, both deterministic:

  EXACT      sha1 of the normalised text (case-folded, whitespace-collapsed).
             Cheap enough to run inside the distill loop, so a duplicate is
             never handed to an LM in the first place.

  NEAR       MinHash + LSH over word 5-grams, verified with real Jaccard.
             This is the one that catches an assistant answering the same
             research question twice in slightly different words.  It runs as a
             sweep (`dedupe` op) rather than inline, and marks nothing unless
             asked, because "almost the same" is a judgement call about content
             — a revised document is near-identical to the one it supersedes.

Nothing here deletes text.  A duplicate keeps its row (search still finds it by
either path) and is simply excluded from distillation, with the mapping recorded
so provenance can say where else the text lives.
"""
from __future__ import annotations

import hashlib
import re

_WS = re.compile(r"\s+")
_ZERO_WIDTH = re.compile(r"[​-‏﻿]")
# Layout-only differences that must not make two copies look distinct: quote
# style, dash width, and the soft hyphen a PDF extractor leaves mid-word.
_PUNCT_FOLD = str.maketrans({
    "‘": "'", "’": "'", "“": '"', "”": '"',
    "–": "-", "—": "-", "­": "", " ": " ",
})


def norm(text: str) -> str:
    """The comparable form of a chunk's text."""
    t = (text or "").translate(_PUNCT_FOLD)
    t = _ZERO_WIDTH.sub("", t)
    return _WS.sub(" ", t).strip().casefold()


def text_hash(text: str) -> str:
    return hashlib.sha1(norm(text).encode("utf-8", "replace")).hexdigest()[:20]


# ── near-duplicate: MinHash over word shingles, banded for candidate lookup ──

SHINGLE = 5             # words per shingle
PERMS = 32              # signature length
BANDS = 8               # PERMS must divide evenly by BANDS
_MASK = (1 << 61) - 1   # a Mersenne prime keeps the mixing cheap and stable


def shingles(text: str, k: int = SHINGLE) -> set:
    """Hashed word k-grams.  A chunk shorter than k words shingles to its words,
    so short chunks still compare rather than silently matching everything."""
    words = norm(text).split()
    if not words:
        return set()
    if len(words) <= k:
        return {hash_str(" ".join(words))}
    return {hash_str(" ".join(words[i:i + k])) for i in range(len(words) - k + 1)}


def hash_str(s: str) -> int:
    return int.from_bytes(hashlib.blake2b(s.encode("utf-8", "replace"),
                                          digest_size=8).digest(), "big")


def _perm(i: int, x: int) -> int:
    """Permutation i of a shingle hash — the standard two-parameter family, with
    fixed odd constants so signatures are reproducible across runs and machines."""
    a = (i * 0x9E3779B97F4A7C15) | 1
    b = (i * 0xC2B2AE3D27D4EB4F) | 1
    return ((a * x + b) & _MASK)


def signature(sh: set, perms: int = PERMS) -> tuple:
    if not sh:
        return ()
    return tuple(min(_perm(i, x) for x in sh) for i in range(perms))


def bands(sig: tuple, n_bands: int = BANDS) -> list:
    """Band keys: two texts sharing ANY band key are worth comparing properly."""
    if not sig:
        return []
    rows = max(1, len(sig) // n_bands)
    out = []
    for b in range(n_bands):
        part = sig[b * rows:(b + 1) * rows]
        if part:
            out.append((b, hash_str(",".join(str(v) for v in part))))
    return out


def jaccard(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    return inter / float(len(a) + len(b) - inter)


def near_pairs(items, threshold: float = 0.9, k: int = SHINGLE):
    """items: iterable of (id, text).  Yields (id_a, id_b, similarity) for pairs
    at or above `threshold`, a is the FIRST seen (the one worth keeping).

    LSH first, so this is linear-ish in corpus size rather than quadratic; the
    exact Jaccard is then computed only for banded candidates."""
    buckets: dict = {}
    shing: dict = {}
    order: list = []
    for cid, text in items:
        sh = shingles(text, k)
        if not sh:
            continue
        shing[cid] = sh
        order.append(cid)
        sig = signature(sh)
        seen_here = set()
        for key in bands(sig):
            for other in buckets.get(key, ()):      # earlier chunks only
                if other in seen_here or other == cid:
                    continue
                seen_here.add(other)
                sim = jaccard(shing[other], sh)
                if sim >= threshold:
                    yield (other, cid, round(sim, 4))
            buckets.setdefault(key, []).append(cid)
