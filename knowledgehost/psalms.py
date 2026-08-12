"""VINUR — Vulgate↔Hebrew Psalm reconciliation, recovered from the texts themselves.

The Psalms are numbered differently in the Latin Vulgate tradition (Douay-Rheims) and the
Hebrew/Masoretic tradition (KJV and most English Bibles).  The psalm NUMBERS shift —
Vulgate 9 is Hebrew 9+10 combined, Vulgate 10-112 are Hebrew 11-113, and so on — and,
worse, the VERSE numbers drift because the Douay counts the Latin historical titles as
verses (Vulgate Ps 50:3 'Have mercy on me' is Hebrew Ps 51:1).  The offset is therefore
per-psalm and edition-specific: it cannot be tabulated once and trusted, and a wrong guess
lands a Latin title on top of the KJV's first real verse — a precise-looking mis-alignment.

So we do NOT hard-code the verse offsets.  We recover them from the two editions actually
ingested.  The psalm-NUMBER correspondence (below) is fixed and authoritative; within each
corresponding psalm we find the constant verse shift that best lines the Douay wording up
with the Hebrew wording (they translate from Latin vs Hebrew, so the words differ but the
content overlaps).  A title verse simply fails to match anything and falls off the front of
the alignment — which is exactly how its offset is discovered.  Low-confidence psalms are
left unaligned and reported, never guessed.

The result is a set of `key_aliases` (bible:Ps.<vulgate> → bible:Ps.<hebrew>) written into
the Douay document's reference map, after which the citation graph and the parallel-reading
surface line the two editions up verse-for-verse on the Hebrew frame.  Deterministic, no LM.
"""
from __future__ import annotations

import json
import logging
import os
import re

log = logging.getLogger("knowledgehost.psalms")

# editions that number the Psalms in the Vulgate frame (their psalms get reconciled onto
# the Hebrew frame of a reference edition).  Extend as more Vulgate editions are added.
VULGATE_EDITIONS = frozenset({"douay-rheims"})

_PS_KEY = re.compile(r"^bible:Ps\.(\d+)\.(\d+)$")

# tiny stoplist — the connective words that carry no identifying content, so a Douay verse
# and its Hebrew counterpart are compared on the words that actually distinguish a verse.
_STOP = frozenset(
    "the a an and or of to in on for with by is are was were be he she it they them his her "
    "thy thou thee my me we us our you your that this which who whom what shall will not so as "
    "at from unto into hath have hast had do did with out up off down all any every there here "
    "but if then when than lest yea also o".split())


def hebrew_targets(vulg_psalm: int) -> list[int]:
    """The Hebrew psalm(s) a Vulgate psalm number corresponds to — the authoritative
    Septuagint/Vulgate ↔ Masoretic concordance (this part never varies by edition):

        Vulgate 1-8      = Hebrew 1-8
        Vulgate 9        = Hebrew 9 + 10        (one Vulgate psalm, two Hebrew)
        Vulgate 10-112   = Hebrew 11-113        (Hebrew = Vulgate + 1)
        Vulgate 113      = Hebrew 114 + 115
        Vulgate 114, 115 = Hebrew 116           (two Vulgate psalms, one Hebrew)
        Vulgate 116-145  = Hebrew 117-146       (Hebrew = Vulgate + 1)
        Vulgate 146, 147 = Hebrew 147
        Vulgate 148-150  = Hebrew 148-150

    Returns the ordered list of Hebrew psalm numbers whose verses, concatenated, form the
    target the Vulgate psalm aligns into."""
    p = vulg_psalm
    if p <= 8 or p >= 148:
        return [p]
    if p == 9:
        return [9, 10]
    if p <= 112:
        return [p + 1]
    if p == 113:
        return [114, 115]
    if p in (114, 115):
        return [116]
    if p <= 145:
        return [p + 1]
    return [147]                                          # 146, 147


def _tokens(text: str) -> frozenset:
    return frozenset(w for w in re.findall(r"[a-z]+", text.lower())
                     if len(w) > 2 and w not in _STOP)


def _sim(a: frozenset, b: frozenset) -> float:
    """Overlap coefficient of two token sets — |a∩b| / min(|a|,|b|).  Robust to the two
    translations padding a verse to different lengths; 0 when either side is empty."""
    if not a or not b:
        return 0.0
    return len(a & b) / min(len(a), len(b))


def _target_verses(reference: dict, vulg_psalm: int) -> list[tuple[int, int, frozenset]]:
    """The ordered (hebrew_psalm, hebrew_verse, tokens) list a Vulgate psalm aligns into."""
    out: list[tuple[int, int, frozenset]] = []
    for h in hebrew_targets(vulg_psalm):
        for v in sorted(reference.get(h, {})):
            out.append((h, v, _tokens(reference[h][v])))
    return out


def align_psalm(douay_verses: dict, reference: dict, vulg_psalm: int, *,
                sim_floor: float) -> dict:
    """Recover the single verse offset that best lines a Douay psalm up with its Hebrew
    target, by maximising total wording overlap.  A renumbering is a CONSTANT shift, so one
    offset maps the whole psalm; the offset is proven by the aggregate score, and the gap
    to the next-best offset (`margin`) is how sure we are.  Returns
    {offset, matched, total, best_score, margin, pairs:[(vulg_v, heb_psalm, heb_v, sim)]}
    where `pairs` are ALL in-range verses at the chosen offset (a numbered Latin title maps
    off the front of the target and is simply absent — that is how its offset is found)."""
    src = [(v, _tokens(douay_verses[v])) for v in sorted(douay_verses)]
    tgt = _target_verses(reference, vulg_psalm)
    if not src or not tgt:
        return {"offset": 0, "matched": 0, "total": len(src), "best_score": 0.0,
                "margin": 0.0, "pairs": []}
    scored: list[tuple[float, int, int]] = []            # (score, matched, offset)
    for off in range(-6, len(tgt)):                       # leading titles (−) and mid-psalm combines (+)
        score = 0.0
        matched = 0
        for i, (_v, tk) in enumerate(src):
            j = i + off
            if 0 <= j < len(tgt):
                s = _sim(tk, tgt[j][2])
                score += s
                if s >= sim_floor:
                    matched += 1
        scored.append((score, matched, off))
    scored.sort(key=lambda c: (round(c[0], 6), c[1], -abs(c[2])), reverse=True)
    best_score, matched, off = scored[0]
    second = scored[1][0] if len(scored) > 1 else 0.0
    pairs = []
    titles = []                                           # verses BEFORE the target start —
    for i, (v, _tk) in enumerate(src):                    # the psalm's numbered Latin title(s)
        j = i + off
        if j < 0:
            titles.append(v)
        elif j < len(tgt):
            hp, hv, _ = tgt[j]
            pairs.append((v, hp, hv, round(_sim(src[i][1], tgt[j][2]), 3)))
    return {"offset": off, "matched": matched, "total": len(src),
            "best_score": round(best_score, 3), "margin": round(best_score - second, 3),
            "pairs": pairs, "titles": titles}


def recover_aliases(douay: dict, reference: dict, *, sim_floor: float = 0.34,
                    margin_abs: float = 0.5, margin_rel: float = 1.5,
                    frac_ok: float = 0.6) -> tuple[dict, list]:
    """Recover the Vulgate→Hebrew verse-key aliases for every Douay psalm that diverges
    from its Hebrew counterpart, by wording alignment against the reference edition.

    `douay`, `reference`: {psalm:int -> {verse:int -> text}}.  Returns (aliases, report):
      * aliases — {"bible:Ps.<vulg>.<v>": "bible:Ps.<heb>.<v>"} for CONFIDENT psalms;
      * report  — one row per psalm that needed work: {psalm, targets, offset, matched,
                  total, best_score, margin, aliases, confident, reason}.

    A psalm's offset is CONFIDENT when the best-scoring offset clearly wins: the gap to the
    next best is ≥ margin_abs AND the offset is either well-supported across the psalm
    (matched/total ≥ frac_ok) or decisively higher-scoring (best ≥ margin_rel × second).
    That continuous-score test is robust to refrains and internal repetition, which fool a
    match-count test.  For a confident psalm the offset is applied to EVERY verse (the shift
    is constant), so alignment is complete, not just at the verses whose wording happened to
    match.  A psalm whose offset is ambiguous contributes NO aliases and is reported for
    review — never a guess.  Verses already sharing a key with their Hebrew counterpart
    (identity psalms) yield no alias."""
    aliases: dict[str, str] = {}
    report: list[dict] = []
    for p in sorted(douay):
        targets = hebrew_targets(p)
        a = align_psalm(douay[p], reference, p, sim_floor=sim_floor)
        second = a["best_score"] - a["margin"]
        supported = a["total"] and a["matched"] / a["total"] >= frac_ok
        confident = bool(a["best_score"] > 0 and a["margin"] >= margin_abs
                         and (supported or a["best_score"] >= margin_rel * second))
        made = {}
        if confident:
            for v, hp, hv, _s in a["pairs"]:
                src_key, dst_key = f"bible:Ps.{p}.{v}", f"bible:Ps.{hp}.{hv}"
                if src_key != dst_key:
                    made[src_key] = dst_key
            # a numbered Latin title falls off the FRONT of the alignment; its home is the
            # superscription slot — verse 0 of its Hebrew psalm — so it travels with its
            # psalm instead of masquerading as the Hebrew psalm that shares its number.
            for v in a["titles"]:
                made[f"bible:Ps.{p}.{v}"] = f"bible:Ps.{targets[0]}.0"
        aliases.update(made)
        # report anything that isn't a trivial already-aligned identity psalm
        if made or not confident or targets != [p]:
            report.append({
                "psalm": p, "targets": targets, "offset": a["offset"],
                "matched": a["matched"], "total": a["total"],
                "best_score": a["best_score"], "margin": a["margin"],
                "aliases": len(made), "confident": confident,
                "reason": "" if confident else (
                    "no wording overlap" if a["best_score"] == 0 else
                    "ambiguous offset — left on Vulgate keys for review"),
            })
    return aliases, report


# ── store-facing orchestration ───────────────────────────────────────────────
def _psalms_by_doc(store) -> dict:
    """{path: {psalm:int -> {verse:int -> text}}} for every scripture document's Psalms."""
    docs: dict = {}
    for ch in store.iter_chunks():
        if ch.get("source_type") != "scripture":
            continue
        m = _PS_KEY.match(ch.get("section") or "")
        if not m:
            continue
        path = ch.get("path_or_url") or ""
        docs.setdefault(path, {}).setdefault(int(m.group(1)), {})[int(m.group(2))] = \
            ch.get("text") or ""
    return docs


def _edition_id(meta: dict):
    """The edition id from a doc_meta, whether stored as a plain id or a {id: …} dict."""
    ed = (meta or {}).get("edition")
    return ed.get("id") if isinstance(ed, dict) else ed


def _sidecar_path(cfg: dict, edition: str) -> str:
    base = cfg.get("kb_path") or cfg.get("db_path") or "."
    return os.path.join(os.path.dirname(os.path.abspath(base)),
                        f"psalm_aliases.{edition}.json")


def reconcile(store, cfg, *, edition: str = "douay-rheims", reference_path: str | None = None,
              apply: bool = True, sim_floor: float = 0.34, log=log) -> dict:
    """Align a Vulgate-numbered edition's Psalms onto the Hebrew frame of a reference
    edition present in the same store, by wording (no LM, no external table).

    Finds the edition's document(s) via doc_meta, picks the reference edition (the
    non-Vulgate scripture document with the most Psalm verses, unless `reference_path` is
    given), recovers the verse aliases, and — when `apply` — merges them into the edition
    document's `reference_map.key_aliases` and writes a human-readable sidecar.  The
    citation graph and parallel reading then line the editions up on the Hebrew keys.

    Returns {edition, reference, docs, aliases, psalms_aligned, low_confidence:[...],
    report:[...], applied}."""
    metas = store.all_doc_meta()
    by_doc = _psalms_by_doc(store)
    # classify the Psalm-bearing scripture docs into the target edition vs Hebrew reference
    targets = [p for p, m in metas.items() if _edition_id(m) == edition and p in by_doc]
    if reference_path:
        refs = [reference_path] if reference_path in by_doc else []
    else:
        refs = [p for p in by_doc if _edition_id(metas.get(p)) not in VULGATE_EDITIONS]
    result = {"edition": edition, "reference": None, "docs": 0, "aliases": 0,
              "psalms_aligned": 0, "low_confidence": [], "report": [], "applied": False}
    if not targets:
        result["note"] = f"no ingested '{edition}' document with Psalms found"
        return result
    if not refs:
        result["note"] = "no Hebrew-numbered reference edition (e.g. KJV) with Psalms found"
        return result
    ref_path = max(refs, key=lambda p: sum(len(v) for v in by_doc[p].values()))
    reference = by_doc[ref_path]
    result["reference"] = ref_path

    all_aliases: dict[str, str] = {}
    full_report: list[dict] = []
    for path in targets:
        aliases, report = recover_aliases(by_doc[path], reference, sim_floor=sim_floor)
        full_report.extend(report)
        result["docs"] += 1
        result["psalms_aligned"] += sum(1 for r in report if r["confident"])
        result["low_confidence"].extend(r["psalm"] for r in report if not r["confident"])
        all_aliases.update(aliases)
        if apply and aliases:
            meta = store.get_doc_meta(path) or {}
            rm = meta.get("reference_map") or {"book_aliases": {}, "key_aliases": {}}
            rm.setdefault("key_aliases", {}).update(aliases)
            rm.setdefault("book_aliases", {})
            meta["reference_map"] = rm
            meta["psalms_reconciled"] = {"reference": ref_path, "aliases": len(aliases)}
            store.set_doc_meta(path, meta)
    result["aliases"] = len(all_aliases)
    result["report"] = full_report
    result["applied"] = bool(apply and all_aliases)
    if apply and all_aliases:
        try:
            with open(_sidecar_path(cfg, edition), "w", encoding="utf-8") as f:
                json.dump({"edition": edition, "reference": ref_path,
                           "key_aliases": all_aliases, "report": full_report},
                          f, indent=2, sort_keys=True)
        except OSError as e:
            log.warning("psalms: could not write sidecar: %s", e)
    log.info("psalms: reconciled %s → %s — %d psalm(s), %d alias(es), %d low-confidence",
             edition, os.path.basename(ref_path), result["psalms_aligned"],
             result["aliases"], len(result["low_confidence"]))
    return result
