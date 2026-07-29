"""Edge audit — cull nonsense / improper graph associations with a fast, LM-free pass.

The distiller creates relations from text and sometimes over-links, or links on how
two labels LOOK/SOUND rather than what they mean.  Both are detectable from the graph
itself, no model and no source chunks required:

  * SOUND-ALIKE — a relation asserted because two labels are orthographically similar
    but mean unrelated things (apnea↔anemia, ileum↔ilium, complement↔compliment).
    Flagged by HIGH orthographic label similarity together with LOW semantic
    (node-embedding cosine) similarity.  The AND is what keeps legitimately
    similar-AND-related pairs (type-1 ↔ type-2 diabetes: also high-semantic) off it.

  * UNGROUNDED OVER-LINK — a relation with no basis in the corpus: endpoints are
    semantically distant, cite nothing, AND never co-occur in the corpus (the STAT
    pass's PPMI ``concept_cooccurrence``).  All three are required, so a
    co-occurrence table that isn't populated (or whose ids don't match) can't
    over-cull on its own — we only trust "no co-occurrence" when the table has data.

Culling is a soft ``status='retracted'`` — ``edges_from`` filters ``status='active'``,
so a retracted edge leaves retrieval at once — reversible and auditable via
``edge_audit_log``, never a delete.  Report-only by default; ``apply=True`` retracts.
"""

import difflib
import json
import struct
import time


def _unpack(b):
    return list(struct.unpack(f"<{len(b) // 4}f", b)) if b else []


def _cos(a_blob, b_blob):
    """Cosine of two stored node embeddings (both are L2-normalised, so dot == cosine),
    or None when either embedding is missing — an edge we cannot judge semantically."""
    u, v = _unpack(a_blob), _unpack(b_blob)
    if not u or not v or len(u) != len(v):
        return None
    return sum(x * y for x, y in zip(u, v))


def _norm(s):
    return " ".join((s or "").lower().split())


def _orth_sim(a, b):
    """Orthographic similarity in [0,1] (difflib ratio on the normalised labels) — a
    cheap, dependency-free stand-in for 'these two words look/sound alike'."""
    a, b = _norm(a), _norm(b)
    if not a or not b:
        return 0.0
    return difflib.SequenceMatcher(None, a, b).ratio()


def _has_support(support_json):
    try:
        return bool(json.loads(support_json)) if support_json else False
    except (ValueError, TypeError):
        return False


def _cooc_available(db):
    """True only when concept_cooccurrence exists AND has rows — an empty/absent table
    is 'no data', not 'nothing co-occurs' (that distinction gates the ungrounded rule)."""
    try:
        return db.execute("SELECT 1 FROM concept_cooccurrence LIMIT 1").fetchone() is not None
    except Exception:
        return False


def _ppmi(db, a, b):
    lo, hi = (a, b) if a < b else (b, a)          # table stores concept_a < concept_b
    try:
        r = db.execute("SELECT ppmi FROM concept_cooccurrence WHERE concept_a=? AND concept_b=?",
                       (lo, hi)).fetchone()
    except Exception:
        return None
    return float(r[0]) if r else 0.0


def thresholds(cfg):
    ea = cfg.get("edge_audit", {}) or {}
    return {"orth_high": float(ea.get("orth_high", 0.82)),
            "sem_low": float(ea.get("sem_low", 0.35)),
            "sem_vlow": float(ea.get("sem_vlow", 0.15)),
            "min_label_len": int(ea.get("min_label_len", 3)),
            "sample": int(ea.get("sample", 25))}


def _classify(orth, sem, cooc, has_support, t):
    """(verdict, reason) or (None, '').  sem is None ⇒ can't judge ⇒ never flag."""
    if sem is None:
        return None, ""
    if orth >= t["orth_high"] and sem <= t["sem_low"]:
        return "sound_alike", f"labels {orth:.2f} alike but semantics only {sem:.2f}"
    if sem <= t["sem_vlow"] and not has_support and cooc is not None and cooc <= 0.0:
        return "ungrounded", f"semantics {sem:.2f}, no corpus co-occurrence, uncited"
    return None, ""


_VERDICT_ORDER = {"sound_alike": 0, "ungrounded": 1}


def audit_edges(kb, cfg, *, apply=False, limit=None):
    """Scan active edges; flag (and optionally retract) sound-alike / ungrounded ones.
    Returns a report dict — never raises on a single bad row."""
    t = thresholds(cfg)
    db = kb.db
    cooc_on = _cooc_available(db)
    q = ("SELECT e.id, e.src_id, e.dst_id, e.family, e.type, e.support, "
         "s.label AS slabel, s.embedding AS semb, d.label AS dlabel, d.embedding AS demb "
         "FROM edges e JOIN nodes s ON s.id=e.src_id JOIN nodes d ON d.id=e.dst_id "
         "WHERE e.status='active' AND e.family!='meta'")
    params = []
    if limit:
        q += " LIMIT ?"
        params.append(int(limit))

    flagged, scanned, cooc_hits = [], 0, 0
    for r in db.execute(q, params):
        scanned += 1
        sl, dl = r["slabel"] or "", r["dlabel"] or ""
        if len(_norm(sl)) < t["min_label_len"] or len(_norm(dl)) < t["min_label_len"]:
            continue
        if _norm(sl) == _norm(dl):                # identical labels aren't a sound-alike
            continue
        sem = _cos(r["semb"], r["demb"])
        cooc = _ppmi(db, r["src_id"], r["dst_id"]) if cooc_on else None
        if cooc and cooc > 0:
            cooc_hits += 1
        verdict, reason = _classify(_orth_sim(sl, dl), sem, cooc, _has_support(r["support"]), t)
        if verdict:
            flagged.append({"id": r["id"], "src": sl, "dst": dl, "family": r["family"],
                            "type": r["type"], "verdict": verdict, "reason": reason,
                            "orth": round(_orth_sim(sl, dl), 3),
                            "sem": None if sem is None else round(sem, 3),
                            "cooc": None if cooc is None else round(cooc, 3)})

    flagged.sort(key=lambda f: (_VERDICT_ORDER.get(f["verdict"], 9), -f["orth"]))
    by_verdict = {}
    for f in flagged:
        by_verdict[f["verdict"]] = by_verdict.get(f["verdict"], 0) + 1

    # A cooc table with rows that matches NO edge endpoint means its id-space isn't node
    # ids — say so, because the ungrounded rule is then silently doing nothing.
    cooc_note = None
    if cooc_on and scanned and cooc_hits == 0:
        cooc_note = ("concept_cooccurrence has rows but matched no edge endpoints — its ids "
                     "may not be node ids; the 'ungrounded' rule is effectively inactive")

    applied = 0
    if apply and flagged:
        now = time.time()
        _ensure_log(db)
        for f in flagged:
            cur = db.execute("UPDATE edges SET status='retracted', updated_at=? "
                             "WHERE id=? AND status='active'", (now, f["id"]))
            if cur.rowcount:
                db.execute("INSERT OR REPLACE INTO edge_audit_log"
                           "(edge_id,src_label,dst_label,verdict,orth,sem,cooc,at) "
                           "VALUES (?,?,?,?,?,?,?,?)",
                           (f["id"], f["src"], f["dst"], f["verdict"],
                            f["orth"], f["sem"], f["cooc"], now))
                applied += 1
        db.commit()

    return {"scanned": scanned, "flagged": len(flagged), "by_verdict": by_verdict,
            "applied": applied, "cooc_available": cooc_on, "cooc_note": cooc_note,
            "sample": flagged[:t["sample"]], "thresholds": t}


def restore(kb, edge_id=None):
    """Undo a retraction: reactivate a specific audited edge, or all of them.  Lets a
    calibration run be walked back without touching anything the audit didn't cull."""
    db = kb.db
    try:
        if edge_id:
            n = db.execute("UPDATE edges SET status='active' WHERE id=? AND status='retracted'"
                           " AND id IN (SELECT edge_id FROM edge_audit_log)", (edge_id,)).rowcount
        else:
            n = db.execute("UPDATE edges SET status='active' WHERE status='retracted' "
                           "AND id IN (SELECT edge_id FROM edge_audit_log)").rowcount
        db.commit()
    except Exception:
        return 0
    return n


def _ensure_log(db):
    db.execute("CREATE TABLE IF NOT EXISTS edge_audit_log ("
               "edge_id TEXT PRIMARY KEY, src_label TEXT, dst_label TEXT, verdict TEXT, "
               "orth REAL, sem REAL, cooc REAL, at REAL)")
