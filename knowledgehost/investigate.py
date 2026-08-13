"""VINUR — iterative agentic reasoning over the knowledge graph (VIN-INVESTIGATE).

Think-on-Graph, adapted to this host: instead of one retrieval or one symbolic op,
the LM *walks* the graph — anchor on the question's concepts, look at the typed edges
on offer, choose which to follow, accumulate an evidence subgraph, decide when it has
enough, then answer grounded ONLY in the trail.  The division of labour is the same
paradigm as calendar_resolve and kb_reason, iterated:

    engine (deterministic)             LM (chooses / judges)
    ──────────────────────             ─────────────────────
    anchor spans → seed nodes
    enumerate + score candidate edges
                                       pick which edges to follow; say when done
    expand, dedupe, note contradictions
    render the evidence subgraph
                                       synthesise an answer citing evidence lines
    verify the answer's claims (op_verify)

The LM can never invent an edge — it only chooses among real ones — and the final
claims are fact-checked against the graph before the answer ships (contradicted →
'contested').  Two tiers mirror distillation: a cheap NAVIGATOR picks edges each hop
(multiple-choice, ~200 tokens — well within the fast extractor's range) and the big
LM synthesises once at the end.

Degrades honestly at every rung: no navigator → the engine's own score-ordered greedy
walk; no synthesiser → the structured evidence subgraph is returned for the caller to
reason over; no anchors, or a dead-ended walk → a knowledge gap is logged so the
research loop goes and learns what was missing.  The KB stays always-answering.

LMs are injected as chat_json-shaped callables ((system, user, schema, max_tokens) →
dict|None), so the whole loop runs on a bare interpreter under test; ``live_lms``
builds them from the serving tiers for real use.  Pure python, zero deps.
"""
from __future__ import annotations

import logging
import math
import re

from . import reason as R
from . import understand

log = logging.getLogger("knowledgehost.investigate")

_STOP = frozenset(
    "a an and are as at be but by can could do does did for from had has have how i if in "
    "is it its may might much of on or our so should that the their them then this to was "
    "we what when where which who why will with would you your".split())

# speech act (understand.py) → edge families the walk should lean toward.  A *why*
# question wants causal chains; feasibility wants function; comparison wants shared
# structure.  A preference, not a filter — off-family edges still compete on score.
_FAMILY_PREF = {
    "diagnostic": ("causal",),
    "hypothetical": ("causal",),
    "counterfactual_omit": ("causal",),
    "observation": ("causal",),
    "feasibility": ("functional", "causal"),
    "comparison": ("functional", "taxonomic"),
}

_HUB_DEGREE = 400          # far nodes above this degree are usually generic — penalised

_NAV_SCHEMA = {
    "type": "object",
    "properties": {
        "picks": {"type": "array", "items": {"type": "integer"}},
        "done": {"type": "boolean"},
        "why": {"type": "string"},
    },
    "required": ["picks", "done"],
}

_NAV_SYSTEM = (
    "You are navigating a knowledge graph to gather evidence for a question. You see "
    "the evidence collected so far and a numbered list of edges that could be followed "
    "next. Reply with JSON only: picks = the numbers (up to 3) of the edges most likely "
    "to lead toward the answer; done = true when the evidence already suffices to answer "
    "(picks may then be empty); why = one short clause. Follow chains that connect the "
    "question's concepts; ignore tangents, however interesting.")

_SYNTH_SCHEMA = {
    "type": "object",
    "properties": {
        "answer": {"type": "string"},
        "confident": {"type": "boolean"},
        "claims": {"type": "array", "items": {"type": "object", "properties": {
            "a": {"type": "string"}, "b": {"type": "string"},
            "relation": {"type": "string"}, "polarity": {"type": "string"}},
            "required": ["a", "b"]}},
    },
    "required": ["answer", "confident"],
}

_SYNTH_SYSTEM = (
    "Answer the question STRICTLY from the numbered evidence — it is the complete set "
    "of facts you may use; cite lines like [3]. Reason over chains (A affects B, B "
    "affects C) but never add outside facts. If the evidence cannot answer the "
    "question, say plainly what is missing and set confident=false. Also list your "
    "answer's key factual claims as {a, b, relation, polarity} objects (a=subject, "
    "b=object, polarity positive=increases/promotes, negative=decreases/suppresses) "
    "so each can be checked against the graph. Reply with JSON only.")


def _tokens(text: str) -> list:
    return [t for t in re.findall(r"[a-z0-9]+", (text or "").lower())
            if t not in _STOP and len(t) > 1]


# ── anchoring: the question's spans → seed nodes ─────────────────────────────
def _anchor(g, question: str, limit: int = 4) -> list:
    """Longest exact label/alias spans in the question win (a 2-word match beats the
    two 1-word matches inside it); resolve()'s containment scan is the fallback when
    nothing matches exactly."""
    words = re.findall(r"[a-z0-9']+", (question or "").lower())
    found, used = [], set()
    for n in (4, 3, 2, 1):
        for s in range(0, len(words) - n + 1):
            if any((s + k) in used for k in range(n)):
                continue
            if n == 1 and (words[s] in _STOP or len(words[s]) < 3):
                continue
            hits = g.by_label.get(R._norm(" ".join(words[s:s + n])))
            if hits:
                found.append((s, hits[0]))
                used.update(s + k for k in range(n))
    seeds = list(dict.fromkeys(i for _s, i in sorted(found)))
    if not seeds:
        for w in sorted(set(_tokens(question)), key=len, reverse=True)[:3]:
            for i in g.resolve(w, limit=1):
                if i not in seeds:
                    seeds.append(i)
    return seeds[:limit]


# ── candidate proposal: what the walk could do next ──────────────────────────
def _score(g, e: int, far: int, qtok: set, prefer: tuple) -> float:
    """Cheap, deterministic relevance: where the edge LEADS (overlap with the question),
    how well-attested it is, family fit to the question's speech act; generic hubs and
    derived edges score down.  This ordering is also the greedy fallback's policy."""
    fam, _typ, _pol, derived, nsup, _eid = g.eattr[e]
    far_tok = set(_tokens(g.labels[far]))
    ov = len(far_tok & qtok) / max(1, len(far_tok))
    s = 2.0 * ov + 0.3 * math.log1p(nsup)
    if fam in prefer:
        s += 0.5
    if derived:
        s -= 0.3
    if len(g.out[far]) + len(g.inc[far]) > _HUB_DEGREE:
        s -= 0.4
    return s


def _candidates(g, frontier, visited_edges, visited_nodes, mode, qtok, prefer, cap):
    """Score-ordered unvisited edges off the frontier, deduped per (pair, type).
    Only EDGES are excluded once walked — an edge into an already-visited node stays
    on offer, because those are the closing links ('does A affect B?' is answered by
    the edge that finally CONNECTS the two anchors); visited nodes are merely never
    re-added to the frontier.  An edge leading somewhere already visited scores a
    small bonus: closing a loop between question concepts beats opening a tangent."""
    cand = []
    for i in frontier:
        for other, e, direction in R._neigh(g, i, mode):
            if e in visited_edges:
                continue
            s = _score(g, e, other, qtok, prefer)
            if other in visited_nodes:
                s += 0.8                       # a closing link between question concepts
            cand.append((s, {"src": i, "dst": other, "e": e, "dir": direction}))
    cand.sort(key=lambda t: -t[0])
    out, seen = [], set()
    for _s, c in cand:
        key = (min(c["src"], c["dst"]), max(c["src"], c["dst"]), g.eattr[c["e"]][1])
        if key in seen:
            continue
        seen.add(key)
        out.append(c)
        if len(out) >= cap:
            break
    return out


def _ends(c) -> tuple:
    """(a, b) in the edge's OWN direction (dir='in' means the edge runs other→frontier)."""
    return (c["src"], c["dst"]) if c["dir"] == "out" else (c["dst"], c["src"])


def _edge_line(g, c) -> str:
    a, b = _ends(c)
    _fam, typ, pol, derived, _nsup, _eid = g.eattr[c["e"]]
    sign = {"positive": "↑", "negative": "↓"}.get(pol, "")
    return (f"{g.labels[a]} —[{typ}{sign}]→ {g.labels[b]}"
            + (" (inferred)" if derived else ""))


def _note_contradiction(g, c, mode, out: list) -> None:
    """The graph asserting the OPPOSITE sign for the same pair+type as a chosen edge —
    surfaced immediately so a contested fact can't quietly anchor the answer."""
    _fam, typ, pol, *_rest = g.eattr[c["e"]]
    if pol not in R._SIGN:
        return
    a, b = _ends(c)
    for d2, e2 in g.out[a]:
        if d2 == b and e2 != c["e"] and g.use(e2, mode):
            _f2, t2, p2, *_r2 = g.eattr[e2]
            if t2 == typ and p2 in R._SIGN and p2 != pol:
                out.append({"about": f"{g.labels[a]} {typ} {g.labels[b]}",
                            "note": "the graph asserts BOTH polarities"})
                return


# ── the loop ─────────────────────────────────────────────────────────────────
def investigate(kb, cfg, question: str, *, navigator=None, synthesizer=None,
                mode=None, max_hops=None, log_gaps=True) -> dict:
    """Walk the graph toward an answer.  navigator/synthesizer are chat_json-shaped
    callables or None (None → greedy walk / evidence-only result).  Returns the
    answer (or an honest abstain) with the full audited trail."""
    g = R.view(kb, cfg)
    mode = R._mode(cfg, {"mode": mode} if mode else {})
    act, _conf = understand.classify_speech_act(question)
    prefer = _FAMILY_PREF.get(act, ())
    qtok = set(_tokens(question))
    hops = min(int(max_hops or cfg.get("investigate_max_hops", 4)), 6)
    beam = int(cfg.get("investigate_beam", 6))
    ccap = int(cfg.get("investigate_candidates", 14))
    ecap = int(cfg.get("investigate_evidence_cap", 24))
    gpicks = int(cfg.get("investigate_greedy_picks", 2))

    seeds = _anchor(g, question)
    base = {"ok": True, "op": "investigate", "question": question, "mode": mode,
            "speech_act": act, "anchors": [g.labels[i] for i in seeds],
            "lm": {"navigator": navigator is not None,
                   "synthesizer": synthesizer is not None}}
    if not seeds:
        gap = _log_gap(kb, question, act) if log_gaps else False
        return {**base, "answer": None, "abstain": True, "confidence": "none",
                "evidence": [], "trail": [], "contradictions": [],
                **({"gap_logged": True} if gap else {}),
                "note": "no KB node matches anything in the question"}

    frontier = list(seeds)
    visited_nodes = set(frontier)
    visited_edges: set = set()
    evidence: list = []
    trail: list = []
    contradictions: list = []
    done = False
    nav_failures = 0

    for hop in range(1, hops + 1):
        cands = _candidates(g, frontier, visited_edges, visited_nodes,
                            mode, qtok, prefer, ccap)
        if not cands:
            trail.append({"hop": hop, "offered": 0, "picked": [], "by": "none",
                          "note": "frontier exhausted"})
            break
        picked, why, by = None, "", "greedy"
        if navigator is not None:
            lines = "\n".join(f"{k}. {_edge_line(g, c)}" for k, c in enumerate(cands, 1))
            ev_txt = "\n".join(f"{ev['n']}. {ev['text']}" for ev in evidence) or "(none yet)"
            user = (f"Question: {question}\n\nHop {hop} of {hops}.\n\n"
                    f"Evidence so far:\n{ev_txt}\n\nCandidate edges:\n{lines}\n\nJSON:")
            resp = navigator(_NAV_SYSTEM, user, _NAV_SCHEMA, 300)
            if isinstance(resp, dict):
                by = "lm"
                done = bool(resp.get("done"))
                why = str(resp.get("why") or "")[:200]
                try:
                    idxs = [int(x) for x in (resp.get("picks") or [])]
                except (TypeError, ValueError):
                    idxs = []
                picked = [cands[k - 1] for k in idxs if 1 <= k <= len(cands)][:3]
            else:
                nav_failures += 1       # transient LM trouble → this hop goes greedy
        if picked is None:
            picked = cands[:gpicks]
        elif not picked and not done:
            by = "lm+greedy"            # the LM chose nothing yet wasn't done — a wasted
            picked = cands[:gpicks]     # hop would just re-offer the same list
        new_front = []
        for c in picked:
            visited_edges.add(c["e"])
            a, b = _ends(c)
            evidence.append({"n": len(evidence) + 1, "hop": hop,
                             "text": _edge_line(g, c), **g.edge_dict(c["e"], a, b)})
            _note_contradiction(g, c, mode, contradictions)
            if c["dst"] not in visited_nodes:
                visited_nodes.add(c["dst"])
                new_front.append(c["dst"])
        trail.append({"hop": hop, "offered": len(cands), "by": by,
                      "picked": [_edge_line(g, c) for c in picked],
                      **({"why": why} if why else {}),
                      **({"done": True} if done else {})})
        if done or len(evidence) >= ecap:
            break
        frontier = (new_front + frontier)[:beam]

    base["lm"]["navigator_failures"] = nav_failures
    result = {**base, "evidence": evidence, "trail": trail, "hops": len(trail),
              "contradictions": contradictions}
    if mode == "permissive":
        result["caveat"] = ("items marked 'inferred' are bounded derivations "
                            "(proposed, with parent chains) — not observed facts")

    if not evidence:
        gap = _log_gap(kb, question, act) if log_gaps else False
        return {**result, "answer": None, "abstain": True, "confidence": "none",
                **({"gap_logged": True} if gap else {}),
                "note": "anchored, but no usable relations from there"}
    # a dead-ended walk with almost nothing to show is a sharp research question
    if not done and len(evidence) < 2 and trail and trail[-1].get("by") == "none":
        if log_gaps and _log_gap(kb, question, act):
            result["gap_logged"] = True

    if synthesizer is None:
        return {**result, "answer": None, "abstain": True, "confidence": "evidence_only",
                "note": ("no synthesis LM — the evidence subgraph is returned for the "
                         "caller to reason over")}

    ev_txt = "\n".join(f"{ev['n']}. {ev['text']}" for ev in evidence)
    extra = ""
    if contradictions:
        extra = ("\n\nContradictions the graph itself records:\n"
                 + "\n".join(f"- {c['about']}: {c['note']}" for c in contradictions))
    resp = synthesizer(_SYNTH_SYSTEM,
                       f"Question: {question}\n\nEvidence:\n{ev_txt}{extra}\n\nJSON:",
                       _SYNTH_SCHEMA, 700)
    answer = str((resp or {}).get("answer") or "").strip() if isinstance(resp, dict) else ""
    if not answer:
        return {**result, "answer": None, "abstain": True, "confidence": "evidence_only",
                "note": "synthesis failed — the evidence subgraph is still good"}

    confident = bool(resp.get("confident"))
    verified = {"supported": 0, "contradicted": 0, "unsupported": 0, "unresolved": 0}
    checks = []
    for cl in (resp.get("claims") or [])[:8]:
        if not isinstance(cl, dict):
            continue
        v = R.op_verify(g, cfg, {"a": cl.get("a") or "", "b": cl.get("b") or "",
                                 "relation": cl.get("relation") or "",
                                 "polarity": cl.get("polarity") or "", "mode": mode})
        if not v.get("ok"):
            verified["unresolved"] += 1
            continue
        verdict = v["verdict"]
        checks.append({"claim": f"{cl.get('a')} {cl.get('relation') or 'relates to'} "
                                f"{cl.get('b')}", "verdict": verdict})
        if verdict in ("supported", "supported_indirectly"):
            verified["supported"] += 1
        elif verdict in ("contradicted", "contradicted_indirectly", "mixed"):
            verified["contradicted"] += 1
        else:
            verified["unsupported"] += 1

    if verified["contradicted"] or contradictions:
        band = "contested"
    elif confident and not verified["unsupported"]:
        band = "grounded"
    else:
        band = "partial"
    return {**result, "answer": answer, "abstain": False, "confident": confident,
            "confidence": band, "verified": verified, "claims_checked": checks,
            **({"note": "the answer rests on contested ground — the graph disagrees "
                        "with itself or with a claim"} if band == "contested" else {})}


def _log_gap(kb, question: str, act: str) -> bool:
    try:
        kb.log_gap(question, intent=act)
        return True
    except Exception:                     # a gap is never worth failing the answer
        return False


# ── live plumbing ────────────────────────────────────────────────────────────
def live_lms(cfg, log_fn=None):
    """(navigator, synthesizer) chat_json callables from the serving tiers — each None
    when its tier is down (the loop degrades per-leg).  The navigator prefers the fast
    extractor tier (config investigate_navigator = fast|big), mirroring two-tier
    distillation: cheap multiple-choice hops, one big-LM synthesis."""
    from . import distill as distill_mod
    fast = distill_mod.fast_endpoints(cfg, log_fn)
    big = distill_mod.verify_endpoints(cfg, log_fn)
    tier = (cfg.get("investigate_navigator") or "fast").strip().lower()
    nav_pool = (big or fast) if tier == "big" else (fast or big)
    syn_pool = big or fast
    return (nav_pool[0].chat_json if nav_pool else None,
            syn_pool[0].chat_json if syn_pool else None)
