"""VINUR — deterministic reasoning over the knowledge graph (VIN-REASON).

The KB's edges are typed facts — family/type/polarity/provenance — and this module turns
them into REASONED ANSWERS without a language model in the loop: the LM (via the
``kb_reason`` tool) only chooses the question SHAPE (compare / paths / about / effects /
siblings / contradictions) and renders the structured result; every relation in that
result is an edge somebody can audit, or a bounded derivation whose full parent chain is
attached.  Same paradigm as calendar_resolve: language → symbolic op → deterministic
engine → language.

Two consumption MODES gate what a query may lean on (config ``reasoning_mode``,
overridable per call — so A/B runs need no global flip):

  * conservative — observed edges only.  Derived edges exist but never enter results.
  * permissive   — bounded derivations join results, ALWAYS marked ``inferred`` with
                   their parent chain; an answer resting on derived-only support says so.

The ``derive`` pass (idle op, wipe-and-rebuild like the mind-graph fold) materialises:
  * taxonomic inheritance — is_a(X→Y) + causal/functional(Y→Z) ⇒ X inherits the edge;
  * causal sign composition — causes(A→B,+) ∘ causes(B→C,−) ⇒ A suppresses C (depth-
    bounded qualitative sign algebra over the polarity column);
each as ``family='derived', status='proposed'`` — invisible to every existing read path
(which filter status='active'), consumed only here in permissive mode, wiped and rebuilt
deterministically each run, never mutating into observed knowledge.  It also mines
STRUCTURAL findings that improve the KB in BOTH modes without touching answers:
  * contradictions — the same relation asserted with opposite polarity → surface_questions;
  * sibling completion gaps — every is_a-sibling has a relation this node lacks →
    knowledge_gaps (feeds the research loop with sharp, targeted questions).

Pure python, zero deps.  The graph view is int-interned adjacency built from kb.db and
cached on the database's (path, mtime, edge-count) — ~2s to build at 1.2M edges, sub-ms
per query after that, and always fresh without any scheduled refresh.
"""
from __future__ import annotations

import json
import logging
import os
import re
import time
from collections import defaultdict

log = logging.getLogger("knowledgehost.reason")

_SIGN = {"positive": 1, "negative": -1}
_SIGN_LABEL = {1: "positive", -1: "negative"}
# families that carry propagatable meaning; meta (claim-to-claim) never traverses
_TRAVERSE_DEFAULT = ("causal", "functional", "taxonomic", "citation", "derived")
_INHERIT_FAMILIES = ("causal", "functional")     # what a child inherits through is_a


def _norm(label: str) -> str:
    return re.sub(r"\s+", " ", (label or "").strip().lower())


# ── the graph view ────────────────────────────────────────────────────────────
class GraphView:
    """Int-interned adjacency over the KB's edges: nodes/edges loaded once, queried in
    microseconds.  Loads ACTIVE edges plus the derived layer (family='derived',
    status='proposed'); every edge carries a ``derived`` flag so query functions can
    honour the mode.  Rebuilt whenever kb.db changes (see view())."""

    def __init__(self, kb):
        t0 = time.time()
        self.idx: dict[str, int] = {}
        self.ids: list[str] = []
        self.labels: list[str] = []
        self.kinds: list[str] = []
        self.by_label: dict[str, list[int]] = defaultdict(list)
        for r in kb.db.execute("SELECT id,label,kind,aliases FROM nodes WHERE status='active'"):
            i = len(self.ids)
            self.idx[r["id"]] = i
            self.ids.append(r["id"])
            self.labels.append(r["label"] or "")
            self.kinds.append(r["kind"] or "")
            self.by_label[_norm(r["label"])].append(i)
            try:
                for al in json.loads(r["aliases"] or "[]"):
                    self.by_label[_norm(str(al))].append(i)
            except (ValueError, TypeError):
                pass
        # eattr rows: (family, type, polarity, derived, support_n, edge_id)
        self.eattr: list[tuple] = []
        self.out: list[list] = [[] for _ in self.ids]
        self.inc: list[list] = [[] for _ in self.ids]
        for r in kb.db.execute(
                "SELECT id,src_id,dst_id,family,type,polarity,support,status FROM edges "
                "WHERE (status='active' AND family!='meta') "
                "   OR (status='proposed' AND family='derived')"):
            s, d = self.idx.get(r["src_id"]), self.idx.get(r["dst_id"])
            if s is None or d is None or s == d:
                continue
            try:
                nsup = len(json.loads(r["support"] or "[]"))
            except (ValueError, TypeError):
                nsup = 0
            e = len(self.eattr)
            self.eattr.append((r["family"] or "", r["type"] or "", r["polarity"] or "",
                               r["family"] == "derived", nsup, r["id"]))
            self.out[s].append((d, e))
            self.inc[d].append((s, e))
        self.built_ms = int((time.time() - t0) * 1000)
        self.n_edges = len(self.eattr)

    # ---- resolution ------------------------------------------------------
    def resolve(self, label: str, limit: int = 5) -> list[int]:
        """Label → candidate node ints: exact/alias match first, then whole-word
        containment ranked by degree (a hub named exactly that beats a mention)."""
        key = _norm(label)
        hits = list(dict.fromkeys(self.by_label.get(key, [])))
        if hits:
            return hits[:limit]
        pat = re.compile(r"\b" + re.escape(key) + r"\b")
        scored = [(len(self.out[i]) + len(self.inc[i]), i)
                  for i, lb in enumerate(self.labels) if pat.search(_norm(lb))]
        scored.sort(reverse=True)
        return [i for _, i in scored[:limit]]

    def use(self, e: int, mode: str) -> bool:
        return mode == "permissive" or not self.eattr[e][3]

    def edge_dict(self, e: int, src: int, dst: int) -> dict:
        fam, typ, pol, derived, nsup, eid = self.eattr[e]
        d = {"from": self.labels[src], "to": self.labels[dst],
             "relation": f"{fam}:{typ}", "polarity": pol, "support": nsup}
        if derived:
            d["inferred"] = True
        return d


_CACHE: dict = {}


def view(kb, cfg) -> GraphView:
    """The cached GraphView for this kb.db — rebuilt when the file changes (mtime +
    edge count), so it is always fresh with no scheduled refresh."""
    path = cfg.get("kb_path") or ""
    try:
        mtime = os.stat(path).st_mtime_ns
    except OSError:
        mtime = 0
    n = kb.db.execute("SELECT COUNT(*) FROM edges").fetchone()[0]
    key = (path, mtime, n)
    if _CACHE.get("key") != key:
        _CACHE["key"] = key
        _CACHE["view"] = GraphView(kb)
        log.info("reason: graph view rebuilt — %d nodes, %d edges, %d ms",
                 len(_CACHE["view"].ids), _CACHE["view"].n_edges, _CACHE["view"].built_ms)
    return _CACHE["view"]


def _mode(cfg, args) -> str:
    m = (args.get("mode") or cfg.get("reasoning_mode") or "conservative").strip().lower()
    return m if m in ("conservative", "permissive") else "conservative"


# ── query ops ─────────────────────────────────────────────────────────────────
def _resolve_or_err(g: GraphView, label: str):
    c = g.resolve(label)
    if not c:
        return None, {"ok": False, "error": f"no KB node matches '{label}'"}
    return c[0], None


def _neigh(g, i, mode, families=None):
    """[(other_int, edge_int, 'out'|'in')] honouring mode + family filter."""
    out = []
    for d, e in g.out[i]:
        if g.use(e, mode) and (not families or g.eattr[e][0] in families):
            out.append((d, e, "out"))
    for s, e in g.inc[i]:
        if g.use(e, mode) and (not families or g.eattr[e][0] in families):
            out.append((s, e, "in"))
    return out


def op_about(g: GraphView, cfg, args) -> dict:
    """The local picture of one node: its relations grouped by family:type, each
    direction, top-supported first."""
    mode = _mode(cfg, args)
    i, err = _resolve_or_err(g, args.get("a") or "")
    if err:
        return err
    grouped: dict[str, list] = defaultdict(list)
    inferred_any = False
    for other, e, direction in _neigh(g, i, mode):
        fam, typ, pol, derived, nsup, _ = g.eattr[e]
        inferred_any = inferred_any or derived
        arrow = "→" if direction == "out" else "←"
        grouped[f"{fam}:{typ}"].append(
            {"direction": arrow, "node": g.labels[other], "polarity": pol,
             "support": nsup, **({"inferred": True} if derived else {})})
    for k in grouped:
        grouped[k].sort(key=lambda x: -x["support"])
        grouped[k] = grouped[k][:12]
    return {"ok": True, "op": "about", "node": g.labels[i], "kind": g.kinds[i],
            "mode": mode, "relations": dict(grouped),
            "degree": len(g.out[i]) + len(g.inc[i]),
            **({"note": "some relations are inferred (derived), marked as such"}
               if inferred_any else {})}


def _paths(g, a, b, mode, max_hops, families, cap=8):
    """Up to `cap` shortest typed chains a↔b (undirected walk, direction kept per edge)."""
    if max_hops < 1:
        return []
    frontier = {a: [[]]}
    seen = {a}
    for _hop in range(max_hops):
        nxt: dict[int, list] = {}
        found: list = []
        for n, plist in frontier.items():
            for other, e, direction in _neigh(g, n, mode, families):
                step = (n, other, e, direction)
                if other == b:
                    for p in plist:
                        found.append(p + [step])
                        if len(found) >= cap:
                            break
                if other not in seen and other != b:
                    if other not in nxt:
                        nxt[other] = []
                    if len(nxt[other]) < 2:                 # keep it bounded
                        nxt[other] = [p + [step] for p in plist[:2]]
        if found:
            return found[:cap]
        seen.update(nxt)
        frontier = nxt
        if not frontier:
            break
    return []


def _render_path(g, path) -> list:
    out = []
    for n, other, e, direction in path:
        d = g.edge_dict(e, n, other) if direction == "out" else g.edge_dict(e, other, n)
        d["direction"] = direction
        out.append(d)
    return out


def op_paths(g: GraphView, cfg, args) -> dict:
    mode = _mode(cfg, args)
    a, err = _resolve_or_err(g, args.get("a") or "")
    if err:
        return err
    b, err = _resolve_or_err(g, args.get("b") or "")
    if err:
        return err
    hops = min(int(args.get("max_hops") or cfg.get("reasoning_max_depth", 3)), 5)
    fams = tuple(args.get("families") or _TRAVERSE_DEFAULT)
    ps = _paths(g, a, b, mode, hops, fams)
    return {"ok": True, "op": "paths", "a": g.labels[a], "b": g.labels[b], "mode": mode,
            "paths": [_render_path(g, p) for p in ps],
            **({} if ps else {"note": f"no connection within {hops} hops"})}


def op_compare(g: GraphView, cfg, args) -> dict:
    """The full comparison surface: direct relations, meeting nodes (what BOTH relate
    to, via what), connecting paths, and the per-relation contrast (what one has that
    the other lacks)."""
    mode = _mode(cfg, args)
    a, err = _resolve_or_err(g, args.get("a") or "")
    if err:
        return err
    b, err = _resolve_or_err(g, args.get("b") or "")
    if err:
        return err

    direct = []
    for other, e, direction in _neigh(g, a, mode):
        if other == b:
            direct.append(g.edge_dict(e, a, b) if direction == "out"
                          else g.edge_dict(e, b, a))

    na = {}
    for other, e, _d in _neigh(g, a, mode):
        na.setdefault(other, []).append(e)
    meets = []
    for other, e, _d in _neigh(g, b, mode):
        if other in na and other not in (a, b):
            fam_a = ", ".join(sorted({f"{g.eattr[x][0]}:{g.eattr[x][1]}" for x in na[other]}))
            meets.append({"node": g.labels[other],
                          "a_via": fam_a, "b_via": f"{g.eattr[e][0]}:{g.eattr[e][1]}"})
    meets = meets[:15]

    # contrast: same relation kind, different targets
    def sig(i):
        m: dict[str, set] = defaultdict(set)
        for other, e, direction in _neigh(g, i, mode):
            fam, typ, *_ = g.eattr[e]
            m[f"{fam}:{typ}:{direction}"].add(g.labels[other])
        return m
    sa, sb = sig(a), sig(b)
    contrast = []
    for rel in sorted(set(sa) | set(sb)):
        if rel.startswith(("citation:", "commentary:")):
            continue                                       # structural, not comparative
        only_a = sorted(sa.get(rel, set()) - sb.get(rel, set()))[:8]
        only_b = sorted(sb.get(rel, set()) - sa.get(rel, set()))[:8]
        shared = sorted(sa.get(rel, set()) & sb.get(rel, set()))[:8]
        if only_a or only_b or shared:
            contrast.append({"relation": rel, "shared": shared,
                             "only_a": only_a, "only_b": only_b})

    hops = min(int(args.get("max_hops") or cfg.get("reasoning_max_depth", 3)), 5)
    ps = _paths(g, a, b, mode, hops, tuple(_TRAVERSE_DEFAULT))
    return {"ok": True, "op": "compare", "a": g.labels[a], "b": g.labels[b], "mode": mode,
            "direct_relations": direct, "meeting_nodes": meets,
            "contrast": contrast[:20], "paths": [_render_path(g, p) for p in ps[:4]]}


def op_siblings(g: GraphView, cfg, args) -> dict:
    """What is LIKE this node: is_a co-children first, then nodes sharing the most
    relation targets (relational-signature neighbours)."""
    mode = _mode(cfg, args)
    i, err = _resolve_or_err(g, args.get("a") or "")
    if err:
        return err
    parents = [d for d, e in g.out[i]
               if g.eattr[e][0] == "taxonomic" and g.use(e, mode)]
    sibs = set()
    for p in parents:
        for s, e in g.inc[p]:
            if g.eattr[e][0] == "taxonomic" and s != i and g.use(e, mode):
                sibs.add(s)
    mine = {(g.eattr[e][0], g.eattr[e][1], other)
            for other, e, _d in _neigh(g, i, mode)}
    overlap = defaultdict(int)
    for other, _e, _d in _neigh(g, i, mode):
        for o2, e2, _d2 in _neigh(g, other, mode):
            if o2 != i:
                overlap[o2] += 1
    analogs = sorted(((n, c) for n, c in overlap.items() if c >= 2 and n not in sibs),
                     key=lambda x: -x[1])[:10]
    return {"ok": True, "op": "siblings", "node": g.labels[i], "mode": mode,
            "is_a_parents": [g.labels[p] for p in parents],
            "siblings": sorted(g.labels[s] for s in sibs)[:20],
            "shares_relations_with": [
                {"node": g.labels[n], "shared_neighbours": c} for n, c in analogs],
            "signature": len(mine)}


def op_effects(g: GraphView, cfg, args) -> dict:
    """Signed causal reach: everything downstream (or upstream) of a node within N hops,
    with the NET SIGN multiplied along each path — qualitative what-if.  A node reachable
    with conflicting signs is reported as 'conflicted', never silently averaged."""
    mode = _mode(cfg, args)
    i, err = _resolve_or_err(g, args.get("a") or "")
    if err:
        return err
    direction = "up" if (args.get("direction") or "down") == "up" else "down"
    hops = min(int(args.get("max_hops") or cfg.get("reasoning_max_depth", 3)), 4)
    reach: dict[int, dict] = {}
    frontier = {i: (1, [])}
    for hop in range(1, hops + 1):
        nxt: dict[int, tuple] = {}
        for n, (sign, path) in frontier.items():
            steps = g.out[n] if direction == "down" else g.inc[n]
            for other, e in steps:
                fam, typ, pol, derived, nsup, _ = g.eattr[e]
                if fam not in ("causal", "derived") or not g.use(e, mode):
                    continue
                s2 = sign * _SIGN.get(pol, 0)
                if s2 == 0:
                    continue                               # unsigned edges don't propagate
                p2 = path + [(n, other, e)]
                r = reach.get(other)
                if r is None:
                    reach[other] = {"sign": s2, "hops": hop, "path": p2,
                                    "inferred": derived or any(g.eattr[x[2]][3] for x in path)}
                    if other not in frontier:
                        nxt[other] = (s2, p2)
                elif r["sign"] != s2:
                    r["sign"] = 0                          # conflicting routes — say so
        frontier = nxt
        if not frontier:
            break
    items = []
    for n, r in sorted(reach.items(), key=lambda kv: (kv[1]["hops"], -abs(kv[1]["sign"]))):
        chain = [(g.labels[s], f"{g.eattr[e][1]}({g.eattr[e][2] or '?'})", g.labels[d])
                 for s, d, e in r["path"]]
        items.append({"node": g.labels[n], "hops": r["hops"],
                      "net": {1: "increases", -1: "decreases", 0: "conflicted"}[r["sign"]],
                      "via": chain, **({"inferred": True} if r["inferred"] else {})})
    return {"ok": True, "op": "effects", "node": g.labels[i], "direction": direction,
            "mode": mode, "reach": items[:40]}


def op_contradictions(g: GraphView, cfg, args) -> dict:
    """The same relation asserted with OPPOSITE polarity — locally (around a node) or
    the corpus-wide worst offenders."""
    mode = _mode(cfg, args)
    scope = None
    if args.get("a"):
        scope, err = _resolve_or_err(g, args["a"])
        if err:
            return err
    pairs: dict[tuple, dict] = {}
    nodes = [scope] if scope is not None else range(len(g.ids))
    for n in nodes:
        for other, e in g.out[n]:
            fam, typ, pol, derived, nsup, _ = g.eattr[e]
            if pol not in _SIGN or not g.use(e, mode):
                continue
            k = (n, other, fam, typ)
            r = pairs.setdefault(k, {"signs": set(), "support": 0})
            r["signs"].add(pol)
            r["support"] += nsup
    out = [{"from": g.labels[k[0]], "to": g.labels[k[1]],
            "relation": f"{k[2]}:{k[3]}", "support": v["support"]}
           for k, v in pairs.items() if len(v["signs"]) > 1]
    out.sort(key=lambda x: -x["support"])
    return {"ok": True, "op": "contradictions", "mode": mode,
            **({"node": g.labels[scope]} if scope is not None else {}),
            "contradictions": out[:25]}


def _signed_reach(g, a, b, mode, hops):
    """Net sign of the causal chains a→…→b within `hops`, or None if unreachable.
    Returns (sign, chain) — sign 0 = conflicting routes.  Mirrors op_effects' walk."""
    frontier = {a: (1, [])}
    best = None
    for _hop in range(hops):
        nxt = {}
        for n, (sign, path) in frontier.items():
            for other, e in g.out[n]:
                fam, typ, pol, derived, _n, _ = g.eattr[e]
                if fam not in ("causal", "derived") or not g.use(e, mode):
                    continue
                s2 = sign * _SIGN.get(pol, 0)
                if s2 == 0:
                    continue
                p2 = path + [(n, other, e)]
                if other == b:
                    if best is None:
                        best = {"sign": s2, "path": p2}
                    elif best["sign"] != s2:
                        best["sign"] = 0                   # conflicting routes
                elif other not in frontier and other not in nxt:
                    nxt[other] = (s2, p2)
        frontier = nxt
        if not frontier:
            break
    if best is None:
        return None, []
    chain = [(g.labels[s], f"{g.eattr[e][1]}({g.eattr[e][2] or '?'})", g.labels[d])
             for s, d, e in best["path"]]
    return best["sign"], chain


def op_verify(g: GraphView, cfg, args) -> dict:
    """Check a CLAIMED relation against the graph.  The claim arrives decomposed —
    a (subject), b (object), optional relation type ('causes', 'treats', 'is_a', …) and
    polarity ('positive'=increases/promotes, 'negative'=decreases/suppresses).  Verdicts:
      supported / contradicted / mixed  — a direct edge of the claimed kind (mixed = the
                                          graph itself asserts BOTH signs);
      supported_indirectly / contradicted_indirectly — no direct edge, but the net sign
                                          of the causal chain a→…→b (dis)agrees;
      related_but_different — a and b are directly related, just not as claimed;
      unsupported — the graph is SILENT.  Absence of evidence, never evidence of
                                          absence: the answer must say 'not recorded',
                                          not 'false'."""
    mode = _mode(cfg, args)
    a, err = _resolve_or_err(g, args.get("a") or "")
    if err:
        return err
    b, err = _resolve_or_err(g, args.get("b") or "")
    if err:
        return err
    claimed_rel = _norm(args.get("relation") or "") or None
    claimed_pol = (args.get("polarity") or "").strip().lower() or None
    base = {"ok": True, "op": "verify", "a": g.labels[a], "b": g.labels[b], "mode": mode,
            "claim": {"relation": claimed_rel or "(any)", "polarity": claimed_pol or "(any)"}}

    direct = []                                           # every a↔b edge, both directions
    for d, e in g.out[a]:
        if d == b and g.use(e, mode):
            direct.append(("a→b", e))
    for s, e in g.inc[a]:
        if s == b and g.use(e, mode):
            direct.append(("b→a", e))
    graph_says = [dict(g.edge_dict(e, a, b) if way == "a→b" else g.edge_dict(e, b, a),
                       direction=way) for way, e in direct]

    def matches(e):                                       # claimed type ⊆ edge type, so
        return claimed_rel is None or claimed_rel in _norm(g.eattr[e][1])   # 'causes' hits
                                                          # inherited_/composed_causes too
    matched = [(way, e) for way, e in direct if way == "a→b" and matches(e)]
    if matched:
        signs = {g.eattr[e][2] for _w, e in matched if g.eattr[e][2] in _SIGN}
        inferred = any(g.eattr[e][3] for _w, e in matched)
        if len(signs) > 1:
            verdict = "mixed"
        elif claimed_pol and signs and claimed_pol not in signs:
            verdict = "contradicted"
        else:
            verdict = "supported"
        return {**base, "verdict": verdict, "graph_says": graph_says,
                **({"inferred": True} if inferred else {}),
                **({"note": "the graph asserts BOTH polarities — genuinely contested"}
                   if verdict == "mixed" else {})}

    # no direct edge of the claimed kind — try the signed causal chain
    hops = min(int(args.get("max_hops") or cfg.get("reasoning_max_depth", 3)), 4)
    sign, chain = _signed_reach(g, a, b, mode, hops)
    if sign is not None and (claimed_rel is None or "caus" in claimed_rel
                             or claimed_rel in ("increases", "decreases", "affects")):
        if sign == 0:
            verdict = "mixed"
        elif claimed_pol:
            verdict = ("supported_indirectly" if _SIGN_LABEL[sign] == claimed_pol
                       else "contradicted_indirectly")
        else:
            verdict = "supported_indirectly"
        return {**base, "verdict": verdict, "graph_says": graph_says,
                "chain": chain, "net": {1: "positive", -1: "negative", 0: "conflicted"}[sign],
                **({"inferred": True} if any("inherited" in c[1] or "composed" in c[1]
                                             for c in chain) else {})}

    if graph_says:                                        # related, just not as claimed
        return {**base, "verdict": "related_but_different", "graph_says": graph_says,
                "note": "a and b ARE directly related, but not by the claimed relation"}
    ps = _paths(g, a, b, mode, hops, tuple(_TRAVERSE_DEFAULT), cap=2)
    return {**base, "verdict": "unsupported",
            "note": ("the graph records NOTHING for this claim — absence of evidence, "
                     "not evidence of absence; do not report the claim as false"),
            **({"nearest_context": [_render_path(g, p) for p in ps]} if ps else {})}


OPS = {"about": op_about, "compare": op_compare, "paths": op_paths,
       "siblings": op_siblings, "effects": op_effects,
       "contradictions": op_contradictions, "verify": op_verify}


def query(kb, cfg, args: dict) -> dict:
    """The kb_reason entry point: {op, a, b?, ...} → a grounded structured answer."""
    op = (args.get("op") or "").strip().lower()
    fn = OPS.get(op)
    if fn is None:
        return {"ok": False, "error": f"unknown op '{op}'", "ops": sorted(OPS)}
    g = view(kb, cfg)
    res = fn(g, cfg, args)
    if res.get("ok") and _mode(cfg, args) == "permissive":
        res["caveat"] = ("items marked 'inferred' are bounded derivations "
                        "(proposed, with parent chains) — not observed facts")
    return res


# ── the derive pass (idle op) ─────────────────────────────────────────────────
def derive(kb, cfg, *, log=log) -> dict:
    """Wipe-and-rebuild the derived layer + mine structural findings.  Deterministic:
    same KB in, same derivations out.  Everything written is quarantined —
    family='derived' status='proposed' — and one DELETE reverses the whole layer."""
    cap = int(cfg.get("derive_max_edges", 20000))
    max_children = int(cfg.get("derive_max_children", 40))
    t0 = time.time()
    kb.db.execute("DELETE FROM edges WHERE family='derived'")
    g = GraphView(kb)                                     # fresh, post-wipe

    def observed(a, b):
        return any(not g.eattr[e][3] for d, e in g.out[a] if d == b)

    n_inherit = n_sign = 0
    derived: list[tuple] = []                             # (src, dst, type, polarity, chain)

    # 1) taxonomic inheritance: child ← is_a — parent —causal/functional→ target
    children_of: dict[int, list] = defaultdict(list)
    for i in range(len(g.ids)):
        for d, e in g.out[i]:
            if g.eattr[e][0] == "taxonomic" and not g.eattr[e][3]:
                children_of[d].append(i)
    for parent, kids in children_of.items():
        if len(kids) > max_children:
            continue                                      # a hub class over-generalises
        for d, e in g.out[parent]:
            fam, typ, pol, drv, nsup, _ = g.eattr[e]
            if drv or fam not in _INHERIT_FAMILIES:
                continue
            for kid in kids:
                if len(derived) >= cap:
                    break
                if kid != d and not observed(kid, d):
                    derived.append((kid, d, f"inherited_{typ}", pol,
                                    [g.labels[kid], f"is_a {g.labels[parent]}",
                                     f"{typ}({pol or '?'}) {g.labels[d]}"]))
                    n_inherit += 1

    # 2) causal sign composition, depth 2: A→B (s1) ∘ B→C (s2) ⇒ A→C (s1·s2)
    for a in range(len(g.ids)):
        if len(derived) >= cap:
            break
        for b, e1 in g.out[a]:
            f1, t1, p1, drv1, _n1, _ = g.eattr[e1]
            if drv1 or f1 != "causal" or p1 not in _SIGN:
                continue
            for c, e2 in g.out[b]:
                if len(derived) >= cap:
                    break
                f2, t2, p2, drv2, _n2, _ = g.eattr[e2]
                if drv2 or f2 != "causal" or p2 not in _SIGN or c == a:
                    continue
                if observed(a, c):
                    continue
                sign = _SIGN[p1] * _SIGN[p2]
                derived.append((a, c, "composed_causes", _SIGN_LABEL[sign],
                                [g.labels[a], f"{t1}({p1}) {g.labels[b]}",
                                 f"{t2}({p2}) {g.labels[c]}"]))
                n_sign += 1

    now = time.time()
    for src, dst, typ, pol, chain in derived:
        eid = kb._edge_hash(g.ids[src], g.ids[dst], "derived", typ, pol, "", "derived",
                            None, None)
        kb.db.execute(
            "INSERT OR IGNORE INTO edges(id,src_id,dst_id,family,type,mechanism,"
            "mechanism_basis,modifiers,polarity,edge_hash,support,regime,scope,status,"
            "created_at,updated_at) VALUES(?,?,?,?,?,'','derived','{}',?,?,?,?,'{}',"
            "'proposed',?,?)",
            (eid, g.ids[src], g.ids[dst], "derived", typ, pol, eid,
             json.dumps([{"derivation": chain}]), "derived", now, now))

    # 3) contradiction mining → surface_questions (both modes benefit; answers untouched)
    n_contra = 0
    for row in kb.db.execute(
            "SELECT src_id, dst_id, family, type FROM edges "
            "WHERE status='active' AND polarity IN ('positive','negative') "
            "GROUP BY src_id, dst_id, family, type "
            "HAVING COUNT(DISTINCT polarity) > 1 LIMIT 200"):
        sa = g.idx.get(row["src_id"]); da = g.idx.get(row["dst_id"])
        if sa is None or da is None:
            continue
        kb.add_surface_question(
            "edge", row["src_id"],
            f"Contradiction: '{g.labels[sa]}' {row['type']} '{g.labels[da]}' is asserted "
            f"with BOTH polarities ({row['family']}) — which holds, and under what "
            f"conditions?")
        n_contra += 1

    # 4) sibling-completion gaps → knowledge_gaps (the research loop's sharpest questions)
    n_gaps = 0
    frac = float(cfg.get("derive_sibling_frac", 0.8))
    for parent, kids in children_of.items():
        if not (3 <= len(kids) <= max_children) or n_gaps >= 100:
            continue
        rels: dict[tuple, list] = defaultdict(list)
        for kid in kids:
            for d, e in g.out[kid]:
                fam, typ, *_ = g.eattr[e]
                if fam in _INHERIT_FAMILIES and not g.eattr[e][3]:
                    rels[(fam, typ)].append(kid)
        for (fam, typ), have in rels.items():
            have_set = set(have)
            if len(have_set) / len(kids) >= frac:
                for kid in kids:
                    if kid not in have_set and n_gaps < 100:
                        kb.log_gap(f"{g.labels[kid]}: {typ}?",
                                   intent="sibling_completion",
                                   effect_label=f"siblings of {g.labels[parent]} all "
                                                f"have {fam}:{typ}")
                        n_gaps += 1

    kb.db.commit()
    _CACHE.clear()                                        # the view must see the new layer
    out = {"derived_edges": len(derived), "inherited": n_inherit, "sign_composed": n_sign,
           "contradictions_flagged": n_contra, "sibling_gaps": n_gaps,
           "capped": len(derived) >= cap, "elapsed_s": round(time.time() - t0, 2)}
    log.info("derive: %s", out)
    return out
