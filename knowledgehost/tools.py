"""The ``kb_search`` tool, behind the GET /tools + POST /call contract.

Retrieval flow per call (KNOWLEDGE.md):
  1. embed the query (``search_query:`` prefix)  **and** run a BM25/FTS query;
  2. **fuse** the two candidate lists (Reciprocal Rank Fusion) -> a shortlist;
  3. **rerank** the shortlist (intent-conditioned — the ``intent`` stays local,
     never added to any outbound query);
  4. return top-k passages, each cited, plus a **confidence** = top rerank score.

A low confidence is the signal for Vinkona to fall back to web search rather than
answer from a weak passage.  Every returned ``text`` is sanitized (defence in
depth — Vinkona fences it as low-trust again on its side).
"""
from __future__ import annotations

import json
import logging
import os
import time

from . import query as query_mod
from . import sanitize
from .rerank import make_reranker, rrf_fuse

log = logging.getLogger("knowledgehost.tools")


def _dot(a, b):
    """Cosine of two L2-normalised embedding lists (the embedder returns them normalised)."""
    return sum(x * y for x, y in zip(a, b))
# End-to-end per-call latency (the single chokepoint every tool call passes through —
# the number Vinkona actually feels, including JSON serialisation).  Pairs with the
# per-stage `ask …` line from query.py on the same `knowledgehost.perf` logger.
perf = logging.getLogger("knowledgehost.perf")

CATALOGUE = [
    {
        "name": "kb_search",
        "description": (
            "Search the local general-knowledge base (a Wikipedia snapshot plus "
            "the user's books, papers and documents). Returns cited passages. "
            "Use for established/reference knowledge; it is offline and may be "
            "months out of date for current events — prefer web search for "
            "anything recent or time-sensitive."),
        "parameters": {"type": "object", "properties": {
            "query": {"type": "string", "description": "what to look up"},
            "k": {"type": "integer", "description": "max passages (default 5)"},
            "intent": {"type": "string", "description":
                       "optional: why you're asking, to focus ranking (stays local)"},
            "filters": {"type": "object", "description":
                        "optional: {source_type, title}"}},
            "required": ["query"]},
    },
    {
        "name": "kb_ask",
        "description": (
            "Ask the structured knowledge base a what/how/why/who/where question. "
            "Returns DISTILLED, grounded items (concepts, relations, procedures) with "
            "provenance, a confidence band, and any recorded contradictions — not raw "
            "passages. Pass rigor='high' for high-stakes questions (safety-critical steps, etc.) "
            "to firewall out non-empirical sources and weigh by source trust."),
        "parameters": {"type": "object", "properties": {
            "query": {"type": "string", "description": "the question"},
            "context_features": {"type": "object", "description":
                "optional but powerful: the structured context of the question as "
                "{feature: value} (e.g. {\"trigger\": \"cold start\", \"context\": "
                "\"diesel engine\", \"sign\": \"white smoke\"}). Answers are scored on "
                "how well they MATCH these — a card describing a different situation is "
                "demoted or abstained on, so you get the right entity, not the nearest topic."},
            "intent": {"type": "string", "description":
                "optional: how/why_diag/why_mech/what_if/what — overrides intent detection "
                "when the query is a noun phrase rather than a question"},
            "rigor": {"type": "string", "description": "low (default) | high (stakes)"},
            "mode": {"type": "string", "description":
                     "optional knowledge mode: general (default) | science | scholarly | "
                     "humanities | fiction — restricts answers by source kind"},
            "strict": {"type": "boolean", "description":
                       "with a mode: exclude by SOURCE ORIGIN (drop everything from e.g. "
                       "the fiction folder), not just by claim type (default false)"},
            "facets": {"type": "object", "description":
                       "optional additive facet filter {axis: [values]} (e.g. domain, "
                       "time_frame, trust_tier). Naming an axis also lifts any "
                       "config-default exclusion on that axis for this request."}},
            "required": ["query"]},
    },
]

# Advertised only when the serving host wires itself in (Tools.catalogue) —
# load/unload needs the live server's hot-swap, so the bare CLI can't offer it.
BRAIN_TOOL = {
    "name": "kb_brain",
    "description": (
        "List, load, or unload knowledge 'brains' — modular bundles of the "
        "knowledge base (e.g. a field-geology brain, a base brain). "
        "action='list' shows each brain, its size, and whether it is loaded. "
        "action='load'/'unload' switches one on or off for answering: "
        "non-destructive, reversible, persists across restarts. Use when the "
        "user asks what knowledge is available, or to load/unload a brain by "
        "name."),
    "parameters": {"type": "object", "properties": {
        "action": {"type": "string", "description": "list | load | unload"},
        "brain": {"type": "string", "description":
                  "the brain (bundle) name — required for load/unload"}},
        "required": ["action"]},
}

# Advertised only when external-oracle id regions are configured (VINUR-OPS-01 §3.4) —
# a purely conversational deployment never grows this surface.
OPS_TOOL = {
    "name": "ops_annotate",
    "description": (
        "Annotate candidate operation ids produced by an EXTERNAL legality oracle "
        "(a compiler front-end, analyser, or rules engine that enumerated the legal "
        "moves). For each requested id: attached hazard/knowledge cards and the "
        "learned-rank fields once a learned layer populates them. The response is a "
        "map keyed by exactly the requested ids — this surface never adds, removes, "
        "or orders candidates."),
    "parameters": {"type": "object", "properties": {
        "ops": {"type": "array", "items": {"type": "string"},
                "description": "the oracle's candidate ids (1-500 per call)"},
        "context_features": {"type": "object", "description":
            "optional {feature: value} typed context, reserved for the learned "
            "layer's conditional rank — it never affects which ids are returned"}},
        "required": ["ops"]},
}

# Advertised only when a library corpus is loaded (Tools.catalogue).
LIBRARY_TOOL = {
    "name": "library_search",
    "description": (
        "Search the LOCAL DOCUMENT LIBRARY — a large collection of source texts (science, "
        "fiction, history, …) kept for lookup but NOT distilled into the knowledge base. "
        "Fast lexical search with the best passages reranked; returns cited tracts of text. "
        "Use it to gather source material for research/reading; filter by `collection` to "
        "stay in one register (e.g. 'science') and keep fiction out of a factual search."),
    "parameters": {"type": "object", "properties": {
        "query": {"type": "string", "description": "what to look for"},
        "k": {"type": "integer", "description": "max passages (default 8)"},
        "collection": {"type": "string", "description":
                       "optional topical filter (science | fiction | history | …)"},
        "intent": {"type": "string", "description":
                   "optional: why you're asking, to focus the rerank (stays local)"}},
        "required": ["query"]},
}


class Tools:
    def __init__(self, store, embedder, cfg: dict, kb=None, library_store=None):
        self.store = store
        self.embedder = embedder
        self.cfg = cfg
        self.kb = kb
        self.library_store = library_store        # the search-only document library (optional)
        self.brain_host = None                    # the serving KnowledgeHostServer (kb_brain)
        self.reranker = make_reranker(cfg)

    def catalogue(self):
        tools = list(CATALOGUE)
        if self.library_store is not None:        # only advertise it when a library is loaded
            tools = tools + [LIBRARY_TOOL]
        if self.brain_host is not None:           # only under a live server (needs hot-swap)
            tools = tools + [BRAIN_TOOL]
        if self.kb is not None and getattr(self.kb, "ops_regions", None):
            tools = tools + [OPS_TOOL]            # only with configured id regions (§3.4)
        return {"tools": tools}

    def call(self, name: str, arguments: dict) -> dict:
        arguments = arguments or {}
        handler = getattr(self, f"_t_{name}", None)
        if handler is None:
            return {"ok": False, "error": f"unknown tool: {name}"}
        t0 = time.perf_counter()
        try:
            return handler(arguments)
        except Exception as e:               # never hang / never 500 the caller
            log.exception("tool %s failed", name)
            return {"ok": False, "error": f"{type(e).__name__}: {e}"}
        finally:
            perf.info("call=%s total=%.1fms", name, (time.perf_counter() - t0) * 1e3)

    def _wiki_arm(self, query, k, filters, qvec=None):
        """Live Wikipedia arm: Xapian retrieves+slices the top articles, then we embed the
        slices and rank them by cosine to the query so the most relevant SECTION surfaces
        (not just the article lead).  `qvec` is the already-computed query vector, reused so
        we don't re-embed the query.  Empty list if no ZIM/ libzim, or a source_type filter
        excludes Wikipedia — the chunk-store arms still answer."""
        zim = self.cfg.get("zim_path")
        if not zim or not os.path.isfile(zim):
            return []
        st = (filters or {}).get("source_type")
        if st and st != "wikipedia":
            return []
        try:
            from .sources import wikipedia as wiki_src
            chunks = wiki_src.search(
                zim, query,
                articles=int(self.cfg.get("wiki_articles", 8)),
                chunk_chars=int(self.cfg.get("wiki_chunk_chars", 1000)),
                max_chunks=int(self.cfg.get("wiki_max_chunks", 120)))
        except Exception as e:
            log.debug("wikipedia arm unavailable (%s)", e)
            return []
        if not chunks:
            return []
        # semantic slice-rerank: score each candidate section against the query vector.
        if qvec is not None and self.embedder and self.cfg.get("wiki_semantic", True):
            try:
                vecs = self.embedder.embed_many([c["text"] for c in chunks], "document")
                for c, v in zip(chunks, vecs):
                    c["score"] = round(_dot(qvec, v), 4) if v else 0.0
                chunks.sort(key=lambda c: c["score"], reverse=True)
            except Exception as e:
                log.debug("wiki semantic rerank skipped (%s) — using Xapian order", e)
        return chunks[:k]

    def _t_kb_search(self, args):
        query = (args.get("query") or "").strip()
        if not query:
            return {"ok": False, "error": "kb_search needs a query"}
        k = int(args.get("k") or self.cfg["default_k"])
        intent = args.get("intent")
        filters = args.get("filters") or {}
        shortlist = self.cfg["shortlist"]

        # 1. arms — dense (if embeddings available) + sparse FTS over the chunk store,
        #    plus a live full-text arm over the Wikipedia ZIM's own Xapian index (Tier 0:
        #    no ingestion/embedding, auto-current on a ZIM swap).  The ZIM arm is skipped
        #    when a chunk-store source_type filter excludes Wikipedia.
        qvec = self.embedder.embed_one(query, "query") if self.embedder else None
        dense = self.store.search_vector(qvec, shortlist, filters) if qvec else []
        sparse = self.store.search_text(query, shortlist, filters)
        wiki = self._wiki_arm(query, shortlist, filters, qvec)

        # 2. fuse, 3. rerank (intent-conditioned, local only)
        fused = rrf_fuse(dense, sparse, wiki, rrf_k=self.cfg["rrf_k"])
        # Exclusion firewall at document level (VINUR-OPS-01 §5.1): a chunk is dropped
        # when its source document carries an excluded facet (producers facet region
        # drops as ("doc", <path_or_url>)).  Vacuous until docs are faceted; kb_search
        # has no facets argument, so there is no opt-in on this path.
        excl = self._ask_exclusions(None) if self.kb is not None else []
        ex = [tuple(e.split(":", 1)) for e in excl]
        if ex and fused:
            fmap = self.kb.facets_for([("doc", c.get("path_or_url") or "") for c in fused])
            fused = [c for c in fused
                     if not any(v in fmap.get(("doc", c.get("path_or_url") or ""),
                                              {}).get(a, set()) for a, v in ex)]
        ranked = self.reranker.rerank(query, intent, fused[:shortlist])
        top = ranked[:k]

        # 4. cite + confidence gate
        smax = self.cfg["snippet_max_len"]
        passages = [{
            "text": sanitize.clean(c.get("text"), smax),
            "title": c.get("title") or "",
            "section": c.get("section") or "",
            "path_or_url": c.get("path_or_url") or "",
            "source_type": c.get("source_type") or "",
            "score": round(float(c.get("score", 0.0)), 4),
        } for c in top]
        confidence = passages[0]["score"] if passages else 0.0
        low = confidence < self.cfg["min_confidence"] or not passages
        result = {
            "passages": passages,
            "confidence": round(confidence, 4),
            "low_confidence": bool(low),
            "dense_used": bool(qvec),
        }
        if low:
            result["note"] = ("Weak or no match in the local knowledge base — "
                              "consider web search for this.")
        return {"ok": True, "result": json.dumps(result, ensure_ascii=False)}

    def _t_library_search(self, args):
        """Search-only document library (ingest-library): lexical FTS5 primary, then the
        top slices reranked for precision — the cheap tier that scales to many GB with no
        bulk embedding.  Collection filter keeps registers apart (fiction ≠ science)."""
        if self.library_store is None:
            return {"ok": False, "error": "no document library configured"}
        query = (args.get("query") or "").strip()
        if not query:
            return {"ok": False, "error": "library_search needs a query"}
        k = int(args.get("k") or 8)
        intent = args.get("intent")
        filters = {}
        if args.get("collection"):
            filters["source_type"] = str(args["collection"])   # collection is stored as source_type
        shortlist = self.cfg["shortlist"]

        # 1. lexical retrieval (cheap, scales to GB) + dense arm only if the library was
        #    embedded (library_dense) and the endpoint is up.
        sparse = self.library_store.search_text(query, shortlist, filters)
        dense = []
        if self.library_store.has_vectors() and self.embedder:
            qvec = self.embedder.embed_one(query, "query")
            if qvec:
                dense = self.library_store.search_vector(qvec, shortlist, filters)
        fused = rrf_fuse(dense, sparse, rrf_k=self.cfg["rrf_k"]) if dense else sparse
        # 2. slice-rerank the top candidates (cross-encoder if up, else heuristic/lexical).
        ranked = self.reranker.rerank(query, intent, fused[:shortlist])
        top = ranked[:k]

        smax = self.cfg["snippet_max_len"]
        passages = [{
            "text": sanitize.clean(c.get("text"), smax),
            "title": c.get("title") or "",
            "section": c.get("section") or "",
            "path_or_url": c.get("path_or_url") or "",
            "collection": c.get("source_type") or "",
            "score": round(float(c.get("score", 0.0)), 4),
        } for c in top]
        confidence = passages[0]["score"] if passages else 0.0
        result = {"passages": passages, "confidence": round(confidence, 4),
                  "low_confidence": bool(not passages),
                  "dense_used": bool(dense)}
        if not passages:
            result["note"] = "No match in the local library for this query."
        return {"ok": True, "result": json.dumps(result, ensure_ascii=False)}

    def _t_kb_brain(self, args):
        """List/load/unload knowledge brains — delegated to the serving host,
        which owns the working-DB hot-swap and the persistence of the toggle."""
        host = self.brain_host
        if host is None:
            return {"ok": False, "error":
                    "kb_brain needs the running knowledge-host server"}
        action = (args.get("action") or "list").strip().lower()
        if action == "list":
            return {"ok": True,
                    "result": json.dumps(host.brain_summary(), ensure_ascii=False)}
        if action in ("load", "unload"):
            name = (args.get("brain") or "").strip()
            if not name:
                return {"ok": False, "error": f"{action} needs a brain name"}
            try:
                out = host.brain_toggle(name, load=(action == "load"))
            except ValueError as e:
                return {"ok": False, "error": str(e)}
            return {"ok": True, "result": json.dumps(out, ensure_ascii=False)}
        return {"ok": False, "error": f"kb_brain: unknown action {action!r} "
                                      "(list | load | unload)"}

    def _ask_exclusions(self, facets_arg) -> list:
        """Config-shipped conversational exclusions (VINUR-OPS-01 §5.1), minus any axis
        the request explicitly names in `facets` — naming an axis IS the opt-in."""
        excl = [str(e) for e in (self.cfg.get("ask_exclude_facets") or []) if ":" in str(e)]
        if isinstance(facets_arg, dict) and facets_arg:
            named = {str(a) for a in facets_arg}
            excl = [e for e in excl if e.split(":", 1)[0] not in named]
        return excl

    def _t_kb_ask(self, args):
        query = (args.get("query") or "").strip()
        if not query:
            return {"ok": False, "error": "kb_ask needs a query"}
        if self.kb is None:
            return {"ok": False, "error": "structured KB not available"}
        strict = args["strict"] if "strict" in args else self.cfg.get("strict", False)
        facets_arg = args.get("facets") if isinstance(args.get("facets"), dict) else None
        bundle = query_mod.answer(self.kb, self.embedder, query, rigor=args.get("rigor"),
                                  k=int(args.get("k") or self.cfg["default_k"]),
                                  mode=args.get("mode") or self.cfg.get("mode"),
                                  modes=self.cfg.get("modes"), strict=bool(strict),
                                  prior_penalty=float(self.cfg.get("ask_prior_penalty", 0.5)),
                                  pool=int(self.cfg.get("ask_pool", 50)),
                                  context_features=args.get("context_features"),
                                  intent=args.get("intent"),
                                  fit_gate=bool(self.cfg.get("ask_fit_gate", True)),
                                  vinkona_penalty=float(self.cfg.get("ask_vinkona_penalty", 0.85)),
                                  facets=facets_arg,
                                  exclude_facets=self._ask_exclusions(facets_arg))
        return {"ok": True, "result": json.dumps(bundle, ensure_ascii=False)}

    def _t_ops_annotate(self, args):
        """VINUR-OPS-01 §3 — batch id-join annotation for an external oracle's
        candidates.  Deterministic serialisation (sorted keys) so identical
        (request, graph_version) yields identical bytes (§6 gate 2)."""
        if self.kb is None:
            return {"ok": False, "error": "structured KB not available"}
        if not getattr(self.kb, "ops_regions", None):
            return {"ok": False, "error": "no id regions configured (ops_regions)"}
        ops = args.get("ops")
        if not isinstance(ops, list) or not ops:
            return {"ok": False, "error": "ops_annotate needs a non-empty ops list"}
        if len(ops) > 500:
            return {"ok": False, "error": "ops_annotate accepts at most 500 ids per call"}
        if not all(isinstance(o, str) and o.strip() for o in ops):
            return {"ok": False, "error": "every op id must be a non-empty string"}
        res = self.kb.annotate_ops(ops, args.get("context_features"))
        return {"ok": True, "result": json.dumps(res, ensure_ascii=False, sort_keys=True)}
