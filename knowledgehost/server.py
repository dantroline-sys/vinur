"""Stdlib HTTP server exposing the tool contract — light, fast, always up.

Endpoints:
- ``GET  /health``   liveness + index stats (chunk count, backend, dense?)
- ``GET  /tools``    the tool catalogue
- ``POST /call``     run a tool  {name, arguments}  ->  {ok, result|error}

Localhost-bound by default.  If ``auth_token`` is set, ``/call`` and the
control-panel routes require ``Authorization: Bearer <token>``; co-located
with the cascade none is needed.  Binding a non-loopback ``host`` (Vinkona on
another machine) REQUIRES a token — ``serve`` refuses otherwise, because the
``/ops`` surface runs maintenance jobs.  Threaded so concurrent tool calls
don't queue.
"""
from __future__ import annotations

import hmac
import json
import logging
import os
import signal
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

# Cap on a POST body we will read into memory.  The control routes carry small
# JSON (/drop the largest — a whole research doc); refusing anything larger stops
# a client from OOMing the server with a giant Content-Length.
_MAX_BODY = 4 * 1024 * 1024

from . import __version__
from . import lm_lease
from .ops import COMMANDS as OPS_COMMANDS, HELP as OPS_HELP, OpsRunner
from .viewer import INDEX_HTML

log = logging.getLogger("knowledgehost.server")


def _import_formats() -> list:
    """What can THIS install ingest right now?  Live probes (find_spec/which),
    never assumptions — the viewer's Import table renders exactly this."""
    import importlib.util as _iu
    import shutil as _sh

    def has(mod):
        try:
            return _iu.find_spec(mod) is not None
        except Exception:
            return False

    traf = has("trafilatura")
    ocr = bool(_sh.which("tesseract")) and bool(_sh.which("ocrmypdf"))
    return [
        {"format": "Text / Markdown", "matches": ".txt  .md", "ready": True,
         "how": "always available (stdlib) — Vinkona's research drops are .md"},
        {"format": "HTML", "matches": ".html  .htm", "ready": True,
         "how": ("trafilatura installed — full boilerplate stripping" if traf else
                 "works via the stdlib fallback; ./install.sh --html upgrades extraction")},
        {"format": "PDF", "matches": ".pdf", "ready": has("fitz"),
         "how": "./install.sh --pdf   (PyMuPDF)"},
        {"format": "Scanned-PDF OCR", "matches": "(fallback inside PDF ingest)", "ready": ocr,
         "how": "system packages: tesseract + ocrmypdf (the --pdf install offers them)"},
        {"format": "EPUB", "matches": ".epub", "ready": has("ebooklib"),
         "how": "./install.sh --epub   (ebooklib)"},
        {"format": "Wikipedia ZIM", "matches": ".zim — run: ingest --wikipedia",
         "ready": has("libzim"),
         "how": "./install.sh --wikipedia   (libzim; set zim_path in Settings)"},
    ]


def _external_datasets(cfg: dict, kb=None) -> list:
    """The bulk-importable external datasets: each with its ops verb, the config
    key holding its file path, a LIVE does-the-file-exist probe, and (when a KB
    handle is available) whether it is currently IMPORTED — so the Ops tab shows
    ready-to-import / needs-download / already-in-the-graph at a glance."""
    from pathlib import Path
    from .unimport import DATASETS
    sets = [
        ("ConceptNet 5.7", "import-conceptnet", "conceptnet_path",
         "commonsense triples, assertions.csv (~10 GB) — regime=conventional, low trust"),
        ("ATOMIC v4", "import-atomic", "atomic_path",
         "social if-then commonsense, v4_atomic_all_agg.csv — same epistemics as ConceptNet"),
        ("GLUCOSE", "import-glucose", "glucose_path",
         "general causal rules (variable-slot), training CSV — commonsense backbone"),
        ("CauseNet-precision", "import-causenet", "causenet_path",
         "grounded cause→effect graph (JSONL) — has_reference=1, corroboration counts"),
    ]
    out = []
    for name, verb, key, note in sets:
        p = str(cfg.get(key) or "").strip()
        present = bool(p) and Path(p).expanduser().exists()
        dataset = verb.replace("import-", "")
        imported = None
        if kb is not None:
            try:
                imported = bool(kb.db.execute(
                    "SELECT 1 FROM source_registry WHERE doc_id=?",
                    (DATASETS[dataset],)).fetchall())
            except Exception:
                imported = None
        out.append({"name": name, "verb": verb, "dataset": dataset, "config_key": key,
                    "path": p or None, "present": present, "imported": imported,
                    "note": note})
    return out


def _help_payload(cfg: dict, kb=None) -> dict:
    """Tab help (help.json, read per request so edits show on refresh) + the
    live import-format probes and external-dataset probes above."""
    from pathlib import Path
    try:
        tabs = json.loads((Path(__file__).parent / "help.json").read_text())
    except Exception:
        tabs = {}
    return {"help": tabs, "formats": _import_formats(),
            "datasets": _external_datasets(cfg, kb)}


class Handler(BaseHTTPRequestHandler):
    server_version = f"knowledgehost/{__version__}"
    protocol_version = "HTTP/1.1"

    @property
    def cfg(self):
        return self.server.cfg

    def log_message(self, fmt, *a):
        log.info("%s - %s", self.address_string(), fmt % a)

    def _authed(self) -> bool:
        token = self.cfg.get("auth_token")
        if not token:
            return True
        got = self.headers.get("Authorization", "").strip()
        # constant-time compare so a wrong token can't be recovered by timing.
        return hmac.compare_digest(got, f"Bearer {token}")

    def _send(self, body: bytes, status=200, ctype="application/json; charset=utf-8"):
        try:
            self.send_response(status)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            if self.command != "HEAD":
                self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            # client navigated away / closed the tab before we finished writing — benign,
            # and not worth a stack trace (the viewer polls, so this happens routinely).
            pass

    def _send_json(self, obj, status=200):
        self._send(json.dumps(obj, ensure_ascii=False).encode("utf-8"), status)

    def _read_json(self):
        try:
            length = int(self.headers.get("Content-Length", 0))
        except (TypeError, ValueError):
            return None
        if length < 0 or length > _MAX_BODY:            # refuse an absurd body
            return None
        try:
            raw = self.rfile.read(length) if length else b"{}"
            return json.loads(raw or b"{}")
        except (ValueError, json.JSONDecodeError):
            return None

    def _health(self) -> dict:
        """Cheap status for the panel: lease state (is Vinkona on the GPU?) + KB counts."""
        kb = getattr(self.server, "kb", None)
        h = {"counts": kb.counts() if kb else {}}
        try:
            h["lease_fast"] = lm_lease.is_held(lm_lease.FAST, self.cfg)
            h["lease_big"] = lm_lease.is_held(lm_lease.BIG, self.cfg)
        except Exception:
            pass
        return h

    # A handler exception must NEVER drop the socket without a response —
    # the browser then reports only "NetworkError when attempting to fetch
    # resource", which names nothing.  Answer 500 with the real error and
    # keep the traceback in the server log.
    def do_GET(self):
        try:
            return self._do_GET()
        except (BrokenPipeError, ConnectionResetError):
            pass                                   # the client went away — fine
        except Exception as e:
            self._crash_reply(e)

    def do_POST(self):
        try:
            return self._do_POST()
        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception as e:
            self._crash_reply(e)

    def _crash_reply(self, e):
        import traceback
        log.error("handler crashed on %s %s:\n%s", self.command, self.path,
                  traceback.format_exc())
        try:
            self._send_json({"ok": False, "error":
                             f"server error: {type(e).__name__}: {e} "
                             "(full traceback in the kb log)"}, 500)
        except Exception:
            pass                                   # headers already gone — nothing to save

    def _do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path
        q = parse_qs(parsed.query)
        store = self.server.store
        if path == "/":                            # the human-facing viewer
            return self._send(INDEX_HTML.encode("utf-8"), ctype="text/html; charset=utf-8")
        if path == "/health":
            return self._send_json({
                "ok": True, "version": __version__,
                "backend": store.backend, "chunks": store.count(),
                "dense": store.has_vectors(),
                "auth_required": bool(self.cfg.get("auth_token"))})
        if path == "/metrics/history":             # Stats tab: banked telemetry
            try:
                mins = min(float((q.get("mins") or ["60"])[0] or 60), 14 * 1440)
                step = float((q.get("step") or ["0"])[0] or 0)
            except ValueError:
                return self._send_json({"ok": False, "error": "bad mins/step"}, 400)
            return self._send_json(self.server.metrics_store().history(mins, step))
        if path == "/stats":                       # viewer: index breakdown
            by_source = store.stats_by_source() if hasattr(store, "stats_by_source") else {}
            return self._send_json({
                "ok": True, "backend": store.backend, "chunks": store.count(),
                "dense": store.has_vectors(),
                "version": store.manifest.meta_get("version", "1"),
                "by_source": by_source})
        if path == "/sample":                      # viewer: eyeball stored chunks
            n = min(int((q.get("n") or ["20"])[0] or 20), 100)
            src = (q.get("source_type") or [None])[0] or None
            rows = store.sample(n, src) if hasattr(store, "sample") else []
            return self._send_json({"ok": True, "passages": rows})
        if path == "/kb":                          # viewer: structured-KB counts
            kb = getattr(self.server, "kb", None)
            return self._send_json({"ok": True, "counts": kb.counts() if kb else {}})
        if path == "/knowledge":                   # viewer: distilled nodes (the 'learnings')
            kb = getattr(self.server, "kb", None)
            n = min(int((q.get("n") or ["20"])[0] or 20), 100)
            rows = kb.sample_nodes(n) if kb else []
            return self._send_json({"ok": True, "nodes": rows})
        if path == "/facets":                      # viewer: facet coverage per axis (facets.py)
            from .facets import AXES
            kb = getattr(self.server, "kb", None)
            return self._send_json({"ok": True, "axes": list(AXES),
                                    "counts": kb.facet_counts() if kb else {}})
        if path == "/browse":                      # viewer: peruse any produced table
            kb = getattr(self.server, "kb", None)
            kind = (q.get("kind") or ["nodes"])[0]
            n = min(int((q.get("n") or ["50"])[0] or 50), 200)
            fn = {
                "nodes": lambda: kb.sample_nodes(n),
                "edges": lambda: kb.list_edges(n),
                "cards": lambda: kb.list_cards(n),
                "sources": lambda: kb.list_sources(n),
                "adjudication": lambda: kb.list_merge_candidates(n),
                "gaps": lambda: kb.list_gaps(n),
            }.get(kind)
            rows = fn() if (kb and fn) else []
            if kind != "sources":
                return self._send_json({"ok": True, "kind": kind, "rows": rows})

            # Sources get three enrichments: distillation progress per doc,
            # the source FILE's own date (registry rows have no timestamp;
            # URLs/ZIM entries show none), and the QUEUE — ingested docs the
            # distiller hasn't touched, which the registry can't see at all.
            def stamp(r):
                try:
                    r["file_time"] = time.strftime(
                        "%Y-%m-%d %H:%M", time.localtime(os.path.getmtime(r["doc_id"])))
                except (OSError, ValueError):
                    r["file_time"] = ""

            if rows and hasattr(store, "source_progress"):
                try:
                    prog = store.source_progress(self.server.master_kb_path(),
                                                 [r["doc_id"] for r in rows])
                except Exception:                  # pragma: no cover - defensive
                    prog = {}
                for r in rows:
                    p = prog.get(r["doc_id"]) or {}
                    r["chunks"] = p.get("chunks", 0)
                    r["distilled"] = p.get("distilled", 0)
                    r["pct"] = (round(r["distilled"] / r["chunks"] * 100)
                                if r["chunks"] else None)
                    stamp(r)
            pend, totals = [], {}
            if hasattr(store, "pending_sources"):
                try:
                    pq = store.pending_sources(self.server.master_kb_path(), n) or {}
                except Exception:                  # pragma: no cover - defensive
                    pq = {}
                for r in pq.get("rows") or []:
                    r.update(status="queued", distilled=0, pct=0)
                    stamp(r)
                    pend.append(r)
                if "total_docs" in pq:
                    totals = {"docs": pq["total_docs"], "queued": pq["pending_docs"]}
            return self._send_json({"ok": True, "kind": kind, "rows": rows,
                                    "pending": pend, "totals": totals})
        if path == "/search":                      # viewer: run kb_search (no auth, read-only)
            query = (q.get("q") or [""])[0]
            k = int((q.get("k") or ["8"])[0] or 8)
            res = self.server.tools.call("kb_search", {"query": query, "k": k})
            if res.get("ok"):
                return self._send_json({"ok": True, **json.loads(res["result"])})
            return self._send_json(res)
        if path == "/ask":                         # viewer: structured grounded answer
            args = {"query": (q.get("q") or [""])[0]}
            if q.get("rigor"):
                args["rigor"] = q["rigor"][0]
            if q.get("mode"):
                args["mode"] = q["mode"][0]
            if q.get("strict"):
                args["strict"] = q["strict"][0].lower() in ("1", "true", "yes", "on")
            res = self.server.tools.call("kb_ask", args)
            if res.get("ok"):
                return self._send_json({"ok": True, **json.loads(res["result"])})
            return self._send_json(res)
        if path == "/library":                     # viewer/curl: search the document library
            args = {"query": (q.get("q") or [""])[0],
                    "k": int((q.get("k") or ["8"])[0] or 8)}
            if q.get("collection"):
                args["collection"] = q["collection"][0]
            res = self.server.tools.call("library_search", args)
            if res.get("ok"):
                return self._send_json({"ok": True, **json.loads(res["result"])})
            return self._send_json(res)
        if path == "/help":                        # viewer: tab help + import/dataset probes
            return self._send(json.dumps(
                _help_payload(self.cfg, getattr(self.server, "kb", None))).encode())
        if path == "/tools":
            return self._send_json(self.server.tools.catalogue())
        # ── control panel (auth-gated when a token is set) ──
        if path == "/net":                         # Settings › Network: the egress broker
            if not self._authed():
                return self._send_json({"ok": False, "error": "unauthorized"}, 401)
            import shutil as _sh
            from . import serving as sv
            from .amiga_net import audit as _audit
            from .amiga_net import broker as _broker
            from .amiga_net import policy as _policy
            from .config import net_view
            rules = _policy.load()
            live = {d["rule"]: d for d in _policy.live_leases(rules)}
            rule_rows = [{"name": r.name, "purpose": r.purpose,
                          "hosts": r.hosts, "port": r.port, "methods": r.methods,
                          "leased": r.leased, "enabled": r.enabled,
                          "auth": bool(r.auth), "lease": live.get(r.name)}
                         for r in rules]
            from . import posture as _posture
            try:
                posture = _posture.scan(self.cfg)
            except Exception as e:                 # a broken check must not
                posture = {"checks": [], "summary": {   # take the tab down
                    "overall": "unknown", "error": f"{type(e).__name__}: {e}"}}
            return self._send_json({
                "ok": True, "settings": net_view(self.cfg),
                "writable": bool(self.cfg.get("_config_path")),
                "engines": {"aria2c": bool(_sh.which("aria2c")),
                            "wget": bool(_sh.which("wget"))},
                "engine_resolved": _broker._engine(),
                "rules": rule_rows,
                "stats": _audit.summarize(),
                "events": _audit.tail(10),
                "audit_path": str(_audit.LOG_PATH),
                "posture": posture,
                "warning": sv.proxy_warning(self.cfg)})
        if path == "/serving/swap":                # exclusive-model swap state (poll target)
            if not self._authed():
                return self._send_json({"ok": False, "error": "unauthorized"}, 401)
            from . import serving as sv
            return self._send_json({"ok": True, **sv.swap_state()})
        if path == "/serving/status":              # Serving tab: models + weights + state
            if not self._authed():
                return self._send_json({"ok": False, "error": "unauthorized"}, 401)
            from . import serving as sv
            try:
                return self._send_json({"ok": True, **sv.serving_status(self.cfg),
                                        "job": self.server.ops.status(),
                                        "downloads": self.server.downloads.status()})
            except Exception as e:                 # pragma: no cover - defensive
                return self._send_json({"ok": False, "error": f"{type(e).__name__}: {e}"}, 500)
        if path == "/serving/find":                # Ops › pull: hub-search pick-list
            # Synchronous by design: a search is a few broker calls (one
            # lease), and the caller wants a table, not a log to tail.  By
            # default rows are filtered to what THIS box can actually run —
            # engines declared in [serving] and models that fit its memory —
            # with the hidden counts reported, never silent.  all=1 lifts it.
            if not self._authed():
                return self._send_json({"ok": False, "error": "unauthorized"}, 401)
            from . import modelfind
            from . import serving as sv
            query = (q.get("q") or [""])[0].strip()
            if not query:
                return self._send_json({"ok": False, "error": "q required"}, 400)
            try:
                limit = min(int((q.get("limit") or ["8"])[0]), 12)
            except ValueError:
                limit = 8
            show_all = (q.get("all") or ["0"])[0] == "1"
            declared = {str(e.get("engine") or "") for e in self.cfg["serving"]["llms"]}
            declared.discard("")
            os.environ.update(sv.proxy_env(self.cfg))
            try:
                g = modelfind.gather(query, limit=limit,
                                     engines=None if show_all or not declared else declared,
                                     fit_only=not show_all)
            except Exception as e:
                return self._send_json(
                    {"ok": False, "error": f"{type(e).__name__}: {e}"}, 502)
            return self._send_json({"ok": True, **g,
                                    "filtered": not show_all and bool(declared)})
        if path == "/serving/log":                 # Serving tab: one service's log tail
            if not self._authed():
                return self._send_json({"ok": False, "error": "unauthorized"}, 401)
            from . import serving as sv
            try:
                n = int((q.get("n") or ["300"])[0])
            except ValueError:
                n = 300
            try:
                return self._send_json(
                    {"ok": True, **sv.log_tail((q.get("name") or [""])[0], min(n, 2000))})
            except ValueError as e:
                return self._send_json({"ok": False, "error": str(e)}, 400)
        if path == "/drop":                        # exporter handshake: accepts? + inventory
            if not self._authed():
                return self._send_json({"ok": False, "error": "unauthorized"}, 401)
            from . import research
            payload = research.drop_inventory(self.cfg)
            # The return leg of the handshake: open knowledge gaps (queries
            # kb_ask couldn't answer), most-asked first, VERBATIM — the remote
            # Vinkona seeds her research queue with them and the eventual
            # drop's question closes the gap on lower/trim match (close_gap).
            kb = getattr(self.server, "kb", None)
            if payload.get("accepts") and kb is not None:
                try:
                    payload["gaps"] = [
                        {"query": g["query_text"], "count": g.get("count", 1),
                         "intent": g.get("intent") or ""}
                        for g in kb.list_gaps(200)
                        if g.get("status") == "open" and (g.get("query_text") or "").strip()
                    ][:25]
                except Exception:                  # a gapless/older kb never breaks drops
                    pass
            return self._send_json(payload)
        if path == "/bundles":                     # modular §16: groups + scenarios + active
            if not self._authed():
                return self._send_json({"ok": False, "error": "unauthorized"}, 401)
            from . import bundles as B
            scenarios = self.cfg.get("scenarios") or {}
            # Read from the MASTER, not the served working copy: the panel is an
            # authoring surface, and an unloaded/scenario-excluded bundle must
            # still be visible (or it could never be switched back on).
            try:
                mkb = self.server.open_master_kb()
                try:
                    bsum, srcs = mkb.bundle_summary(), mkb.list_sources(500)
                finally:
                    mkb.close()
            except Exception:                      # no master yet — empty panel
                bsum, srcs = [], []
            return self._send_json({
                "ok": True,
                "bundles": bsum,
                "sources": srcs,
                "scenarios": {n: (scenarios[n] if isinstance(scenarios[n], dict) else {})
                              for n in scenarios},
                "active": B.active_scenario_name(self.cfg),
                "unloaded": sorted(B.unloaded_set(self.cfg)),
                "master": self.server.master_kb_path(),
                "working": self.cfg.get("kb_path"),
                "modular": B.is_modular(self.cfg),
                "encrypted_bundles": self.cfg.get("encrypted_bundles") or []})
        if path == "/ops/autopilot":                # Prioritizer tab: the plan + live state
            if not self._authed():
                return self._send_json({"ok": False, "error": "unauthorized"}, 401)
            from . import autopilot as A
            ap = getattr(self.server, "autopilot", None)
            try:                                   # bundle names, for the args editor
                mkb = self.server.open_master_kb()
                try:
                    bnames = [b["bundle"] for b in mkb.bundle_summary()]
                finally:
                    mkb.close()
            except Exception:
                bnames = []
            plan = A.load_plan(self.cfg)
            excl = [str(e.get("name")) for e in self.cfg["serving"]["llms"]
                    if e.get("exclusive")]
            auto = [A.auto_model(self.cfg, s.get("command", ""), s.get("args") or {}) or ""
                    for s in plan["steps"]]
            return self._send_json({"ok": True, "plan": plan,
                                    "state": ap.status() if ap else {"enabled": False},
                                    "commands": OPS_COMMANDS, "help": OPS_HELP,
                                    "bundles": bnames,
                                    "serving_models": excl, "auto_models": auto})
        if path in ("/ops/status", "/ops/log", "/config", "/settings/paths"):
            if not self._authed():
                return self._send_json({"ok": False, "error": "unauthorized"}, 401)
            if path == "/ops/status":
                return self._send_json({"ok": True, "status": self.server.ops.status(),
                                        "health": self._health(),
                                        "commands": OPS_COMMANDS, "help": OPS_HELP})
            if path == "/ops/log":
                n = int((q.get("tail") or ["300"])[0] or 300)
                return self._send_json({"ok": True, "log": self.server.ops.tail(n),
                                        "status": self.server.ops.status(),
                                        "health": self._health()})
            if path == "/config":
                from .config import settings_schema
                schema = settings_schema()
                return self._send_json({
                    "ok": True, "schema": schema,
                    "values": {k: self.cfg.get(k) for k in schema},
                    "config_path": self.cfg.get("_config_path")})
            if path == "/settings/paths":
                from .config import paths_status
                return self._send_json({
                    "ok": True, **paths_status(self.cfg),
                    "writable": bool(self.cfg.get("_config_path"))})
        if path == "/library/config":               # Library panel: trusted root + subfolder toggles
            if not self._authed():
                return self._send_json({"ok": False, "error": "unauthorized"}, 401)
            from .config import library_status
            return self._send_json({"ok": True, **library_status(self.cfg)})
        return self._send_json({"ok": False, "error": "not found"}, 404)

    def _do_POST(self):
        path = urlparse(self.path).path
        if path not in ("/call", "/ops/run", "/ops/stop", "/ops/reload", "/config",
                        "/ops/autopilot", "/library/config", "/library/root",
                        "/source", "/scenario", "/brain", "/drop", "/serving/swap",
                        "/serving/control", "/serving/model", "/serving/add",
                        "/serving/pull", "/serving/download", "/net",
                        "/metrics/mark", "/gaps/close", "/settings/paths"):
            return self._send_json({"ok": False, "error": "not found"}, 404)
        if not self._authed():
            return self._send_json({"ok": False, "error": "unauthorized"}, 401)
        req = self._read_json()
        if req is None:
            return self._send_json({"ok": False, "error": "bad request"}, 400)
        if path == "/metrics/mark":                    # an A/B boundary, user-labelled
            label = str(req.get("label") or "").strip()
            if not label:
                return self._send_json({"ok": False, "error": "label required"}, 400)
            self.server.metrics_store().event("mark", label[:200])
            return self._send_json({"ok": True, "label": label[:200]})
        if path == "/gaps/close":                      # Curation: retire one gap by hand
            kb = getattr(self.server, "kb", None)
            if kb is None:
                return self._send_json({"ok": False, "error": "no KB loaded"}, 400)
            status = str(req.get("status") or "dismissed")
            if status not in ("dismissed", "acquired"):
                return self._send_json(
                    {"ok": False, "error": "status must be dismissed|acquired"}, 400)
            n = kb.close_gap(req.get("query") or "", status=status)
            return self._send_json({"ok": True, "closed": n})
        if path == "/call":
            name = req.get("name")
            if not name:
                return self._send_json({"ok": False, "error": "missing tool name"}, 400)
            return self._send_json(self.server.tools.call(name, req.get("arguments", {})))
        if path == "/serving/swap":                    # request an exclusive-model swap
            # Async by design: weights take minutes to load, so this returns at
            # once and the caller polls GET /serving/swap (e.g. oleum's phased
            # DST runs between its primary and secondary passes).
            from . import serving as sv
            name = str(req.get("name") or "")
            names = [str(e.get("name")) for e in self.cfg["serving"]["llms"]
                     if e.get("exclusive")]
            if name not in names:
                return self._send_json(
                    {"ok": False, "error": f"'{name}' is not an exclusive serving.llms "
                     f"entry (have: {', '.join(names) or 'none'})"}, 400)
            if not sv.swap_state():
                return self._send_json(
                    {"ok": False, "error": "no swap state — supervisor not running"}, 409)
            sv.request_swap(name)
            return self._send_json({"ok": True, "requested": name,
                                    "note": "poll GET /serving/swap until status=ready"})
        if path == "/serving/control":                 # start/stop/restart one service
            # Async, like the swap lane: the supervisor acts on its next tick
            # and the panel re-polls /serving/status.  Only services the
            # supervisor actually knows are addressable — a stop that names
            # nothing must not look like it worked.
            from . import serving as sv
            from . import supervisor as sup
            st = sup.read_state()
            if not sup.alive(st.get("supervisor", 0)):
                return self._send_json(
                    {"ok": False, "error": "the supervisor is not running "
                                           "(./vinur.sh start)"}, 409)
            name = str(req.get("service") or "")
            action = str(req.get("action") or "")
            known = set(st.get("services") or {}) \
                | set((st.get("standby") or {}).values()) \
                | set(st.get("failed") or {}) | set(st.get("held") or [])
            if name not in known:
                return self._send_json(
                    {"ok": False, "error": f"no such service: {name} "
                     f"(have: {', '.join(sorted(known)) or 'none'})"}, 400)
            try:
                sv.request_service(name, action)
            except ValueError as e:
                return self._send_json({"ok": False, "error": str(e)}, 400)
            return self._send_json({"ok": True, "service": name, "action": action,
                                    "note": "the supervisor acts within a few seconds — "
                                            "re-poll /serving/status"})
        if path == "/net":                             # broker: setting write OR action
            act = str(req.get("action") or "")
            if act:                                    # operator actions, audited
                from .amiga_net import audit as _audit
                from .amiga_net import policy as _policy
                rule = str(req.get("rule") or "")
                if act == "revoke_lease":
                    _policy.lease_close(rule)
                    _audit.write("POLICY", rule=rule or "-",
                                 detail="lease revoked by operator (Network tab)")
                    return self._send_json({"ok": True, "note":
                        f"lease on '{rule}' revoked — whatever holds it is refused "
                        "on its next request (partial downloads are kept, resumable)"})
                if act == "rule":
                    on = bool(req.get("enabled"))
                    try:
                        _policy.set_rule_enabled(rule, on)
                    except (ValueError, OSError) as e:
                        return self._send_json({"ok": False, "error": str(e)}, 400)
                    if not on:
                        _policy.lease_close(rule)      # a disabled rule keeps no lease
                    _audit.write("POLICY", rule=rule,
                                 detail=("rule enabled" if on else "rule disabled")
                                        + " by operator (Network tab)")
                    return self._send_json({"ok": True, "note":
                        f"rule '{rule}' " + ("enabled" if on else
                        "disabled — nothing can use or lease it until re-enabled")})
                return self._send_json({"ok": False, "error": f"unknown action {act}"}, 400)
            from .config import set_net_setting
            cp = self.cfg.get("_config_path")
            if not cp:
                return self._send_json(
                    {"ok": False, "error": "server started without -c; no config file to write"}, 400)
            key = str(req.get("key") or "")
            try:
                v = set_net_setting(cp, key, req.get("value"))
            except (ValueError, OSError) as e:
                return self._send_json({"ok": False, "error": str(e)}, 400)
            self.cfg[key] = v                          # live for this process too
            note = ("the broker attaches it per egress.toml's rule auth — engines never see it"
                    if key == "hf_token" else
                    "applies to the next pull / search (jobs read config at launch)")
            return self._send_json({"ok": True, "key": key, "note": note})
        if path == "/serving/pull":                    # start/resume a download
            return self._send_json(self.server.downloads.start(
                str(req.get("model") or ""), include=str(req.get("include") or ""),
                revision=str(req.get("revision") or "main")))
        if path == "/serving/download":                # pause | discard one download
            act = str(req.get("action") or "")
            model = str(req.get("model") or "")
            if act == "pause":
                return self._send_json(self.server.downloads.stop(model))
            if act == "discard":
                return self._send_json(self.server.downloads.discard(model))
            return self._send_json({"ok": False, "error": f"unknown action {act} "
                                    "(pause | discard; resume = /serving/pull)"}, 400)
        if path == "/serving/add":                     # create a [[serving.llms]] entry
            # The fraught part of config.toml is INVENTING an entry — name,
            # port, engine, exclusive-or-not.  This derives all of it from
            # the model on disk and the entries that already exist, writes a
            # minimal commented block, and says what to do next.
            import re as _re
            from . import serving as sv
            from .config import add_llm_entry
            cp = self.cfg.get("_config_path")
            if not cp:
                return self._send_json(
                    {"ok": False, "error": "server started without -c; no config file to write"}, 400)
            model = str(req.get("model") or "").strip()
            if not model:
                return self._send_json({"ok": False, "error": "model required"}, 400)
            llms = self.cfg["serving"]["llms"]
            engine = str(req.get("engine") or "").strip()
            if not engine:
                if model.lower().endswith(".gguf"):
                    engine = "llama"
                else:                                  # follow the house style for
                    engine = next((str(e.get("engine")) for e in llms   # safetensors
                                   if e.get("engine") == "container"), "vllm")
            if model not in {c["model"] for c in sv.eligible_models(engine, cfg=self.cfg)}:
                return self._send_json(
                    {"ok": False, "error": f"'{model}' is not on this disk in a form "
                     f"{engine} can serve — pull it first (the search below)"}, 400)
            entry: dict = {"engine": engine, "model": model}
            if engine == "container":                  # image/runtime copied from a sibling
                tmpl = next((e for e in llms
                             if e.get("engine") == "container" and e.get("image")), None)
                if tmpl is None:
                    return self._send_json(
                        {"ok": False, "error": "no existing container entry to copy "
                         "image/runtime from — add the first one by hand"}, 400)
                entry["image"] = str(tmpl.get("image"))
                if tmpl.get("runtime"):
                    entry["runtime"] = str(tmpl.get("runtime"))
            raw = str(req.get("name") or "").strip()
            if not raw:
                stem = model.rsplit("/", 1)[-1]
                if stem.lower().endswith(".gguf"):
                    stem = stem[:-5]
                raw = _re.sub(r"[^A-Za-z0-9_-]+", "-", stem).strip("-").lower()[:32] or "model"
            names = {str(e.get("name")) for e in llms}
            name, i = raw, 2
            while name in names:
                name, i = f"{raw}{i}", i + 1
            entry["name"] = name
            ports = [int(e.get("port") or 0) for e in llms] + [11439]
            port = int(req.get("port") or (max(ports) + 1))
            if any(int(e.get("port") or 0) == port for e in llms):
                return self._send_json({"ok": False, "error": f"port {port} is taken"}, 400)
            entry["port"] = port
            entry["exclusive"] = (bool(req.get("exclusive")) if "exclusive" in req
                                  else any(e.get("exclusive") for e in llms))
            try:
                add_llm_entry(cp, entry)
            except (ValueError, OSError) as e:
                return self._send_json({"ok": False, "error": str(e)}, 400)
            llms.append(entry)                         # the live view stays truthful
            return self._send_json({"ok": True, "name": name, "port": port,
                "engine": engine, "exclusive": entry["exclusive"],
                "note": f"llm-{name} added on :{port} ({engine}, "
                        f"{'exclusive — swaps with its siblings' if entry['exclusive'] else 'resident'}). "
                        "Restart the supervisor (./vinur.sh restart) to bring it "
                        "under management, then Start or Swap it in from the table."})
        if path == "/serving/model":                   # repoint one entry at another model
            # The Serving tab's picker: rewrite the entry's model line in
            # config.toml (the launcher re-reads config on every spawn, so a
            # restart/swap-in is all it takes) and restart the service if it
            # is up right now.  Only models actually on this disk are
            # accepted — the picker is not a download button.
            from . import serving as sv
            from . import supervisor as sup
            from .config import update_llm_model
            cp = self.cfg.get("_config_path")
            if not cp:
                return self._send_json(
                    {"ok": False, "error": "server started without -c; no config file to write"}, 400)
            name = str(req.get("name") or "")
            model = str(req.get("model") or "").strip()
            entry = next((e for e in self.cfg["serving"]["llms"]
                          if str(e.get("name")) == name), None)
            if entry is None:
                return self._send_json(
                    {"ok": False, "error": f"'{name}' is not a serving.llms entry"}, 400)
            if not model:
                return self._send_json({"ok": False, "error": "model required"}, 400)
            engine = str(entry.get("engine") or "")
            if model != str(entry.get("model") or "") and \
                    model not in {c["model"] for c in sv.eligible_models(engine, cfg=self.cfg)}:
                return self._send_json(
                    {"ok": False, "error": f"'{model}' is not on this disk in a form "
                     f"{engine} can serve — pull it first (Ops › find / pull)"}, 400)
            try:
                old = update_llm_model(cp, name, model)
            except (ValueError, OSError) as e:
                return self._send_json({"ok": False, "error": str(e)}, 400)
            entry["model"] = model                     # the live view stays truthful
            svc = f"llm-{name}"
            st = sup.read_state()
            pid = (st.get("services") or {}).get(svc)
            if sup.alive(st.get("supervisor", 0)) and pid and sup.alive(int(pid)):
                sv.request_service(svc, "restart")
                note = (f"restarting {svc} with {model} — weights load, "
                        "this can take minutes")
            else:
                note = "saved — applies when the service next starts or swaps in"
            return self._send_json({"ok": True, "name": name, "model": model,
                                    "was": old, "note": note})
        if path == "/drop":                            # research hand-off over HTTP
            # A remote Vinkona's exporter posts solved/*.md here instead of
            # writing a shared folder; ingest mines research_solved_dir either way.
            from . import research
            try:
                return self._send_json(
                    research.write_drop(self.cfg, req.get("name"), req.get("content")))
            except ValueError as e:
                return self._send_json({"ok": False, "error": str(e)}, 400)
            except OSError as e:
                return self._send_json({"ok": False, "error": f"write failed: {e}"}, 500)
        if path == "/ops/run":                         # launch a maintenance verb
            try:
                return self._send_json(
                    self.server.ops.start(req.get("command", ""), req.get("args") or {}))
            except ValueError as e:                    # unknown verb / bad option
                return self._send_json({"ok": False, "error": str(e)}, 400)
        if path == "/ops/stop":
            return self._send_json(self.server.ops.stop())
        if path == "/ops/reload":                      # re-warm caches after a write-job / crash
            kb = getattr(self.server, "kb", None)
            if kb is None:
                return self._send_json({"ok": False, "error": "no KB loaded"}, 400)
            try:
                return self._send_json({"ok": True, "counts": kb.reload()})
            except Exception as e:                     # pragma: no cover - defensive
                return self._send_json({"ok": False, "error": f"{type(e).__name__}: {e}"}, 500)
        if path == "/ops/autopilot":                   # Prioritizer tab: save the plan
            from . import autopilot as A
            try:
                saved = A.save_plan(self.cfg, req.get("plan") or {})
            except (ValueError, OSError) as e:
                return self._send_json({"ok": False, "error": str(e)}, 400)
            ap = getattr(self.server, "autopilot", None)
            if ap is not None:                         # apply enable/disable live
                ap.start() if saved["enabled"] else None
            return self._send_json({"ok": True, "plan": saved})
        if path == "/config":                          # persist scalar settings to config.toml
            from .config import update_config_file
            cp = self.cfg.get("_config_path")
            if not cp:
                return self._send_json(
                    {"ok": False, "error": "server started without -c; no config file to write"}, 400)
            try:
                applied = update_config_file(cp, req.get("updates") or {})
            except (ValueError, OSError) as e:
                return self._send_json({"ok": False, "error": str(e)}, 400)
            return self._send_json({"ok": True, "applied": applied, "note":
                "saved — restart to apply, or 'Reload KB' for read-path keys"})
        if path == "/settings/paths":                  # one validated path key
            from .config import paths_status, set_path_setting
            cp = self.cfg.get("_config_path")
            if not cp:
                return self._send_json(
                    {"ok": False, "error": "server started without -c; no config file to write"}, 400)
            try:
                value, live = set_path_setting(self.cfg, cp, str(req.get("key") or ""),
                                               req.get("value"))
            except (ValueError, OSError) as e:
                return self._send_json({"ok": False, "error": str(e)}, 400)
            return self._send_json({
                "ok": True, "value": value, "live": live, **paths_status(self.cfg),
                "note": "applied live" if live else
                        "saved — restart the host (or supervisor) to apply"})
        if path == "/library/root":                    # set the trusted root itself
            from .config import library_status, set_library_root
            cp = self.cfg.get("_config_path")
            if not cp:
                return self._send_json(
                    {"ok": False, "error": "server started without -c; no config file to write"}, 400)
            try:
                set_library_root(self.cfg, cp, req.get("root"))
            except (ValueError, OSError) as e:
                return self._send_json({"ok": False, "error": str(e)}, 400)
            return self._send_json({"ok": True, **library_status(self.cfg), "note":
                "root saved — tick the subfolders to index, Save selection, then index"})
        if path == "/library/config":                  # persist WHICH subfolders are indexed
            from .config import (resolve_library_selection, write_library_sources,
                                 library_status)
            cp = self.cfg.get("_config_path")
            if not cp:
                return self._send_json(
                    {"ok": False, "error": "server started without -c; no config file to write"}, 400)
            try:
                paths = resolve_library_selection(self.cfg, req.get("active") or [])
                write_library_sources(cp, paths)       # containment-validated names only
            except (ValueError, OSError) as e:
                return self._send_json({"ok": False, "error": str(e)}, 400)
            self.cfg["library_sources"] = paths        # reflect immediately for the next GET
            return self._send_json({"ok": True, **library_status(self.cfg), "note":
                "saved — run 'ingest-library' in Operations to (re)index the selection"})
        if path == "/source":                          # rename / regroup / relicense (on the MASTER)
            doc_id = req.get("doc_id")
            if not doc_id:
                return self._send_json({"ok": False, "error": "missing doc_id"}, 400)
            # only pass fields the client actually sent (None = don't touch)
            kw = {k: req[k] for k in ("title", "bundle", "license", "license_holder",
                                      "license_url", "license_text") if k in req}
            mkb = self.server.open_master_kb()
            try:
                row = mkb.set_source(doc_id, **kw)
                if row is not None:
                    row["shippable_flags"] = mkb.license_of(doc_id)["flags"]
            finally:
                mkb.close()
            if row is None:
                return self._send_json({"ok": False, "error": f"no such source: {doc_id}"}, 404)
            return self._send_json({"ok": True, "source": row, "note":
                "saved to master — Apply a scenario (or restart) to fold into the live session"})
        if path == "/scenario":                        # switch scenario + hot-swap the KB
            try:
                info = self.server.swap_scenario(req.get("scenario"))
            except Exception as e:
                return self._send_json(
                    {"ok": False, "error": f"{type(e).__name__}: {e}"}, 500)
            return self._send_json({"ok": True, **info})
        if path == "/brain":                           # runtime brain toggle (non-destructive)
            action = (req.get("action") or "").strip().lower()
            try:
                if action == "list":
                    return self._send_json({"ok": True, **self.server.brain_summary()})
                if action in ("load", "unload"):
                    out = self.server.brain_toggle((req.get("brain") or "").strip(),
                                                   load=(action == "load"))
                    return self._send_json({"ok": True, **out})
            except ValueError as e:
                return self._send_json({"ok": False, "error": str(e)}, 400)
            except Exception as e:                     # swap failure — report, don't 500-html
                return self._send_json(
                    {"ok": False, "error": f"{type(e).__name__}: {e}"}, 500)
            return self._send_json(
                {"ok": False, "error": "action must be list | load | unload"}, 400)


class KnowledgeHostServer(ThreadingHTTPServer):
    daemon_threads = True
    # The stdlib default accept backlog is FIVE.  A panel polling every 2.5s
    # (sometimes from several windows) while a status response is slow can
    # overflow that, and refused loopback connections surface in the browser
    # as bare "NetworkError" with nothing in any log.
    request_queue_size = 32
    allow_reuse_address = True

    def __init__(self, cfg, store, tools, kb=None):
        self.cfg = cfg
        self.store = store
        self.tools = tools
        self.kb = kb
        self.ops = OpsRunner(cfg)                   # single-slot maintenance-job runner
        from . import downloads as _D
        from .serving import ROOT as _ROOT
        # pulls get their OWN lane: a transfer must never queue behind a distill
        self.downloads = _D.Downloads(_ROOT, str(cfg.get("_config_path") or ""))
        from . import autopilot as _A
        from . import lm_lease as _L
        self.autopilot = _A.Autopilot(cfg, self.ops, lease_mod=_L)   # priority-driven verb runner
        self._swap_lock = __import__("threading").Lock()
        tools.brain_host = self                     # lets the kb_brain tool reach the hot-swap
        super().__init__((cfg["host"], cfg["port"]), Handler)

    def master_kb_path(self) -> str:
        return self.cfg.get("_master_kb_path") or self.cfg["kb_path"]

    # ── telemetry (VINUR-UI-01 Stage 6) ───────────────────────────────────────
    def metrics_store(self):
        """Lazy: route reads work with or without a live sampler (and tests
        that construct the server never touch the disk unless they ask)."""
        if getattr(self, "_mstore", None) is None:
            from .metrics import MetricsStore, db_path
            self._mstore = MetricsStore(db_path(self.cfg))
        return self._mstore

    def start_metrics(self):
        """Start the always-on sampler.  Called from serve() — the production
        entry — NOT from __init__, so test-constructed servers stay inert."""
        iv = float(self.cfg.get("stats_interval_s", 5) or 0)
        if iv <= 0 or getattr(self, "_sampler", None) is not None:
            return None
        from .metrics import Sampler
        self._sampler = Sampler(
            self.cfg, self.metrics_store(),
            counts_fn=self._metric_counts,
            slow_fn=lambda: {"kb.chunks": self.store.count()},
            runner=self.ops)
        self._sampler.start()
        return self._sampler

    def _metric_counts(self) -> dict:
        """kb.* series from the cached counts() — reads the CURRENT handle so
        a scenario hot-swap just changes what gets sampled next tick."""
        kb = self.kb
        if kb is None:
            return {}
        c = kb.counts()
        return {"kb.nodes": c.get("nodes", 0), "kb.edges": c.get("edges", 0),
                "kb.cards": c.get("cards", 0),
                "kb.distilled": c.get("distilled_chunks", 0),
                "kb.merge_q": c.get("merge_candidates", 0),
                "kb.gaps": c.get("gaps", 0)}

    def open_master_kb(self):
        """A short-lived KB handle on the MASTER (not the served working copy) for admin
        edits like renaming/regrouping a source — changes must land in the authoring source
        of truth, not the disposable session DB.  Caller must close it."""
        from .kb import KB
        return KB({**self.cfg, "kb_path": self.master_kb_path(), "ann_search": False})

    def swap_scenario(self, name: str | None = None) -> dict:
        """Switch the live session to a scenario: reassemble its working DB from the master
        and hot-swap the KB the server + tools read from (old handle closed).  Serialised so
        two panel clicks can't race a half-built swap."""
        from . import bundles
        from .kb import KB
        with self._swap_lock:
            if name:
                self.cfg["active_scenario"] = name
            self.cfg["kb_path"] = self.master_kb_path()      # assemble from the master
            work = bundles.assemble_working_db(self.cfg, force=True)
            if work != self.cfg["kb_path"]:
                self.cfg["kb_path"] = work
                self.cfg["ann_path"] = ""
            new_kb = KB(self.cfg)
            try:
                new_kb.warm()
                new_kb._get_ann()
            except Exception:                                # pragma: no cover - best effort
                pass
            old, self.kb, self.tools.kb = self.kb, new_kb, new_kb
            if old is not None:
                try:
                    old.close()
                except Exception:                            # pragma: no cover
                    pass
            return {"scenario": bundles.active_scenario_name(self.cfg),
                    "working_db": work, "counts": new_kb.counts()}

    # ── brains: the runtime load/unload surface (kb_brain tool + /brain) ──────
    def brain_summary(self) -> dict:
        """Every bundle in the MASTER with its size and loaded state — 'loaded'
        is the runtime toggle only; a scenario may exclude it independently."""
        from . import bundles as B
        mkb = self.open_master_kb()
        try:
            summ = mkb.bundle_summary()
        finally:
            mkb.close()
        unloaded = B.unloaded_set(self.cfg)
        return {"brains": [{"name": b["bundle"], "sources": b["sources"],
                            "loaded": b["bundle"] not in unloaded}
                           for b in summ],
                "unloaded": sorted(unloaded),
                "active_scenario": B.active_scenario_name(self.cfg)}

    def brain_toggle(self, name: str, *, load: bool) -> dict:
        """Flip one brain on/off: update unloaded_bundles, persist it when a
        config file exists, and hot-swap the working DB.  Non-destructive —
        the master is untouched; this only changes what the session serves."""
        from . import bundles as B
        mkb = self.open_master_kb()
        try:
            known = {b["bundle"] for b in mkb.bundle_summary()}
        finally:
            mkb.close()
        if name not in known:
            raise ValueError(f"no such brain: '{name}' "
                             f"(available: {', '.join(sorted(known))})")
        unloaded = B.unloaded_set(self.cfg)
        already = (name not in unloaded) if load else (name in unloaded)
        if already:
            return {**self.brain_summary(),
                    "note": f"'{name}' is already {'loaded' if load else 'unloaded'}"}
        (unloaded.discard if load else unloaded.add)(name)
        self.cfg["unloaded_bundles"] = ",".join(sorted(unloaded))
        persisted = False
        cp = self.cfg.get("_config_path")
        if cp:
            try:
                from .config import update_config_file
                update_config_file(cp, {"unloaded_bundles":
                                        self.cfg["unloaded_bundles"]})
                persisted = True
            except (ValueError, OSError) as e:     # session still switches; say so
                log.warning("unloaded_bundles not persisted: %s", e)
        swap = self.swap_scenario(None)
        return {**self.brain_summary(), "swap": swap, "persisted": persisted,
                "note": f"{'loaded' if load else 'unloaded'} '{name}'"
                        + ("" if persisted else " (this session only — no config file)")}


def check_bind_auth(cfg) -> None:
    """A non-loopback bind without a token exposes /ops (maintenance jobs) to
    the whole LAN — refuse it.  Deliberate override: VINUR_ALLOW_UNAUTHED_LAN=1."""
    import os
    host = cfg.get("host") or ""
    if host in ("127.0.0.1", "localhost", "::1", ""):
        return
    if cfg.get("auth_token") or os.environ.get("VINUR_ALLOW_UNAUTHED_LAN") == "1":
        return
    raise SystemExit(
        f"refusing to bind {host}:{cfg['port']} without auth_token — the control\n"
        "panel runs maintenance jobs.  Set auth_token in config.toml (clients send\n"
        "Authorization: Bearer <it>), or bind host = \"127.0.0.1\".")


def serve(cfg, store, tools, kb=None):
    check_bind_auth(cfg)
    httpd = KnowledgeHostServer(cfg, store, tools, kb)
    # Warm the ANN index now (one-time resident load of ~index-size RAM) so the first
    # `ask` doesn't eat the load — every query is RAM-speed from the first one.
    if kb is not None:
        try:
            ann = kb._get_ann()
            if ann is not None:
                log.info("ANN warmed: %d node vectors resident", len(ann))
        except Exception as e:
            log.warning("ANN warm failed (%s) — falling back to brute force", e)
        # Pull the node/card tables into the page cache so the first ask's candidate
        # hydration doesn't fault hundreds of rows from disk.
        import time as _t
        _t0 = _t.perf_counter()
        kb.warm()
        log.info("KB tables warmed into page cache (%.1fs)", _t.perf_counter() - _t0)
    log.info("listening on http://%s:%s (backend=%s, %d chunks, dense=%s)",
             cfg["host"], cfg["port"], store.backend, store.count(),
             store.has_vectors())
    if cfg.get("auth_token"):
        log.info("auth: Bearer token required on /call")
    # Telemetry: the always-on sampler (Stats tab).  stats_interval_s = 0 disables.
    try:
        if httpd.start_metrics() is not None:
            log.info("metrics: sampling every %ss into %s",
                     cfg.get("stats_interval_s"), httpd.metrics_store().path)
    except Exception as e:                              # pragma: no cover
        log.warning("metrics sampler failed to start (%s) — Stats stays empty", e)
    # Autopilot: start the thread; it no-ops until the saved plan is enabled (Prioritizer tab).
    try:
        httpd.autopilot.start()
        from . import autopilot as _A
        if _A.load_plan(cfg).get("enabled"):
            log.info("autopilot: enabled — running maintenance verbs on a priority basis")
    except Exception as e:                              # pragma: no cover
        log.warning("autopilot failed to start (%s) — maintenance stays manual", e)
    # SIGTERM (service managers) and SIGHUP (tmux killing the pane's pty) must
    # run the same janitor as Ctrl-C.  Python's default action for both is
    # immediate death with NO unwinding — no finally, no atexit — and the ops
    # job runs in its OWN session (so killpg can manage its whole tree), which
    # also means the pty's HUP never reaches it.  Without this, stopping the
    # stack mid-job orphaned the job and its process-pool workers at 100% CPU.
    # Raising SystemExit turns the signal into a normal unwind through the
    # finally below, where ops.shutdown() kills the job's process group.
    def _die(signum, _frame):
        raise SystemExit(128 + signum)
    signal.signal(signal.SIGTERM, _die)
    signal.signal(signal.SIGHUP, _die)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        log.info("shutting down")
    except SystemExit as e:
        log.info("shutting down (signal %s)", e.code)
    finally:
        try:
            httpd.autopilot.stop()                 # stop the priority driver before its job runner
        except Exception:                          # pragma: no cover
            pass
        httpd.ops.shutdown()                       # don't leave a job orphaned past the server
        httpd.shutdown()
        store.close()
        lib = getattr(tools, "library_store", None)
        if lib is not None:
            try:
                lib.close()
            except Exception:                      # pragma: no cover
                pass
        if kb is not None:
            kb.close()
