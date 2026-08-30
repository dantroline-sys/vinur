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

# How long a FINISHED job keeps its place in the viewer's header strip.  The
# Operations tab still names it as the last job; the header is for "right now".
_JOB_BRIEF_LINGER_S = 900


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


def _fs_roots(cfg: dict) -> list:
    """Curated jump-off points for the file browser: home + the box's configured
    roots (sources, library, quarantine, drops, pack/bundle output).  Shortcuts
    only — the browser can still navigate up to '/'."""
    roots, seen = [], set()

    def add(label, p):
        if not p:
            return
        rp = os.path.realpath(os.path.expanduser(str(p)))
        if rp not in seen and os.path.isdir(rp):
            seen.add(rp)
            roots.append({"label": label, "path": rp})

    add("Home", os.path.expanduser("~"))
    for s in (cfg.get("sources") or []):
        add("source", s)
    for s in (cfg.get("library_sources") or []):
        add("library", s)
    add("library root", cfg.get("library_root"))
    add("quarantine", cfg.get("quarantine_dir"))
    add("drops", cfg.get("research_solved_dir"))
    add("packs", cfg.get("pack_dir") or "packs")
    add("bundles", cfg.get("bundle_dir"))
    return roots


def _fs_list(cfg: dict, raw: str) -> dict:
    """List one directory for the browser: dirs first, dotfiles hidden, capped.
    An empty/at-a-file path resolves to the nearest readable directory."""
    roots = _fs_roots(cfg)
    base = (os.path.realpath(os.path.expanduser(raw)) if raw.strip()
            else (roots[0]["path"] if roots else os.path.expanduser("~")))
    while base and not os.path.isdir(base) and base != "/":
        base = os.path.dirname(base)                       # a file / gone → nearest dir
    entries, cap, truncated = [], 2000, False
    try:
        with os.scandir(base) as it:
            for e in it:
                if e.name.startswith("."):
                    continue
                try:
                    isdir = e.is_dir()
                except OSError:
                    isdir = False
                size = None
                if not isdir:
                    try:
                        size = e.stat().st_size
                    except OSError:
                        size = None
                entries.append({"name": e.name, "dir": isdir, "size": size})
                if len(entries) >= cap:
                    truncated = True
                    break
    except OSError as ex:
        return {"ok": False, "error": f"cannot read {base}: {ex}"}
    entries.sort(key=lambda r: (not r["dir"], r["name"].lower()))
    parent = os.path.dirname(base)
    return {"ok": True, "path": base,
            "parent": (parent if parent and parent != base else None),
            "roots": roots, "entries": entries, "truncated": truncated,
            "exts": [str(x).lower() for x in (cfg.get("extensions") or [])]}


def _write_answers_file(cfg: dict, answers: dict) -> str:
    """Persist the wizard's structured-text confirmation answers to a scratch JSON file
    the collect subprocess reads (the answers are too structured for a CLI flag).  Kept
    under the control dir so it lives with the run, not the user's data."""
    import json
    import tempfile
    from pathlib import Path
    ctrl = cfg.get("control_dir") or str(Path(__file__).resolve().parent.parent / "var")
    d = Path(ctrl).expanduser() / "run"
    d.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix="collect-answers-", suffix=".json", dir=str(d))
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        json.dump(answers, f, ensure_ascii=False)
    return name


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

    def _job_brief(self) -> dict | None:
        """What the maintenance slot is doing, for the viewer's header — on every
        tab, not just Operations.  Rides /stats (which the header already polls),
        so it carries NO argv: the command name, its age and its progress record
        say what is happening without publishing paths on an unauthed route."""
        ops = getattr(self.server, "ops", None)
        if ops is None:
            return None
        st = ops.status()
        if not st.get("command"):
            return None
        if not st.get("running") and time.time() - (st.get("ended") or 0) > _JOB_BRIEF_LINGER_S:
            return None          # yesterday's job is not "what this host is doing"
        out = {"running": bool(st.get("running")), "command": st.get("command"),
               "elapsed_s": st.get("elapsed_s"), "exit_code": st.get("exit_code")}
        if out["running"]:
            out["progress"] = ops.progress()
        return out

    def _mode_brief(self) -> dict:
        """The box's posture, for the header's mode selector: the pinned
        override (or none = Automatic), the mechanism states, and what the
        prioritizer is doing — so the Automatic label can say what automation
        is up to right now.  Rides the unauthed /stats like _job_brief: names
        only, no paths or argv."""
        from . import serving as sv
        try:
            d = {"override": (sv.override_state() or {}).get("mode"),
                 "minimal": bool(sv.minimal_state().get("on")),
                 "endpoint": bool(sv.endpoint_state().get("on"))}
        except Exception:                          # pragma: no cover - defensive
            return {}
        ap = getattr(self.server, "autopilot", None)
        try:
            st = ap.status() if ap else {}
            d["prioritizer"] = {"enabled": bool(st.get("enabled")),
                                "running_step": st.get("running_step")}
        except Exception:                          # pragma: no cover - defensive
            d["prioritizer"] = {}
        return d

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
                "by_source": by_source, "job": self._job_brief(),
                "mode": self._mode_brief()})
        if path == "/yield_audit":                 # Sources: documents that distilled to ~nothing
            kb = getattr(self.server, "kb", None)
            if kb is None:
                return self._send_json({"ok": False, "error": "no KB loaded"}, 400)
            from . import yield_audit
            try:
                res = yield_audit.audit(
                    store, kb,
                    min_chunks=int((q.get("min_chunks") or [yield_audit.DEFAULT_MIN_CHUNKS])[0]),
                    ratio=float((q.get("ratio") or [yield_audit.DEFAULT_RATIO])[0]),
                    limit=min(int((q.get("limit") or ["200"])[0]), 1000))
            except ValueError as e:
                return self._send_json({"ok": False, "error": f"bad parameter: {e}"}, 400)
            return self._send_json(res)
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
            bsel = (q.get("bundle") or [""])[0]
            fn = {
                "nodes": lambda: kb.sample_nodes(n),
                "edges": lambda: kb.list_edges(n),
                "cards": lambda: kb.list_cards(n),
                "sources": lambda: kb.list_sources(n, bundle=bsel or None),
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
                    r["dupes"] = p.get("dupes", 0)
                    r["zoned"] = p.get("zoned", 0)
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
                                    "pending": pend, "totals": totals,
                                    "bundle": bsel,
                                    "bundles": (kb.source_bundle_counts()
                                                if hasattr(kb, "source_bundle_counts")
                                                else {})})
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
            # Auth-gated: the payload's dataset probes carry raw config filesystem
            # paths + exists-flags.  Serving it pre-auth contradicted the README's
            # "Bearer on every control route" and leaked paths to any LAN client.
            if not self._authed():
                return self._send_json({"ok": False, "error": "unauthorized"}, 401)
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
        if path == "/serving/schedule":            # Serving › Schedule: weekly minimal-mode plan
            if not self._authed():
                return self._send_json({"ok": False, "error": "unauthorized"}, 401)
            from datetime import datetime
            from . import serving as sv
            sched = sv.read_schedule()
            now = datetime.now()
            loc = now.astimezone()                       # the box's local tz (what the
            #                                              schedule is evaluated against)
            return self._send_json(
                {"ok": True, "schedule": sched,
                 "minimal": sv.minimal_state(),          # current live posture
                 "override": sv.override_state(),        # schedule is held while pinned
                 # server-local clock so the editor can show "now" against the grid —
                 # tz included so a UTC-vs-wall-clock mismatch (a classic "timer fires at
                 # the wrong hour" cause) is visible at a glance
                 "now": {"dow": now.weekday(), "hhmm": now.strftime("%H:%M"),
                         "clock": now.strftime("%H:%M:%S"),
                         "tz": loc.strftime("%Z") or "local", "offset": loc.strftime("%z")},
                 "wants_minimal": sv.schedule_wants_minimal(sched, now)})
        if path == "/serving/status":              # Serving tab: models + weights + state
            if not self._authed():
                return self._send_json({"ok": False, "error": "unauthorized"}, 401)
            from . import serving as sv
            try:
                return self._send_json({"ok": True, **sv.serving_status(self.cfg),
                                        "job": self.server.ops.status(),
                                        "downloads": self.server.downloads.status(),
                                        "tuning_schema": sv.TUNING_SCHEMA})
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
        if path == "/collection/preview":          # wizard: is a target file compatible?
            if not self._authed():
                return self._send_json({"ok": False, "error": "unauthorized"}, 401)
            from . import pack as pack_mod
            tgt = (q.get("to") or [""])[0]
            bundle = (q.get("bundle") or [""])[0]
            doc = (q.get("doc") or [""])[0]        # optional: analyze it → confirm questions
            if not tgt or not bundle:
                return self._send_json({"ok": False, "error": "to and bundle required"}, 400)
            try:
                return self._send_json(pack_mod.collection_preview(self.cfg, tgt, bundle, doc))
            except Exception as e:
                return self._send_json({"ok": False, "error": str(e)}, 400)
        if path == "/fs/list":                     # file browser: list one directory
            if not self._authed():
                return self._send_json({"ok": False, "error": "unauthorized"}, 401)
            res = _fs_list(self.cfg, (q.get("path") or [""])[0])
            return self._send_json(res, 200 if res.get("ok") else 400)
        if path == "/pending":                     # 'Needs your input': deferred structured docs
            if not self._authed():
                return self._send_json({"ok": False, "error": "unauthorized"}, 401)
            from . import pending as pending_mod
            try:
                p = pending_mod.open_pending(self.cfg)
                try:
                    items = p.list("pending")
                finally:
                    p.close()
            except Exception as e:
                return self._send_json({"ok": False, "error": str(e)}, 500)
            # strip the bulky proposed profile; the UI only needs kind/questions/docs
            for it in items:
                it.pop("profile", None); it.pop("answers", None); it.pop("confirmed", None)
            return self._send_json({"ok": True, "count": len(items), "pending": items})
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
                                        "progress": self.server.ops.progress(),
                                        "health": self._health(),
                                        "commands": OPS_COMMANDS, "help": OPS_HELP})
            if path == "/ops/log":
                n = int((q.get("tail") or ["300"])[0] or 300)
                ap = getattr(self.server, "autopilot", None)
                return self._send_json({"ok": True, "log": self.server.ops.tail(n),
                                        "status": self.server.ops.status(),
                                        "progress": self.server.ops.progress(),
                                        "auto": ap.status() if ap else None,
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
                        "/serving/control", "/serving/minimal", "/serving/mode",
                        "/serving/schedule",
                        "/serving/model", "/serving/add",
                        "/serving/pull", "/serving/download", "/serving/tune", "/net",
                        "/metrics/mark", "/gaps/close", "/queue/delete", "/queue/clear",
                        "/pending/answer", "/pending/dismiss", "/settings/paths",
                        "/redistil"):
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
        if path == "/pending/answer":                  # 'Needs your input': confirm a deferred doc group
            from . import pending as pending_mod, structure as S
            try:
                rid = int(req.get("id"))
            except (TypeError, ValueError):
                return self._send_json({"ok": False, "error": "id required"}, 400)
            answers = req.get("answers")
            if not isinstance(answers, dict):
                return self._send_json({"ok": False, "error": "answers object required"}, 400)
            try:
                p = pending_mod.open_pending(self.cfg)
                try:
                    row = p.get(rid)
                    if not row:
                        return self._send_json({"ok": False, "error": "no such request"}, 404)
                    confirmed = S.apply_answers(row.get("profile") or {}, answers)
                    ok = p.answer(rid, answers, confirmed)
                finally:
                    p.close()
            except Exception as e:
                return self._send_json({"ok": False, "error": str(e)}, 500)
            if not ok:
                return self._send_json({"ok": False, "error": "already dismissed"}, 400)
            started = False
            if req.get("ingest_now"):                  # let the confirmed docs flow in now
                try:
                    self.server.ops.start("ingest", {})
                    started = True
                except Exception:
                    started = False
            return self._send_json({"ok": True, "kind": confirmed.get("kind"),
                                    "ingest_as": confirmed.get("ingest_as"),
                                    "docs": row.get("doc_count", 0), "ingest_started": started})
        if path == "/pending/dismiss":                 # 'Needs your input': set a request aside
            from . import pending as pending_mod
            try:
                rid = int(req.get("id"))
            except (TypeError, ValueError):
                return self._send_json({"ok": False, "error": "id required"}, 400)
            try:
                p = pending_mod.open_pending(self.cfg)
                try:
                    ok = p.dismiss(rid)
                finally:
                    p.close()
            except Exception as e:
                return self._send_json({"ok": False, "error": str(e)}, 500)
            return self._send_json({"ok": ok})
        if path == "/redistil":                        # Sources: re-queue low-yield documents
            kb = getattr(self.server, "kb", None)
            store = getattr(self.server, "store", None)
            if kb is None or store is None:
                return self._send_json({"ok": False, "error": "no KB/store loaded"}, 400)
            from . import yield_audit
            docs = [str(d) for d in (req.get("docs") or []) if str(d).strip()]
            if req.get("all_flagged"):
                try:
                    aud = yield_audit.audit(
                        store, kb,
                        min_chunks=int(req.get("min_chunks") or yield_audit.DEFAULT_MIN_CHUNKS),
                        ratio=float(req.get("ratio") or yield_audit.DEFAULT_RATIO),
                        limit=1000)
                except ValueError as e:
                    return self._send_json({"ok": False, "error": f"bad parameter: {e}"}, 400)
                docs = sorted(set(docs) | {f["doc_id"] for f in aud.get("flagged", [])})
            if not docs:
                return self._send_json({"ok": False, "error": "nothing to re-queue"}, 400)
            if self.server.ops.running():
                return self._send_json(
                    {"ok": False, "error": "a job is running — re-queueing under a live "
                     "distil pass would race its checkpoint; wait for it to finish"}, 409)
            res = yield_audit.reset_docs(store, kb, docs)
            log.info("redistil: %d document(s), %d chunk(s) re-queued", res.get("docs", 0),
                     res.get("chunks", 0))
            if res.get("ok") and req.get("distill"):
                try:
                    res["job"] = self.server.ops.start("distill", {})
                except ValueError as e:
                    res["job"] = {"ok": False, "error": str(e)}
            return self._send_json(res)
        if path == "/queue/delete":                    # Sources: drop a QUEUED doc + its chunks
            store = getattr(self.server, "store", None)
            if store is None or not hasattr(store, "purge_source"):
                return self._send_json({"ok": False, "error": "no chunk store on this box"}, 400)
            doc_id = str(req.get("doc_id") or "").strip()
            if not doc_id:
                return self._send_json({"ok": False, "error": "doc_id required"}, 400)
            # Only for QUEUED (undistilled) docs: a distilled doc is in the source registry
            # with nodes/cards/edges hanging off it — dropping just its chunks would orphan
            # those, so refuse and point at the heavier unimport/eject path.
            kb = getattr(self.server, "kb", None)
            if kb is not None:
                try:
                    seen = kb.db.execute("SELECT 1 FROM source_registry WHERE doc_id=?",
                                         (doc_id,)).fetchone()
                except Exception:
                    seen = None
                if seen:
                    return self._send_json(
                        {"ok": False, "error": "already distilled — remove it with "
                         "unimport/eject, not the queue"}, 409)
            res = store.purge_source(doc_id)
            log.info("queue delete: purged %s (%d chunk(s))", doc_id, res.get("chunks", 0))
            return self._send_json({"ok": True, **res})
        if path == "/queue/clear":                     # Sources: bulk-drop the whole queue
            store = getattr(self.server, "store", None)
            if store is None or not hasattr(store, "clear_queue"):
                return self._send_json({"ok": False, "error": "no chunk store on this box"}, 400)
            from . import ingest as ingest_mod
            include_partial = bool(req.get("include_partial"))
            quarantine = req.get("quarantine")
            quarantine = True if quarantine is None else bool(quarantine)   # default on
            dry_run = bool(req.get("dry_run"))
            res = ingest_mod.clear_ingest_queue(
                self.cfg, store, self.server.master_kb_path(),
                include_partial=include_partial, quarantine=quarantine, dry_run=dry_run)
            if not res.get("ok"):
                return self._send_json(res, 400)
            if not dry_run:
                q = res.get("quarantine") or {}
                log.info("queue clear: removed %d chunk(s) (untouched %d doc(s), partial=%s); "
                         "quarantined %d file(s), %d error(s)",
                         res.get("chunks_removed", 0), res.get("queued_docs", 0),
                         include_partial, q.get("moved", 0), q.get("errors", 0))
            return self._send_json(res)
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
            excl = [e for e in self.cfg["serving"]["llms"] if e.get("exclusive")]
            names = [str(e.get("name")) for e in excl]
            if name not in names:
                # accept the names a CLIENT knows: the model id or its
                # served_model_name alias (Vinkona's big_lm.model) — the
                # caller shouldn't need to learn this box's entry names
                hit = next((e for e in excl
                            if name and name in (str(e.get("model") or ""),
                                                 str(e.get("served_model_name") or ""))),
                           None)
                if hit is None:
                    return self._send_json(
                        {"ok": False, "error": f"'{name}' matches no exclusive "
                         "serving.llms entry, model or served_model_name "
                         f"(have: {', '.join(names) or 'none'})"}, 400)
                name = str(hit.get("name"))
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
        if path == "/serving/minimal":                 # vacate/restore VRAM, keep serving KB
            # Lets a remote (Vinkona) tell the box to release its GPU while still
            # answering kb_ask/kb_search.  Async, like the swap/control lanes.
            from . import supervisor as sup
            st = sup.read_state()
            if not sup.alive(st.get("supervisor", 0)):
                return self._send_json(
                    {"ok": False, "error": "the supervisor is not running "
                                           "(./vinur.sh start)"}, 409)
            action = str(req.get("action") or "")
            if action not in ("on", "off"):
                return self._send_json(
                    {"ok": False, "error": "action must be on|off"}, 400)
            summary = sup.apply_minimal(self.cfg, action)
            return self._send_json({"ok": True, **summary,
                                    "note": "the supervisor acts within a few seconds — "
                                            "re-poll /serving/status"})
        if path == "/serving/mode":                    # header selector: pin a posture / unset
            # 'automatic' hands control back to the schedule/prioritizer (and
            # reconciles immediately); full|minimal|endpoint PIN that posture —
            # no automated scheduler changes it until unset.  Endpoint is the
            # permanent yield-all (LM served to outside apps only, own jobs
            # held).  Flag flips live, survives restarts.
            from . import supervisor as sup
            st = sup.read_state()
            if not sup.alive(st.get("supervisor", 0)):
                return self._send_json(
                    {"ok": False, "error": "the supervisor is not running "
                                           "(./vinur.sh start)"}, 409)
            mode = str(req.get("mode") or "")
            if mode not in ("automatic", "full", "minimal", "endpoint"):
                return self._send_json(
                    {"ok": False, "error": "mode must be "
                                           "automatic|full|minimal|endpoint"}, 400)
            summary = sup.apply_override(self.cfg,
                                         None if mode == "automatic" else mode)
            return self._send_json({"ok": True, **summary,
                                    "note": "takes effect immediately — "
                                            "re-poll /stats"})
        if path == "/serving/schedule":                # save the weekly minimal-mode plan
            # Validated + normalised, then written to var/run/schedule.json; the
            # supervisor re-reads it live and flips minimal at window boundaries.
            from . import serving as sv
            from . import supervisor as sup
            cleaned = sv.clean_schedule(req if isinstance(req, dict) else {})
            sv.write_schedule(cleaned)
            return self._send_json(
                {"ok": True, "schedule": cleaned,
                 "note": f"saved — the supervisor applies it within ~{int(sup.SCHED_TICK_S)}s"})
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
        if path == "/serving/tune":                    # the Tune editor's save
            from . import serving as sv
            from .config import update_llm_entry
            cp = self.cfg.get("_config_path")
            if not cp:
                return self._send_json(
                    {"ok": False, "error": "server started without -c; no config file to write"}, 400)
            name = str(req.get("name") or "")
            entry = next((e for e in self.cfg["serving"]["llms"]
                          if str(e.get("name")) == name), None)
            if entry is None:
                return self._send_json(
                    {"ok": False, "error": f"'{name}' is not a serving.llms entry"}, 400)
            try:
                coerced = sv.validate_tuning(str(entry.get("engine") or ""),
                                             req.get("updates") or {})
            except ValueError as e:
                return self._send_json({"ok": False, "error": str(e)}, 400)
            if not coerced:
                return self._send_json({"ok": False, "error": "nothing to change"}, 400)
            # two destinations: model-fit knobs go to tune.toml BESIDE THE
            # WEIGHTS (they travel with the model); slot properties (swap
            # group, port, image, runtime) stay on the config.toml entry
            rows = {t["key"]: t for t in sv.TUNING_SCHEMA}
            model_lane = {k: v for k, v in coerced.items()
                          if rows[k].get("scope", "model") == "model"}
            entry_lane = {k: v for k, v in coerced.items()
                          if rows[k].get("scope", "model") == "entry"}
            if entry_lane.get("port") is not None and \
                    any(int(x.get("port") or 0) == entry_lane["port"] and x is not entry
                        for x in self.cfg["serving"]["llms"]):
                return self._send_json(
                    {"ok": False,
                     "error": f"port {entry_lane['port']} is taken by another entry"}, 400)
            bits = []
            tp = sv.tuning_path(entry)
            if model_lane and tp is None:
                # legacy lane: a hub-cache id with no folder of its own —
                # tuning stays on the entry until pull/adopt gives it a home
                entry_lane.update(model_lane)
                model_lane = {}
                bits.append("this model has no folder under models/, so tuning "
                            "stays on the config entry — pull or adopt it to make "
                            "tuning travel with the weights")
            if model_lane:
                # one-shot graceful upgrade: legacy model-fit keys still on the
                # entry migrate into the file alongside this save
                legacy = {k: entry[k] for k, t in rows.items()
                          if t.get("scope", "model") == "model"
                          and k in entry and k not in model_lane}
                try:
                    sv.write_model_tuning(tp, {**legacy, **model_lane})
                except OSError as e:
                    return self._send_json(
                        {"ok": False, "error": f"could not write {tp}: {e}"}, 400)
                entry_lane.update({k: None for k in set(legacy) | set(model_lane)
                                   if k in entry})
                if legacy:
                    bits.append(f"moved {len(legacy)} older setting(s) out of "
                                f"config.toml into {tp.name} — tuning now travels "
                                "with the model")
            if entry_lane:
                try:
                    update_llm_entry(cp, name, entry_lane)
                except (ValueError, OSError) as e:
                    return self._send_json({"ok": False, "error": str(e)}, 400)
                for k, v in entry_lane.items():        # the live view stays truthful
                    if v is None:
                        entry.pop(k, None)
                    else:
                        entry[k] = v
            topo = [k for k in coerced if rows[k].get("applies") == "supervisor"]
            bits.append("exclusive/boots-first/port/runtime change what the "
                        "supervisor froze at start — ./vinur.sh restart to apply"
                        if topo else
                        f"Restart llm-{name} (or its next swap-in) applies it")
            return self._send_json({"ok": True, "applied": coerced,
                                    "tune_file": str(tp or ""),
                                    "note": "saved — " + "; ".join(bits)})
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
            if engine == "container":
                # image/runtime copied from a sibling when one declares them;
                # otherwise the entry simply runs the built-in default image
                # (llm_argv fills it in) — a sibling running the default is a
                # working template too, so refusing here helped nobody
                tmpl = next((e for e in llms
                             if e.get("engine") == "container" and e.get("image")), None)
                if tmpl is not None:
                    entry["image"] = str(tmpl.get("image"))
                if tmpl is not None and tmpl.get("runtime"):
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
                command = req.get("command", "")
                run_args = dict(req.get("args") or {})
                answers = req.get("answers")           # collect: structured-text confirmations
                if command == "collect" and isinstance(answers, dict) and answers:
                    # too structured for a CLI flag — persist it and pass the path
                    run_args["answers_file"] = _write_answers_file(self.cfg, answers)
                return self._send_json(self.server.ops.start(command, run_args))
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
