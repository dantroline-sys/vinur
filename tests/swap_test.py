"""Exclusive GPU group + swap protocol: services_for autostart selection, the
/health readiness probe, the client-side ensure_active handshake, the kb
server's /serving/swap routes, and the autopilot's per-step "model" phase key.

The full supervisor swap (stop A, spawn B, wait ready) is exercised LIVE by
tests/swap_live.sh with stub llama-servers; this file covers the pieces that
are deterministic without processes.

Run:  python tests/swap_test.py     (stdlib only)
"""
import json
import os
import sys
import tempfile
import threading
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from knowledgehost import autopilot as ap
from knowledgehost import serving as sv
from knowledgehost.config import load_config
from knowledgehost.server import KnowledgeHostServer
from knowledgehost.supervisor import services_for

PASS = 0


def ok(label):
    global PASS
    PASS += 1
    print(f"  ok {PASS:2d}  {label}")


def excl_cfg(td):
    toml = Path(td) / "c.toml"
    toml.write_text('[[serving.llms]]\nname = "primary"\nengine = "vllm"\n'
                    'model = "a"\nserved_model_name = "brain"\n'
                    'port = 11438\nexclusive = true\n'
                    '[[serving.llms]]\nname = "secondary"\nengine = "vllm"\n'
                    'model = "b"\nport = 11435\nexclusive = true\ndefault = true\n'
                    '[[serving.llms]]\nname = "tiny"\nengine = "llama"\n'
                    'model = "m.gguf"\nport = 11441\n')
    return load_config(str(toml))


class Health(BaseHTTPRequestHandler):
    code = 503

    def do_GET(self):
        self.send_response(Health.code)
        self.send_header("Content-Length", "0")
        self.end_headers()

    def log_message(self, *_):
        pass


def main():
    with tempfile.TemporaryDirectory() as td:
        # ── services_for: exclusive group, default picks the boot model ──
        cfg = excl_cfg(td)
        svcs = services_for(cfg)
        auto = {s["name"]: s["autostart"] for s in svcs}
        assert auto == {"llm-primary": False, "llm-secondary": True,
                        "llm-tiny": True, "kb": True}, auto
        assert next(s for s in svcs if s["name"] == "llm-primary")["probe"] == \
            ("127.0.0.1", 11438)
        ok("exclusive group: only default=true autostarts; non-exclusive unaffected")

        cfg2 = excl_cfg(td)
        for e in cfg2["serving"]["llms"]:
            e.pop("default", None)
        auto2 = {s["name"]: s["autostart"] for s in services_for(cfg2)}
        assert auto2["llm-primary"] is True and auto2["llm-secondary"] is False
        ok("no default marked: first exclusive entry boots")

        assert cfg["serving"]["swap_timeout_s"] == 900
        ok("swap_timeout_s default present")

        # ── readiness probe: 503 (loading) vs 200 vs not listening ──────
        httpd = ThreadingHTTPServer(("127.0.0.1", 0), Health)
        threading.Thread(target=httpd.serve_forever, daemon=True).start()
        port = httpd.server_address[1]
        assert sv.probe_ready("127.0.0.1", port) is False        # 503: loading
        Health.code = 200
        assert sv.probe_ready("127.0.0.1", port) is True
        assert sv.probe_ready("127.0.0.1", 1) is False           # nothing there
        httpd.shutdown()
        ok("probe_ready: 200=ready, 503=loading, refused=down")

        # ── ensure_active handshake against a scripted 'supervisor' ─────
        req0, st0 = sv.SWAP_REQ, sv.SWAP_STATE
        sv.SWAP_REQ = Path(td) / "swap.req"
        sv.SWAP_STATE = Path(td) / "swap.state"
        try:
            try:
                sv.ensure_active("primary", timeout_s=1)
                raise AssertionError("no state file must raise")
            except RuntimeError as e:
                assert "supervisor" in str(e)
            ok("ensure_active without a supervisor fails fast")

            sv.SWAP_STATE.write_text(json.dumps({"active": "primary", "status": "ready"}))
            assert sv.ensure_active("primary", timeout_s=1)["active"] == "primary"
            assert not sv.SWAP_REQ.exists(), "already-active must not write a request"
            ok("ensure_active no-ops when the model is already resident")

            def fake_supervisor():
                for _ in range(100):
                    if sv.SWAP_REQ.exists():
                        break
                    time.sleep(0.02)
                want = json.loads(sv.SWAP_REQ.read_text())["name"]
                sv.SWAP_REQ.unlink()
                sv.SWAP_STATE.write_text(json.dumps(
                    {"active": "primary", "status": "swapping", "request": want}))
                time.sleep(0.1)
                sv.SWAP_STATE.write_text(json.dumps(
                    {"active": want, "status": "ready"}))

            t = threading.Thread(target=fake_supervisor)
            t.start()
            seen = []
            st = sv.ensure_active("secondary", timeout_s=5, poll_s=0.05,
                                  progress=lambda s: seen.append(s.get("status")))
            t.join()
            assert st["active"] == "secondary" and "swapping" in seen
            ok("ensure_active: request -> swapping -> ready observed")

            sv.SWAP_STATE.write_text(json.dumps(
                {"active": "primary", "status": "error",
                 "request": "secondary", "error": "boom"}))
            try:
                sv.ensure_active("secondary", timeout_s=1, poll_s=0.05)
                raise AssertionError("error state must raise")
            except RuntimeError as e:
                assert "boom" in str(e)
            ok("ensure_active surfaces the supervisor's error")

            # ── kb server routes ─────────────────────────────────────────
            scfg = excl_cfg(td)
            scfg.update({"host": "127.0.0.1", "port": 0, "auth_token": "tk",
                         "control_dir": str(Path(td) / "ctrl")})
            khs = KnowledgeHostServer(scfg, SimpleNamespace(), SimpleNamespace(), kb=None)
            kport = khs.server_address[1]
            threading.Thread(target=khs.serve_forever, daemon=True).start()

            def call(method, body=None):
                req = urllib.request.Request(
                    f"http://127.0.0.1:{kport}/serving/swap",
                    data=json.dumps(body).encode() if body is not None else None,
                    headers={"Authorization": "Bearer tk",
                             "Content-Type": "application/json"},
                    method=method)
                try:
                    with urllib.request.urlopen(req, timeout=5) as r:
                        return r.status, json.loads(r.read())
                except urllib.error.HTTPError as e:
                    return e.code, json.loads(e.read())

            sv.SWAP_STATE.write_text(json.dumps({"active": "primary", "status": "ready"}))
            code, res = call("GET")
            assert code == 200 and res["active"] == "primary"
            code, res = call("POST", {"name": "secondary"})
            assert code == 200 and res["requested"] == "secondary"
            assert json.loads(sv.SWAP_REQ.read_text())["name"] == "secondary"
            sv.SWAP_REQ.unlink()
            code, res = call("POST", {"name": "tiny"})       # not exclusive
            assert code == 400, (code, res)
            # aliases a CLIENT knows (Vinkona's auto-swap sends its model name):
            # the model id and served_model_name resolve to the entry
            code, res = call("POST", {"name": "b"})
            assert code == 200 and res["requested"] == "secondary", (code, res)
            sv.SWAP_REQ.unlink()
            code, res = call("POST", {"name": "brain"})
            assert code == 200 and res["requested"] == "primary", (code, res)
            sv.SWAP_REQ.unlink()
            code, res = call("POST", {"name": "ghost"})
            assert code == 400 and "served_model_name" in res["error"], (code, res)
            sv.SWAP_STATE.unlink()
            code, res = call("POST", {"name": "secondary"})  # no supervisor
            assert code == 409, (code, res)
            khs.shutdown()
            ok("/serving/swap: GET state, POST request, 400 non-exclusive, 409 no supervisor")

            # ── autopilot: per-step model key ───────────────────────────
            plan = ap.save_plan(scfg, {"steps": [
                {"command": "distill", "model": "primary"},
                {"command": "distill", "args": {}, "enabled": True}]})
            assert plan["steps"][0]["model"] == "primary"
            assert plan["steps"][1]["model"] == ""
            assert ap.step_key(plan["steps"][0]) != ap.step_key(plan["steps"][1])
            ok("save_plan keeps the model key; step identity includes it")

            # ── automatic model routing: verb lane -> exclusive entry ────
            assert sv.exclusive_entry_for_url(scfg, "http://127.0.0.1:11438") == "primary"
            assert sv.exclusive_entry_for_url(scfg, "http://localhost:11435") == "secondary"
            assert sv.exclusive_entry_for_url(scfg, "http://127.0.0.1:11441") is None
            assert sv.exclusive_entry_for_url(scfg, "http://10.0.0.7:11438") is None
            assert sv.exclusive_entry_for_url(scfg, "nonsense") is None
            ok("exclusive_entry_for_url: local port match; non-exclusive/foreign -> None")

            # ── entry_for_url: the any-entry lookup (distill fan-out needs the engine) ──
            e = sv.entry_for_url(scfg, "http://127.0.0.1:11441")
            assert e and e["name"] == "tiny" and e["engine"] == "llama"
            assert sv.entry_for_url(scfg, "http://0.0.0.0:11438")["name"] == "primary"
            assert sv.entry_for_url(scfg, "http://10.0.0.7:11441") is None
            assert sv.entry_for_url(scfg, "http://127.0.0.1:11441",
                                    exclusive_only=True) is None
            assert sv.entry_for_url({}, "http://127.0.0.1:11441") is None
            ok("entry_for_url: any entry incl. non-exclusive; engine visible; stub cfg safe")

            assert ap.auto_model(scfg, "distill") == "primary"
            assert ap.auto_model(scfg, "refine") == "primary"
            assert ap.auto_model(scfg, "link") == "primary"
            assert ap.auto_model(scfg, "link", {"fast": True}) == "secondary"
            assert ap.auto_model(scfg, "adjudicate", {"fast": True}) == "secondary"
            assert ap.auto_model(scfg, "ingest") is None
            assert ap.auto_model(scfg, "ingest", {"distill": True}) == "primary"
            assert ap.auto_model(scfg, "import-conceptnet") is None
            noext = json.loads(json.dumps(scfg))
            noext["extract_urls"] = []
            assert ap.auto_model(noext, "link", {"fast": True}) == "primary"
            plaincfg = json.loads(json.dumps(scfg))
            plaincfg["serving"]["llms"] = []
            assert ap.auto_model(plaincfg, "distill") is None
            ok("auto_model: lanes map to entries; fast flag, fallbacks, no-serving -> None")

            p2 = ap.save_plan(scfg, {"auto_models": False, "steps": plan["steps"]})
            assert p2["auto_models"] is False
            assert ap.load_plan(scfg)["auto_models"] is False, "persisted flag survives"
            p3 = ap.save_plan(scfg, {"steps": plan["steps"]})
            assert p3["auto_models"] is True, "default is on"
            ok("auto_models plan flag: persists, defaults on, load_plan backfills")

            # Dead-port config for the hold tests: ports 1/9 answer nothing even
            # on a box whose REAL services occupy the standard 11438/11435 —
            # the residency-evidence fallback must not turn holds into runs.
            deadcfg = json.loads(json.dumps(scfg))
            deadcfg["serving"]["llms"][0]["port"] = 1       # primary
            deadcfg["serving"]["llms"][1]["port"] = 9       # secondary
            deadcfg["distill_urls"] = ["http://127.0.0.1:1"]
            deadcfg["extract_urls"] = ["http://127.0.0.1:9"]
            calls = []
            pilot = ap.Autopilot(deadcfg, SimpleNamespace(
                start=lambda c, a: calls.append(c) or {"ok": True},
                running=lambda: False, result=lambda: {},
                status=lambda: {"exit_code": 0}))
            sv.SWAP_STATE.write_text(json.dumps({"active": "primary", "status": "ready"}))
            pilot._run_step({"command": "distill", "model": "primary", "label": "d"},
                            {"idle_interval_s": 60})
            assert calls == ["distill"], "resident model: step must run without a swap"
            ok("autopilot runs the step when its model is already resident")

            sv.SWAP_STATE.unlink()
            pilot._run_step({"command": "distill", "model": "secondary", "label": "d2"},
                            {"idle_interval_s": 60})
            assert calls == ["distill"], "failed swap must NOT launch the verb"
            assert "swap to secondary failed" in pilot._state["last_reason"]
            assert pilot._hold_until, "failed swap must back the step off"
            ok("autopilot holds the step when the swap fails (verb never launched)")

            # ── _run_step auto-routing: no model key needed on the step ──
            calls.clear()                        # SWAP_STATE is still absent here:
            pilot._run_step({"command": "distill", "label": "auto"},
                            {"idle_interval_s": 60, "auto_models": True})
            assert calls == [], "derived model unavailable -> verb must not launch"
            assert "swap to primary failed" in pilot._state["last_reason"], \
                pilot._state["last_reason"]      # ...which proves 'primary' was derived
            calls.clear()
            pilot._run_step({"command": "stats", "label": "s"},
                            {"idle_interval_s": 60, "auto_models": True})
            assert calls == ["stats"], "embed-only verb: no model derived, runs freely"
            calls.clear()
            pilot._run_step({"command": "distill", "label": "off"},
                            {"idle_interval_s": 60, "auto_models": False})
            assert calls == ["distill"], "auto_models off: legacy behavior"
            ok("autopilot derives per-verb models; embed-only and opt-out unaffected")

            # ── residency evidence beats a broken handshake ──────────────
            # No swap.state (unsupervised/manual serving), but the entry's own
            # endpoint answers /health: the verb must run, not hold — this is
            # what keeps card generation alive on a manually-run container.
            class OkHealth(BaseHTTPRequestHandler):
                def do_GET(self):
                    self.send_response(200)
                    self.send_header("Content-Length", "0")
                    self.end_headers()

                def log_message(self, *_):
                    pass
            hsrv = ThreadingHTTPServer(("127.0.0.1", 0), OkHealth)
            threading.Thread(target=hsrv.serve_forever, daemon=True).start()
            hport = hsrv.server_address[1]
            livecfg = json.loads(json.dumps(scfg))
            livecfg["serving"]["llms"][0]["port"] = hport   # primary answers here
            livecfg["distill_urls"] = [f"http://127.0.0.1:{hport}"]
            calls.clear()
            pilot_live = ap.Autopilot(livecfg, SimpleNamespace(
                start=lambda c, a: calls.append(c) or {"ok": True},
                running=lambda: False, result=lambda: {},
                status=lambda: {"exit_code": 0}))
            assert not sv.SWAP_STATE.exists()
            pilot_live._run_step({"command": "distill", "label": "manual"},
                                 {"idle_interval_s": 60, "auto_models": True})
            assert calls == ["distill"], "answering endpoint: verb must run"
            # (the transient 'proceeding' reason is replaced on completion)
            assert pilot_live._state["last_reason"] == "ran distill", \
                pilot_live._state["last_reason"]
            hsrv.shutdown()
            ok("handshake down but model answering -> verb runs (manual serving works)")
            # ── serving_status: weights on disk + service states ────────
            from knowledgehost import supervisor as sup
            state0, sup_state0 = os.environ.get("HF_HOME"), sup.STATE
            os.environ["HF_HOME"] = str(Path(td) / "hf")
            sup.STATE = Path(td) / "sup.json"          # no supervisor state file
            try:
                hub = Path(td) / "hf" / "hub"
                ok_repo = hub / "models--org--good" / "snapshots" / "s1"
                ok_repo.mkdir(parents=True)
                (ok_repo / "config.json").write_text("{}")
                (ok_repo / "model.safetensors").write_bytes(b"x" * 1024)
                bad_repo = hub / "models--org--stuck" / "blobs"
                bad_repo.mkdir(parents=True)
                (bad_repo / "abc.incomplete").write_bytes(b"x")

                gguf2 = Path(td) / "w.gguf"
                gguf2.write_bytes(b"GGUF")
                wtoml = Path(td) / "w.toml"
                wtoml.write_text(
                    '[[serving.llms]]\nname = "good"\nengine = "vllm"\n'
                    'model = "org/good"\nport = 1\nexclusive = true\ndefault = true\n'
                    '[[serving.llms]]\nname = "stuck"\nengine = "vllm"\n'
                    'model = "org/stuck"\nport = 2\nexclusive = true\n'
                    '[[serving.llms]]\nname = "nowhere"\nengine = "vllm"\n'
                    'model = "org/absent"\nport = 3\n'
                    f'[[serving.llms]]\nname = "gg"\nengine = "llama"\n'
                    f'model = "{gguf2}"\nport = 4\n')
                wcfg = load_config(str(wtoml))
                res = sv.serving_status(wcfg)
                by = {m["name"]: m for m in res["llms"]}
                assert res["hosting"] is True and res["supervisor"]["running"] is False
                assert by["good"]["weights"]["status"] == "ready"
                assert by["stuck"]["weights"]["status"] == "incomplete"
                assert "mid-download" in by["stuck"]["weights"]["detail"]
                assert by["nowhere"]["weights"]["status"] == "missing"
                assert "pull --model org/absent" in by["nowhere"]["weights"]["detail"], \
                    "a missing model names the broker pull, not an engine download"
                assert by["gg"]["weights"]["status"] == "ready"
                assert all(m["service"] == "supervisor-down" for m in res["llms"])
                assert isinstance(res.get("unserved"), list), \
                    "the Add-service list must always be present"
                ok("serving_status: ready / mid-download / missing weights + supervisor-down")

                # ── stale .incomplete litter must NOT mask a complete snapshot ──
                # (interrupted first fetch leaves blobs/*.incomplete; the retry
                # completes under a fresh temp name, so the litter outlives it)
                lit = hub / "models--org--good" / "blobs"
                lit.mkdir(parents=True, exist_ok=True)
                (lit / "old.incomplete").write_bytes(b"x")
                ws = sv.weights_status("container", "org/good")
                assert ws["status"] == "ready" and "stale" in ws["detail"], ws
                # sharded: every shard the index names must resolve, else incomplete
                snap = hub / "models--org--good" / "snapshots" / "s1"
                (snap / "model.safetensors.index.json").write_text(json.dumps(
                    {"weight_map": {"a": "model-00001-of-00002.safetensors",
                                    "b": "model-00002-of-00002.safetensors"}}))
                (snap / "model-00001-of-00002.safetensors").write_bytes(b"x")
                assert sv.weights_status("vllm", "org/good")["status"] == "incomplete"
                (snap / "model-00002-of-00002.safetensors").write_bytes(b"x")
                assert sv.weights_status("vllm", "org/good")["status"] == "ready"
                ok("weights_status: ready beats stale .incomplete; index shards all checked")

                # ── the cache location is REPORTED, not folklore ────────────
                # ("where did the 200 GB go?" is answered in the panel)
                cache = sv.hf_cache_status()
                assert cache["path"] == str(hub) and cache["exists"] is True, cache
                assert cache["repos"] == 2, cache          # good + stuck
                assert cache["incomplete_gb"] >= 0 and cache["env"] == "HF_HOME", cache
                assert res.get("cache") or sv.serving_status(wcfg)["cache"]["path"] == str(hub)
                ok("hf_cache_status: hub path, repo count and stale-partial bytes reported")

                # ── a download that stopped is not a download in progress ───
                stuck_blobs = hub / "models--org--stuck" / "blobs"
                old = time.time() - 3600
                os.utime(stuck_blobs / "abc.incomplete", (old, old))
                ws = sv.weights_status("vllm", "org/stuck")
                assert ws["status"] == "stalled" and ws["idle_s"] >= 3500, ws
                assert "NOTHING has been written" in ws["detail"], ws
                fresh = time.time()
                os.utime(stuck_blobs / "abc.incomplete", (fresh, fresh))
                assert sv.weights_status("vllm", "org/stuck")["status"] == "incomplete"
                ok("weights_status: stalled vs downloading decided by partial-file mtime")

                # ── the line that NAMES the failure, not the last one printed ─
                vllm_tail = (
                    "INFO 07-20 12:00:00 [api_server.py:1] vLLM API server version 0.11\n"
                    "  Value error, speculative_config must be a JSON object\n"
                    "For further information visit https://errors.pydantic.dev/2.13/v/value_error\n"
                    "(APIServer pid=1) For further information visit "
                    "https://errors.pydantic.dev/2.13/v/value_error\n")
                cause = sv.cause_lines(vllm_tail)
                assert cause and "speculative_config" in cause[-1], cause
                assert not any("pydantic.dev" in c for c in cause), cause
                assert "REJECTED ITS OWN CONFIG" in (sv.failure_hint(vllm_tail) or "")
                ok("cause_lines: pydantic 'Value error' beats the docs-URL sign-off")

                assert "rate-limited" in (sv.failure_hint("429 Client Error: Too Many") or "")
                assert "disk" in (sv.failure_hint("No space left on device") or "")
                ok("failure_hint covers the download failures (429, disk, timeout)")

                # ── per-service control requests round-trip ─────────────────
                req0 = sv.SVC_REQ_DIR
                sv.SVC_REQ_DIR = Path(td) / "svcreq"
                try:
                    sv.request_service("llm-good", "stop")
                    sv.request_service("embed", "restart")
                    got = {d["service"]: d["action"] for d in sv.take_service_requests()}
                    assert got == {"llm-good": "stop", "embed": "restart"}, got
                    assert sv.take_service_requests() == []   # consumed, not replayed
                    # one file per service: a second press supersedes, never
                    # collides with another service's pending request
                    sv.request_service("llm-good", "stop")
                    sv.request_service("llm-good", "start")
                    assert [d["action"] for d in sv.take_service_requests()] == ["start"]
                    for bad in ("../etc/passwd", "", "a b"):
                        try:
                            sv.request_service(bad, "stop"); raise AssertionError(bad)
                        except ValueError:
                            pass
                    try:
                        sv.request_service("embed", "obliterate"); raise AssertionError("action")
                    except ValueError:
                        pass
                finally:
                    sv.SVC_REQ_DIR = req0
                ok("request_service: per-service files, consumed once, names/actions validated")

                # ── what the supervisor DOES with each request ──────────────
                plain = {"name": "embed", "exclusive": False, "entry": ""}
                excl_a = {"name": "llm-a", "exclusive": True, "entry": "a"}
                excl_b = {"name": "llm-b", "exclusive": True, "entry": "b"}
                P = sup.control_plan
                # a stop must STICK — hold is what stops the watchdog reviving it
                assert P("stop", plain, running=True, active_excl="a") == \
                    ["stop", "hold", "clear"]
                # start after a give-up verdict = "try again": clear, then spawn
                assert P("start", plain, running=False, active_excl="a") == \
                    ["unhold", "clear", "spawn"]
                # already up: start un-holds and clears but must NOT double-spawn
                assert P("start", plain, running=True, active_excl="a") == ["unhold", "clear"]
                assert P("restart", plain, running=True, active_excl="a") == \
                    ["stop", "unhold", "clear", "spawn"]
                # an exclusive sibling can't be spawned beside the resident one —
                # its VRAM is taken, so the request becomes a swap
                assert P("start", excl_b, running=False, active_excl="a") == \
                    ["unhold", "clear", "swap"]
                assert P("restart", excl_a, running=True, active_excl="a") == \
                    ["stop", "unhold", "clear", "spawn"]
                ok("control_plan: stop holds, start clears+revives, exclusive start -> swap")

                sup.STATE.write_text(json.dumps({
                    "supervisor": os.getpid(),           # a live pid
                    "services": {"llm-good": os.getpid(), "llm-gg": 999999},
                    "standby": {"stuck": "llm-stuck"},
                    "held": ["llm-nowhere"],
                    "failed": {"llm-nowhere": "gave up after 5 restarts"}}))
                by0 = {m["name"]: m for m in sv.serving_status(wcfg)["llms"]}
                # held wins over failed: it was stopped on purpose, and the
                # panel must offer Start rather than shouting about a crash
                assert by0["nowhere"]["service"] == "stopped", by0["nowhere"]
                assert by0["good"]["service_name"] == "llm-good"
                ok("serving_status: a held service reads 'stopped', rows carry service_name")

                sup.STATE.write_text(json.dumps({
                    "supervisor": os.getpid(),           # a live pid
                    "services": {"llm-good": os.getpid(), "llm-gg": 999999},
                    "standby": {"stuck": "llm-stuck"},
                    "failed": {"llm-nowhere": "gave up after 5 restarts"}}))
                res = sv.serving_status(wcfg)
                by = {m["name"]: m for m in res["llms"]}
                assert by["good"]["service"] == "up"
                assert by["stuck"]["service"] == "standby"
                assert by["nowhere"]["service"] == "failed" and "gave up" in by["nowhere"]["reason"]
                assert by["gg"]["service"] == "dead"
                ok("serving_status: up / standby / failed(reason) / dead from supervisor state")

                # the route the panel polls
                scfg2 = dict(wcfg)
                scfg2.update({"host": "127.0.0.1", "port": 0, "auth_token": "tk",
                              "control_dir": str(Path(td) / "ctrl2")})
                khs2 = KnowledgeHostServer(scfg2, SimpleNamespace(), SimpleNamespace(), kb=None)
                threading.Thread(target=khs2.serve_forever, daemon=True).start()
                req = urllib.request.Request(
                    f"http://127.0.0.1:{khs2.server_address[1]}/serving/status",
                    headers={"Authorization": "Bearer tk"})
                with urllib.request.urlopen(req, timeout=5) as r:
                    body = json.loads(r.read())
                assert body["ok"] and body["hosting"] and len(body["llms"]) == 4
                ok("GET /serving/status serves the panel payload (authed)")

                # ── the Serving tab's buttons: control + log tail over HTTP ──
                base = f"http://127.0.0.1:{khs2.server_address[1]}"

                def call(p, obj=None):
                    rq = urllib.request.Request(
                        base + p, headers={"Authorization": "Bearer tk",
                                           "Content-Type": "application/json"},
                        data=json.dumps(obj).encode() if obj is not None else None)
                    try:
                        with urllib.request.urlopen(rq, timeout=5) as r:
                            return r.status, json.loads(r.read())
                    except urllib.error.HTTPError as e:
                        return e.code, json.loads(e.read())

                req0 = sv.SVC_REQ_DIR
                sv.SVC_REQ_DIR = Path(td) / "svcreq-http"
                try:
                    code, b = call("/serving/control",
                                   {"service": "llm-good", "action": "restart"})
                    assert code == 200 and b["ok"], (code, b)
                    pending = {d["service"]: d["action"] for d in sv.take_service_requests()}
                    assert pending == {"llm-good": "restart"}, pending
                    # a service the supervisor doesn't know must FAIL, not
                    # silently queue a request nothing will ever act on
                    code, b = call("/serving/control", {"service": "ghost", "action": "stop"})
                    assert code == 400 and "no such service" in b["error"], (code, b)
                    assert sv.take_service_requests() == []
                    # standby and failed services are addressable too (that is
                    # exactly when you need Start)
                    for svc_name in ("llm-stuck", "llm-nowhere"):
                        code, b = call("/serving/control",
                                       {"service": svc_name, "action": "start"})
                        assert code == 200 and b["ok"], (svc_name, code, b)
                    sv.take_service_requests()
                finally:
                    sv.SVC_REQ_DIR = req0

                (sup.LOGS).mkdir(parents=True, exist_ok=True)
                logp = sup.LOGS / "llm-good.log"
                keep = logp.read_bytes() if logp.exists() else None
                logp.write_text("".join(f"line {i}\n" for i in range(500)))
                try:
                    code, b = call("/serving/log?name=llm-good&n=5")
                    assert code == 200 and b["ok"], (code, b)
                    assert b["text"].splitlines() == [f"line {i}" for i in range(495, 500)], b
                    code, b = call("/serving/log?name=../../etc/passwd")
                    assert code == 400 and "bad service name" in b["error"], (code, b)
                finally:
                    if keep is None:
                        logp.unlink()
                    else:
                        logp.write_bytes(keep)
                khs2.shutdown()
                ok("POST /serving/control + GET /serving/log: buttons and log tail wired")

                # ── known-failure hints + toolkit preflight ──────────────
                assert "toolkit missing" in sv.failure_hint(
                    "x\nRuntimeError: Could not find nvcc and default cuda_home...")
                assert "gated HF repo" in sv.failure_hint("401 Client Error: Unauthorized")
                assert "VRAM" in sv.failure_hint("torch.cuda: CUDA out of memory")
                assert "NVCC_APPEND_FLAGS" in sv.failure_hint(
                    "#error -- unsupported GNU version! gcc versions later than 15")
                assert sv.failure_hint("something novel") is None
                ok("failure_hint maps known crash signatures, ignores the rest")

                # a dead vllm service whose log tail holds the nvcc error gets the hint
                logs0, sup.LOGS = sup.LOGS, Path(td) / "logs"
                try:
                    sup.LOGS.mkdir(parents=True, exist_ok=True)
                    (sup.LOGS / "llm-good.log").write_text(
                        "RuntimeError: Could not find nvcc and default cuda_home="
                        "'/usr/local/cuda' doesn't exist\n[rank0] NCCL teardown noise\n")
                    sup.STATE.write_text(json.dumps({
                        "supervisor": os.getpid(),
                        "services": {"llm-good": 999999}, "standby": {}, "failed": {}}))
                    res2 = sv.serving_status(wcfg)
                    dead = {m["name"]: m for m in res2["llms"]}["good"]
                    assert dead["service"] == "dead", dead
                    assert "toolkit missing" in dead.get("hint", ""), dead
                    ok("serving_status attaches the hint from the dead service's log tail")
                finally:
                    sup.LOGS = logs0

                fp4cfg = load_config()
                fp4cfg["serving"] = {**fp4cfg["serving"], "llms": [
                    {"name": "big", "engine": "vllm", "model": "org/M-NVFP4", "port": 1}]}
                w = sv.toolkit_warning(fp4cfg, toolkit_present=False)
                assert w and "WILL fail" in w and "big" in w
                assert sv.toolkit_warning(fp4cfg, toolkit_present=True) is None
                fp4cfg["serving"]["llms"][0]["model"] = "org/M-FP8"
                w = sv.toolkit_warning(fp4cfg, toolkit_present=False)
                assert w and "WILL fail" not in w
                fp4cfg["serving"]["llms"] = []
                assert sv.toolkit_warning(fp4cfg, toolkit_present=False) is None
                ok("toolkit_warning: loud for NVFP4/modelopt, gentle otherwise, quiet w/o vllm")
            finally:
                sup.STATE = sup_state0
                if state0 is None:
                    os.environ.pop("HF_HOME", None)
                else:
                    os.environ["HF_HOME"] = state0
        finally:
            sv.SWAP_REQ, sv.SWAP_STATE = req0, st0

    # ── container stop is authoritative: runtime stop by NAME, never killpg ──
    # (killing the attached podman/docker client orphans the workload: conmon/
    # containerd owns it, so the model keeps its VRAM and the next exclusive
    # load fails its free-memory check — the swap-doesn't-unload bug.)
    import subprocess

    with tempfile.TemporaryDirectory() as td:
        calls = Path(td) / "calls.log"
        rt = Path(td) / "fakert"
        rt.write_text("#!/bin/sh\n"
                      f"echo \"$@\" >> {calls}\n"
                      "if [ \"$1\" = ps ]; then echo running; fi\n")
        rt.chmod(0o755)
        rt_quiet = Path(td) / "fakert-quiet"
        rt_quiet.write_text("#!/bin/sh\nexit 0\n")
        rt_quiet.chmod(0o755)

        ccfg = load_config()
        ccfg["serving"] = {**ccfg["serving"], "llms": [
            {"name": "big", "engine": "container", "model": "org/m", "port": 1,
             "exclusive": True, "runtime": str(rt)},
            {"name": "bare", "engine": "vllm", "model": "org/m2", "port": 2}]}
        assert sv.container_name("big") == "vinur-llm-big"
        ref = sv.container_ref(ccfg, "big")
        assert ref == (str(rt), "vinur-llm-big")
        assert sv.container_ref(ccfg, "bare") is None
        assert sv.container_ref(ccfg, "nope") is None
        argv = sv.llm_argv(ccfg["serving"]["llms"][0])
        assert argv[0] == str(rt) and "vinur-llm-big" in argv, \
            "llm_argv must name the container with container_name()"
        ok("container_name/container_ref: shared handle; bare/unknown -> None")

        # image provenance ENV (VLLM_BUILD_*) trips vLLM's own unknown-var
        # warning — podman strips it at run; docker has no unset flag
        pod = Path(td) / "podman"
        pod.write_text(rt.read_text())
        pod.chmod(0o755)
        pargv = sv.llm_argv({**ccfg["serving"]["llms"][0], "runtime": str(pod)})
        assert pargv.count("--unsetenv") == len(sv._IMAGE_NOISE_ENV)
        for k in sv._IMAGE_NOISE_ENV:
            assert pargv[pargv.index(k) - 1] == "--unsetenv", k
        assert "--unsetenv" not in argv, "docker path has no unset flag"
        ok("podman argv --unsetenv's the image's VLLM_BUILD_* provenance noise")

        svcs2 = services_for(ccfg)
        big = next(s for s in svcs2 if s["name"] == "llm-big")
        bare = next(s for s in svcs2 if s["name"] == "llm-bare")
        assert big["container"] == ref and bare["container"] is None
        ok("services_for carries the (runtime, container-name) stop handle")

        sup._stop_container(ref)
        got = [ln.split() for ln in calls.read_text().splitlines()]
        assert got[0] == ["stop", "-t", str(sup.CONTAINER_STOP_S), "vinur-llm-big"]
        assert got[1] == ["rm", "-f", "vinur-llm-big"]
        ok("_stop_container: <runtime> stop -t then rm -f, by name")

        assert sup._container_alive(ref) is True
        assert sup._container_alive((str(rt_quiet), "vinur-llm-big")) is False
        ok("_container_alive: runtime ps output decides; quiet/missing -> False")

        calls.write_text("")
        sup._stop_one({}, {"name": "llm-big", "container": ref})
        assert any(ln.startswith("stop -t") for ln in calls.read_text().splitlines()), \
            "an untracked (zombie-client) container service must still be stopped"
        p = subprocess.Popen(["sleep", "30"], start_new_session=True)
        procs2 = {"llm-bare": p}
        t0 = time.time()
        sup._stop_one(procs2, {"name": "llm-bare", "container": None})
        assert p.poll() is not None and not procs2 and time.time() - t0 < sup.GRACE_S
        ok("_stop_one: container svc -> runtime stop even with no client; bare -> killpg")

    # ── engines run OFFLINE: acquisition moved to the egress broker ─────────
    envkeys = ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN")
    saved_env = {k: os.environ.pop(k, None) for k in envkeys}
    try:
        hcfg = {"hf_token": "hf_abc123"}
        for eng in ("container", "vllm"):
            e = sv.hf_env(hcfg, eng)
            assert e["HF_HUB_OFFLINE"] == "1" and e["TRANSFORMERS_OFFLINE"] == "1", e
            # the null endpoint: even a code path that ignores the offline
            # flags dials a dead loopback port, not the hub
            assert e["HF_ENDPOINT"].startswith("http://127.0.0.1:"), e
            assert e["VLLM_DO_NOT_TRACK"] == "1" and e["HF_HUB_DISABLE_TELEMETRY"] == "1"
            assert e["VLLM_NO_USAGE_STATS"] == "1" and e["DO_NOT_TRACK"] == "1", \
                "phone-home stats must be off at launch (B-14)"
            assert "HF_TOKEN" not in e and "HUGGING_FACE_HUB_TOKEN" not in e, \
                "engines never hold the token — the broker attaches it to pulls"
        assert sv.hf_env(hcfg, "llama") == {}, "llama engines take local GGUFs only"
        os.environ["HF_TOKEN"] = "hf_fromhost"
        assert "HF_TOKEN" not in sv.hf_env({}, "container"), \
            "a host-env token must not leak into an engine either"
    finally:
        for k, v in saved_env.items():
            os.environ.pop(k, None)
            if v is not None:
                os.environ[k] = v
    red = sv.redact_argv(["podman", "run", "-e", "HF_TOKEN=hf_abc123",
                          "-e", "MY_API_KEY=zz", "-e", "FOO=1", "image"])
    assert red[3] == "HF_TOKEN=***" and red[5] == "MY_API_KEY=***" and red[7] == "FOO=1"
    assert "hf_abc123" not in " ".join(red) and "zz" not in red[5]
    from knowledgehost.config import settings_schema
    assert "hf_token" not in settings_schema(), \
        "a secret must never surface in the panel schema"
    # a proxy URL can carry credentials: the GENERIC schema never carries it —
    # the Network tab lane (net_view/set_net_setting) is the deliberate,
    # REDACTED exception, tested below
    assert "http_proxy" not in settings_schema()
    ok("hf_env: engines offline + statless, no tokens; secrets stay off panel/logs")

    # ── the model store: resolve_model + container mount ────────────────────
    with tempfile.TemporaryDirectory() as td3:
        root3 = Path(td3)
        stored = root3 / "models" / "org--tiny"
        stored.mkdir(parents=True)
        (stored / "config.json").write_text("{}")
        (stored / "model.safetensors").write_bytes(b"w" * 64)
        import json as _json
        (stored / ".pull.json").write_text(_json.dumps(
            {"model": "org/tiny", "files": {
                "config.json": {"size": 2}, "model.safetensors": {"size": 64}}}))
        ent = {"name": "t", "engine": "vllm", "model": "org/tiny", "port": 1}
        r = sv.resolve_model(ent, root=root3)
        assert r["model"] == str(stored), r
        assert sv.resolve_model({**ent, "engine": "llama"}, root=root3)["model"] == "org/tiny"
        # an incomplete pull must NOT resolve (engine would boot on half a model)
        (stored / "model.safetensors").write_bytes(b"w" * 10)
        assert sv.resolve_model(ent, root=root3)["model"] == "org/tiny"
        (stored / "model.safetensors").write_bytes(b"w" * 64)
        # container engine: a local-dir model is mounted read-only at /model
        vdir3 = root3 / "serving" / ".venv" / "bin"; vdir3.mkdir(parents=True)
        (vdir3 / "vllm").write_text("")
        centry3 = {"name": "t", "engine": "container", "model": str(stored),
                   "port": 11438, "runtime": "podman", "image": "img:v1"}
        argv3 = sv.llm_argv(centry3, root=root3)
        s3 = " ".join(argv3)
        assert f"-v {stored}:/model:ro,z" in s3, s3
        assert argv3[argv3.index("img:v1") + 1] == "/model", argv3
    ok("resolve_model: pulled store wins, incomplete never resolves; container mounts ro")

    # ── eligible_models: the Serving tab's picker (disk only, per engine) ────
    with tempfile.TemporaryDirectory() as td4:
        root4 = Path(td4)
        st4 = root4 / "models" / "org--dense"
        st4.mkdir(parents=True)
        (st4 / "model.safetensors").write_bytes(b"w" * 32)
        import json as _json
        (st4 / ".pull.json").write_text(_json.dumps(
            {"model": "org/dense", "files": {"model.safetensors": {"size": 32}}}))
        half = root4 / "models" / "org--half"          # incomplete: never offered
        half.mkdir()
        (half / "model.safetensors").write_bytes(b"w")
        (half / ".pull.json").write_text(_json.dumps(
            {"model": "org/half", "files": {"model.safetensors": {"size": 999}}}))
        gg = root4 / "models" / "org--tiny-GGUF"
        gg.mkdir()
        (gg / "tiny-Q4_K_M.gguf").write_bytes(b"g" * 16)
        (gg / "big-00001-of-00002.gguf").write_bytes(b"g")
        (gg / "big-00002-of-00002.gguf").write_bytes(b"g")
        (gg / ".pull.json").write_text(_json.dumps(
            {"model": "org/tiny-GGUF", "files": {"tiny-Q4_K_M.gguf": {"size": 16}}}))
        (root4 / "models" / "nomic-embed-text-v1.5.f16.gguf").write_bytes(b"e")
        vll = sv.eligible_models("vllm", root=root4)
        assert [c["model"] for c in vll] == ["org/dense"], vll
        lla = [c["model"] for c in sv.eligible_models("llama", root=root4)]
        assert "models/org--tiny-GGUF/tiny-Q4_K_M.gguf" in lla, lla
        assert "models/org--tiny-GGUF/big-00001-of-00002.gguf" in lla, lla
        assert not any("00002-of" in m for m in lla), lla
        assert not any("nomic-embed" in m for m in lla), lla
    ok("eligible_models: complete safetensors stores for vllm; GGUFs for llama "
       "(first split part only, embed model excluded)")

    # ── a broker pull in flight is a LIVE download on the Serving tab ───────
    with tempfile.TemporaryDirectory() as td6:
        root6 = Path(td6)
        sd6 = root6 / "models" / "org--pulling"
        sd6.mkdir(parents=True)
        (sd6 / "config.json").write_bytes(b"{}")
        (sd6 / "model.safetensors.part").write_bytes(b"x" * 2048)
        root0 = sv.ROOT
        try:
            sv.ROOT = root6
            ws = sv.weights_status("vllm", "org/pulling")
            assert ws["status"] == "incomplete" and "downloading now" in ws["detail"], ws
            assert "1 file(s) done, 1 in flight" in ws["detail"], ws
            # the manifest is written first, so mid-pull the tab knows the %
            (sd6 / ".pull.json").write_text(_json.dumps(
                {"model": "org/pulling", "files": {
                    "config.json": {"size": 2},
                    "model.safetensors": {"size": 4096}}}))
            ws = sv.weights_status("vllm", "org/pulling")
            assert ws["pct"] == 50 and "(50%)" in ws["detail"], ws
            old6 = time.time() - 3600
            os.utime(sd6 / "model.safetensors.part", (old6, old6))
            ws = sv.weights_status("vllm", "org/pulling")
            assert ws["status"] == "stalled" and ws["idle_s"] >= 3500, ws
            assert "re-run it" in ws["detail"], ws
            (sd6 / "model.safetensors.part").unlink()
            os.utime(sd6 / "config.json", (old6, old6))   # nothing fresh left:
            ws = sv.weights_status("vllm", "org/pulling")  # that's INTERRUPTED
            assert ws["status"] == "incomplete" and "never finished" in ws["detail"], ws
            assert sv.weights_status("vllm", "org/absent6")["status"] == "missing"
        finally:
            sv.ROOT = root0
    ok("weights_status: a store pull reads downloading / stalled / interrupted "
       "live — never 'missing' mid-pull")

    # ── update_llm_model: the picker's config.toml rewrite ───────────────────
    from knowledgehost.config import update_llm_model
    with tempfile.TemporaryDirectory() as td5:
        cp5 = Path(td5) / "config.toml"
        cp5.write_text('port = 8770\n'
                       '[[serving.llms]]\n'
                       'name   = "primary"\n'
                       'model  = "org/old"     # the resident model\n'
                       'port   = 11438\n'
                       '[[serving.llms]]\n'
                       'name = "secondary"\n'
                       "model = 'org/other'\n"
                       'port = 11439\n'
                       '[serving.embed]\n'
                       'enabled = true\n')
        old = update_llm_model(str(cp5), "primary", "org/new")
        assert old == "org/old"
        txt = cp5.read_text()
        assert 'model  = "org/new"     # the resident model' in txt, txt
        assert "'org/other'" in txt, "the OTHER entry must be untouched"
        assert "# the resident model" in txt and "port = 8770" in txt
        try:
            update_llm_model(str(cp5), "nope", "x")
            raise AssertionError("unknown entry must raise")
        except ValueError as e:
            assert "nope" in str(e)
    ok("update_llm_model: rewrites ONE entry's model in place — comments, "
       "spacing, and the other entries untouched")

    # ── the Tune editor: schema, validation, flag mapping, config writer ─────
    keys_in_table = {k for k, _, _ in sv._VLLM_KEYS}
    for t in sv.TUNING_SCHEMA:
        assert t["key"] and t["label"] and t["help"] and t["applies"], t
        assert t["type"] in ("int", "float", "bool", "bool3", "choice", "str"), t
        assert t.get("scope") in ("model", "entry"), t
        # model-scope knobs must be forwarded flags; entry-scope keys are
        # consumed by llm_argv/supervisor directly (port, image, runtime …)
        if "vllm" in t["engines"] and t.get("scope") == "model":
            assert t["key"] in keys_in_table, \
                f"schema offers {t['key']} but llm_argv never forwards it"
        r = t.get("recommended")
        if isinstance(r, (int, float)) and not isinstance(r, bool):
            assert t.get("min", r) <= r <= t.get("max", r), f"{t['key']}: rec out of bounds"
    ok("TUNING_SCHEMA: every knob typed, explained, and actually forwarded")

    fl = sv._mapped_flags({"enable_prefix_caching": True, "max_num_batched_tokens": 8192},
                          sv._VLLM_KEYS)
    assert "--enable-prefix-caching" in fl and "8192" in fl, fl
    fl = sv._mapped_flags({"enable_prefix_caching": False}, sv._VLLM_KEYS)
    assert "--no-enable-prefix-caching" in fl, fl
    assert "--enable-prefix-caching" not in fl
    fl = sv._mapped_flags({}, sv._VLLM_KEYS)
    assert not any("prefix" in f for f in fl), "absent = the engine's default"
    ok("prefix caching: tri-state on/off/absent maps to the right vLLM flags")

    co = sv.validate_tuning("vllm", {"max_model_len": "16384", "kv_cache_dtype": "fp8",
                                     "enable_prefix_caching": "false",
                                     "gpu_memory_utilization": 0.9,
                                     "exclusive": False, "max_num_seqs": None})
    assert co["max_model_len"] == 16384 and co["kv_cache_dtype"] == "fp8"
    assert co["enable_prefix_caching"] is False and co["gpu_memory_utilization"] == 0.9
    assert co["exclusive"] is None, "bool false = remove (absent IS false)"
    assert co["max_num_seqs"] is None
    for bad in ({"ctx_size": 4096},                      # llama key on vllm
                {"kv_cache_dtype": "int4"},              # not a choice
                {"gpu_memory_utilization": 1.5},         # above ceiling
                {"nonsense": 1}):
        try:
            sv.validate_tuning("vllm", bad)
            raise AssertionError(f"must refuse {bad}")
        except ValueError:
            pass
    ok("validate_tuning: coercion + engine gating + bounds + choice checks")

    from knowledgehost.config import update_llm_entry
    with tempfile.TemporaryDirectory() as td12:
        cp12 = Path(td12) / "config.toml"
        cp12.write_text('[[serving.llms]]\nname   = "primary"\n'
                        'model  = "org/a"\nport   = 11438\n'
                        'max_model_len = 8192     # tuned by hand once\n'
                        '\n[[serving.llms]]\nname = "secondary"\nmodel = "org/b"\n'
                        'port = 11439\n\n[serving.embed]\nenabled = false\n')
        update_llm_entry(str(cp12), "primary",
                         {"max_model_len": 16384,          # rewrite, keep comment
                          "kv_cache_dtype": "fp8",         # insert new key
                          "enable_prefix_caching": True})  # insert bool
        txt12 = cp12.read_text()
        assert "max_model_len = 8192" not in txt12
        assert "max_model_len = 16384     # tuned by hand once" in txt12, \
            "a rewritten line keeps its trailing comment"
        assert txt12.index('kv_cache_dtype = "fp8"') < txt12.index('name = "secondary"'), \
            "new keys land inside PRIMARY's block"
        assert "enable_prefix_caching = true" in txt12
        cfg12 = load_config(str(cp12))["serving"]["llms"]
        assert cfg12[0]["max_model_len"] == 16384 and cfg12[1].get("kv_cache_dtype") is None
        update_llm_entry(str(cp12), "primary", {"kv_cache_dtype": None})   # remove
        cfg12 = load_config(str(cp12))["serving"]["llms"]
        assert "kv_cache_dtype" not in cfg12[0], "None removes the key (engine default)"
        try:
            update_llm_entry(str(cp12), "primary", {"name": "x"})
            raise AssertionError("name must not be editable here")
        except ValueError:
            pass
        try:
            update_llm_entry(str(cp12), "ghost", {"max_model_len": 1024})
            raise AssertionError("unknown entry must raise")
        except ValueError:
            pass
    ok("update_llm_entry: rewrite keeps comments, insert stays in-block, "
       "None removes, siblings untouched")

    # ── add_llm_entry: the Add-service flow's config writer ─────────────────
    from knowledgehost.config import add_llm_entry
    with tempfile.TemporaryDirectory() as td8:
        cp8 = Path(td8) / "config.toml"
        cp8.write_text('[[serving.llms]]\nname = "a"\nengine = "vllm"\n'
                       'model = "org/x"\nport = 11438\n'
                       '[serving.embed]\nenabled = false\n# tail comment\n')
        add_llm_entry(str(cp8), {"name": "new-one", "engine": "vllm",
                                 "model": "org/new", "port": 11440,
                                 "exclusive": True})
        txt8 = cp8.read_text()
        assert txt8.index('name   = "new-one"') < txt8.index("[serving.embed]"), \
            "the new block joins the llms group, not the end of the file"
        assert "# tail comment" in txt8 and 'name = "a"' in txt8
        es = load_config(str(cp8))["serving"]["llms"]
        assert len(es) == 2 and es[1]["name"] == "new-one"
        assert es[1]["exclusive"] is True and es[1]["port"] == 11440
        try:
            add_llm_entry(str(cp8), {"name": "bad name!", "engine": "vllm",
                                     "model": "m", "port": 2})
            raise AssertionError("bad name must raise")
        except ValueError:
            pass
        cp9 = Path(td8) / "empty.toml"           # first entry on a fresh box
        cp9.write_text("")
        add_llm_entry(str(cp9), {"name": "solo", "engine": "llama",
                                 "model": "models/x.gguf", "port": 11441})
        assert 'name   = "solo"' in cp9.read_text()
    ok("add_llm_entry: joins the llms group in place, validates name/engine, "
       "first entry into an empty file works")

    # ── container images: fetched deliberately, under a policy lease ────────
    from knowledgehost.amiga_net import audit as au10
    from knowledgehost.amiga_net import policy as po10
    with tempfile.TemporaryDirectory() as td10:
        r10 = Path(td10)
        (r10 / "bin").mkdir()
        fake = r10 / "bin" / "podman"
        fake.write_text('#!/bin/bash\necho "$@" >> "$PODLOG"\n'
                        '[ "$1" = image ] && exit 1\nexit 0\n')
        fake.chmod(0o755)
        pol10 = r10 / "egress.toml"
        pol10.write_text('[[rule]]\nname = "container-images"\n'
                         'hosts = ["docker.io"]\nport = 443\n'
                         'methods = ["GET", "HEAD"]\npurpose = "t"\n'
                         'ttl_seconds = 60\nmax_uses = 5\n')
        keep10 = (po10.POLICY_PATH, po10.LEASE_DIR, au10.LOG_PATH)
        po10.POLICY_PATH, po10.LEASE_DIR = pol10, r10 / "run"
        au10.LOG_PATH = r10 / "egress.jsonl"
        old10 = {"PODLOG": os.environ.get("PODLOG"), "PATH": os.environ["PATH"]}
        os.environ["PODLOG"] = str(r10 / "calls.log")
        os.environ["PATH"] = f"{r10 / 'bin'}:{os.environ['PATH']}"
        try:
            ent10 = {"engine": "container", "runtime": str(fake),
                     "image": "docker.io/vllm/vllm-openai:v1"}
            sv._ensure_image(ent10)
            calls = (r10 / "calls.log").read_text()
            assert "pull docker.io/vllm/vllm-openai:v1" in calls, calls
            evs = au10.tail(10)
            assert any(e["verdict"] == "ALLOWED" and e["rule"] == "container-images"
                       and "not byte-accounted" in e.get("detail", "") for e in evs)
            assert any(e["verdict"] == "LEASE_OPEN" for e in evs) and \
                any(e["verdict"] == "LEASE_CLOSE" for e in evs)
            # a present image costs one inspect and NO egress
            fake.write_text('#!/bin/bash\necho "$@" >> "$PODLOG"\nexit 0\n')
            (r10 / "calls.log").write_text("")
            sv._ensure_image(ent10)
            assert "pull" not in (r10 / "calls.log").read_text()
        finally:
            po10.POLICY_PATH, po10.LEASE_DIR, au10.LOG_PATH = keep10
            os.environ["PATH"] = old10["PATH"]
            if old10["PODLOG"] is None:
                os.environ.pop("PODLOG", None)
            else:
                os.environ["PODLOG"] = old10["PODLOG"]
    ok("container image pull: policy-leased + audited, never a side effect; "
       "a present image touches no network")

    # ── the download lane: queue, pause/resume/discard, disk truth ──────────
    from knowledgehost.downloads import Downloads
    with tempfile.TemporaryDirectory() as td11:
        r11 = Path(td11)
        dl = Downloads(r11, "")
        import subprocess as sp11

        def fake_spawn(model, include, revision):     # a sleeping stand-in child
            dl.logdir.mkdir(parents=True, exist_ok=True)
            lf = open(dl._logfile(model), "wb", buffering=0)
            lf.write(b"pull started\n")
            proc = sp11.Popen([sys.executable, "-c", "import time; time.sleep(60)"],
                              stdout=lf, stderr=sp11.STDOUT, start_new_session=True)
            dl._live[model] = {"proc": proc, "started": time.time(),
                               "include": include, "revision": revision}
        dl._spawn = fake_spawn
        assert dl.start("org/one")["state"] == "pulling"
        assert dl.start("org/two")["state"] == "queued"
        assert dl.start("org/one")["ok"] is False          # already downloading
        sd11 = r11 / "models" / "org--one"                 # manifest-first pulls:
        sd11.mkdir(parents=True)                           # % is disk truth
        (sd11 / ".pull.json").write_text(_json.dumps(
            {"model": "org/one", "include": "m*",
             "files": {"m.safetensors": {"size": 1000}}}))
        (sd11 / "m.safetensors.part").write_bytes(b"x" * 250)
        st = {d["model"]: d for d in dl.status()}
        assert st["org/one"]["state"] == "pulling" and st["org/one"]["pct"] == 25, st
        assert st["org/two"]["state"] == "queued"
        dl.stop("org/one")                                 # pause -> next promoted
        for _ in range(40):
            st = {d["model"]: d for d in dl.status()}
            if st.get("org/two", {}).get("state") == "pulling":
                break
            time.sleep(0.1)
        assert st["org/two"]["state"] == "pulling", st
        assert st["org/one"]["state"] == "paused", "our SIGTERM is a pause, not an error"
        dl.stop("org/two")
        assert dl.discard("org/one")["ok"] and not sd11.exists()
        done11 = r11 / "models" / "org--done"              # complete = untouchable
        done11.mkdir()
        (done11 / "w.safetensors").write_bytes(b"x" * 10)
        (done11 / ".pull.json").write_text(_json.dumps(
            {"model": "org/done", "files": {"w.safetensors": {"size": 10}}}))
        assert dl.start("org/done")["ok"] is False
        assert dl.discard("org/done")["ok"] is False and done11.exists()
    ok("download lane: own slot + visible queue, pause promotes the next, "
       "manifest %, discard only ever touches incomplete folders")

    # ── adopt: legacy hub-cache snapshots -> the models/ store ───────────────
    with tempfile.TemporaryDirectory() as td9:
        root9 = Path(td9)
        hub9 = root9 / "cache" / "hub"
        blobs = hub9 / "models--org--legacy" / "blobs"
        snap9 = hub9 / "models--org--legacy" / "snapshots" / "rev1"
        blobs.mkdir(parents=True)
        snap9.mkdir(parents=True)
        (blobs / "b1").write_bytes(b"W" * 128)
        (snap9 / "config.json").write_text("{}")
        os.symlink(blobs / "b1", snap9 / "model.safetensors")
        env_hf = os.environ.get("HF_HOME")
        os.environ["HF_HOME"] = str(root9 / "cache")
        try:
            said9 = []
            assert sv.adopt_cached(root=root9, say=said9.append) == 1, said9
            from knowledgehost.amiga_net import pull as pull9
            got9 = pull9.pulled(root9, "org/legacy")
            assert got9 and (got9 / "model.safetensors").read_bytes() == b"W" * 128
            man9 = _json.loads((got9 / ".pull.json").read_text())
            assert man9["adopted_from"] and \
                man9["files"]["model.safetensors"]["size"] == 128
            # the picker offers the STORE copy; the cache is never offered
            vll9 = sv.eligible_models("vllm", root=root9)
            assert [c["model"] for c in vll9] == ["org/legacy"], vll9
            assert all(c["via"] == "store" for c in vll9)
            assert sv.adopt_cached(root=root9, say=lambda m: None) == 0  # idempotent
        finally:
            if env_hf is None:
                os.environ.pop("HF_HOME", None)
            else:
                os.environ["HF_HOME"] = env_hf
    ok("adopt_cached: cache snapshot -> models/ (links/copies + manifest), "
       "idempotent; pickers offer the store only")

    # ── Settings › Network: the broker's deliberate, REDACTED settings lane ──
    from knowledgehost.config import net_view, set_net_setting
    with tempfile.TemporaryDirectory() as td7:
        cp7 = Path(td7) / "config.toml"
        cp7.write_text('# mine\nport = 8770\n\n[serving.embed]\nenabled = false\n')
        set_net_setting(str(cp7), "https_proxy", "http://bob:hunter2@proxy.corp:3128")
        set_net_setting(str(cp7), "fetch_engine", "wget")
        txt7 = cp7.read_text()
        assert "hunter2" in txt7 and "# mine" in txt7 and "port = 8770" in txt7
        assert txt7.index("fetch_engine") < txt7.index("[serving.embed]"), \
            "new keys must stay top-level (before the first table)"
        cfg7 = load_config(str(cp7))
        view7 = net_view(cfg7)
        assert view7["https_proxy"] == "http://***:***@proxy.corp:3128", view7
        assert view7["hf_token_set"] is False and "hf_token" not in view7
        try:                       # the redacted echo must never clobber the real value
            set_net_setting(str(cp7), "https_proxy", view7["https_proxy"])
            raise AssertionError("redacted echo must be refused")
        except ValueError as e:
            assert "REDACTED" in str(e)
        try:
            set_net_setting(str(cp7), "fetch_engine", "curl")
            raise AssertionError("unknown engine must be refused")
        except ValueError:
            pass
        try:
            set_net_setting(str(cp7), "http_proxy", "proxy.corp:3128")
            raise AssertionError("bare host:port must be refused (URL wanted)")
        except ValueError:
            pass
        set_net_setting(str(cp7), "hf_token", "hf_secretsecret")
        view7 = net_view(load_config(str(cp7)))
        assert view7["hf_token_set"] is True and view7["hf_token_hint"] == "…cret"
        # and config-file proxies now actually REACH the engines/broker env
        env7 = sv.proxy_env(load_config(str(cp7)))
        assert env7["https_proxy"] == "http://bob:hunter2@proxy.corp:3128", env7
        assert "127.0.0.1" in env7["no_proxy"]
    ok("network settings: redacted view, redacted-echo/bad values refused, "
       "token write-only, config proxies reach proxy_env")

    # ── proxy: nothing in the stack reads OS proxy settings, so we pass it ───
    pcfg = {"serving": {"llms": [{"name": "a", "host": "10.0.0.5"}]},
            "http_proxy": "http://proxy.corp:3128"}
    px = sv.proxy_env(pcfg)
    assert px["http_proxy"] == px["HTTP_PROXY"] == "http://proxy.corp:3128", px
    # loopback must never go through a proxy — the kb calls its own LMs there —
    # and a declared serving host is local traffic too
    for h in ("localhost", "127.0.0.1", "::1", "10.0.0.5"):
        assert h in px["no_proxy"].split(","), (h, px)
    assert px["no_proxy"] == px["NO_PROXY"]
    env0 = {k: os.environ.get(k) for k in ("http_proxy", "HTTP_PROXY", "no_proxy")}
    try:
        for k in env0:
            os.environ.pop(k, None)
        assert sv.proxy_env({"serving": {"llms": []}}) == {}   # unproxied box: untouched
        assert sv.proxy_warning({}) is None
        # a shell proxy with no loopback exemption is the trap worth naming
        os.environ["http_proxy"] = "http://proxy.corp:3128"
        assert "no_proxy doesn't exempt loopback" in (sv.proxy_warning({}) or "")
        os.environ["no_proxy"] = "localhost,127.0.0.1"
        assert sv.proxy_warning({}) is None
        # the shell's proxy is inherited by an engine when config doesn't set one
        inherited = sv.proxy_env({"serving": {"llms": []}})
        assert inherited["http_proxy"] == "http://proxy.corp:3128", inherited
        # …and engine_env is what the exec path applies: the OFFLINE block +
        # proxy together (a container gets no host env) — and never the token
        merged = sv.engine_env({"serving": {"llms": []}, "hf_token": "hf_x"}, "container")
        assert merged["HF_HUB_OFFLINE"] == "1" and merged["http_proxy"], merged
        assert "HF_TOKEN" not in merged
    finally:
        for k, v in env0.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
    # credentials in a proxy URL must not land in the exec: log line
    red = sv.redact_argv(["podman", "-e", "http_proxy=http://bob:hunter2@proxy:3128"])
    assert "hunter2" not in red[-1] and "***:***@proxy:3128" in red[-1], red
    ok("proxy_env: passed to engines, loopback exempt, warning + credentials redacted")

    # ── /serving/model end-to-end: the picker's Deploy button ───────────────
    # GET offers choices → POST repoints the entry → file, live view and the
    # supervisor (a restart request for a running service) all update.
    with tempfile.TemporaryDirectory() as td12:
        root12 = Path(td12)
        import json as _json
        for nm in ("alpha", "beta"):
            d = root12 / "models" / f"org--{nm}"
            d.mkdir(parents=True)
            (d / "model.safetensors").write_bytes(b"w" * 32)
            (d / ".pull.json").write_text(_json.dumps(
                {"model": f"org/{nm}", "files": {"model.safetensors": {"size": 32}}}))
        toml12 = root12 / "config.toml"
        toml12.write_text('[[serving.llms]]\nname = "big"\nengine = "container"\n'
                          'model = "org/alpha"   # keep me\nport = 11438\n'
                          'image = "img:v1"\nexclusive = true\n')
        cfg12 = load_config(str(toml12))
        cfg12.update({"host": "127.0.0.1", "port": 0, "auth_token": "tk",
                      "_config_path": str(toml12)})
        from knowledgehost import supervisor as sup_mod
        keep = (sv.ROOT, sv.eligible_models.__defaults__, sv.SVC_REQ_DIR,
                sup_mod.read_state)
        try:
            sv.ROOT = root12
            sv.eligible_models.__defaults__ = (root12, None)
            sv.SVC_REQ_DIR = root12 / "var" / "run" / "svcreq"
            sup_mod.read_state = lambda: {}        # no supervisor: save-only lane
            khs12 = KnowledgeHostServer(cfg12, SimpleNamespace(), SimpleNamespace(),
                                        kb=None)
            p12 = khs12.server_address[1]
            threading.Thread(target=khs12.serve_forever, daemon=True).start()

            def call12(method, path, body=None):
                req = urllib.request.Request(
                    f"http://127.0.0.1:{p12}{path}",
                    data=_json.dumps(body).encode() if body is not None else None,
                    headers={"Authorization": "Bearer tk",
                             "Content-Type": "application/json"}, method=method)
                try:
                    with urllib.request.urlopen(req, timeout=10) as r:
                        return r.status, _json.loads(r.read())
                except urllib.error.HTTPError as e:
                    return e.code, _json.loads(e.read())

            code, st = call12("GET", "/serving/status")
            assert code == 200 and st["llms"][0]["model"] == "org/alpha"
            assert [c["model"] for c in st["llms"][0]["choices"]] == \
                ["org/alpha", "org/beta"], st["llms"][0]["choices"]
            code, res = call12("POST", "/serving/model",
                               {"name": "big", "model": "org/beta"})
            assert code == 200 and res["was"] == "org/alpha", res
            assert "saved" in res["note"]          # nothing running: no restart
            code, st = call12("GET", "/serving/status")
            assert st["llms"][0]["model"] == "org/beta"   # runtime view updated
            line = next(ln for ln in toml12.read_text().splitlines() if "model" in ln)
            assert line == 'model = "org/beta"   # keep me', line   # comment kept
            # a running service gets a live restart request (the hot path)
            me = os.getpid()
            sup_mod.read_state = lambda: {"supervisor": me,
                                          "services": {"llm-big": me}}
            code, res = call12("POST", "/serving/model",
                               {"name": "big", "model": "org/alpha"})
            assert code == 200 and "restarting llm-big" in res["note"], res
            reqf = sv.SVC_REQ_DIR / "llm-big.req"
            assert _json.loads(reqf.read_text())["action"] == "restart"
            # a model that is not on this disk is refused, nothing written
            code, res = call12("POST", "/serving/model",
                               {"name": "big", "model": "org/ghost"})
            assert code == 400 and "pull it first" in res["error"], res
            assert 'model = "org/alpha"' in toml12.read_text()
            khs12.shutdown()
        finally:
            (sv.ROOT, sv.eligible_models.__defaults__, sv.SVC_REQ_DIR,
             sup_mod.read_state) = keep
    ok("/serving/model deploy: choices offered, file+runtime repointed "
       "(comment kept), running service restarted, off-disk model refused")

    # ── local-dir weights: every layout a real model folder comes in ────────
    # The old check was a top-level *.safetensors glob — a fully downloaded
    # model in any other layout read "incomplete: local dir has no
    # *.safetensors" (Dan hit this on a complete download).
    with tempfile.TemporaryDirectory() as td13:
        import json as _json
        r13 = Path(td13)

        def mk(name):
            d = r13 / name
            d.mkdir(parents=True)
            return d

        flat = mk("flat")                              # flat snapshot: ready
        (flat / "config.json").write_text("{}")
        (flat / "model.safetensors").write_bytes(b"w" * 8)
        assert sv._local_dir_status_now(flat)["status"] == "ready"

        hub = mk("models--org--big")                   # hub-cache repo root
        snap = hub / "snapshots" / "aaa"
        snap.mkdir(parents=True)
        (snap / "config.json").write_text("{}")
        (snap / "model.safetensors").write_bytes(b"w" * 8)
        ws = sv._local_dir_status_now(hub)
        assert ws["status"] == "ready" and ws["path"] == str(snap), ws
        assert "hub-cache layout" in ws["detail"]
        (snap / "model.safetensors").unlink()          # gutted snapshot
        ws = sv._local_dir_status_now(hub)
        assert ws["status"] == "incomplete" and "no complete snapshot" in ws["detail"]

        nest = mk("nest")                              # weights one level down
        sub = nest / "w4a16"
        sub.mkdir()
        (sub / "model.safetensors").write_bytes(b"w" * 8)
        ws = sv._local_dir_status_now(nest)
        assert ws["status"] == "ready" and "weights live in w4a16/" in ws["detail"], ws

        shard = mk("shard")                            # index is the honest test
        (shard / "model.safetensors.index.json").write_text(_json.dumps(
            {"weight_map": {"a": "model-00001-of-00002.safetensors",
                            "b": "model-00002-of-00002.safetensors"}}))
        (shard / "model-00001-of-00002.safetensors").write_bytes(b"w")
        ws = sv._local_dir_status_now(shard)
        assert ws["status"] == "incomplete" and \
            "model-00002-of-00002.safetensors" in ws["detail"], ws
        (shard / "model-00002-of-00002.safetensors").write_bytes(b"w")
        assert sv._local_dir_status_now(shard)["status"] == "ready"

        legacy = mk("legacy")                          # .bin models still serve
        (legacy / "pytorch_model.bin").write_bytes(b"w" * 8)
        assert sv._local_dir_status_now(legacy)["status"] == "ready"

        store = mk("store")                            # store dir by PATH: manifest %
        (store / ".pull.json").write_text(_json.dumps(
            {"model": "org/s", "files": {"model.safetensors": {"size": 100}}}))
        (store / "model.safetensors").write_bytes(b"w" * 50)
        ws = sv._local_dir_status_now(store)
        assert ws["status"] == "incomplete" and "(50%)" in ws["detail"], ws
        (store / "model.safetensors").write_bytes(b"w" * 100)
        assert sv._local_dir_status_now(store)["status"] == "ready"

        mid = mk("mid")                                # nothing but a .part yet
        (mid / "model.safetensors.part").write_bytes(b"w")
        ws = sv._local_dir_status_now(mid)
        assert ws["status"] == "incomplete" and "mid-download" in ws["detail"], ws

        empty = mk("empty")                            # wrong path: say what's here
        (empty / "config.json").write_text("{}")
        (empty / "README.md").write_text("x")
        ws = sv._local_dir_status_now(empty)
        assert ws["status"] == "incomplete", ws
        assert "README.md" in ws["detail"] and "right path" in ws["detail"], ws

        # end-to-end wiring: weights_status routes a dir path here
        assert sv.weights_status("container", str(flat))["status"] == "ready"
    ok("local-dir weights: flat/hub-root/nested/sharded-index/.bin/store-"
       "manifest/mid-download all judged honestly; wrong path names contents")

    # ── per-model tuning: tune.toml lives WITH the weights ──────────────────
    with tempfile.TemporaryDirectory() as td14:
        import json as _json
        r14 = Path(td14)
        for nm, tune in (("alpha", 'max_model_len = 8192\nkv_cache_dtype = "fp8"\n'),
                         ("beta", "max_model_len = 32768\n")):
            d = r14 / "models" / f"org--{nm}"
            d.mkdir(parents=True)
            (d / "model.safetensors").write_bytes(b"w" * 16)
            (d / ".pull.json").write_text(_json.dumps(
                {"model": f"org/{nm}", "files": {"model.safetensors": {"size": 16}}}))
            (d / "tune.toml").write_text(tune)
        gg = r14 / "models" / "org--gg"
        gg.mkdir()
        (gg / "m-Q4.gguf").write_bytes(b"g")
        (gg / "m-Q4.gguf.tune.toml").write_text("ctx_size = 2048\n")

        e_id = {"name": "a", "engine": "container", "model": "org/alpha", "port": 1}
        assert sv.tuning_path(e_id, root=r14) == \
            r14 / "models" / "org--alpha" / "tune.toml"
        bdir = r14 / "models" / "org--beta"
        assert sv.tuning_path({"engine": "vllm", "model": str(bdir)}) == \
            bdir / "tune.toml"
        e_gg = {"name": "g", "engine": "llama", "model": str(gg / "m-Q4.gguf"),
                "port": 2}
        assert sv.tuning_path(e_gg) == gg / "m-Q4.gguf.tune.toml"
        assert sv.tuning_path({"engine": "container", "model": "org/ghost"},
                              root=r14) is None
        ok("tuning_path: store id / local dir / per-GGUF sibling; homeless id -> None")

        vals, note = sv.read_model_tuning(e_id, root=r14)
        assert vals == {"max_model_len": 8192, "kv_cache_dtype": "fp8"} and not note
        tp = sv.tuning_path(e_id, root=r14)
        tp.write_text('max_model_len = 8192\nexclusive = true\nbogus = 3\n'
                      'dtype = "bfloat16"\nmax_num_seqs = 999999\n')
        vals, note = sv.read_model_tuning(e_id, root=r14)
        assert vals["max_model_len"] == 8192 and vals["dtype"] == "bfloat16"
        assert "max_num_seqs" not in vals              # out of bounds: dropped
        assert "exclusive" in note and "bogus" in note, note
        tp.write_text("not = [valid")
        vals, note = sv.read_model_tuning(e_id, root=r14)
        assert vals == {} and "unreadable" in note
        ok("read_model_tuning: schema-validated, entry-only/unknown/bad keys "
           "dropped AND named, syntax errors degrade to engine defaults")

        sv.write_model_tuning(tp, {"max_model_len": 4096, "max_num_seqs": 32})
        got = sv.read_model_tuning(e_id, root=r14)[0]
        assert got == {"max_model_len": 4096, "max_num_seqs": 32}   # repaired too
        sv.write_model_tuning(tp, {"max_num_seqs": None})
        assert "max_num_seqs" not in sv.read_model_tuning(e_id, root=r14)[0]
        sv.write_model_tuning(tp, {"max_model_len": None})
        assert not tp.exists()                         # empty file removed
        ok("write_model_tuning: merge, None removes, empty removes the file")

        vdir = r14 / "serving" / ".venv" / "bin"
        vdir.mkdir(parents=True)
        (vdir / "vllm").write_text("")
        ev = {"name": "b", "engine": "vllm", "model": str(bdir), "port": 9}
        s = " ".join(sv.llm_argv(ev, root=r14))
        assert "--max-model-len 32768" in s, s          # from beta's tune.toml
        s = " ".join(sv.llm_argv({**ev, "max_model_len": 1024}, root=r14))
        assert "--max-model-len 1024" in s and "32768" not in s   # entry wins
        env0 = os.environ.get("LLAMA_SERVER")
        os.environ["LLAMA_SERVER"] = "/usr/bin/env"
        try:
            s = " ".join(sv.llm_argv(e_gg, root=r14))
            assert "-c 2048" in s, s                    # per-GGUF file applies
        finally:
            os.environ.pop("LLAMA_SERVER", None)
            if env0 is not None:
                os.environ["LLAMA_SERVER"] = env0
        ok("llm_argv: tune.toml supplies flags, a config-entry key overrides")

        # E2E /serving/tune: lanes split, legacy keys migrate, tuning follows
        # the model through Deploy
        toml14 = r14 / "config.toml"
        toml14.write_text('[[serving.llms]]\nname = "big"\nengine = "container"\n'
                          'model = "org/alpha"\nport = 11438\n'
                          'image = "img:v1"\nexclusive = true\n'
                          'max_num_seqs = 64   # legacy: predates tune.toml\n')
        cfg14 = load_config(str(toml14))
        cfg14.update({"host": "127.0.0.1", "port": 0, "auth_token": "tk",
                      "_config_path": str(toml14)})
        from knowledgehost import supervisor as sup_mod
        keep14 = (sv.ROOT, sv.eligible_models.__defaults__, sup_mod.read_state)
        try:
            sv.ROOT = r14
            sv.eligible_models.__defaults__ = (r14, None)
            sup_mod.read_state = lambda: {}
            khs14 = KnowledgeHostServer(cfg14, SimpleNamespace(), SimpleNamespace(),
                                        kb=None)
            p14 = khs14.server_address[1]
            threading.Thread(target=khs14.serve_forever, daemon=True).start()

            def call14(method, path, body=None):
                req = urllib.request.Request(
                    f"http://127.0.0.1:{p14}{path}",
                    data=_json.dumps(body).encode() if body is not None else None,
                    headers={"Authorization": "Bearer tk",
                             "Content-Type": "application/json"}, method=method)
                try:
                    with urllib.request.urlopen(req, timeout=10) as r:
                        return r.status, _json.loads(r.read())
                except urllib.error.HTTPError as e:
                    return e.code, _json.loads(e.read())

            code, res = call14("POST", "/serving/tune",
                               {"name": "big",
                                "updates": {"gpu_memory_utilization": 0.85}})
            assert code == 200 and "moved 1 older setting" in res["note"], res
            atoml = (r14 / "models" / "org--alpha" / "tune.toml").read_text()
            assert "gpu_memory_utilization = 0.85" in atoml
            assert "max_num_seqs = 64" in atoml         # legacy key migrated in
            assert "max_num_seqs" not in toml14.read_text()   # …and out of config
            code, st = call14("GET", "/serving/status")
            t14 = st["llms"][0]
            assert t14["tuning"]["gpu_memory_utilization"] == 0.85
            assert t14["tuning"]["max_num_seqs"] == 64  # effective, from the file
            assert t14["tune_file"].endswith("org--alpha/tune.toml")
            # entry lane: image edits the config entry; empty port refused
            code, res = call14("POST", "/serving/tune",
                               {"name": "big", "updates": {"image": "img:v2"}})
            assert code == 200 and 'image = "img:v2"' in toml14.read_text(), res
            code, res = call14("POST", "/serving/tune",
                               {"name": "big", "updates": {"port": ""}})
            assert code == 400 and "required" in res["error"], res
            # Deploy: the OTHER model's tune.toml rides along
            code, res = call14("POST", "/serving/model",
                               {"name": "big", "model": "org/beta"})
            assert code == 200, res
            code, st = call14("GET", "/serving/status")
            assert st["llms"][0]["tuning"]["max_model_len"] == 32768, \
                st["llms"][0]["tuning"]      # beta's own tuning, not alpha's
            assert st["llms"][0]["tuning"].get("gpu_memory_utilization") is None
            khs14.shutdown()
        finally:
            (sv.ROOT, sv.eligible_models.__defaults__,
             sup_mod.read_state) = keep14
    ok("/serving/tune: model knobs -> tune.toml (legacy keys migrate out of "
       "config.toml), slot keys -> entry, required port refused, and Deploy "
       "brings the new model's own tuning")

    print(f"swap_test: {PASS} checks OK")


if __name__ == "__main__":
    main()
