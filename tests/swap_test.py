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
                    'model = "a"\nport = 11438\nexclusive = true\n'
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

            calls = []
            pilot = ap.Autopilot(scfg, SimpleNamespace(
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
        finally:
            sv.SWAP_REQ, sv.SWAP_STATE = req0, st0

    print(f"swap_test: {PASS} checks OK")


if __name__ == "__main__":
    main()
