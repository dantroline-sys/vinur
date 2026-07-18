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
                assert "hf download org/absent" in by["nowhere"]["weights"]["detail"]
                assert by["gg"]["weights"]["status"] == "ready"
                assert all(m["service"] == "supervisor-down" for m in res["llms"])
                ok("serving_status: ready / mid-download / missing weights + supervisor-down")

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
                khs2.shutdown()
                ok("GET /serving/status serves the panel payload (authed)")

                # ── known-failure hints + toolkit preflight ──────────────
                assert "toolkit missing" in sv.failure_hint(
                    "x\nRuntimeError: Could not find nvcc and default cuda_home...")
                assert "gated HF repo" in sv.failure_hint("401 Client Error: Unauthorized")
                assert "VRAM" in sv.failure_hint("torch.cuda: CUDA out of memory")
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

    print(f"swap_test: {PASS} checks OK")


if __name__ == "__main__":
    main()
