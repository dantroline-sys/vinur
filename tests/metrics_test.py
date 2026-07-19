"""Telemetry collector (VINUR-UI-01 Stage 6): the metrics store's downsampled
history + retention, nvidia-smi / vLLM-Prometheus parsing (both stubbed), the
sampler's tick + ops-transition events, and the server's /metrics routes.

Run:  python tests/metrics_test.py     (stdlib only)
"""
import json
import os
import stat
import subprocess
import sys
import tempfile
import threading
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from knowledgehost import metrics as M
from knowledgehost.config import load_config
from knowledgehost.server import KnowledgeHostServer

PASS = 0


def ok(label):
    global PASS
    PASS += 1
    print(f"  ok {PASS:2d}  {label}")


def main():
    with tempfile.TemporaryDirectory() as td:
        # ── MetricsStore: downsampled history ────────────────────────────
        st = M.MetricsStore(str(Path(td) / "m.db"))
        t0 = 1_000_000.0
        for i in range(120):                       # 1 sample/s for 2 minutes
            st.add(t0 + i, {"a": float(i), "b": 5.0})
        h = st.history(mins=2, step=10, now=t0 + 120)
        assert set(h["series"]) == {"a", "b"} and h["bucket"] == 10
        assert len(h["series"]["a"]) == 12, len(h["series"]["a"])
        # bucket 0 covers samples 0..9 -> avg 4.5
        assert abs(h["series"]["a"][0][1] - 4.5) < 1e-6
        assert all(v == 5.0 for _, v in h["series"]["b"])
        # auto step targets <= ~600 points/series
        h2 = st.history(mins=2, now=t0 + 120)
        assert len(h2["series"]["a"]) <= 600
        ok("history: bucket averages, auto step bounds the payload")

        # ── events + retention ───────────────────────────────────────────
        st.event("mark", "A: distill_parallel=4", ts=t0 + 30)
        st.event("mark", "B: distill_parallel=8", {"note": "x"}, ts=t0 + 90)
        h = st.history(mins=2, now=t0 + 120)
        assert [e["label"][:1] for e in h["events"]] == ["A", "B"]
        assert h["events"][1]["data"] == {"note": "x"}
        # prune is on wall-clock age: t0 (epoch 1970s) is ancient, so ALL of it
        # goes; only rows younger than now-14d survive
        st.add(time.time(), {"fresh": 2.0})
        st.event("mark", "fresh-mark")
        st.prune(keep_days=14)
        with st._con() as con:
            kept_s = con.execute("SELECT series FROM samples").fetchall()
            kept_e = con.execute("SELECT label FROM events").fetchall()
        assert kept_s == [("fresh",)], kept_s
        assert kept_e == [("fresh-mark",)], kept_e
        ok("events round-trip; prune drops aged rows, keeps fresh ones")

        # ── nvidia-smi parsing via a PATH stub ───────────────────────────
        bindir = Path(td) / "bin"
        bindir.mkdir()
        smi = bindir / "nvidia-smi"
        smi.write_text("#!/bin/sh\n"
                       "echo '0, 87, 81234, 97871, 312.45, 67'\n"
                       "echo '1, [N/A], 100, 200, [N/A], 55'\n")
        smi.chmod(smi.stat().st_mode | stat.S_IEXEC)
        path0 = os.environ["PATH"]
        os.environ["PATH"] = f"{bindir}:{path0}"
        try:
            g = M.sample_gpu()
        finally:
            os.environ["PATH"] = path0
        assert g["gpu0.util"] == 87 and g["gpu0.power_w"] == 312.45
        assert g["gpu1.vram_mb"] == 100 and g["gpu1.temp_c"] == 55
        assert "gpu1.util" not in g and "gpu1.power_w" not in g   # [N/A] skipped
        os.environ["PATH"] = str(bindir.parent / "nowhere")       # no binary at all
        try:
            assert M.sample_gpu() == {}
        finally:
            os.environ["PATH"] = path0
        ok("sample_gpu: per-GPU series, [N/A] skipped, missing binary -> {}")

        # ── Prometheus parsing ───────────────────────────────────────────
        prom = "\n".join([
            "# HELP vllm:num_requests_running ...",
            'vllm:num_requests_running{model_name="big"} 3.0',
            'vllm:num_requests_running{model_name="other"} 2.0',   # summed
            'vllm:num_requests_waiting{model_name="big"} 7.0',
            'vllm:gpu_cache_usage_perc{model_name="big"} 0.42',
            'vllm:generation_tokens_total{model_name="big"} 123456.0',
            "garbage line without value",
            'some_other_metric{x="y"} 9',
        ])
        p = M.parse_prom(prom)
        assert p == {"running": 5.0, "waiting": 7.0, "kv_pct": 42.0,
                     "gen_toks": 123456.0}, p
        ok("parse_prom: sums label sets, scales kv%, ignores noise")

        # ── vllm_targets + live scrape of a stub endpoint ────────────────
        class Prom(BaseHTTPRequestHandler):
            def do_GET(self):
                body = prom.encode()
                self.send_response(200)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *_):
                pass

        psrv = ThreadingHTTPServer(("127.0.0.1", 0), Prom)
        threading.Thread(target=psrv.serve_forever, daemon=True).start()
        pport = psrv.server_address[1]
        vcfg = {"serving": {"llms": [
            {"name": "primary", "engine": "container", "port": pport},
            {"name": "gg", "engine": "llama", "port": 11441},
            {"name": "faraway", "engine": "vllm", "port": 9, "host": "10.0.0.7"},
            {"name": "dead", "engine": "vllm", "port": 1}]}}
        tg = M.vllm_targets(vcfg)
        assert [n for n, _ in tg] == ["primary", "dead"], tg   # llama+foreign skipped
        kv = M.sample_vllm(vcfg)
        assert kv["vllm.primary.running"] == 5.0 and kv["vllm.primary.kv_pct"] == 42.0
        assert not any(k.startswith("vllm.dead") for k in kv)  # down -> skipped
        ok("vllm targets: batching+local only; scrape prefixes by entry name")

        # ── Sampler tick: counts_fn/slow_fn cadence + ops transitions ────
        st2 = M.MetricsStore(str(Path(td) / "m2.db"))

        op_t0 = time.time() - 30                 # a realistic job-start stamp

        class FakeRunner:
            def __init__(self):
                self.seq = [
                    {"running": False, "command": None},
                    {"running": True, "command": "distill", "argv": ["--limit", "5"],
                     "started": op_t0, "exit_code": None},
                    {"running": False, "command": "distill", "argv": ["--limit", "5"],
                     "started": op_t0, "exit_code": 0},
                ]
                self.i = 0

            def status(self):
                s = self.seq[min(self.i, len(self.seq) - 1)]
                self.i += 1
                return s

            def result(self):
                return {"command": "distill", "exit_code": 0,
                        "did_work": True, "cards": 3}

        runner = FakeRunner()
        scfg = {"stats_interval_s": 5, "stats_keep_days": 14, "serving": {"llms": []}}
        sm = M.Sampler(scfg, st2, counts_fn=lambda: {"kb.nodes": 10},
                       slow_fn=lambda: {"kb.chunks": 999}, runner=runner)
        os.environ["PATH"] = str(bindir.parent / "nowhere")   # no GPU in ticks
        try:
            sm.tick()      # tick 1: slow_fn fires (ticks % 12 == 1), op idle
            sm.tick()      # tick 2: op running -> op_start
            sm.tick()      # tick 3: op finished -> op_end (with OPS_RESULT)
        finally:
            os.environ["PATH"] = path0
        h = st2.history(mins=10)
        assert h["series"]["kb.nodes"] and h["series"]["kb.chunks"]
        assert len([p for p in h["series"]["kb.chunks"]]) >= 1
        evs = [(e["kind"], e["label"]) for e in h["events"]]
        assert evs == [("op_start", "distill"), ("op_end", "distill")], evs
        start_ev = h["events"][0]
        assert start_ev["ts"] == op_t0                        # back-stamped
        assert h["events"][1]["data"]["cards"] == 3
        assert h["events"][1]["data"]["exit_code"] == 0
        ok("sampler: counts every tick, scans on the slow cadence, op events")

        # a job that starts AND ends between ticks still gets both events
        st3 = M.MetricsStore(str(Path(td) / "m3.db"))
        r2 = FakeRunner()
        r2.seq = [{"running": False, "command": "link", "argv": [],
                   "started": time.time() - 5, "exit_code": 0}]
        sm2 = M.Sampler(scfg, st3, runner=r2)
        sm2.tick()
        evs = [(e["kind"], e["label"]) for e in st3.history(mins=10)["events"]]
        assert evs == [("op_start", "link"), ("op_end", "link")], evs
        ok("sampler: sub-tick job -> start (back-stamped) + end in one tick")

        # ── server routes: /metrics/history (open read) + /metrics/mark (authed) ──
        toml = Path(td) / "c.toml"
        toml.write_text('auth_token = "tk"\nhost = "127.0.0.1"\nport = 0\n'
                        f'metrics_db = "{Path(td) / "srv.db"}"\n'
                        'stats_interval_s = 0.0\n')
        cfg = load_config(str(toml))
        assert cfg["stats_interval_s"] == 0.0 and cfg["stats_keep_days"] == 14
        httpd = KnowledgeHostServer(cfg, SimpleNamespace(), SimpleNamespace(), kb=None)
        assert httpd.start_metrics() is None                  # 0 = disabled
        threading.Thread(target=httpd.serve_forever, daemon=True).start()
        base = f"http://127.0.0.1:{httpd.server_address[1]}"

        def post(path, body, token=None):
            req = urllib.request.Request(
                base + path, data=json.dumps(body).encode(), method="POST",
                headers={"Content-Type": "application/json",
                         **({"Authorization": f"Bearer {token}"} if token else {})})
            try:
                with urllib.request.urlopen(req, timeout=5) as r:
                    return r.status, json.loads(r.read())
            except urllib.error.HTTPError as e:
                return e.code, json.loads(e.read() or b"{}")

        code, _ = post("/metrics/mark", {"label": "A"})
        assert code == 401
        code, body = post("/metrics/mark", {"label": ""}, token="tk")
        assert code == 400
        code, body = post("/metrics/mark", {"label": "A: fanout=8"}, token="tk")
        assert code == 200 and body["ok"]
        with urllib.request.urlopen(base + "/metrics/history?mins=5", timeout=5) as r:
            h = json.loads(r.read())
        assert h["ok"] and [e["label"] for e in h["events"]] == ["A: fanout=8"]
        assert h["series"] == {}                              # sampler off -> no series
        code, _ = post("/metrics/mark", {"label": "x"}, token="wrong")
        assert code == 401
        httpd.shutdown()
        ok("/metrics/mark authed + validated; /metrics/history serves the marks")

        psrv.shutdown()
    print(f"metrics_test: {PASS} checks OK")


if __name__ == "__main__":
    main()
