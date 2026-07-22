#!/usr/bin/env python
"""modelfind — the hub searcher: search parsing, tree-API sizing, GGUF quant
expansion (split files grouped), fit verdicts against an injected memory
budget, gated repos marked instead of fatal, numbered picks -> pull --include,
and everything through the broker (leased + audited)."""
import hashlib
import json
import os
import sys
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ["AMIGA_FETCH_ENGINE"] = "stdlib"

from knowledgehost import modelfind  # noqa: E402
from knowledgehost.amiga_net import audit, policy, pull  # noqa: E402

OK = 0


def ok(label):
    global OK
    OK += 1
    print(f"  ok {OK:2d}  {label}")


GiB = 2 ** 30
QBLOB = os.urandom(50_000)                    # the one file actually downloaded
QBLOB_SHA = hashlib.sha256(QBLOB).hexdigest()

SEARCH = [
    {"modelId": "org/dense-fp8", "downloads": 3_100_000, "gated": False,
     "tags": ["fp8", "text-generation"]},
    {"modelId": "org/dense-huge", "downloads": 900_000, "gated": False, "tags": []},
    {"modelId": "org/tiny-GGUF", "downloads": 412_000, "gated": False, "tags": ["gguf"]},
    {"modelId": "org/locked", "downloads": 88_000, "gated": "manual", "tags": []},
    {"modelId": "org/hidden", "private": True, "downloads": 1, "tags": []},
]

TREES = {
    "org/dense-fp8": [
        {"type": "file", "path": "config.json", "size": 1_000},
        {"type": "file", "path": "model.safetensors", "size": 35 * GiB},
        {"type": "file", "path": "README.md", "size": 5},
    ],
    "org/dense-huge": [
        {"type": "file", "path": "config.json", "size": 1_000},
        {"type": "file", "path": "model.safetensors", "size": 200 * GiB},
    ],
    "org/tiny-GGUF": [
        {"type": "file", "path": "README.md", "size": 5},
        {"type": "file", "path": "tiny-Q8_0.gguf", "size": 34 * GiB},
        {"type": "file", "path": "tiny-Q4_K_M-00001-of-00002.gguf", "size": 10 * GiB},
        {"type": "file", "path": "tiny-Q4_K_M-00002-of-00002.gguf", "size": 10 * GiB},
        {"type": "file", "path": "tiny-F16.gguf", "size": 120 * GiB},
        {"type": "file", "path": "tiny-Q2_K.gguf", "size": len(QBLOB),
         "lfs": {"oid": QBLOB_SHA}},
    ],
}


class Hub(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _json(self, obj):
        body = json.dumps(obj).encode()
        self.send_response(200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path.startswith("/api/models?"):
            return self._json(SEARCH)
        if self.path.startswith("/api/models/org/locked/tree/"):
            self.send_response(401)
            self.end_headers()
            return
        for mid, tree in TREES.items():
            if self.path.startswith(f"/api/models/{mid}/tree/"):
                return self._json(tree)
        if self.path == "/org/tiny-GGUF/resolve/main/tiny-Q2_K.gguf":
            self.send_response(200)
            self.send_header("Content-Length", str(len(QBLOB)))
            self.end_headers()
            self.wfile.write(QBLOB)
            return
        self.send_response(404)
        self.end_headers()


srv = ThreadingHTTPServer(("127.0.0.1", 0), Hub)
threading.Thread(target=srv.serve_forever, daemon=True).start()
PORT = srv.server_address[1]

TD = Path(tempfile.mkdtemp())
POL = TD / "egress.toml"
POL.write_text(f"""
[[rule]]
name = "huggingface"
hosts = ["127.0.0.1"]
port = {PORT}
methods = ["GET", "HEAD"]
purpose = "test hub"
ttl_seconds = 60
max_uses = 1000
""")
policy.POLICY_PATH = POL
policy.LEASE_DIR = TD / "run"
audit.LOG_PATH = TD / "egress.jsonl"
pull.HF = f"http://127.0.0.1:{PORT}"

BUDGET = 96 * GiB                             # pretend: Dan's 96 GB box

# ── fit verdicts are plain arithmetic ────────────────────────────────────────
assert modelfind.fit(35 * GiB, BUDGET)[0] == "fits"
assert modelfind.fit(80 * GiB, BUDGET)[0] == "tight"      # 88+2 GB of 96
assert modelfind.fit(200 * GiB, BUDGET)[0] == "too big"
assert modelfind.fit(35 * GiB, 0)[0] == "?"
ok("fit(): fits / tight / too big bands, honest '?' with no hardware")

# ── budget() answers something on any machine ────────────────────────────────
b, label = modelfind.budget()
assert isinstance(b, int) and isinstance(label, str) and label
ok(f"budget(): detects this machine without crashing ({label})")

# ── the full find ────────────────────────────────────────────────────────────
lines = []
n = modelfind.find("tiny", root=TD, limit=8, say=lines.append,
                   budget_bytes=BUDGET, budget_label="96 GB VRAM (test)")
out = "\n".join(lines)
assert n == json.loads((TD / "var/run/find.json").read_text())["picks"].__len__()
assert "org/hidden" not in out, "private repos never listed"
ok("find(): runs end-to-end, picks saved, private repos dropped")

assert "org/dense-fp8" in out and "fits" in out
assert "35.0 GB" in out and "3.1M pulls" in out and "[fp8" in out
ok("a dense repo: exact size from the tree API, verdict, format hint, pulls")

assert "org/dense-huge" in out and "too big" in out and "200.0 GB" in out
ok("an oversized dense repo says 'too big', with the arithmetic shown")

assert "GGUF repo, pick a file" in out
assert "Q8_0" in out and "Q4_K_M" in out and "20.0 GB" in out
assert "F16" not in out.replace("96 GB VRAM", ""), "a 120 GB quant must be hidden"
assert "1 more quantisation" in out
ok("a GGUF repo: quants expanded, split files summed (2x10 -> 20 GB), "
   "oversized ones hidden but counted")

assert "org/locked" in out and "gated" in out and "licence" in out
ok("a gated repo is marked and numbered, not fatal to the whole find")

# ── numbered picks resolve to id (+ include glob for quants) ────────────────
picks = json.loads((TD / "var/run/find.json").read_text())["picks"]
assert picks[0] == {"id": "org/dense-fp8", "include": ""}
q4 = next(p for p in picks if "Q4_K_M" in p.get("include", ""))
assert q4["id"] == "org/tiny-GGUF" and q4["include"] == "tiny-Q4_K_M*"
assert modelfind.pick(1, root=TD) == ("org/dense-fp8", "")
assert modelfind.pick(99, root=TD) is None
ok("picks: row -> (id, include); out-of-range and missing file give None")

# ── the whole find ran under ONE lease, all requests audited ────────────────
evs = audit.tail(200)
verdicts = [e["verdict"] for e in evs]
assert verdicts.count("LEASE_OPEN") == 1 and verdicts.count("LEASE_CLOSE") == 1
allowed = [e for e in evs if e["verdict"] == "ALLOWED"]
assert len(allowed) >= 4, "search + one tree call per sizeable candidate"
assert any("model search" in e.get("purpose", "") for e in allowed)
ok("one lease wraps search + sizing; every request audited")

# ── pull --include: only the chosen quant lands ──────────────────────────────
qk = next(p for p in picks if p.get("include") == "tiny-Q2_K*")
got = pull.pull(qk["id"], root=TD, include=qk["include"], say=lambda m: None)
assert (got / "tiny-Q2_K.gguf").read_bytes() == QBLOB
assert not (got / "tiny-Q8_0.gguf").exists()
assert not (got / "README.md").exists()
manifest = json.loads((got / ".pull.json").read_text())
assert list(manifest["files"]) == ["tiny-Q2_K.gguf"]
assert pull.pulled(TD, qk["id"]) == got
ok("pull --include fetches ONLY the chosen quant (sha-verified, manifest ok)")

# a second include-pull merges the manifest instead of orphaning the first
(got / "fake-Q8.gguf").write_bytes(b"x")
mf = json.loads((got / ".pull.json").read_text())
mf["files"]["fake-Q8.gguf"] = {"size": 1, "sha256": ""}
(got / ".pull.json").write_text(json.dumps(mf))
pull.pull(qk["id"], root=TD, include=qk["include"], say=lambda m: None)
manifest = json.loads((got / ".pull.json").read_text())
assert set(manifest["files"]) == {"tiny-Q2_K.gguf", "fake-Q8.gguf"}
ok("a later quant pull MERGES the manifest — earlier quants stay verified")

# ── search honours limit and drops empties ───────────────────────────────────
from knowledgehost.amiga_net import broker  # noqa: E402
with broker.lease("t", "huggingface"):
    cands = modelfind.search("tiny", limit=2)
assert [c["id"] for c in cands] == ["org/dense-fp8", "org/dense-huge"]
ok("search(): honours limit, keeps hub order (most downloaded first)")

srv.shutdown()
print(f"modelfind_test: {OK} checks OK")
