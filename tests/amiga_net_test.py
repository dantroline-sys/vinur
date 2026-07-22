#!/usr/bin/env python
"""The egress broker (amiga_net) — G-9's required coverage and then some:
ALLOW, DENY, lease expiry/exhaustion, AUTH_REJECT, plus resume, sha256
verification, audit hygiene (no bodies), and the pull path end-to-end against
a fake hub on loopback (allowed here because the TEST policy declares a rule
for 127.0.0.1 — deny-by-default means the tests must say so too)."""
import hashlib
import json
import os
import sys
import tempfile
import threading
import time
import urllib.error
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ["AMIGA_FETCH_ENGINE"] = "stdlib"     # deterministic engine in tests

from knowledgehost.amiga_net import audit, broker, policy, pull, status  # noqa: E402

OK = 0


def ok(label):
    global OK
    OK += 1
    print(f"  ok {OK:2d}  {label}")


# ── a fake hub on loopback ───────────────────────────────────────────────────
BLOB = os.urandom(300_000)
BLOB_SHA = hashlib.sha256(BLOB).hexdigest()
CONFIG = b'{"architectures": ["Fake"]}'


class Hub(BaseHTTPRequestHandler):
    require_auth = False

    def log_message(self, *a):
        pass

    def do_GET(self):
        if Hub.require_auth and self.headers.get("Authorization") != "Bearer hf_good":
            self.send_response(401)
            self.end_headers()
            return
        if self.path.startswith("/api/models/org/tiny/tree/"):
            body = json.dumps([
                {"type": "file", "path": "config.json", "size": len(CONFIG)},
                {"type": "file", "path": "model.safetensors", "size": len(BLOB),
                 "lfs": {"oid": BLOB_SHA}},
                {"type": "file", "path": "README.md", "size": 5},
                {"type": "file", "path": "pytorch_model.bin", "size": 9},
                {"type": "file", "path": "original/ckpt.pt", "size": 9},
            ]).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if self.path == "/org/tiny/resolve/main/slow.bin":
            self.send_response(200)            # dribbled out so the watcher ticks
            self.send_header("Content-Length", str(len(BLOB)))
            self.end_headers()
            for i in range(0, len(BLOB), 60_000):
                self.wfile.write(BLOB[i:i + 60_000])
                self.wfile.flush()
                time.sleep(0.08)
            return
        data = {"/org/tiny/resolve/main/config.json": CONFIG,
                "/org/tiny/resolve/main/model.safetensors": BLOB}.get(self.path)
        if data is None:
            self.send_response(404)
            self.end_headers()
            return
        rng = self.headers.get("Range")
        if rng and rng.startswith("bytes="):
            start = int(rng.split("=")[1].rstrip("-"))
            self.send_response(206)
            body = data[start:]
        else:
            self.send_response(200)
            body = data
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


srv = ThreadingHTTPServer(("127.0.0.1", 0), Hub)
threading.Thread(target=srv.serve_forever, daemon=True).start()
PORT = srv.server_address[1]
BASE = f"http://127.0.0.1:{PORT}"

# ── test policy + isolated state ─────────────────────────────────────────────
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
max_uses = 100
auth = "hf_token"

[[rule]]
name = "standing"
hosts = ["127.0.0.1"]
port = {PORT}
methods = ["GET"]
purpose = "a standing (non-leased) rule"
""")
policy.POLICY_PATH = POL
policy.LEASE_DIR = TD / "run"
audit.LOG_PATH = TD / "egress.jsonl"
pull.HF = BASE


def verdicts():
    return [e["verdict"] for e in audit.tail(200)]


# ── the shipped production policy parses and is lease-only ──────────────────
prod = policy.load(Path(__file__).resolve().parent.parent / "egress.toml")
assert prod, "the repo's egress.toml must parse"
assert all(r.leased for r in prod), "vinur ships NO standing egress rules"
assert all(r.purpose for r in prod), "every rule carries a plain-language purpose"
ok("shipped egress.toml: parses, lease-only (idle Vinur has zero standing egress)")

# ── DENY: no rule ────────────────────────────────────────────────────────────
try:
    broker.request("nope", "https://evil.example.com/x")
    raise AssertionError("must deny")
except broker.EgressDenied as e:
    assert "no rule" in str(e)
assert verdicts()[-1] == "DENIED"
ok("deny-by-default: unlisted destination refused and audited")

# ── standing rule allows without a lease ─────────────────────────────────────
body = broker.request("standing fetch", f"{BASE}/org/tiny/resolve/main/config.json")
assert body == CONFIG
assert verdicts()[-1] == "ALLOWED"
ok("a standing rule allows without a lease")

# ── leased rule: nothing until opened; open/close are paired events ─────────
# make the leased rule the only match by narrowing the standing rule away
POL.write_text(POL.read_text().replace('methods = ["GET"]\npurpose = "a standing',
                                       'methods = ["HEAD"]\npurpose = "a standing'))
try:
    broker.request("no lease", f"{BASE}/org/tiny/resolve/main/config.json")
    raise AssertionError("must deny without an open lease")
except broker.EgressDenied as e:
    assert "lease" in str(e)
ok("a leased rule grants NOTHING between operations")

with broker.lease("test op", "huggingface"):
    assert broker.request("in lease", f"{BASE}/org/tiny/resolve/main/config.json") == CONFIG
    assert policy.live_leases(policy.load())[0]["rule"] == "huggingface"
assert policy.live_leases(policy.load()) == []
v = verdicts()
assert v.count("LEASE_OPEN") == 1 and v.count("LEASE_CLOSE") == 1
assert v.index("LEASE_OPEN") < v.index("ALLOWED", v.index("LEASE_OPEN")) < v.index("LEASE_CLOSE")
ok("lease: open -> allowed -> closed, paired events, no live lease after")

# ── lease expiry ─────────────────────────────────────────────────────────────
rule = next(r for r in policy.load() if r.name == "huggingface")
st = policy.lease_open(rule, "expiring")
st["expires"] = 1.0                                   # forced into the past
policy._lease_path(rule.name).write_text(json.dumps(st))
try:
    broker.request("expired", f"{BASE}/org/tiny/resolve/main/config.json")
    raise AssertionError("expired lease must deny")
except broker.EgressDenied:
    pass
policy.lease_close(rule.name)
ok("an expired lease is treated as absent (self-revoking)")

# ── lease exhaustion ─────────────────────────────────────────────────────────
st = policy.lease_open(rule, "exhausting")
st["uses"] = 100
policy._lease_path(rule.name).write_text(json.dumps(st))
try:
    broker.request("exhausted", f"{BASE}/org/tiny/resolve/main/config.json")
    raise AssertionError("exhausted lease must deny")
except broker.EgressDenied:
    pass
policy.lease_close(rule.name)
ok("an exhausted lease (max_uses) is treated as absent")

# ── AUTH_REJECT ──────────────────────────────────────────────────────────────
Hub.require_auth = True
os.environ["HF_TOKEN"] = "hf_bad"                     # rule auth resolves via env
with broker.lease("auth test", "huggingface"):
    try:
        broker.request("auth test", f"{BASE}/org/tiny/resolve/main/config.json")
        raise AssertionError("401 must raise")
    except urllib.error.HTTPError:
        pass
assert "AUTH_REJECT" in verdicts()
os.environ["HF_TOKEN"] = "hf_good"
with broker.lease("auth ok", "huggingface"):
    assert broker.request("auth ok", f"{BASE}/org/tiny/resolve/main/config.json") == CONFIG
ok("AUTH_REJECT audited on 401; the broker attaches the rule's token itself")
Hub.require_auth = False

# ── download: resume + sha256 ────────────────────────────────────────────────
dest = TD / "blob.bin"
part = dest.with_suffix(".bin.part")
part.write_bytes(BLOB[:100_000])                      # a prior interrupted fetch
with broker.lease("dl", "huggingface"):
    broker.download("dl", f"{BASE}/org/tiny/resolve/main/model.safetensors",
                    dest, sha256=BLOB_SHA)
assert dest.read_bytes() == BLOB and not part.exists()
dl_ev = [e for e in audit.tail(5) if e["verdict"] == "ALLOWED"][-1]
assert dl_ev["bytes_in"] == len(BLOB) - 100_000, dl_ev  # only the resumed tail
assert "resumed" in dl_ev.get("detail", "")
ok("download resumes from a .part file (Range) and verifies sha256")

with broker.lease("dl-bad", "huggingface"):
    try:
        broker.download("dl-bad", f"{BASE}/org/tiny/resolve/main/config.json",
                        TD / "bad.bin", sha256="0" * 64)
        raise AssertionError("sha mismatch must raise")
    except broker.EgressDenied as e:
        assert "sha256 mismatch" in str(e)
assert not (TD / "bad.bin").exists() and not (TD / "bad.bin.part").exists()
ok("a sha256 mismatch discards the file — corrupt data never lands")

# ── progress: one format, engine-agnostic, measured from the .part file ─────
assert broker._progress_line(10 * 2**30, 20 * 2**30, 50 * 2**20) == \
    "      … 10.00 / 20.00 GB (50%) · 50 MB/s · ~3m24s left"
assert broker._progress_line(3 * 2**29, 0, 2**20).endswith("1.50 GB · 1 MB/s")
os.environ["AMIGA_PROGRESS_S"] = "0.05"
plines = []
with broker.lease("slow", "huggingface"):
    broker.download("slow", f"{BASE}/org/tiny/resolve/main/slow.bin",
                    TD / "slow.bin", size=len(BLOB), progress=plines.append)
del os.environ["AMIGA_PROGRESS_S"]
assert (TD / "slow.bin").read_bytes() == BLOB
assert plines and all("GB" in ln and "MB/s" in ln for ln in plines), plines
ok("download progress: periodic '… have / total GB (%) · MB/s · ETA' lines "
   "from the broker itself — identical for aria2c, wget, and stdlib")

# ── the transfer engine: env override > config fetch_engine > auto-detect ───
engcfg = TD / "engcfg.toml"
engcfg.write_text('fetch_engine = "wget"\n')
del os.environ["AMIGA_FETCH_ENGINE"]
os.environ["KNOWLEDGEHOST_CONFIG"] = str(engcfg)
assert broker._engine() == "wget"
engcfg.write_text('fetch_engine = ""\n')
assert broker._engine() in ("aria2c", "wget", "stdlib")
del os.environ["KNOWLEDGEHOST_CONFIG"]
os.environ["AMIGA_FETCH_ENGINE"] = "stdlib"          # tests stay deterministic
assert broker._engine() == "stdlib"
ok("fetch engine: AMIGA_FETCH_ENGINE > fetch_engine key (Network tab) > auto")

# ── audit hygiene ────────────────────────────────────────────────────────────
raw = audit.LOG_PATH.read_text()
assert b"architectures"[0:0] == b"" and "architectures" not in raw
assert "hf_good" not in raw and "hf_bad" not in raw
ok("the audit log holds no bodies and no tokens")

# ── pull end-to-end ──────────────────────────────────────────────────────────
said = []
got = pull.pull("org/tiny", root=TD, say=said.append)
assert (got / "config.json").read_bytes() == CONFIG
assert (got / "model.safetensors").read_bytes() == BLOB
assert not (got / "README.md").exists()
assert not (got / "pytorch_model.bin").exists(), "legacy pickle weights: never fetched"
assert not (got / "original").exists()
manifest = json.loads((got / ".pull.json").read_text())
assert manifest["files"]["model.safetensors"]["sha256"] == BLOB_SHA
ok("pull: snapshot lands in the store, junk skipped, manifest written")

assert pull.pulled(TD, "org/tiny") == got
(got / "model.safetensors").write_bytes(b"short")     # truncate one file
assert pull.pulled(TD, "org/tiny") is None
ok("pulled(): complete-only — a truncated snapshot never reads as ready")

# a second pull heals the truncation (size mismatch -> refetch)
pull.pull("org/tiny", root=TD, say=lambda m: None)
assert pull.pulled(TD, "org/tiny") == got
ok("re-pull heals a damaged file (idempotent, size-checked)")

# ── the kill switches: disable a rule, revoke a lease ───────────────────────
policy.set_rule_enabled("huggingface", False, POL)
assert next(r for r in policy.load() if r.name == "huggingface").enabled is False
try:
    with broker.lease("nope", "huggingface"):
        pass
    raise AssertionError("a disabled rule must refuse a lease")
except broker.EgressDenied as e:
    assert "disabled" in str(e)
policy.set_rule_enabled("huggingface", True, POL)
with broker.lease("re-enabled", "huggingface"):
    assert broker.request("re-enabled", f"{BASE}/org/tiny/resolve/main/config.json") == CONFIG
txt = POL.read_text()
assert "enabled = true" in txt and txt.count("[[rule]]") == 2, \
    "the toggle edits ONE rule block in place"
# revoke: an open lease dies mid-operation, the next request is refused
st = policy.lease_open(next(r for r in policy.load() if r.name == "huggingface"), "doomed")
policy.lease_close("huggingface")
try:
    broker.request("after revoke", f"{BASE}/org/tiny/resolve/main/config.json")
    raise AssertionError("a revoked lease must deny")
except broker.EgressDenied:
    pass
ok("kill switches: rule disable refuses leases+requests; revoke ends a live lease")

# ── traffic rollup: some statistics, nothing too detailed ───────────────────
stats = audit.summarize(5000)
hf = next(x for x in stats["rules"] if x["rule"] == "huggingface")
assert hf["requests"] > 0 and hf["bytes_in"] > 0 and hf["last_ts"], hf
assert stats["denied_total"] > 0, "the denials above must be counted"
assert not any("architectures" in json.dumps(x) for x in stats["rules"]), \
    "stats carry counts, never content"
ok("summarize(): per-rule requests/bytes/denials with last activity")

# ── status is readable ───────────────────────────────────────────────────────
out = status.render(10)
assert "deny by default" in out and "test hub" in out
assert "LEASE_OPEN" in out or "ALLOWED" in out
ok("status: policy in plain language + recent events")

srv.shutdown()
print(f"amiga_net_test: {OK} checks OK")
