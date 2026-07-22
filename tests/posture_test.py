#!/usr/bin/env python
"""Posture (AMIGA-OPS-01 B-18..23): the parsers, each grader's bands, and the
whole scan — with the contract's core assertion held everywhere: UNKNOWN IS
NEVER A PASS."""
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from knowledgehost import posture  # noqa: E402
from knowledgehost.amiga_net import policy  # noqa: E402

OK = 0


def ok(label):
    global OK
    OK += 1
    print(f"  ok {OK:2d}  {label}")


# ── /proc/net/tcp parsing ────────────────────────────────────────────────────
TCP = (
    "  sl  local_address rem_address   st tx_queue rx_queue tr tm->when retrnsmt uid\n"
    "   0: 0100007F:2296 00000000:0000 0A 00000000:00000000 00:00000000 00000000 1000\n"
    "   1: 00000000:1F90 00000000:0000 0A 00000000:00000000 00:00000000 00000000 1000\n"
    "   2: 0500000A:0016 00000000:0000 0A 00000000:00000000 00:00000000 00000000    0\n"
    "   3: 0100007F:AAAA 00000000:0000 01 00000000:00000000 00:00000000 00000000 1000\n")
rows = posture.parse_proc_tcp(TCP)
assert ("loopback", 0x2296) in rows and ("wildcard", 0x1F90) in rows
assert ("local", 22) in rows
assert not any(p == 0xAAAA for _, p in rows), "ESTABLISHED is not LISTEN"
TCP6 = (
    "  sl  local_address                         rem_address st\n"
    "   0: 00000000000000000000000001000000:22B8 00000000000000000000000000000000:0000 0A x x x x 0\n"
    "   1: 00000000000000000000000000000000:22B9 00000000000000000000000000000000:0000 0A x x x x 0\n")
rows6 = posture.parse_proc_tcp(TCP6, v6=True)
assert ("loopback", 0x22B8) in rows6 and ("wildcard", 0x22B9) in rows6
ok("parse_proc_tcp: LISTEN only; loopback/wildcard/local for v4 and v6")

# ── engine environment ───────────────────────────────────────────────────────
assert posture.environ_offline(b"PATH=/x\0HF_HUB_OFFLINE=1\0Y=2\0")
assert not posture.environ_offline(b"PATH=/x\0HF_HUB_OFFLINE=11\0")
assert not posture.environ_offline(b"")
ok("environ_offline: exact HF_HUB_OFFLINE=1 in a null-separated environ")

# ── wireguard detection + grading ────────────────────────────────────────────
with tempfile.TemporaryDirectory() as td:
    net = Path(td) / "net"
    (net / "wg0").mkdir(parents=True)
    (net / "wg0" / "uevent").write_text("DEVTYPE=wireguard\nINTERFACE=wg0\n")
    (net / "wg0" / "operstate").write_text("up\n")
    (net / "eth0").mkdir()
    (net / "eth0" / "uevent").write_text("INTERFACE=eth0\n")
    assert posture.wg_interfaces(net) == [("wg0", "up")]
    assert posture.wg_interfaces(Path(td) / "nope") is None
assert posture.check_wireguard(None, False)["state"] == "unknown", \
    "can't-tell must NOT read as fine"
assert posture.check_wireguard([], True)["state"] == "warn"
assert posture.check_wireguard([], False)["state"] == "good"
assert posture.check_wireguard([("wg0", "up")], True)["state"] == "good"
assert posture.check_wireguard([("wg0", "down")], False)["state"] == "warn"
ok("wireguard: sysfs detection; unknown/absent-but-exposed/up graded honestly")

# ── listener grading ─────────────────────────────────────────────────────────
CFG = {"port": 8770, "auth_token": "",
       "serving": {"llms": [{"name": "a", "port": 11438}], "embed": {"port": 11437}}}
assert posture.check_listeners(CFG, None)["state"] == "unknown"
assert posture.check_listeners(
    CFG, [("loopback", 8770), ("loopback", 11438)])["state"] == "good"
bad = posture.check_listeners(CFG, [("wildcard", 8770)])
assert bad["state"] == "bad" and "auth_token" in bad["fix"]
warn = posture.check_listeners({**CFG, "auth_token": "t"}, [("wildcard", 8770)])
assert warn["state"] == "warn" and "token-gated" in warn["detail"]
other = posture.check_listeners(CFG, [("wildcard", 22), ("loopback", 8770)])
assert other["state"] == "warn" and ":22" in other["detail"]
ok("listeners: LAN-bound vinur w/o token = UNSAFE; with token = declared; "
   "someone else's open port is named, not blamed")

# ── policy grading (patched policy file) ─────────────────────────────────────
with tempfile.TemporaryDirectory() as td:
    pol = Path(td) / "egress.toml"
    keep = policy.POLICY_PATH
    try:
        policy.POLICY_PATH = pol
        assert posture.check_policy()["state"] == "bad"          # missing file
        pol.write_text("not toml [[[")
        assert posture.check_policy()["state"] == "bad"          # unparseable
        pol.write_text('[[rule]]\nname = "a"\nhosts = ["x.com"]\n'
                       'purpose = "t"\nttl_seconds = 60\n')
        assert posture.check_policy()["state"] == "good"
        pol.write_text('[[rule]]\nname = "a"\nhosts = ["x.com"]\npurpose = "t"\n')
        c = posture.check_policy()
        assert c["state"] == "warn" and "STANDING" in c["detail"]
    finally:
        policy.POLICY_PATH = keep
ok("policy: missing/broken = bad (but closed), standing rules = warn, "
   "lease-only = good")

# ── token storage grading ────────────────────────────────────────────────────
with tempfile.TemporaryDirectory() as td:
    cp = Path(td) / "config.toml"
    cp.write_text('hf_token = "hf_x"\n')
    os.chmod(cp, 0o644)
    c = posture.check_token({"hf_token": "hf_x", "_config_path": str(cp)})
    assert c["state"] == "warn" and "chmod 600" in c["fix"]
    os.chmod(cp, 0o600)
    assert posture.check_token(
        {"hf_token": "hf_x", "_config_path": str(cp)})["state"] == "good"
assert posture.check_token({"hf_token": ""})["state"] == "good"
ok("token: world-readable config with a token = warn + the chmod named")

# ── the whole scan, on this machine ──────────────────────────────────────────
res = posture.scan({"port": 8770, "auth_token": "", "hf_token": "",
                    "serving": {"llms": [], "embed": {}}})
assert len(res["checks"]) == 8
assert all(c["state"] in ("good", "warn", "bad", "unknown") for c in res["checks"])
assert all(c["detail"] for c in res["checks"]), "every light carries its reason"
assert set(res["summary"]) >= {"good", "warn", "bad", "unknown", "overall"}
assert res["summary"]["overall"] != "good", \
    "install-time fetches are a standing, honest warn — overall is never a clean sheet"
for c in res["checks"]:
    if c["state"] in ("warn", "bad"):
        assert c.get("fix") or c["id"] == "listen", f"{c['id']}: non-green needs a fix line"
ok("scan(): 8 checks, every state legal, every light explained, overall honest")

print(f"posture_test: {OK} checks OK")
