"""Posture: the honest "how leaky is this box?" scan (AMIGA-OPS-01 B-18..23).

Read-only and advisory — it changes nothing, it REPORTS: what listens where,
whether the egress policy is really lease-only, whether running engines carry
the offline environment they were promised, whether an encrypted overlay
exists when the box is LAN-exposed, where the token sits.  Three lights:

    good     nothing to do
    warn     works, but worth fixing — the fix is named
    bad      leaking or refusing to work — fix first

and **unknown is its own state**: a check that could not run is never a pass
(B-21).  Lives beside amiga_net/, not inside it, so the broker core stays
under its size gate.
"""
from __future__ import annotations

import os
import stat
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

_ORDER = {"bad": 0, "unknown": 1, "warn": 2, "good": 3}


def _c(cid: str, name: str, state: str, detail: str, fix: str = "") -> dict:
    out = {"id": cid, "name": name, "state": state, "detail": detail}
    if fix:
        out["fix"] = fix
    return out


# ── pure parsers (testable without a live /proc) ─────────────────────────────

def parse_proc_tcp(text: str, v6: bool = False) -> list[tuple[str, int]]:
    """LISTEN sockets from /proc/net/tcp[6] -> [(kind, port)], kind in
    wildcard | loopback | local (a specific non-loopback address)."""
    out = []
    for ln in text.splitlines()[1:]:
        parts = ln.split()
        if len(parts) < 4 or parts[3] != "0A":        # 0A = LISTEN
            continue
        addr, _, port_hex = parts[1].rpartition(":")
        try:
            port = int(port_hex, 16)
        except ValueError:
            continue
        if set(addr) <= {"0"}:
            kind = "wildcard"
        elif (not v6 and addr.endswith("7F")) or \
                (v6 and addr == "00000000000000000000000001000000"):
            kind = "loopback"
        else:
            kind = "local"
        out.append((kind, port))
    return out


def environ_offline(env_bytes: bytes) -> bool:
    return b"HF_HUB_OFFLINE=1" in env_bytes.split(b"\0")


def wg_interfaces(sysroot: Path = Path("/sys/class/net")) -> list | None:
    """[(name, operstate)] of WireGuard interfaces, or None when the OS gives
    no way to tell (None must surface as unknown, never as 'no')."""
    if not sysroot.is_dir():
        return None
    out = []
    for d in sorted(sysroot.iterdir()):
        try:
            if "DEVTYPE=wireguard" in (d / "uevent").read_text():
                oper = "?"
                try:
                    oper = (d / "operstate").read_text().strip()
                except OSError:
                    pass
                out.append((d.name, oper))
        except OSError:
            continue
    return out


def expected_ports(cfg: dict) -> dict[int, str]:
    """port -> the vinur service that legitimately owns it."""
    out = {int(cfg.get("port") or 8770): "kb server"}
    for e in (cfg.get("serving") or {}).get("llms") or []:
        try:
            out[int(e.get("port") or 0)] = f"llm-{e.get('name')}"
        except (TypeError, ValueError):
            continue
    emb = (cfg.get("serving") or {}).get("embed") or {}
    out[int(emb.get("port") or 11437)] = "embed"
    out.pop(0, None)
    return out


# ── the checks ───────────────────────────────────────────────────────────────

def check_policy() -> dict:
    from .amiga_net import policy
    if not policy.POLICY_PATH.exists():
        return _c("policy", "Egress policy", "bad",
                  "egress.toml is missing — the broker denies everything, so "
                  "pulls and searches will fail",
                  "restore egress.toml from the repo")
    rules = policy.load()
    if not rules:
        return _c("policy", "Egress policy", "bad",
                  "egress.toml exists but no rule parses — deny-by-default "
                  "holds (nothing leaks) but every pull will fail",
                  "fix the TOML; `./vinur.sh net` shows what loaded")
    standing = [r.name for r in rules if r.enabled and not r.leased]
    off = [r.name for r in rules if not r.enabled]
    if standing:
        return _c("policy", "Egress policy", "warn",
                  f"rule(s) {', '.join(standing)} are STANDING — open at all "
                  "times, not lease-only",
                  "add ttl_seconds/max_uses to make them leases")
    detail = f"{len(rules)} rule(s), all lease-only — idle Vinur has zero standing egress"
    if off:
        detail += f"; disabled: {', '.join(off)}"
    return _c("policy", "Egress policy", "good", detail)


def check_audit() -> dict:
    from .amiga_net import audit
    d = audit.LOG_PATH.parent
    if not os.access(d if d.is_dir() else d.parent, os.W_OK):
        return _c("audit", "Audit log", "bad",
                  f"{audit.LOG_PATH} is not writable — egress would go unrecorded",
                  "fix permissions on var/log")
    evs = audit.tail(1)
    last = f"last event {evs[0]['ts']}" if evs else "no events yet"
    return _c("audit", "Audit log", "good", f"append-only at {audit.LOG_PATH} ({last})")


def check_listeners(cfg: dict, rows: list | None) -> dict:
    if rows is None:
        return _c("listen", "Listening sockets", "unknown",
                  "cannot read /proc/net/tcp on this OS yet — what binds where "
                  "was NOT verified",
                  "check by hand: ss -tlnp (Linux) / netstat -an")
    known = expected_ports(cfg)
    token = bool(str(cfg.get("auth_token") or "").strip())
    exposed_ours, exposed_other = [], []
    for kind, port in rows:
        if kind == "loopback":
            continue
        if port in known:
            exposed_ours.append(f"{known[port]} (:{port})")
        else:
            exposed_other.append(f":{port}")
    if exposed_ours and not token:
        return _c("listen", "Listening sockets", "bad",
                  f"{', '.join(sorted(set(exposed_ours)))} reachable from the "
                  "network with NO auth_token set",
                  "set auth_token in config.toml, or bind host = \"127.0.0.1\"")
    if exposed_ours:
        return _c("listen", "Listening sockets", "warn",
                  f"{', '.join(sorted(set(exposed_ours)))} reachable from the "
                  "network (token-gated — a declared deployment, not a leak)"
                  + (f"; not vinur's: {', '.join(sorted(set(exposed_other)))}"
                     if exposed_other else ""),
                  "keep it deliberate; prefer a WireGuard overlay for off-LAN")
    detail = "every vinur port is loopback-only"
    if exposed_other:
        detail += (f" — but OTHER software listens openly on "
                   f"{', '.join(sorted(set(exposed_other))[:8])}; not vinur's "
                   "to fix, worth knowing")
        return _c("listen", "Listening sockets", "warn", detail,
                  "identify them: ss -tlnp")
    return _c("listen", "Listening sockets", "good", detail)


def check_wireguard(wgs: list | None, lan_exposed: bool) -> dict:
    if wgs is None:
        return _c("wg", "WireGuard overlay", "unknown",
                  "cannot inspect network interfaces on this OS yet")
    up = [n for n, oper in wgs if oper in ("up", "unknown")]
    if up:
        return _c("wg", "WireGuard overlay", "good",
                  f"interface {', '.join(up)} is up — LAN/off-LAN peers ride "
                  "an encrypted overlay")
    if wgs:
        return _c("wg", "WireGuard overlay", "warn",
                  f"configured ({', '.join(n for n, _ in wgs)}) but not up",
                  "wg-quick up <iface>")
    if lan_exposed:
        return _c("wg", "WireGuard overlay", "warn",
                  "no overlay, and vinur ports are LAN-reachable — traffic to "
                  "them is only as safe as your LAN",
                  "consider WireGuard between your boxes (B-3's other half)")
    return _c("wg", "WireGuard overlay", "good",
              "not configured — and not needed: everything binds loopback")


def check_engines(cfg: dict) -> dict:
    from . import supervisor as sup
    st = sup.read_state()
    if not sup.alive(st.get("supervisor", 0)):
        return _c("engines", "Engines offline", "unknown",
                  "supervisor not running — no engine environments to verify "
                  "(they get the offline block at every spawn)")
    entries = {f"llm-{e.get('name')}": str(e.get("engine") or "")
               for e in (cfg.get("serving") or {}).get("llms") or []}
    checked, failed, containers = [], [], []
    for svc, pid in (st.get("services") or {}).items():
        eng = entries.get(svc)
        if not eng or not sup.alive(int(pid)):
            continue
        if eng == "container":
            containers.append(svc)              # env rides the argv -e flags
            continue
        try:
            env = Path(f"/proc/{pid}/environ").read_bytes()
        except OSError:
            return _c("engines", "Engines offline", "unknown",
                      f"cannot read {svc}'s environment on this OS — offline "
                      "env NOT verified")
        (checked if environ_offline(env) else failed).append(svc)
    if failed:
        return _c("engines", "Engines offline", "bad",
                  f"{', '.join(failed)} is running WITHOUT HF_HUB_OFFLINE=1 — "
                  "it could reach the hub",
                  "restart it (./vinur.sh restart <svc>); spawns always get "
                  "the offline block")
    bits = []
    if checked:
        bits.append(f"{', '.join(checked)}: offline env verified in /proc")
    if containers:
        bits.append(f"{', '.join(containers)}: offline flags ride the launch "
                    "argv (container env not directly inspectable)")
    return _c("engines", "Engines offline", "good",
              "; ".join(bits) or "no engines running — every spawn gets the "
              "offline block + null HF endpoint")


def check_token(cfg: dict) -> dict:
    tok = str(cfg.get("hf_token") or "").strip()
    if not tok:
        return _c("token", "HF token storage", "good",
                  "no token stored — nothing to leak (gated repos will ask for one)")
    cp = str(cfg.get("_config_path") or "")
    if cp and Path(cp).exists():
        mode = stat.S_IMODE(Path(cp).stat().st_mode)
        if mode & 0o044:
            return _c("token", "HF token storage", "warn",
                      f"config.toml holds the token and is group/world-readable "
                      f"(mode {oct(mode)[2:]})",
                      f"chmod 600 {cp}")
        return _c("token", "HF token storage", "good",
                  "in config.toml, owner-readable only; attached by the broker, "
                  "never by engines")
    return _c("token", "HF token storage", "good",
              "from the environment; attached by the broker, never by engines")


def check_proxy(cfg: dict) -> dict:
    from .serving import proxy_warning
    w = proxy_warning(cfg)
    if w:
        return _c("proxy", "Proxy hygiene", "warn", w,
                  "export no_proxy=localhost,127.0.0.1,::1")
    return _c("proxy", "Proxy hygiene", "good",
              "no shell proxy misconfiguration (loopback stays local)")


def check_installs() -> dict:
    return _c("installs", "Install-time fetches", "warn",
              "install/update scripts (uv, pip, git, llama.cpp fetch) reach the "
              "network directly — confinement begins POST-install by design",
              "run installs deliberately; runtime egress is broker-only")


# ── the scan ─────────────────────────────────────────────────────────────────

def scan(cfg: dict) -> dict:
    """All checks + a summary.  Overall = the worst state present; unknown
    outranks good because unverified is not verified."""
    try:
        rows = parse_proc_tcp(Path("/proc/net/tcp").read_text())
        rows += parse_proc_tcp(Path("/proc/net/tcp6").read_text(), v6=True)
    except OSError:
        rows = None
    lan = bool(rows) and any(k != "loopback" and p in expected_ports(cfg)
                             for k, p in rows)
    checks = [check_policy(), check_audit(), check_listeners(cfg, rows),
              check_wireguard(wg_interfaces(), lan), check_engines(cfg),
              check_token(cfg), check_proxy(cfg), check_installs()]
    counts = {"good": 0, "warn": 0, "bad": 0, "unknown": 0}
    for c in checks:
        counts[c["state"]] = counts.get(c["state"], 0) + 1
    overall = min((c["state"] for c in checks), key=lambda s: _ORDER[s])
    return {"checks": checks, "summary": {**counts, "overall": overall,
                                          "at": time.time()}}
