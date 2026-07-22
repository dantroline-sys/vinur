"""The append-only egress audit log (B-10): one JSON line per decision.

Fields: ts (UTC ISO), component, purpose, host, port, rule ('-' when none
matched), verdict (ALLOWED / DENIED / LEASE_OPEN / LEASE_CLOSE / AUTH_REJECT /
POSTURE), bytes_out, bytes_in.  Never request or response bodies — the log
says WHO talked to WHOM and WHY, not what was said.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
LOG_PATH = ROOT / "var" / "log" / "egress.jsonl"

VERDICTS = ("ALLOWED", "DENIED", "LEASE_OPEN", "LEASE_CLOSE", "AUTH_REJECT",
            "POSTURE")


def component() -> str:
    return os.environ.get("AMIGA_COMPONENT", "vinur")


def write(verdict: str, *, purpose: str = "", host: str = "", port: int = 0,
          rule: str = "-", bytes_out: int = 0, bytes_in: int = 0,
          detail: str = "", path: Path | None = None) -> dict:
    assert verdict in VERDICTS, verdict
    ev = {"ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
          "component": component(), "purpose": purpose[:200],
          "host": host, "port": port, "rule": rule, "verdict": verdict,
          "bytes_out": int(bytes_out), "bytes_in": int(bytes_in)}
    if detail:
        ev["detail"] = detail[:300]
    p = path or LOG_PATH
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "a") as f:
        f.write(json.dumps(ev) + "\n")
    return ev


def tail(n: int = 40, path: Path | None = None) -> list[dict]:
    p = path or LOG_PATH
    try:
        lines = p.read_bytes()[-262144:].decode("utf-8", "replace").splitlines()
    except OSError:
        return []
    out = []
    for ln in lines[-max(1, n):]:
        try:
            out.append(json.loads(ln))
        except ValueError:
            continue
    return out
