"""Acceptance test for VINUR-SHIP-01 S1 — the machine seam + OS service layer.

  * supervisor.status_data / `status --json`: one machine-readable collection
    (quiet box, live box with up/dead/failed/standby/held services, stale state).
  * knowledgehost/service.py: systemd unit + launchd agent generation, dry-run
    previews (file + exact commands, nothing written), install writes the file,
    uninstall removes it, Windows refuses with the named interim.

Run:  python tests/ship_test.py     (stdlib only)
"""
import contextlib
import io
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from knowledgehost import service as S
from knowledgehost import supervisor as SUP


def check(label, cond):
    if cond:
        print(f"  ok  {label}")
    else:
        print(f"  FAIL  {label}")
        check.failed += 1
check.failed = 0


def main():
    with tempfile.TemporaryDirectory() as td:
        state0, logs0 = SUP.STATE, SUP.LOGS
        SUP.STATE = Path(td) / "run" / "supervisor.json"
        SUP.LOGS = Path(td) / "log"
        try:
            # ── quiet box ────────────────────────────────────────────────────
            d = SUP.status_data()
            check("quiet box: running=false, no stale flag, services empty",
                  d["running"] is False and d["stale"] is False
                  and d["services"] == [] and d["repo"])
            check("panel url read from config", d["panel"].startswith("http://"))
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                rc = SUP.main(["status", "--json"])
            j = json.loads(buf.getvalue())
            check("status --json prints the same dict; exit 1 when down",
                  rc == 1 and j["running"] is False)

            # ── a live-looking state file ────────────────────────────────────
            dead = subprocess.Popen(["true"])
            dead.wait()
            SUP.STATE.parent.mkdir(parents=True, exist_ok=True)
            SUP.LOGS.mkdir(parents=True, exist_ok=True)
            (SUP.LOGS / "kb.log").write_text("boom: port already in use\n")
            SUP.STATE.write_text(json.dumps({
                "supervisor": os.getpid(),
                "services": {"kb": dead.pid, "embed": os.getpid()},
                "hints": {"embed": ":8069"},
                "failed": {"kb2": "gave up after 5 restarts"},
                "standby": {"secondary": "llm-secondary"},
                "held": ["reranker"],
            }))
            # kb2 must appear in services to be reported as failed
            st = json.loads(SUP.STATE.read_text())
            st["services"]["kb2"] = dead.pid
            SUP.STATE.write_text(json.dumps(st))
            d = SUP.status_data()
            states = {s["name"]: s["state"] for s in d["services"]}
            notes = {s["name"]: s["note"] for s in d["services"]}
            check("live box: up/dead/failed/standby/stopped all classified",
                  d["running"] and states == {"kb": "dead", "embed": "up",
                                              "kb2": "failed",
                                              "llm-secondary": "standby",
                                              "reranker": "stopped"})
            check("dead service carries its last log line; standby names the swap",
                  "port already in use" in notes["kb"]
                  and "./vinur.sh swap secondary" in notes["llm-secondary"])
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                rc = SUP.main(["status"])
            text = buf.getvalue()
            check("text renderer keeps the classic lines from the same data",
                  rc == 0 and "supervisor pid=" in text and "FAILED" in text
                  and "standby — './vinur.sh swap secondary' loads it" in text
                  and "stopped — by request" in text)

            # ── stale state ──────────────────────────────────────────────────
            SUP.STATE.write_text(json.dumps({"supervisor": dead.pid}))
            d = SUP.status_data()
            check("stale state flagged (dead supervisor pid + leftover file)",
                  d["running"] is False and d["stale"] is True)
        finally:
            SUP.STATE, SUP.LOGS = state0, logs0

        # ── OS service layer ─────────────────────────────────────────────────
        os.environ["VINUR_SERVICE_DIR"] = os.path.join(td, "svc")
        try:
            u = S.unit_text()
            check("systemd unit: foreground run + repo workdir + restart policy",
                  "-m knowledgehost.supervisor run" in u
                  and f"WorkingDirectory={S.ROOT}" in u
                  and "Restart=on-failure" in u and "WantedBy=default.target" in u)
            p = S.plist_text()
            check("launchd agent: label + RunAtLoad + KeepAlive + log path",
                  S.AGENT_LABEL in p and "<key>RunAtLoad</key><true/>" in p
                  and "KeepAlive" in p and "supervisor.log" in p)
            check("per-platform file naming",
                  S.service_file("Linux").name == "vinur.service"
                  and S.service_file("Darwin").name == "is.vinur.host.plist")

            r = S.install(dry_run=True, system="Linux")
            check("install --dry-run: nothing written, exact commands previewed",
                  not Path(r["file"]).exists()
                  and r["commands"] == ["systemctl --user daemon-reload",
                                        "systemctl --user enable --now vinur.service"])
            r = S.install(system="Linux")
            check("install writes the unit (manager errors reported, not raised)",
                  Path(r["file"]).exists() and isinstance(r["errors"], list))
            r2 = S.uninstall(dry_run=True, system="Linux")
            check("uninstall --dry-run leaves the unit in place",
                  Path(r["file"]).exists() and "disable" in r2["commands"][0])
            r3 = S.uninstall(system="Linux")
            check("uninstall removes the unit", r3["removed"]
                  and not Path(r["file"]).exists())
            st = S.status(system="Linux")
            check("status: absent file reads 'not installed'",
                  st["installed"] is False and st["active"] == "not installed")
            try:
                S.install(system="Windows")
                check("Windows refuses with the named interim", False)
            except ValueError as e:
                check("Windows refuses with the named interim",
                      "Task Scheduler" in str(e) and "S6" in str(e))
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                rc = S.main(["install", "--dry-run"])
            check("CLI: service install --dry-run exits 0 and previews",
                  rc == 0 and "WOULD write" in buf.getvalue())
        finally:
            del os.environ["VINUR_SERVICE_DIR"]

    print()
    if check.failed:
        print(f"{check.failed} FAILURE(S)")
        return 1
    print("ALL OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
