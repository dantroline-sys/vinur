"""OS-service integration — vinur as a login service (VINUR-SHIP-01 S1).

`./vinur.sh service install` registers the box with the operating system's
own service manager so vinur starts at login and is revived by the OS if the
supervisor itself dies (the supervisor already babysits its children; this
layer babysits the supervisor):

  * Linux — a systemd USER unit (~/.config/systemd/user/vinur.service),
    `Type=simple` over `supervisor run` (the foreground mode; journald gets
    the log stream: `journalctl --user -u vinur`).
  * macOS — a launchd agent (~/Library/LaunchAgents/is.vinur.host.plist),
    RunAtLoad + KeepAlive, logging to var/log/supervisor.log.
  * Windows — not yet (VINUR-SHIP-01 S6); the refusal names the interim
    (Task Scheduler on the same `supervisor run` command).

Everything is inspectable before it acts: `--dry-run` writes nothing and
prints the file and the exact commands.  Headless-first by construction —
the GUI shell calls this same module.

Stdlib only.  Test seams: VINUR_SERVICE_DIR overrides the target directory;
every function takes `system=` so both platforms are testable anywhere.
"""
from __future__ import annotations

import os
import platform
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

UNIT_NAME = "vinur.service"
AGENT_LABEL = "is.vinur.host"


def _python() -> str:
    """The interpreter the service should run — the repo's own venv when it
    exists (what vinur.sh picks), else whoever is running us."""
    venv = ROOT / ".venv" / "bin" / "python3"
    return str(venv) if venv.exists() else sys.executable


def service_dir(system: str | None = None) -> Path:
    env = os.environ.get("VINUR_SERVICE_DIR")
    if env:
        return Path(env)
    system = system or platform.system()
    if system == "Darwin":
        return Path.home() / "Library" / "LaunchAgents"
    return Path.home() / ".config" / "systemd" / "user"


def service_file(system: str | None = None) -> Path:
    system = system or platform.system()
    name = f"{AGENT_LABEL}.plist" if system == "Darwin" else UNIT_NAME
    return service_dir(system) / name


def unit_text() -> str:
    return f"""[Unit]
Description=Vinur knowledge host (supervisor for the kb + declared LMs)
After=network-online.target

[Service]
Type=simple
WorkingDirectory={ROOT}
ExecStart={_python()} -m knowledgehost.supervisor run
Restart=on-failure
RestartSec=5

[Install]
WantedBy=default.target
"""


def plist_text() -> str:
    log = ROOT / "var" / "log" / "supervisor.log"
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key><string>{AGENT_LABEL}</string>
    <key>ProgramArguments</key>
    <array>
        <string>{_python()}</string>
        <string>-m</string>
        <string>knowledgehost.supervisor</string>
        <string>run</string>
    </array>
    <key>WorkingDirectory</key><string>{ROOT}</string>
    <key>RunAtLoad</key><true/>
    <key>KeepAlive</key><dict><key>SuccessfulExit</key><false/></dict>
    <key>StandardOutPath</key><string>{log}</string>
    <key>StandardErrorPath</key><string>{log}</string>
</dict>
</plist>
"""


def _commands(action: str, system: str) -> list[list[str]]:
    f = str(service_file(system))
    if system == "Darwin":
        if action == "install":
            return [["launchctl", "unload", f], ["launchctl", "load", "-w", f]]
        return [["launchctl", "unload", "-w", f]]
    if action == "install":
        return [["systemctl", "--user", "daemon-reload"],
                ["systemctl", "--user", "enable", "--now", UNIT_NAME]]
    return [["systemctl", "--user", "disable", "--now", UNIT_NAME],
            ["systemctl", "--user", "daemon-reload"]]


def _refuse_windows(system: str) -> None:
    if system == "Windows":
        raise ValueError(
            "Windows service install is not wired yet (VINUR-SHIP-01 S6) — "
            "interim: register a Task Scheduler logon task running "
            f"'{_python()} -m knowledgehost.supervisor run' in {ROOT}")


def install(*, dry_run: bool = False, system: str | None = None) -> dict:
    """Write the unit/agent and enable it.  dry_run: print, write nothing."""
    system = system or platform.system()
    _refuse_windows(system)
    f = service_file(system)
    text = plist_text() if system == "Darwin" else unit_text()
    cmds = _commands("install", system)
    res = {"file": str(f), "text": text, "commands": [" ".join(c) for c in cmds],
           "ran": [], "errors": []}
    if dry_run:
        return res
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(text)
    for c in cmds:
        try:
            p = subprocess.run(c, capture_output=True, text=True, timeout=30)
            res["ran"].append(" ".join(c))
            # launchctl unload of a not-loaded agent is a normal first-install path
            if p.returncode != 0 and not (system == "Darwin" and c[1] == "unload"):
                res["errors"].append(f"{' '.join(c)}: {p.stderr.strip() or p.returncode}")
        except (OSError, subprocess.TimeoutExpired) as e:
            res["errors"].append(f"{' '.join(c)}: {e}")
    return res


def uninstall(*, dry_run: bool = False, system: str | None = None) -> dict:
    system = system or platform.system()
    _refuse_windows(system)
    f = service_file(system)
    cmds = _commands("uninstall", system)
    res = {"file": str(f), "commands": [" ".join(c) for c in cmds],
           "ran": [], "errors": [], "removed": False}
    if dry_run:
        return res
    for c in cmds:
        try:
            p = subprocess.run(c, capture_output=True, text=True, timeout=30)
            res["ran"].append(" ".join(c))
            if p.returncode != 0:
                res["errors"].append(f"{' '.join(c)}: {p.stderr.strip() or p.returncode}")
        except (OSError, subprocess.TimeoutExpired) as e:
            res["errors"].append(f"{' '.join(c)}: {e}")
    if f.exists():
        f.unlink()
        res["removed"] = True
    return res


def status(system: str | None = None) -> dict:
    """{installed, file, active} — active is the OS manager's verdict, or a
    plain 'unknown (<why>)' when it can't be asked (no session bus, etc.)."""
    system = system or platform.system()
    f = service_file(system)
    out = {"installed": f.exists(), "file": str(f), "active": "unknown"}
    if not out["installed"]:
        out["active"] = "not installed"
        return out
    try:
        if system == "Darwin":
            p = subprocess.run(["launchctl", "list", AGENT_LABEL],
                               capture_output=True, text=True, timeout=10)
            out["active"] = "loaded" if p.returncode == 0 else "not loaded"
        elif system == "Windows":
            out["active"] = "unknown (Windows: check Task Scheduler)"
        else:
            p = subprocess.run(["systemctl", "--user", "is-active", UNIT_NAME],
                               capture_output=True, text=True, timeout=10)
            out["active"] = (p.stdout or p.stderr).strip() or "unknown"
    except (OSError, subprocess.TimeoutExpired) as e:
        out["active"] = f"unknown ({e})"
    return out


def main(args: list[str]) -> int:
    verb = args[0] if args else "status"
    dry = "--dry-run" in args
    try:
        if verb == "install":
            r = install(dry_run=dry)
            print(("WOULD write " if dry else "wrote ") + r["file"])
            for c in r["commands"]:
                print(("  would run: " if dry else "  ran: ") + c)
            for e in r["errors"]:
                print(f"  ERROR: {e}")
            if not dry and not r["errors"]:
                print("vinur is now a login service — it starts with your session "
                      "and the OS revives it if the supervisor dies.")
            return 1 if r["errors"] else 0
        if verb == "uninstall":
            r = uninstall(dry_run=dry)
            print(("WOULD remove " if dry else
                   ("removed " if r.get("removed") else "no file at ")) + r["file"])
            for e in r["errors"]:
                print(f"  ERROR: {e}")
            return 1 if r["errors"] else 0
        if verb == "status":
            s = status()
            print(f"service file: {s['file']}  ({'installed' if s['installed'] else 'absent'})")
            print(f"OS manager:   {s['active']}")
            return 0
    except ValueError as e:
        print(str(e), file=sys.stderr)
        return 1
    print("usage: ./vinur.sh service install|uninstall|status [--dry-run]")
    return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
