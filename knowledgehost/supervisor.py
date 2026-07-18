"""Process supervisor for a standalone Vinur box — `./vinur.sh` is its shim.

One resident process owns every service this machine serves — the declared
LMs (knowledgehost.serving), the embed endpoint, the CPU reranker, and the
kb server itself — as direct children in their own process groups:

    python -m knowledgehost.supervisor start|stop|restart [svc]|status|logs [svc]

Children log to var/log/<name>.log (truncated on start).  The watchdog
revives a dead service with backoff and gives up after MAX_RESTARTS in
WINDOW_S, leaving the reason visible in `status`.  State (exact pids) lives
in var/run/supervisor.json so a stale run is recovered precisely — no
pattern-matched pkill.

Runs on the .venv interpreter (vinur.sh picks it): config parsing needs
tomllib, so the floor is the package's own (>= 3.11).  Stdlib only.
"""
from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parent.parent
LOGS = ROOT / "var" / "log"
STATE = ROOT / "var" / "run" / "supervisor.json"

GRACE_S = 8            # TERM -> KILL budget at shutdown
TICK_S = 2.0           # watchdog cadence
MAX_RESTARTS = 5       # per service, within WINDOW_S, then give up
WINDOW_S = 600.0


# ── config → service list ────────────────────────────────────────────────────

def load_cfg() -> dict:
    from .config import load_config
    p = ROOT / "config.toml"
    return load_config(str(p) if p.exists() else None)


def services_for(cfg: dict) -> list[dict]:
    """Each entry: name, cmd (argv), env (extra), hint (port for status)."""
    py = sys.executable
    svcs: list[dict] = []
    for e in cfg["serving"]["llms"]:
        name = str(e.get("name") or "")
        if not name:
            raise ValueError("every serving.llms entry needs a name")
        svcs.append({"name": f"llm-{name}", "cmd": [py, "-m", "knowledgehost.serving", name],
                     "env": {}, "hint": f":{e.get('port', '?')}"})
    if cfg["serving"]["embed"].get("enabled"):
        svcs.append({"name": "embed", "cmd": [py, "-m", "knowledgehost.serving", "embed"],
                     "env": {}, "hint": f":{cfg['serving']['embed'].get('port', 11437)}"})
    if cfg["serving"]["reranker"].get("enabled"):
        rr = urlparse(cfg.get("rerank_url") or "http://127.0.0.1:11439")
        svcs.append({"name": "reranker", "cmd": ["./run-reranker.sh"],
                     "env": {"HOST": rr.hostname or "127.0.0.1", "PORT": str(rr.port or 11439)},
                     "hint": f":{rr.port or 11439}"})
    svcs.append({"name": "kb", "cmd": ["./run.sh"], "env": {}, "hint": f":{cfg['port']}"})
    return svcs


# ── state file ────────────────────────────────────────────────────────────────

def read_state() -> dict:
    try:
        return json.loads(STATE.read_text())
    except (OSError, ValueError):
        return {}


def write_state(st: dict) -> None:
    STATE.parent.mkdir(parents=True, exist_ok=True)
    tmp = STATE.with_suffix(".tmp")
    tmp.write_text(json.dumps(st, indent=1))
    os.replace(tmp, STATE)


def alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def last_log_line(name: str) -> str:
    try:
        data = (LOGS / f"{name}.log").read_bytes()[-4096:]
        lines = [ln for ln in data.decode("utf-8", "replace").splitlines() if ln.strip()]
        return lines[-1] if lines else ""
    except OSError:
        return ""


# ── the resident supervisor ──────────────────────────────────────────────────

def spawn(svc: dict) -> subprocess.Popen:
    logf = open(LOGS / f"{svc['name']}.log", "ab")
    env = {**os.environ, **svc["env"]}
    return subprocess.Popen(svc["cmd"], cwd=str(ROOT), env=env,
                            stdout=logf, stderr=subprocess.STDOUT,
                            stdin=subprocess.DEVNULL, start_new_session=True)


def _killpg(pid: int, sig: int) -> None:
    try:
        os.killpg(pid, sig)
    except (ProcessLookupError, PermissionError):
        pass


def _shutdown(procs: dict) -> None:
    for p in procs.values():
        _killpg(p.pid, signal.SIGTERM)
    deadline = time.time() + GRACE_S
    for p in procs.values():
        try:
            p.wait(max(0.1, deadline - time.time()))
        except subprocess.TimeoutExpired:
            _killpg(p.pid, signal.SIGKILL)
    try:
        STATE.unlink()
    except OSError:
        pass


def _run(svcs: list[dict]) -> None:
    """The resident loop — runs detached, children in their own groups."""
    stop_requested = []
    signal.signal(signal.SIGTERM, lambda *_: stop_requested.append(1))
    signal.signal(signal.SIGINT, lambda *_: stop_requested.append(1))

    procs: dict[str, subprocess.Popen] = {}
    restarts: dict[str, list[float]] = {n["name"]: [] for n in svcs}
    failed: dict[str, str] = {}

    def sync_state() -> None:
        write_state({"supervisor": os.getpid(),
                     "services": {n: p.pid for n, p in procs.items()},
                     "hints": {s["name"]: s["hint"] for s in svcs},
                     "failed": failed})

    for svc in svcs:
        (LOGS / f"{svc['name']}.log").write_bytes(b"")     # truncate, like a fresh tee
        procs[svc["name"]] = spawn(svc)
        print(f"started {svc['name']} pid={procs[svc['name']].pid}", flush=True)
    sync_state()

    by_name = {s["name"]: s for s in svcs}
    while not stop_requested:
        time.sleep(TICK_S)
        for name, p in list(procs.items()):
            if p.poll() is None or name in failed:
                continue
            now = time.time()
            hist = [t for t in restarts[name] if now - t < WINDOW_S]
            hist.append(now)
            restarts[name] = hist
            if len(hist) > MAX_RESTARTS:
                failed[name] = (f"gave up after {MAX_RESTARTS} restarts in "
                                f"{int(WINDOW_S / 60)} min — see var/log/{name}.log")
                print(f"{name}: {failed[name]}", flush=True)
            else:
                print(f"{name} exited rc={p.returncode} — restarting", flush=True)
                procs[name] = spawn(by_name[name])
            sync_state()
    _shutdown(procs)


# ── commands ──────────────────────────────────────────────────────────────────

def _loopback(host: str) -> bool:
    return host in ("127.0.0.1", "localhost", "::1", "")


def cmd_start() -> int:
    st = read_state()
    if alive(st.get("supervisor", 0)):
        print(f"already running (supervisor pid={st['supervisor']}) — './vinur.sh status'")
        return 0
    cfg = load_cfg()
    if not _loopback(cfg["host"]) and not cfg.get("auth_token"):
        print("refusing to start: host binds the LAN but auth_token is empty.\n"
              "Set auth_token in config.toml (the /ops surface runs maintenance jobs),\n"
              "or bind host = \"127.0.0.1\".", file=sys.stderr)
        return 1
    svcs = services_for(cfg)
    LOGS.mkdir(parents=True, exist_ok=True)

    if os.fork() != 0:                                 # parent: report and leave
        for _ in range(50):
            time.sleep(0.1)
            if read_state().get("services"):
                break
        st = read_state()
        for name, pid in (st.get("services") or {}).items():
            print(f"  {name:<12} pid={pid}  {st.get('hints', {}).get(name, '')}")
        return 0
    os.setsid()                                        # child: become the supervisor
    logf = open(LOGS / "supervisor.log", "ab", buffering=0)
    os.dup2(logf.fileno(), 1)
    os.dup2(logf.fileno(), 2)
    _run(svcs)
    os._exit(0)


def cmd_stop() -> int:
    st = read_state()
    sup = st.get("supervisor", 0)
    if alive(sup):
        os.kill(sup, signal.SIGTERM)
        for _ in range(int((GRACE_S + 4) * 10)):
            time.sleep(0.1)
            if not alive(sup):
                break
        print("stopped" if not alive(sup) else f"supervisor pid={sup} did not exit — kill it yourself")
        return 0 if not alive(sup) else 1
    # stale state: reap the exact recorded pids, nothing pattern-matched
    for name, pid in (st.get("services") or {}).items():
        if alive(pid):
            _killpg(pid, signal.SIGTERM)
            print(f"reaped stale {name} (pid={pid})")
    try:
        STATE.unlink()
    except OSError:
        pass
    print("not running")
    return 0


def cmd_status() -> int:
    st = read_state()
    sup = st.get("supervisor", 0)
    if not alive(sup):
        print("not running" + (" (stale state — './vinur.sh stop' cleans up)" if st else ""))
        return 1
    print(f"supervisor pid={sup}")
    failed = st.get("failed") or {}
    for name, pid in (st.get("services") or {}).items():
        hint = st.get("hints", {}).get(name, "")
        if name in failed:
            print(f"  {name:<12} FAILED  {failed[name]}")
        elif alive(pid):
            print(f"  {name:<12} up      pid={pid}  {hint}")
        else:
            line = last_log_line(name)
            print(f"  {name:<12} dead    {('— ' + line) if line else ''}")
    return 0


def cmd_restart(target: str | None) -> int:
    if target is None:
        cmd_stop()
        return cmd_start()
    st = read_state()
    if not alive(st.get("supervisor", 0)):
        print("not running — './vinur.sh start'")
        return 1
    pid = (st.get("services") or {}).get(target)
    if not pid:
        print(f"no such service: {target} (have: {', '.join(st.get('services') or {})})")
        return 1
    _killpg(pid, signal.SIGTERM)                       # the watchdog revives it
    print(f"sent TERM to {target} — the supervisor restarts it within ~{int(TICK_S)}s")
    return 0


def cmd_logs(target: str | None) -> int:
    names = [target] if target else list((read_state().get("services") or {}).keys()) or ["kb"]
    files = {n: LOGS / f"{n}.log" for n in names}
    pos = {n: (f.stat().st_size if f.exists() else 0) for n, f in files.items()}
    # print a little context first
    for n, f in files.items():
        if f.exists():
            tail = f.read_bytes()[-2048:].decode("utf-8", "replace").splitlines()[-5:]
            for ln in tail:
                print(f"[{n}] {ln}")
            pos[n] = f.stat().st_size
    print("— following (Ctrl-C detaches) —")
    try:
        while True:
            time.sleep(0.5)
            for n, f in files.items():
                try:
                    size = f.stat().st_size
                except OSError:
                    continue
                if size < pos[n]:
                    pos[n] = 0                          # truncated (service restarted)
                if size > pos[n]:
                    with open(f, "rb") as fh:
                        fh.seek(pos[n])
                        chunk = fh.read()
                    pos[n] = size
                    for ln in chunk.decode("utf-8", "replace").splitlines():
                        print(f"[{n}] {ln}")
    except KeyboardInterrupt:
        return 0


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    cmd = args[0] if args else "status"
    if cmd == "start":
        return cmd_start()
    if cmd == "stop":
        return cmd_stop()
    if cmd == "status":
        return cmd_status()
    if cmd == "restart":
        return cmd_restart(args[1] if len(args) > 1 else None)
    if cmd == "logs":
        return cmd_logs(args[1] if len(args) > 1 else None)
    print(__doc__.split("\n\n")[1])
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
