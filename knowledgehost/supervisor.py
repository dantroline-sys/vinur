"""Process supervisor for a standalone Vinur box — `./vinur.sh` is its shim.

One resident process owns every service this machine serves — the declared
LMs (knowledgehost.serving), the embed endpoint, the CPU reranker, and the
kb server itself — as direct children in their own process groups:

    python -m knowledgehost.supervisor start|stop|restart [svc]|status [--json]|swap <llm>|logs [svc]
    python -m knowledgehost.supervisor minimal on|off|status   # vacate VRAM, keep serving the KB
    python -m knowledgehost.supervisor run              # foreground (systemd/launchd mode)
    python -m knowledgehost.supervisor service install|uninstall|status [--dry-run]

`start`/`stop`/`restart` with a service name act on that one service (the
panel's Serving buttons post the same requests): a stop is HELD — the watchdog
will not revive it until a start — and a start clears a "gave up after N
restarts" verdict.  Without a name they mean the whole box.

Children log to var/log/<name>.log (truncated on start).  The watchdog
revives a dead service with backoff and gives up after MAX_RESTARTS in
WINDOW_S, leaving the reason visible in `status`.  State (exact pids) lives
in var/run/supervisor.json so a stale run is recovered precisely — no
pattern-matched pkill.

Exclusive GPU group: llms entries marked `exclusive = true` cannot co-reside
in VRAM; exactly one runs (the `default = true` one at start) and `swap`
loads another in its place — stop, spawn, then wait for /health before
reporting ready (see serving.py's swap protocol; the autopilot's per-step
"model" key drives it for batched distill-vs-verify phases).

engine = "container" entries are stopped through the runtime (`podman/docker
stop` by the deterministic vinur-llm-<name>), never by signalling the attached
client: conmon/containerd owns the workload, so killing the client would
orphan the model with its VRAM held and the next exclusive load would fail
its free-memory check.  start/swap/stop also sweep orphaned vinur-llm-*
containers left by a dead run.

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
CONTAINER_STOP_S = 60  # in-container TERM -> KILL budget (`<runtime> stop -t`):
                       # a big vLLM engine takes well over GRACE_S to exit, and
                       # the runtime's own KILL still frees VRAM either way
TICK_S = 2.0           # watchdog cadence
SCHED_TICK_S = 30.0    # weekly-schedule evaluation cadence (minute resolution is plenty)
MAX_RESTARTS = 5       # per service, within WINDOW_S, then give up
WINDOW_S = 600.0


# ── config → service list ────────────────────────────────────────────────────

def load_cfg() -> dict:
    from .config import load_config
    if os.environ.get("KNOWLEDGEHOST_CONFIG"):
        return load_config(None)                 # env override (tests, alt deployments)
    p = ROOT / "config.toml"
    return load_config(str(p) if p.exists() else None)


def services_for(cfg: dict) -> list[dict]:
    """Each entry: name, cmd (argv), env (extra), hint (port for status).

    llms entries marked `exclusive = true` form ONE GPU group (models that
    cannot co-reside): only the group's default member autostarts; the rest
    are standby, brought up by the swap protocol (serving.ensure_active)."""
    py = sys.executable
    svcs: list[dict] = []
    exclusives = [e for e in cfg["serving"]["llms"] if e.get("exclusive")]
    default = next((e for e in exclusives if e.get("default")),
                   exclusives[0] if exclusives else None)
    from . import serving as sv
    for e in cfg["serving"]["llms"]:
        name = str(e.get("name") or "")
        if not name:
            raise ValueError("every serving.llms entry needs a name")
        svcs.append({"name": f"llm-{name}", "cmd": [py, "-m", "knowledgehost.serving", name],
                     "env": {}, "hint": f":{e.get('port', '?')}", "entry": name,
                     "exclusive": bool(e.get("exclusive")),
                     "autostart": (not e.get("exclusive")) or e is default,
                     "probe": (str(e.get("host") or "127.0.0.1"), int(e.get("port") or 0)),
                     # (runtime, container-name) for engine="container" — the
                     # authoritative stop handle; None for bare-metal engines
                     "container": sv.container_ref(cfg, name)})
    if cfg["serving"]["embed"].get("enabled"):
        svcs.append({"name": "embed", "cmd": [py, "-m", "knowledgehost.serving", "embed"],
                     "env": {}, "hint": f":{cfg['serving']['embed'].get('port', 11437)}",
                     "entry": "", "exclusive": False, "autostart": True})
    if cfg["serving"]["reranker"].get("enabled"):
        rr = urlparse(cfg.get("rerank_url") or "http://127.0.0.1:11439")
        svcs.append({"name": "reranker", "cmd": ["./run-reranker.sh"],
                     "env": {"HOST": rr.hostname or "127.0.0.1", "PORT": str(rr.port or 11439)},
                     "hint": f":{rr.port or 11439}",
                     "entry": "", "exclusive": False, "autostart": True})
    svcs.append({"name": "kb", "cmd": ["./run.sh"], "env": {}, "hint": f":{cfg['port']}",
                 "entry": "", "exclusive": False, "autostart": True})
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


def dying_words(name: str, max_chars: int = 300) -> str:
    """Why a service died, in one line — the line that NAMES the failure if the
    log tail has one, else simply the last line.  vLLM's last line is a pydantic
    docs URL, so 'exited rc=1 — <last line>' was routinely useless."""
    try:
        tail = (LOGS / f"{name}.log").read_bytes()[-16384:].decode("utf-8", "replace")
    except OSError:
        return ""
    from . import serving as sv
    hits = sv.cause_lines(tail, 1)
    return (hits[-1] if hits else last_log_line(name))[:max_chars]


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


def _container_alive(ref) -> bool:
    """Is the named container running?  Used to log/skip sweeps — the stop path
    itself never depends on this answer (stop/rm are idempotent)."""
    rt, cname = ref
    try:
        r = subprocess.run([rt, "ps", "-q", "--filter", f"name=^{cname}$"],
                           capture_output=True, text=True, timeout=10)
        return bool(r.stdout.strip())
    except (OSError, subprocess.TimeoutExpired):
        return False


def _stop_container(ref) -> None:
    """Authoritative stop for an engine="container" workload.  The attached
    client is only a window onto the container (conmon/containerd owns the real
    process tree), so killing the client's process group ORPHANS the model —
    it keeps serving and keeps its VRAM, and the next exclusive load then fails
    with 'free memory less than desired utilization'.  `<runtime> stop` signals
    the workload itself and BLOCKS until it is dead (VRAM is released with the
    process); `rm -f` is the idempotent belt for a runtime that lost --rm
    cleanup after an unclean exit."""
    rt, cname = ref
    for cmdline in ([rt, "stop", "-t", str(CONTAINER_STOP_S), cname],
                    [rt, "rm", "-f", cname]):
        try:
            subprocess.run(cmdline, stdout=subprocess.DEVNULL,
                           stderr=subprocess.DEVNULL, stdin=subprocess.DEVNULL,
                           timeout=CONTAINER_STOP_S + 30, check=False)
        except (OSError, subprocess.TimeoutExpired):
            pass


def _shutdown(procs: dict, by_name: dict | None = None) -> None:
    from . import serving as sv
    for name in list(procs):
        ref = (by_name or {}).get(name, {}).get("container")
        if ref:
            _stop_container(ref)                       # blocks until VRAM is free
    for p in procs.values():
        _killpg(p.pid, signal.SIGTERM)
    deadline = time.time() + GRACE_S
    for p in procs.values():
        try:
            p.wait(max(0.1, deadline - time.time()))
        except subprocess.TimeoutExpired:
            _killpg(p.pid, signal.SIGKILL)
    for f in (STATE, sv.SWAP_STATE, sv.SWAP_REQ):
        try:
            f.unlink()
        except OSError:
            pass


def _stop_one(procs: dict, svc: dict) -> None:
    p = procs.pop(svc["name"], None)
    ref = svc.get("container")
    if ref:
        _stop_container(ref)          # the workload first — clients only follow
    if p is None:
        return
    if ref:
        try:
            p.wait(10)                # the attached client exits with its container
            return
        except subprocess.TimeoutExpired:
            pass                      # wedged client — fall through to the killpg
    _killpg(p.pid, signal.SIGTERM)
    try:
        p.wait(GRACE_S)
    except subprocess.TimeoutExpired:
        _killpg(p.pid, signal.SIGKILL)
        try:
            p.wait(5)
        except subprocess.TimeoutExpired:
            pass


def control_plan(action: str, svc: dict, *, running: bool, active_excl: str | None) -> list[str]:
    """What start|stop|restart means for one service, as an ordered op list —
    kept out of the loop so the semantics are testable:

      stop     stop it, then HOLD (the watchdog must not undo an operator)
      start    un-hold, clear the failure verdict and the restart budget, and
               bring it up — an exclusive model via SWAP, since its siblings
               hold the VRAM and a bare spawn would just OOM
      restart  the same, with a stop first (and a spawn even though it was up)
    """
    if action == "stop":
        return ["stop", "hold", "clear"]
    ops = ["stop"] if action == "restart" else []
    ops += ["unhold", "clear"]
    if svc.get("exclusive") and svc.get("entry") != active_excl:
        return ops + ["swap"]
    if action == "restart" or not running:
        ops.append("spawn")
    return ops


# ── minimal mode: vacate VRAM, keep serving the KB ────────────────────────────

def _default_excl(cfg: dict) -> str:
    """The exclusive-group boot model (default=true, else the first exclusive)."""
    excl = [str(e.get("name")) for e in cfg["serving"]["llms"] if e.get("exclusive")]
    return next((str(e.get("name")) for e in cfg["serving"]["llms"]
                 if e.get("exclusive") and e.get("default")), excl[0] if excl else "")


def plan_minimal(cfg: dict, action: str, *, active: str = "", restore: str = "") -> dict:
    """Pure: what turning minimal mode on/off means, as request-lane ops — kept
    out of the loop so the semantics are testable.  Returns
    {flag, stop:[svc], restart:[svc], start:[svc], swap:name|None}.

      on   stop the VRAM-holding LMs (the ACTIVE exclusive + every non-exclusive
           entry — standby siblings already hold nothing, so they're left alone);
           the embedder goes to CPU (restart) or off (stop) per config; write the
           flag (embed device + the model to restore).  kb + CPU reranker stay up.
      off  clear the flag, swap the exclusive model back in, start the non-
           exclusive LMs, and restart the embedder (back on GPU)."""
    embed_on = bool(cfg["serving"]["embed"].get("enabled"))
    embed_mode = str(cfg["serving"].get("minimal", {}).get("embed", "cpu"))
    default = _default_excl(cfg)
    nonx = [f"llm-{e.get('name')}" for e in cfg["serving"]["llms"]
            if e.get("name") and not e.get("exclusive")]
    if action == "on":
        keep = active or default
        stop = ([f"llm-{keep}"] if keep else []) + nonx
        plan = {"flag": {"on": True, "embed": embed_mode, "restore": keep},
                "stop": stop, "restart": [], "start": [], "swap": None}
        if embed_on:
            (plan["restart"] if embed_mode == "cpu" else plan["stop"]).append("embed")
        return plan
    tgt = restore or active or default
    return {"flag": None, "stop": [], "start": list(nonx), "swap": (tgt or None),
            "restart": (["embed"] if embed_on else [])}


def _reconcile_schedule(cfg: dict, last_want):
    """Weekly-schedule tick: flip minimal mode at a window boundary.  `last_want`
    is the state the schedule last DECIDED (None = not governing).  We act only on
    a boundary crossing (want != last_want) — so a manual `minimal on/off` inside a
    window sticks until the next boundary, never fought every tick — and even then
    only when the box isn't already in the wanted state.  Returns the new want."""
    from datetime import datetime
    from . import serving as sv
    sched = sv.read_schedule()
    want = sv.schedule_wants_minimal(sched, datetime.now())
    if want is None:                       # schedule disabled — stop governing
        return None
    if want == last_want:                  # inside the same window — respect manual toggles
        return last_want
    if want != bool(sv.minimal_state().get("on")):
        print(f"schedule: window boundary -> minimal {'on' if want else 'off'}", flush=True)
        try:
            apply_minimal(cfg, "on" if want else "off")
        except Exception as e:             # a broken transition must not kill the loop
            print(f"schedule: apply_minimal failed: {e}", flush=True)
    return want


def apply_minimal(cfg: dict, action: str) -> dict:
    """Execute a minimal-mode transition through the same async request lanes the
    panel/CLI use.  Idempotent-ish: safe to re-issue.  Returns the plan summary."""
    from . import serving as sv
    if action not in ("on", "off"):
        raise ValueError("action must be on|off")
    active = str((sv.swap_state() or {}).get("active") or "")
    restore = str((sv.minimal_state() or {}).get("restore") or "")
    plan = plan_minimal(cfg, action, active=active, restore=restore)
    # Order: (re)write the flag FIRST so a re-spawned embedder reads the right
    # device, THEN fire the async service/swap requests.
    if plan["flag"] is not None:
        sv.set_minimal(plan["flag"])
    else:
        sv.clear_minimal()
    for name in plan["stop"]:
        sv.request_service(name, "stop")
    for name in plan["restart"]:
        sv.request_service(name, "restart")
    for name in plan["start"]:
        sv.request_service(name, "start")
    if plan.get("swap"):
        sv.request_swap(plan["swap"])
    return {"action": action, **plan}


def _run(svcs: list[dict], cfg: dict) -> None:
    """The resident loop — runs detached, children in their own groups."""
    from . import serving as sv
    stop_requested = []
    signal.signal(signal.SIGTERM, lambda *_: stop_requested.append(1))
    signal.signal(signal.SIGINT, lambda *_: stop_requested.append(1))

    procs: dict[str, subprocess.Popen] = {}
    restarts: dict[str, list[float]] = {n["name"]: [] for n in svcs}
    failed: dict[str, str] = {}
    excl = {s["entry"]: s for s in svcs if s.get("exclusive")}
    active_excl = next((s["entry"] for s in svcs
                        if s.get("exclusive") and s.get("autostart")), None)
    swap_timeout = float(cfg["serving"].get("swap_timeout_s", 900))

    def write_swap(status: str, request: str = "", error: str = "") -> None:
        d = {"active": active_excl, "status": status, "at": time.time()}
        if request:
            d["request"] = request
        if error:
            d["error"] = error
        sv.SWAP_STATE.parent.mkdir(parents=True, exist_ok=True)
        tmp = sv.SWAP_STATE.with_suffix(".tmp")
        tmp.write_text(json.dumps(d))
        os.replace(tmp, sv.SWAP_STATE)

    held: set[str] = set()          # stopped BY REQUEST — the watchdog leaves these alone

    def sync_state() -> None:
        write_state({"supervisor": os.getpid(),
                     "services": {n: p.pid for n, p in procs.items()},
                     "hints": {s["name"]: s["hint"] for s in svcs},
                     "standby": {e: excl[e]["name"] for e in excl if e != active_excl},
                     "failed": failed, "held": sorted(held)})

    # A dead run (or crashed client) can leave a model container running with
    # its VRAM held — the autostart model would then fail its free-memory check.
    # Sweep our named containers before spawning anything.
    for svc in svcs:
        ref = svc.get("container")
        if ref and _container_alive(ref):
            print(f"start: stopping orphaned container {ref[1]}", flush=True)
            _stop_container(ref)
    # Minimal mode persists across a reboot (the GPU may still be wanted
    # elsewhere): if the flag is set, DON'T autostart the VRAM-holding services —
    # park them held so status shows them stopped and the watchdog leaves them be.
    # The embedder still autostarts when embed="cpu" (it reads the flag → -ngl 0).
    _min = sv.minimal_state()
    _min_on = bool(_min.get("on"))
    _embed_off = _min_on and _min.get("embed") == "off"
    if _min_on:
        print("start: MINIMAL MODE — GPU services stay down (kb serves the KB)", flush=True)
    for svc in svcs:
        if not svc.get("autostart", True):
            continue
        if _min_on and (svc["name"].startswith("llm-")
                        or (svc["name"] == "embed" and _embed_off)):
            held.add(svc["name"])                          # minimal: keep VRAM vacated
            continue
        (LOGS / f"{svc['name']}.log").write_bytes(b"")     # truncate, like a fresh tee
        procs[svc["name"]] = spawn(svc)
        print(f"started {svc['name']} pid={procs[svc['name']].pid}", flush=True)
    sync_state()
    try:
        sv.SWAP_REQ.unlink()                               # a stale request must not fire
    except OSError:
        pass
    sv.take_service_requests()                             # …nor a stale stop/start
    if excl:
        write_swap("ready")

    def check_swap() -> None:
        nonlocal active_excl
        if not excl or not sv.SWAP_REQ.exists():
            return
        try:
            want = str(json.loads(sv.SWAP_REQ.read_text()).get("name") or "")
        except (OSError, ValueError):
            want = ""
        try:
            sv.SWAP_REQ.unlink()
        except OSError:
            pass
        if want not in excl:
            write_swap("error", request=want,
                       error=f"'{want}' is not an exclusive serving.llms entry")
            return
        if want == active_excl and excl[want]["name"] in procs:
            write_swap("ready")
            return
        svc = excl[want]
        write_swap("swapping", request=want)
        print(f"swap: {active_excl} -> {want}", flush=True)
        cur = excl.get(active_excl or "")
        if cur and cur["name"] != svc["name"]:
            _stop_one(procs, cur)
        # Zombie defense: a container whose attached client died (crashed
        # client, previous supervisor run, manual start) is invisible to
        # `procs` but still holds its VRAM — sweep every exclusive sibling's
        # container by name before loading the requested one.
        for entry_name, s in excl.items():
            ref = s.get("container")
            if entry_name != want and ref and _container_alive(ref):
                print(f"swap: stopping orphaned container {ref[1]}", flush=True)
                _stop_container(ref)
        held.discard(svc["name"])                          # swapping in un-holds it
        if svc["name"] not in procs or procs[svc["name"]].poll() is not None:
            procs.pop(svc["name"], None)                   # a re-request after a timeout
            (LOGS / f"{svc['name']}.log").write_bytes(b"")  # keeps a live loader running
            procs[svc["name"]] = spawn(svc)
        sync_state()
        host, port = svc["probe"]
        deadline = time.time() + swap_timeout
        while time.time() < deadline and not stop_requested:
            p = procs.get(svc["name"])
            if p is not None and p.poll() is not None:
                procs.pop(svc["name"], None)
                write_swap("error", request=want,
                           error=f"{svc['name']} exited rc={p.returncode} — "
                                 f"{dying_words(svc['name'])} (full log: "
                                 f"var/log/{svc['name']}.log, or the Log button "
                                 f"on the Serving tab)")
                sync_state()
                return
            if sv.probe_ready(host, port):
                active_excl = want
                restarts[svc["name"]] = []                 # a fresh model, fresh budget
                write_swap("ready")
                sync_state()
                print(f"swap: {want} ready on :{port}", flush=True)
                return
            time.sleep(1.0)
        if not stop_requested:
            # Left running (it may still be loading) — a re-request resumes the wait.
            write_swap("error", request=want,
                       error=f"not answering /health after {int(swap_timeout)}s "
                             f"(still loading? re-run the swap to keep waiting)")

    by_name = {s["name"]: s for s in svcs}

    def check_control() -> None:
        """Panel/CLI start|stop|restart for ONE service (var/run/svcreq/*.req).

        A stop must STICK — the watchdog's whole job is reviving things, so a
        requested stop parks the name in `held` and only a start/restart takes
        it out again.  Both also clear a `failed` verdict and the restart
        budget: an operator saying "start" is saying "try again", and without
        this a service that gave up needed a whole-supervisor restart."""
        for req in sv.take_service_requests():
            name, action = str(req["service"]), str(req["action"])
            svc = by_name.get(name)
            if svc is None:
                print(f"control: no such service '{name}'", flush=True)
                continue
            p = procs.get(name)
            running = p is not None and p.poll() is None
            for op in control_plan(action, svc, running=running, active_excl=active_excl):
                if op == "stop":
                    _stop_one(procs, svc)          # safe when absent: also sweeps
                                                   # a container whose client died
                elif op == "hold":
                    held.add(name)
                    print(f"control: stopped {name} (held — start brings it back)",
                          flush=True)
                elif op == "unhold":
                    held.discard(name)
                elif op == "clear":
                    failed.pop(name, None)
                    restarts[name] = []                # an operator retry, fresh budget
                elif op == "swap":
                    sv.request_swap(svc["entry"])
                    print(f"control: {action} {name} -> swap to {svc['entry']}", flush=True)
                elif op == "spawn":
                    procs.pop(name, None)
                    (LOGS / f"{name}.log").write_bytes(b"")   # a fresh attempt, fresh log
                    procs[name] = spawn(svc)
                    print(f"control: {action}ed {name} pid={procs[name].pid}", flush=True)
            sync_state()

    sched_want = None          # the state the weekly schedule last decided
    sched_next = 0.0           # next schedule evaluation (throttled to SCHED_TICK_S)
    while not stop_requested:
        time.sleep(TICK_S)
        check_control()
        check_swap()
        if time.time() >= sched_next:
            sched_next = time.time() + SCHED_TICK_S
            sched_want = _reconcile_schedule(cfg, sched_want)
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
                ref = by_name[name].get("container")
                if ref and _container_alive(ref):      # client died, workload survived
                    _stop_container(ref)               # — clear it or the name collides
                procs[name] = spawn(by_name[name])
            sync_state()
    _shutdown(procs, by_name)


# ── commands ──────────────────────────────────────────────────────────────────

def _loopback(host: str) -> bool:
    return host in ("127.0.0.1", "localhost", "::1", "")


def _prepare_start():
    """Shared preflight for the detached start and the foreground `run` (the
    OS-service mode).  Returns (cfg, svcs), or an int exit code."""
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
    # Friendly preflight: embed, the reranker, and engine="llama" entries all
    # run on llama-server — say so upfront instead of letting them die at spawn.
    from . import serving as sv
    needs_llama = ([s["name"] for s in svcs if s["name"] in ("embed", "reranker")]
                   + [f"llm-{e.get('name')}" for e in cfg["serving"]["llms"]
                      if e.get("engine") == "llama"])
    if needs_llama and not sv.find_llama_server():
        verb = "needs" if len(needs_llama) == 1 else "need"
        print(f"warning: {', '.join(needs_llama)} {verb} llama-server, which is not\n"
              "installed — they will show as dead in status.  Build it in-tree with\n"
              "'./install.sh --llama' (or set LLAMA_SERVER=/path/to/llama-server).",
              file=sys.stderr)
    tk_warn = sv.toolkit_warning(cfg)
    if tk_warn:
        print(f"warning: {tk_warn}", file=sys.stderr)
    px_warn = sv.proxy_warning(cfg)
    if px_warn:
        print(f"warning: {px_warn}", file=sys.stderr)
    LOGS.mkdir(parents=True, exist_ok=True)
    return cfg, svcs


def cmd_start() -> int:
    r = _prepare_start()
    if isinstance(r, int):
        return r
    cfg, svcs = r

    if os.fork() != 0:                                 # parent: report and leave
        for _ in range(50):
            time.sleep(0.1)
            if read_state().get("services"):
                break
        st = read_state()
        for name, pid in (st.get("services") or {}).items():
            print(f"  {name:<12} pid={pid}  {st.get('hints', {}).get(name, '')}")
        for entry, name in (st.get("standby") or {}).items():
            print(f"  {name:<12} standby — './vinur.sh swap {entry}' loads it")
        return 0
    os.setsid()                                        # child: become the supervisor
    logf = open(LOGS / "supervisor.log", "ab", buffering=0)
    os.dup2(logf.fileno(), 1)
    os.dup2(logf.fileno(), 2)
    _run(svcs, cfg)
    os._exit(0)


def cmd_run() -> int:
    """Foreground supervisor — the OS-service mode (systemd/launchd is the
    daemonizer and captures stdout; see knowledgehost/service.py).  Same
    preflight as `start`, no fork, no log redirect."""
    r = _prepare_start()
    if isinstance(r, int):
        return r
    cfg, svcs = r
    _run(svcs, cfg)
    return 0


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
    # …and our named model containers, which outlive their attached clients
    try:
        cfg = load_cfg()
        from . import serving as sv
        for e in cfg["serving"]["llms"]:
            ref = sv.container_ref(cfg, str(e.get("name") or ""))
            if ref and _container_alive(ref):
                print(f"stopping orphaned container {ref[1]}")
                _stop_container(ref)
    except Exception:
        pass                                           # config broken — pids were reaped
    try:
        STATE.unlink()
    except OSError:
        pass
    print("not running")
    return 0


def status_data() -> dict:
    """Machine-readable box status — THE seam `status --json`, the web panel
    and a GUI shell all share (one collection, several renderers).  Safe on a
    quiet box: running:false with whatever else is knowable."""
    st = read_state()
    sup = st.get("supervisor", 0)
    out: dict = {"running": alive(sup), "supervisor": sup if alive(sup) else 0,
                 "stale": bool(st) and not alive(sup), "repo": str(ROOT),
                 "panel": "", "services": [], "swap": {}}
    try:
        cfg = load_cfg()
        out["panel"] = f"http://{cfg.get('host') or '127.0.0.1'}:{cfg.get('port')}"
    except Exception:
        pass
    failed = st.get("failed") or {}
    for name, pid in (st.get("services") or {}).items():
        if name in failed:
            out["services"].append({"name": name, "state": "failed", "pid": 0,
                                    "note": str(failed[name])})
        elif alive(pid):
            out["services"].append({"name": name, "state": "up", "pid": pid,
                                    "note": st.get("hints", {}).get(name, "")})
        else:
            out["services"].append({"name": name, "state": "dead", "pid": 0,
                                    "note": last_log_line(name)})
    for entry, name in (st.get("standby") or {}).items():
        out["services"].append({"name": name, "state": "standby", "pid": 0,
                                "entry": entry, "note": f"'./vinur.sh swap {entry}' loads it"})
    for name in st.get("held") or []:
        out["services"].append({"name": name, "state": "stopped", "pid": 0,
                                "note": f"by request; './vinur.sh start {name}' brings it back"})
    try:
        from . import serving as sv
        out["swap"] = sv.swap_state() or {}
        out["minimal"] = sv.minimal_state() or {}
    except Exception:
        out["swap"] = {}
        out["minimal"] = {}
    return out


def cmd_status(as_json: bool = False) -> int:
    d = status_data()
    if as_json:
        print(json.dumps(d, indent=1))
        return 0 if d["running"] else 1
    if not d["running"]:
        print("not running" + (" (stale state — './vinur.sh stop' cleans up)"
                               if d["stale"] else ""))
        return 1
    print(f"supervisor pid={d['supervisor']}")
    m = d.get("minimal") or {}
    if m.get("on"):
        print(f"  ** MINIMAL MODE — VRAM vacated, serving the KB (embed: "
              f"{m.get('embed', 'cpu')}).  './vinur.sh minimal off' restores "
              f"{m.get('restore') or 'the default LM'} **")
    for s in d["services"]:
        if s["state"] == "failed":
            print(f"  {s['name']:<12} FAILED  {s['note']}")
        elif s["state"] == "up":
            print(f"  {s['name']:<12} up      pid={s['pid']}  {s['note']}")
        elif s["state"] == "dead":
            print(f"  {s['name']:<12} dead    {('— ' + s['note']) if s['note'] else ''}")
        elif s["state"] == "standby":
            print(f"  {s['name']:<12} standby — {s['note']}")
        else:
            print(f"  {s['name']:<12} stopped — {s['note']}")
    sw = d["swap"]
    if sw.get("status") == "swapping":
        print(f"  (swap in progress: -> {sw.get('request')})")
    elif sw.get("status") == "error":
        print(f"  (last swap failed: {sw.get('error')})")
    return 0


def cmd_restart(target: str | None) -> int:
    if target is None:
        cmd_stop()
        return cmd_start()
    return cmd_service(target, "restart")


def cmd_service(target: str, action: str) -> int:
    """start|stop|restart ONE service, through the same request lane the panel
    uses — so a stop sticks (the watchdog won't revive it) and a start clears a
    'gave up after N restarts' verdict instead of needing a full restart."""
    st = read_state()
    if not alive(st.get("supervisor", 0)):
        print("not running — './vinur.sh start'")
        return 1
    known = set(st.get("services") or {}) | set((st.get("standby") or {}).values()) \
        | set(st.get("failed") or {}) | set(st.get("held") or [])
    if target not in known:
        print(f"no such service: {target} (have: {', '.join(sorted(known)) or 'none'})")
        return 1
    from . import serving as sv
    try:
        sv.request_service(target, action)
    except ValueError as e:
        print(str(e))
        return 2
    print(f"{action} requested for {target} — the supervisor acts within ~{int(TICK_S)}s "
          f"(./vinur.sh status)")
    return 0


def cmd_swap(target: str | None) -> int:
    if not target:
        print("usage: ./vinur.sh swap <serving.llms name>")
        return 2
    if not alive(read_state().get("supervisor", 0)):
        print("not running — './vinur.sh start'")
        return 1
    from . import serving as sv
    cfg = load_cfg()
    names = [str(e.get("name")) for e in cfg["serving"]["llms"] if e.get("exclusive")]
    if target not in names:
        print(f"'{target}' is not an exclusive serving.llms entry "
              f"(exclusive entries: {', '.join(names) or 'none'})")
        return 1

    def progress(st):
        if st.get("status") == "swapping":
            print(f"  swapping -> {st.get('request')} (weights loading; this can take minutes)")

    try:
        sv.ensure_active(target, timeout_s=float(cfg["serving"].get("swap_timeout_s", 900)),
                         progress=progress)
    except (RuntimeError, TimeoutError) as e:
        print(f"swap failed: {e}")
        return 1
    print(f"{target} ready")
    return 0


def cmd_minimal(action: str | None) -> int:
    """Vacate all VRAM but keep serving the KB (or restore).  A hot switch: the
    kb server never goes down, so Vinkona keeps being answered throughout."""
    from . import serving as sv
    action = (action or "status").lower()
    if action == "status":
        m = sv.minimal_state()
        if m.get("on"):
            print(f"minimal mode ON — VRAM vacated (embed: {m.get('embed', 'cpu')}); "
                  f"restore target: {m.get('restore') or '(default LM)'}")
        else:
            print("minimal mode OFF (full serving)")
        return 0
    if action not in ("on", "off"):
        print("usage: ./vinur.sh minimal on|off|status")
        return 2
    if not alive(read_state().get("supervisor", 0)):
        print("not running — './vinur.sh start'")
        return 1
    summary = apply_minimal(load_cfg(), action)
    if action == "on":
        gone = ", ".join(summary["stop"] + summary["restart"]) or "(nothing resident)"
        print(f"minimal mode ON — freeing VRAM: {gone}; embed={summary['flag']['embed']}, "
              f"kb keeps serving.  VRAM frees within ~{CONTAINER_STOP_S}s "
              f"(container) or seconds (bare-metal); './vinur.sh status' to watch.")
    else:
        tgt = summary.get("swap") or "(default LM)"
        print(f"minimal mode OFF — restoring {tgt} (+ embed on GPU).  Weights load "
              f"over minutes; the kb server is served throughout.")
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
        return cmd_service(args[1], "start") if len(args) > 1 else cmd_start()
    if cmd == "run":
        return cmd_run()
    if cmd == "stop":
        return cmd_service(args[1], "stop") if len(args) > 1 else cmd_stop()
    if cmd == "status":
        return cmd_status(as_json="--json" in args)
    if cmd == "service":
        from . import service as service_mod
        return service_mod.main(args[1:])
    if cmd == "restart":
        return cmd_restart(args[1] if len(args) > 1 else None)
    if cmd == "swap":
        return cmd_swap(args[1] if len(args) > 1 else None)
    if cmd == "minimal":
        return cmd_minimal(args[1] if len(args) > 1 else "status")
    if cmd == "logs":
        return cmd_logs(args[1] if len(args) > 1 else None)
    print(__doc__.split("\n\n")[1])
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
