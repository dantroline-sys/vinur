"""
Autopilot — run maintenance verbs automatically, in a user-defined priority order.

The knowledge host has several background verbs (ingest, distill, link, adjudicate,
refine, …).  Left to a human they get run by hand; left to a dumb timer they run in
the wrong order.  The autopilot runs them on a priority basis: the ORDER of the step
list is the priority, and after every step it re-evaluates from the top — so a
high-priority step that just gained work (e.g. fresh Vinkona research drops to
distil) preempts the big uncurated backlog below it.

Why this order matters (the motivating case): Vinkona's own research outputs are
more immediately relevant than mountains of ingested PDFs, so 'distil the vinkona
bundle' sits above 'distil everything else' — the former drains fully (it's small),
the latter runs in bounded batches so the former can preempt it when new drops land.

Starvation guard: a step whose run reports no work (the verbs print an OPS_RESULT
line; ops.result() relays it) — or that exits non-zero — is held aside for one
idle_interval before it's considered again, so a permanently-due 0-interval step
can't monopolise the slot.  That is what makes TWO steps of the same verb with
different args (the two distills above) actually both run.

Coordination: the single-slot OpsRunner means one job at a time (a manual job from
the Operations tab always wins).  And when the assistant is doing its own idle work
it holds the LM leases (lm_fast / lm_big); with respect_leases on, the autopilot
stands down so the two never fight over the GPUs — the mirror of the assistant's
'pause idle work' button.

Exclusive models (a box whose big LMs can't co-reside): with auto_models on
(the default) each step swaps in the model its verb's LM lane points at
(auto_model — distill/refine → the distill_urls entry; link/adjudicate follow
their `fast` flag; ingest only when it distils inline) before the verb runs.
Consecutive steps sharing a model swap once, not per run, and the priority
order doubles as the phase order.  A step's explicit "model" pins it; boxes
with no [serving] models derive nothing and behave exactly as before.

Config lives in var/autopilot.json (live-editable from the Prioritizer tab); this
module owns its defaults and the pure step-selection logic (unit-tested).
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from pathlib import Path

log = logging.getLogger("knowledgehost.autopilot")

# The verbs the autopilot may schedule, and the sensible default plan.  Distill is
# split so Vinkona's drops go first; the long backlog steps carry a limit so each
# invocation is bounded and the loop can re-check priorities between batches.
DEFAULT_PLAN = {
    "enabled": False,               # opt-in; the Prioritizer tab turns it on
    "idle_interval_s": 60,          # wait this long when a full pass found no work
    "respect_leases": True,         # stand down while the assistant holds the LMs
    "auto_models": True,            # derive each step's exclusive model from its verb's
                                    # LM lane (auto_model) unless the step pins one
    "steps": [
        {"command": "ingest",     "args": {},                    "enabled": True,
         "min_interval_s": 900,   "label": "Ingest new/changed documents"},
        {"command": "distill",    "args": {"bundle": "vinkona"}, "enabled": True,
         "min_interval_s": 0,     "label": "Distil Vinkona's research drops (priority)"},
        {"command": "distill",    "args": {"limit": 50},         "enabled": True,
         "min_interval_s": 0,     "label": "Distil the rest of the corpus (batched)"},
        {"command": "link",       "args": {"limit": 200},        "enabled": True,
         "min_interval_s": 3600,  "label": "Link related cards"},
        {"command": "adjudicate", "args": {"limit": 100},        "enabled": True,
         "min_interval_s": 3600,  "label": "Merge duplicate nodes"},
        {"command": "refine",     "args": {"limit": 50},         "enabled": False,
         "min_interval_s": 86400, "label": "Refine cards against their sources"},
    ],
}


def plan_path(cfg: dict) -> Path:
    root = Path(__file__).resolve().parent.parent
    ctrl = cfg.get("control_dir") or str(root / "var")
    return Path(ctrl).expanduser() / "autopilot.json"


def load_plan(cfg: dict) -> dict:
    """The saved plan, or the default.  Missing keys are backfilled so an older file
    keeps working after we add a field."""
    p = plan_path(cfg)
    plan = json.loads(json.dumps(DEFAULT_PLAN))     # deep copy
    try:
        if p.exists():
            saved = json.loads(p.read_text())
            for k in ("enabled", "idle_interval_s", "respect_leases", "auto_models"):
                if k in saved:
                    plan[k] = saved[k]
            if isinstance(saved.get("steps"), list):
                plan["steps"] = saved["steps"]
    except Exception as e:                          # pragma: no cover
        log.warning("autopilot: bad plan file %s (%s) — using defaults", p, e)
    return plan


def save_plan(cfg: dict, plan: dict) -> dict:
    """Validate and persist a plan from the UI.  Only known commands/fields survive."""
    from .ops import COMMANDS
    clean_steps = []
    for s in plan.get("steps", []):
        cmd = str(s.get("command", ""))
        if cmd not in COMMANDS:
            continue
        clean_steps.append({
            "command": cmd,
            "args": s.get("args") if isinstance(s.get("args"), dict) else {},
            "enabled": bool(s.get("enabled", True)),
            "min_interval_s": max(0, int(s.get("min_interval_s", 0) or 0)),
            "label": str(s.get("label", cmd))[:120],
            # Exclusive-model phase batching: run this step under a specific
            # [serving] model (swapped in first).  Empty = whatever is loaded.
            "model": str(s.get("model", "") or "")[:60],
        })
    out = {
        "enabled": bool(plan.get("enabled", False)),
        "idle_interval_s": max(5, int(plan.get("idle_interval_s", 60) or 60)),
        "respect_leases": bool(plan.get("respect_leases", True)),
        "auto_models": bool(plan.get("auto_models", True)),
        "steps": clean_steps or DEFAULT_PLAN["steps"],
    }
    p = plan_path(cfg)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".json.tmp")                # atomic: the loop thread re-reads this
    tmp.write_text(json.dumps(out, indent=2) + "\n")
    os.replace(tmp, p)
    return out


def due_step(steps: list, last_run: dict, now: float, hold_until: dict | None = None):
    """Pure selection: the first enabled step (highest priority) whose min_interval has
    elapsed since it last ran AND that isn't in a no-work hold.  Returns (index, step)
    or (None, None).

    `last_run` maps a step key -> unix ts of its last completion.  Re-evaluating from
    the top each call is what gives priority preemption.

    `hold_until` maps a step key -> unix ts before which the step is skipped because
    its last run reported no work (or failed).  This is what lets two 0-interval steps
    of the SAME verb coexist — e.g. 'distil the vinkona bundle' above 'distil the
    rest': when the first finds nothing, it stands aside instead of monopolising the
    slot forever, and the second actually runs."""
    hold_until = hold_until or {}
    for i, s in enumerate(steps):
        if not s.get("enabled", True):
            continue
        key = step_key(s)
        if now < hold_until.get(key, 0.0):
            continue
        gap = now - last_run.get(key, 0.0)
        if gap >= float(s.get("min_interval_s", 0) or 0):
            return i, s
    return None, None


# Automatic model routing: distill/refine always want the big distiller;
# link/adjudicate follow their `fast` flag to the extract tier; ingest only
# touches an LM when it distils inline.  Verbs not mapped below (imports,
# embeds, stats, …) use no chat LM — no swap needed.
def auto_model(cfg: dict, command: str, args: dict | None = None):
    """The exclusive [serving] entry `command` needs, or None (no swap):
    derived from the first URL of the LM lane the verb drives, matched
    against the exclusive serving entries by host/port.  Returns None for
    embed-only verbs, non-exclusive (always-resident) endpoints, foreign
    hosts, and boxes with no [serving] models — so on a classic deployment
    this changes nothing.  An explicit step "model" always overrides."""
    args = args or {}
    if command in ("distill", "refine"):
        lane = "distill"
    elif command in ("link", "adjudicate"):
        lane = "extract" if args.get("fast") else "distill"
    elif command == "ingest":
        lane = "distill" if args.get("distill") else None
    else:
        lane = None
    if lane is None:
        return None
    urls = cfg.get(lane + "_urls") or []
    if lane == "extract" and not urls:      # no fast tier configured → the big one
        urls = cfg.get("distill_urls") or []
    if not urls:
        return None
    from . import serving
    return serving.exclusive_entry_for_url(cfg, str(urls[0]))


def step_key(step: dict) -> str:
    """Stable identity for a step (command + its args + its model) so last-run
    tracking survives reordering in the UI."""
    args = step.get("args") or {}
    key = step["command"] + "(" + ",".join(f"{k}={args[k]}" for k in sorted(args)) + ")"
    if step.get("model"):
        key += f"@{step['model']}"
    return key


class Autopilot:
    """Background driver: picks due steps in priority order and runs them through the
    server's single-slot OpsRunner, yielding to manual jobs and (optionally) to the
    assistant's LM leases."""

    def __init__(self, cfg: dict, ops, lease_mod=None):
        self.cfg = cfg
        self.ops = ops
        self.lease = lease_mod
        self._last_run: dict = {}
        self._hold_until: dict = {}    # step key -> ts; set when a run found no work
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._state = {"running_step": None, "last_reason": "not started"}

    # ── lifecycle ────────────────────────────────────────────────────────────
    def start(self):
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name="autopilot", daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()

    def status(self) -> dict:
        plan = load_plan(self.cfg)
        return {"enabled": plan["enabled"], "running_step": self._state["running_step"],
                "last_reason": self._state["last_reason"],
                "respect_leases": plan["respect_leases"],
                "auto_models": plan.get("auto_models", True)}

    # ── the loop ─────────────────────────────────────────────────────────────
    def _leases_held(self, plan) -> bool:
        if not plan.get("respect_leases", True) or self.lease is None:
            return False
        try:
            return (self.lease.is_held(self.lease.FAST, self.cfg)
                    or self.lease.is_held(self.lease.BIG, self.cfg))
        except Exception:                           # pragma: no cover
            return False

    def _loop(self):
        log.info("autopilot thread started")
        while not self._stop.is_set():
            try:
                plan = load_plan(self.cfg)
                if not plan.get("enabled"):
                    self._state.update(running_step=None, last_reason="disabled")
                    self._sleep(5)
                    continue
                if self.ops.running():              # a manual job owns the single slot
                    self._state["last_reason"] = "a manual job is running"
                    self._sleep(5)
                    continue
                if self._leases_held(plan):         # the assistant is using the LMs
                    self._state.update(running_step=None, last_reason="yielding — assistant holds the LMs")
                    self._sleep(min(30, plan["idle_interval_s"]))
                    continue
                idx, step = due_step(plan["steps"], self._last_run, time.time(),
                                     self._hold_until)
                if step is None:
                    self._state.update(running_step=None, last_reason="all steps idle / not due")
                    self._sleep(plan["idle_interval_s"])
                    continue
                self._run_step(step, plan)
            except Exception as e:                  # pragma: no cover - defensive
                log.warning("autopilot loop error (continuing): %s", e)
                self._sleep(10)
        log.info("autopilot thread stopped")

    def _run_step(self, step, plan):
        label = step.get("label", step["command"])
        key = step_key(step)
        self._state["running_step"] = label
        log.info("autopilot: running %s (%s)", step["command"], label)
        want = str(step.get("model") or "").strip()
        if not want and plan.get("auto_models", True):
            # No pinned model: derive the one this verb's LM lane needs, so
            # exclusive-GPU boxes swap automatically as the plan progresses.
            want = auto_model(self.cfg, step["command"], step.get("args") or {}) or ""
            if want:
                log.info("autopilot: %s auto-routes to model '%s'", step["command"], want)
        if want:
            # Phase batching on an exclusive GPU: make the step's model resident
            # first (a no-op when it already is).  Blocking is fine — this thread
            # runs one step at a time by design, and consecutive steps sharing a
            # model swap once, not per run.
            from . import serving as _sv
            try:
                self._state["last_reason"] = f"{label}: ensuring model '{want}'"
                _sv.ensure_active(want, timeout_s=float(
                    self.cfg["serving"].get("swap_timeout_s", 900)))
            except (RuntimeError, TimeoutError, OSError) as e:
                # The handshake fails while the model is in fact resident when
                # the serving group isn't supervisor-managed (manually-run
                # container, stale/absent swap state).  The gate exists to
                # avoid running against a swapped-OUT model — and the entry's
                # own endpoint answering is direct proof it isn't.  Trust the
                # evidence over the handshake; only hold when it's truly gone.
                if self._entry_answers(want):
                    self._state["last_reason"] = (f"{label}: swap handshake failed ({e}) "
                                                  f"but '{want}' is answering — proceeding")
                    log.warning("autopilot: %s", self._state["last_reason"])
                else:
                    now = time.time()
                    self._last_run[key] = now
                    self._hold_until[key] = now + float(plan.get("idle_interval_s", 60) or 60)
                    self._state["last_reason"] = f"{label}: model swap to {want} failed — {e}"
                    log.warning("autopilot: %s", self._state["last_reason"])
                    return
        try:
            res = self.ops.start(step["command"], step.get("args") or {})
        except ValueError as e:              # bad args (hand-edited plan) — treat as a
            res = {"ok": False, "error": str(e)}  # failed launch, not a loop crash: the
                                             # crash path retried the same step every 10s
                                             # forever and starved everything below it
        if not res.get("ok"):
            # Couldn't launch (bad args, or a job slipped in) — note and back off so we
            # don't hot-loop on a broken step.
            self._last_run[key] = time.time()
            self._state["last_reason"] = f"{step['command']}: {res.get('error','could not start')}"
            self._sleep(10)
            return
        # Wait for completion, staying responsive to stop.
        while not self._stop.is_set() and self.ops.running():
            self._stop.wait(2)
        now = time.time()
        self._last_run[key] = now
        # Work-aware hold: a step that found nothing to do (OPS_RESULT did_work:false)
        # or that exited non-zero stands aside for one idle interval, so the steps
        # below it get the slot.  When new work lands it is picked up again within
        # that interval — priority preemption survives, starvation doesn't.
        result = self.ops.result() or {}
        backoff = float(plan.get("idle_interval_s", 60) or 60)
        if result.get("command") == step["command"] and result.get("did_work") is False:
            self._hold_until[key] = now + backoff
            self._state["last_reason"] = f"{label}: no work — standing aside {int(backoff)}s"
        elif self.ops.status().get("exit_code") not in (0, None):
            self._hold_until[key] = now + backoff
            self._state["last_reason"] = f"{label}: failed — backing off {int(backoff)}s"
        else:
            self._hold_until.pop(key, None)
            self._state["last_reason"] = f"ran {step['command']}"
        # Loop restarts from the top → priority preemption.

    def _entry_answers(self, name: str) -> bool:
        """Is the named [serving] entry's own endpoint up right now?  Direct
        evidence of residency, used when the swap handshake is unavailable."""
        from . import serving as _sv
        for e in self.cfg["serving"]["llms"]:
            if str(e.get("name")) == name:
                host = str(e.get("host") or "127.0.0.1")
                try:
                    return _sv.probe_ready("127.0.0.1" if host == "0.0.0.0" else host,
                                           int(e.get("port") or 0))
                except Exception:
                    return False
        return False

    def _sleep(self, secs):
        self._stop.wait(secs)
