"""Scheduler test for the Prioritizer — the two-distill starvation fix.

The motivating bug: two `distill` steps with different args (bundle=vinkona
above the plain backlog pass), both at min_interval 0.  A 0-interval step is
always "due", so the first one monopolised the slot forever — even when it had
nothing to distil — and the second NEVER ran.  The fix is the no-work hold:
verbs report did_work via an OPS_RESULT line, the runner relays it, and the
autopilot stands a no-work (or failed) step aside for one idle interval.

Covers: step identity by (command,args), the hold in due_step, hold expiry,
the _run_step wiring against a fake ops runner, OPS_RESULT emission and
OpsRunner.result() parsing.

Run:  python tests/autopilot_test.py     (stdlib only)
"""
import io
import json
import os
import subprocess
import sys
import tempfile
import time
from contextlib import redirect_stdout

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from knowledgehost import autopilot as ap
from knowledgehost import ops as ops_mod


def check(label, cond):
    print(("  ok  " if cond else "  FAIL ") + label)
    if not cond:
        check.failed += 1
check.failed = 0


def main():
    A = {"command": "distill", "args": {"bundle": "vinkona"}, "enabled": True,
         "min_interval_s": 0, "label": "vinkona drops"}
    B = {"command": "distill", "args": {"limit": 50}, "enabled": True,
         "min_interval_s": 0, "label": "backlog"}
    steps = [A, B]

    # ── identity: same verb, different args = different steps ───────────────
    check("step_key distinguishes the two distills",
          ap.step_key(A) != ap.step_key(B))

    # ── the starvation scenario ──────────────────────────────────────────────
    now = 1000.0
    last, hold = {}, {}
    i, s = ap.due_step(steps, last, now, hold)
    check("priority: the vinkona distill is picked first", s is A)

    # it ran and found nothing → hold it for one idle interval
    last[ap.step_key(A)] = now
    hold[ap.step_key(A)] = now + 60
    i, s = ap.due_step(steps, last, now + 1, hold)
    check("REGRESSION: with A on no-work hold, B finally runs", s is B)

    # B also drains → both on hold → nothing due (the loop then idles)
    last[ap.step_key(B)] = now + 1
    hold[ap.step_key(B)] = now + 61
    i, s = ap.due_step(steps, last, now + 2, hold)
    check("both held → idle", s is None)

    # hold expires → A (higher priority) picked again: preemption survives
    i, s = ap.due_step(steps, last, now + 62, hold)
    check("hold expiry: A preempts again", s is A)

    # min_interval still throttles independently of holds
    C = {"command": "link", "args": {}, "enabled": True, "min_interval_s": 3600}
    last[ap.step_key(C)] = now
    i, s = ap.due_step([C], last, now + 100, {})
    check("min_interval still honoured", s is None)
    i, s = ap.due_step([C], last, now + 3601, {})
    check("…and elapses", s is C)

    check("disabled steps skipped",
          ap.due_step([{**A, "enabled": False}], {}, now, {})[1] is None)

    # legacy call shape (no hold dict) keeps working
    check("due_step without hold_until behaves as before",
          ap.due_step(steps, {}, now)[1] is A)

    # ── _run_step wiring against a fake ops runner ───────────────────────────
    class FakeOps:
        def __init__(self, did_work, exit_code=0):
            self._r = {"command": "distill", "exit_code": exit_code,
                       "did_work": did_work}
            self._exit = exit_code
        def start(self, command, args):
            return {"ok": True}
        def running(self):
            return False
        def result(self):
            return dict(self._r)
        def status(self):
            return {"exit_code": self._exit}

    plan = {"idle_interval_s": 60}
    pilot = ap.Autopilot({}, FakeOps(did_work=False))
    pilot._run_step(A, plan)
    key = ap.step_key(A)
    check("no-work run sets the hold",
          pilot._hold_until.get(key, 0) > time.time() - 1
          and "no work" in pilot._state["last_reason"])

    pilot = ap.Autopilot({}, FakeOps(did_work=True))
    pilot._run_step(A, plan)
    check("productive run sets no hold", key not in pilot._hold_until)

    class NoResultOps(FakeOps):
        def __init__(self, exit_code):
            super().__init__(True, exit_code)
        def result(self):
            return None
    pilot = ap.Autopilot({}, NoResultOps(exit_code=1))
    pilot._run_step(A, plan)
    check("failed run (no OPS_RESULT) also backs off",
          pilot._hold_until.get(key, 0) > time.time() - 1)
    pilot = ap.Autopilot({}, NoResultOps(exit_code=0))
    pilot._run_step(A, plan)
    check("clean run without OPS_RESULT keeps old always-due behaviour",
          key not in pilot._hold_until)

    # ── the result channel itself ────────────────────────────────────────────
    buf = io.StringIO()
    with redirect_stdout(buf):
        ops_mod.emit_result(False, chunks=0, skipped=3)
    line = buf.getvalue().strip()
    check("emit_result prints one OPS_RESULT line",
          line.startswith(ops_mod.RESULT_PREFIX)
          and json.loads(line[len(ops_mod.RESULT_PREFIX):]) ==
          {"did_work": False, "chunks": 0, "skipped": 3})

    td = tempfile.mkdtemp(prefix="ops-result-")
    runner = ops_mod.OpsRunner({"control_dir": td})
    logfile = os.path.join(td, "fake.log")
    with open(logfile, "w") as f:
        f.write("distill: warming up\n")
        f.write(ops_mod.RESULT_PREFIX + '{"did_work": false, "chunks": 0}\n')
        f.write("bye\n")
    proc = subprocess.Popen(["/bin/true"])
    proc.wait()
    runner._job = {"proc": proc, "logfh": None, "command": "distill",
                   "argv": [], "started": time.time(), "logfile": logfile}
    res = runner.result()
    check("OpsRunner.result parses the line after exit",
          res == {"command": "distill", "exit_code": 0, "did_work": False,
                  "chunks": 0})

    check("every command has panel help",
          all(c in ops_mod.HELP and "_" in ops_mod.HELP[c]
              for c in ops_mod.COMMANDS))
    check("every documented option exists in the spec",
          all(k == "_" or k in ops_mod.COMMANDS[c]
              for c in ops_mod.HELP for k in ops_mod.HELP[c]))

    print()
    if check.failed:
        print(f"{check.failed} FAILURE(S)")
        return 1
    print("ALL OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
