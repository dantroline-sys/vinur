"""Acceptance test for the mode override (header selector) and endpoint mode.

Automatic = the weekly schedule + Prioritizer govern.  Picking full / minimal
/ endpoint PINS that posture — no automated scheduler changes it until unset,
and unset reconciles to the schedule IMMEDIATELY.  Endpoint is the permanent
yield-all: the box hosts its LM(s) purely for OUTSIDE applications while
every self-initiated consumer stands down.

  * serving.override_state / set / clear round-trip; endpoint_state is a
    DERIVED view (endpoint exists only as an override); the autopilot's gate
    reads it.
  * apply_override: endpoint/full restore the LM when minimal was on (an
    endpoint with no model resident serves nobody); minimal engages the
    mechanism then pins; None unsets and reconciles to the schedule's want
    NOW, not at the next boundary.
  * apply_minimal ends a CONTRADICTING pin (the more recent explicit act
    wins) and keeps a matching one.
  * _reconcile_schedule holds while ANY mode is pinned.
  * OpsRunner.start refuses every job under an endpoint pin — panel,
    autopilot and any future caller hit the same wall — and spawns nothing.
  * status_data carries the override; cmd_status prints the MODE PINNED
    banner; cmd_mode status speaks both states.

Run:  python tests/endpoint_test.py     (stdlib only)
"""
import contextlib
import io
import json
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from knowledgehost import autopilot as A
from knowledgehost import ops as OPS
from knowledgehost import serving as SV
from knowledgehost import supervisor as SUP


def check(label, cond):
    if cond:
        print(f"  ok  {label}")
    else:
        print(f"  FAIL  {label}")
        check.failed += 1
check.failed = 0


def _cfg():
    return {"host": "127.0.0.1", "port": 8998,
            "serving": {"llms": [{"name": "big", "exclusive": True, "default": True},
                                 {"name": "aux", "exclusive": False}],
                        "embed": {"enabled": True, "port": 11437},
                        "reranker": {"enabled": True},
                        "minimal": {"embed": "cpu"}}}


def main():
    # ── flag round-trip + the derived endpoint view ──────────────────────────
    with tempfile.TemporaryDirectory() as td:
        fl0 = SV.OVERRIDE_FLAG
        SV.OVERRIDE_FLAG = Path(td) / "run" / "override.flag"
        try:
            check("override_state: absent flag reads as Automatic",
                  SV.override_state() == {})
            SV.set_override({"mode": "endpoint", "since": 123})
            check("set_override/override_state round-trip",
                  SV.override_state() == {"mode": "endpoint", "since": 123})
            check("endpoint_state derives ON from an endpoint pin",
                  SV.endpoint_state().get("on") and SV.endpoint_state()["since"] == 123)
            check("the autopilot's gate reads the derived view",
                  A.Autopilot._endpoint_on(object()) is True)
            SV.set_override({"mode": "minimal", "since": 1})
            check("a minimal pin is NOT endpoint mode", SV.endpoint_state() == {})
            SV.OVERRIDE_FLAG.write_text(json.dumps({"mode": "sideways"}))
            check("an unknown mode in the file reads as Automatic (fail-safe)",
                  SV.override_state() == {})
            SV.clear_override()
            check("clear_override removes the flag", SV.override_state() == {}
                  and A.Autopilot._endpoint_on(object()) is False)
        finally:
            SV.OVERRIDE_FLAG = fl0

    # ── apply_override / apply_minimal interplay (stubbed request lanes) ─────
    calls = {"svc": [], "swap": [], "min": [], "ov": []}
    orig = {n: getattr(SV, n) for n in ("request_service", "request_swap",
                                        "set_minimal", "clear_minimal",
                                        "set_override", "clear_override",
                                        "swap_state", "minimal_state",
                                        "override_state", "read_schedule",
                                        "schedule_wants_minimal")}
    SV.request_service = lambda s, a: calls["svc"].append((s, a))
    SV.request_swap = lambda n: calls["swap"].append(n)
    SV.set_minimal = lambda d: calls["min"].append(("set", d))
    SV.clear_minimal = lambda: calls["min"].append(("clear", None))
    SV.set_override = lambda d: calls["ov"].append(("set", d))
    SV.clear_override = lambda: calls["ov"].append(("clear", None))
    SV.swap_state = lambda: {"active": "big"}

    def clear():
        for c in calls.values():
            c.clear()
    try:
        SV.minimal_state = lambda: {}
        SV.override_state = lambda: {}
        r = SUP.apply_override(_cfg(), "endpoint")
        check("pin endpoint (LM already up): just the flag — no service touched",
              calls["ov"] and calls["ov"][-1][0] == "set"
              and calls["ov"][-1][1]["mode"] == "endpoint"
              and calls["svc"] == [] and r["restored_llm"] is False)

        clear()
        SV.minimal_state = lambda: {"on": True, "restore": "big"}
        SV.override_state = lambda: {"mode": "minimal"}     # pinned minimal before
        r2 = SUP.apply_override(_cfg(), "endpoint")
        check("pin endpoint from pinned-minimal: LM restored first (swap back + "
              "aux started), old pin cleared, new pin set, reported",
              r2["restored_llm"] is True and calls["swap"] == ["big"]
              and ("llm-aux", "start") in calls["svc"]
              and ("clear", None) in calls["ov"]
              and calls["ov"][-1][0] == "set"
              and calls["ov"][-1][1]["mode"] == "endpoint")

        clear()
        SV.minimal_state = lambda: {}
        SV.override_state = lambda: {}
        SUP.apply_override(_cfg(), "minimal")
        check("pin minimal: the mechanism engages (minimal.flag set, LMs "
              "stopped) and THEN the pin is written",
              any(c[0] == "set" for c in calls["min"])
              and ("llm-big", "stop") in calls["svc"]
              and calls["ov"][-1][0] == "set"
              and calls["ov"][-1][1]["mode"] == "minimal")

        clear()
        SV.minimal_state = lambda: {"on": True, "restore": "big"}
        SV.override_state = lambda: {}
        r3 = SUP.apply_override(_cfg(), "full")
        check("pin full from schedule-minimal: restores the LM, pins full",
              r3["restored_llm"] is True and calls["swap"] == ["big"]
              and calls["ov"][-1][1]["mode"] == "full")

        # unset: clears the pin and reconciles to the schedule NOW
        clear()
        SV.minimal_state = lambda: {}                       # box is full…
        SV.override_state = lambda: {}
        SV.read_schedule = lambda: {"enabled": True}
        SV.schedule_wants_minimal = lambda sched, now: True  # …schedule wants minimal
        r4 = SUP.apply_override(_cfg(), None)
        check("unset: pin cleared, box reconciled to the schedule immediately",
              calls["ov"][0] == ("clear", None)
              and any(c[0] == "set" for c in calls["min"])   # minimal engaged
              and r4["reconciled"] == "minimal")

        clear()
        SV.schedule_wants_minimal = lambda sched, now: None  # schedule not governing
        r5 = SUP.apply_override(_cfg(), None)
        check("unset with no governing schedule: nothing to reconcile",
              calls["ov"] == [("clear", None)] and calls["min"] == []
              and r5["reconciled"] is None)

        try:
            SUP.apply_override(_cfg(), "sideways")
            check("apply_override rejects a bad mode", False)
        except ValueError:
            check("apply_override rejects a bad mode", True)

        # manual mechanism flips vs. the pin: contradiction clears, match keeps
        clear()
        SV.minimal_state = lambda: {}
        SV.override_state = lambda: {"mode": "endpoint"}
        rm = SUP.apply_minimal(_cfg(), "on")
        check("minimal on under an endpoint pin ends the pin (recent act wins)",
              rm["cleared_override"] is True and ("clear", None) in calls["ov"])
        clear()
        SV.override_state = lambda: {"mode": "minimal"}
        rm2 = SUP.apply_minimal(_cfg(), "on")
        check("minimal on under a minimal pin keeps the pin (no contradiction)",
              rm2["cleared_override"] is False and calls["ov"] == [])
        clear()
        rm3 = SUP.apply_minimal(_cfg(), "off")
        check("minimal off under a minimal pin ends the pin",
              rm3["cleared_override"] is True and ("clear", None) in calls["ov"])
    finally:
        for n, f in orig.items():
            setattr(SV, n, f)

    # ── the weekly schedule is held while ANY mode is pinned ─────────────────
    sched0, wants0, apply0 = SV.read_schedule, SV.schedule_wants_minimal, SUP.apply_minimal
    ov0 = SV.override_state
    applied = []
    SV.read_schedule = lambda: {"enabled": True, "windows": {"mon": [["09:00", "17:00"]]}}
    SV.schedule_wants_minimal = lambda sched, now: True     # a boundary is due
    SUP.apply_minimal = lambda cfg, a: applied.append(a)
    try:
        SV.override_state = lambda: {"mode": "full"}
        out = SUP._reconcile_schedule(_cfg(), False)
        check("schedule tick under a pin (any mode): held — nothing applied, "
              "last_want passes through",
              applied == [] and out is False)
        SV.override_state = lambda: {}
        SUP._reconcile_schedule(_cfg(), False)
        check("…and acts again once the pin is unset", applied == ["on"])
    finally:
        SV.read_schedule, SV.schedule_wants_minimal = sched0, wants0
        SUP.apply_minimal, SV.override_state = apply0, ov0

    # ── OpsRunner refuses every job under an endpoint pin ────────────────────
    with tempfile.TemporaryDirectory() as td:
        fl0 = SV.OVERRIDE_FLAG
        SV.OVERRIDE_FLAG = Path(td) / "run" / "override.flag"
        try:
            runner = OPS.OpsRunner({"control_dir": td})
            SV.set_override({"mode": "endpoint", "since": 1})
            r = runner.start("distill", {})
            check("ops.start under an endpoint pin: refused, names the way out, "
                  "spawns nothing",
                  r["ok"] is False and "ENDPOINT" in r["error"]
                  and "mode automatic" in r["error"] and not runner.running())
            SV.set_override({"mode": "full", "since": 1})
            try:
                runner.start("no-such-verb", {})
                check("a full pin does NOT gate jobs (the verb validator "
                      "answers)", False)
            except ValueError as e:
                check("a full pin does NOT gate jobs (the verb validator "
                      "answers)", "unknown command" in str(e))
            SV.clear_override()
            try:
                runner.start("no-such-verb", {})
                check("…and Automatic doesn't either", False)
            except ValueError:
                check("…and Automatic doesn't either", True)
        finally:
            SV.OVERRIDE_FLAG = fl0

    # ── status surfaces ──────────────────────────────────────────────────────
    with tempfile.TemporaryDirectory() as td:
        st0, lg0, fl0 = SUP.STATE, SUP.LOGS, SV.OVERRIDE_FLAG
        SUP.STATE = Path(td) / "run" / "supervisor.json"
        SUP.LOGS = Path(td) / "log"
        SV.OVERRIDE_FLAG = Path(td) / "run" / "override.flag"
        SUP.STATE.parent.mkdir(parents=True, exist_ok=True)
        SUP.LOGS.mkdir(parents=True, exist_ok=True)
        SUP.STATE.write_text(json.dumps({"supervisor": os.getpid(),
                                         "services": {"kb": os.getpid()}}))
        SV.set_override({"mode": "endpoint", "since": 1})
        try:
            d = SUP.status_data()
            check("status_data carries the override (and the derived endpoint)",
                  d.get("override", {}).get("mode") == "endpoint"
                  and d.get("endpoint", {}).get("on") is True)
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                SUP.cmd_status()
            check("cmd_status prints the MODE PINNED: ENDPOINT banner with the "
                  "way back",
                  "MODE PINNED: ENDPOINT" in buf.getvalue()
                  and "mode automatic" in buf.getvalue())
            SV.set_override({"mode": "full", "since": 1})
            buf2 = io.StringIO()
            with contextlib.redirect_stdout(buf2):
                SUP.cmd_status()
            check("…and the generic banner for other pins",
                  "MODE PINNED: FULL" in buf2.getvalue())
            buf3 = io.StringIO()
            with contextlib.redirect_stdout(buf3):
                SUP.cmd_mode("status")
            check("cmd_mode status: pinned state speaks plainly",
                  "mode PINNED: full" in buf3.getvalue())
            SV.clear_override()
            buf4 = io.StringIO()
            with contextlib.redirect_stdout(buf4):
                SUP.cmd_mode("status")
            check("cmd_mode status: Automatic names the current posture",
                  "mode automatic" in buf4.getvalue()
                  and "posture" in buf4.getvalue())
        finally:
            SUP.STATE, SUP.LOGS, SV.OVERRIDE_FLAG = st0, lg0, fl0

    print()
    if check.failed:
        print(f"{check.failed} FAILED")
        return 1
    print("ALL PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
