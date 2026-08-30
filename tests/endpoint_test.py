"""Acceptance test for endpoint mode — the permanent yield-all: the box hosts
its LM(s) purely as an endpoint for OUTSIDE applications, and every
self-initiated consumer stands down.

  * serving.endpoint_state / set / clear round-trip (flag file, like minimal's).
  * apply_endpoint: on writes the flag — and first RESTORES the LM when minimal
    was on (an endpoint with no model resident serves nobody); off clears it.
  * apply_minimal("on") deliberately ENDS endpoint mode (the more recent
    explicit act wins) and says so in its summary.
  * _reconcile_schedule holds while endpoint is on: no boundary may vacate the
    model out from under the outside apps using it.
  * the autopilot's endpoint gate reads the flag.
  * OpsRunner.start refuses every job while the flag is on — panel, autopilot
    and any future caller all hit the same wall — and spawns nothing.
  * status_data carries the flag; cmd_status prints the ENDPOINT MODE banner;
    cmd_endpoint status speaks both states.

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
    # ── flag round-trip ──────────────────────────────────────────────────────
    with tempfile.TemporaryDirectory() as td:
        fl0 = SV.ENDPOINT_FLAG
        SV.ENDPOINT_FLAG = Path(td) / "run" / "endpoint.flag"
        try:
            check("endpoint_state: absent flag reads as off", SV.endpoint_state() == {})
            SV.set_endpoint({"on": True, "since": 123})
            check("set_endpoint/endpoint_state round-trip",
                  SV.endpoint_state().get("on") and SV.endpoint_state()["since"] == 123)
            check("the autopilot's gate reads the same flag",
                  A.Autopilot._endpoint_on(object()) is True)
            SV.clear_endpoint()
            check("clear_endpoint removes the flag", SV.endpoint_state() == {})
            check("…and the autopilot gate follows",
                  A.Autopilot._endpoint_on(object()) is False)
        finally:
            SV.ENDPOINT_FLAG = fl0

    # ── apply_endpoint / apply_minimal interplay (stubbed request lanes) ─────
    calls = {"svc": [], "swap": [], "min": [], "ep": []}
    orig = {n: getattr(SV, n) for n in ("request_service", "request_swap",
                                        "set_minimal", "clear_minimal",
                                        "set_endpoint", "clear_endpoint",
                                        "swap_state", "minimal_state",
                                        "endpoint_state")}
    SV.request_service = lambda s, a: calls["svc"].append((s, a))
    SV.request_swap = lambda n: calls["swap"].append(n)
    SV.set_minimal = lambda d: calls["min"].append(("set", d))
    SV.clear_minimal = lambda: calls["min"].append(("clear", None))
    SV.set_endpoint = lambda d: calls["ep"].append(("set", d))
    SV.clear_endpoint = lambda: calls["ep"].append(("clear", None))
    SV.swap_state = lambda: {"active": "big"}
    try:
        SV.minimal_state = lambda: {}
        SV.endpoint_state = lambda: {}
        r = SUP.apply_endpoint(_cfg(), "on")
        check("endpoint on (minimal off): just the flag — no service is touched",
              calls["ep"] and calls["ep"][0][0] == "set" and calls["ep"][0][1]["on"]
              and calls["svc"] == [] and r["restored_llm"] is False)

        for c in calls.values():
            c.clear()
        SV.minimal_state = lambda: {"on": True, "restore": "big"}
        r2 = SUP.apply_endpoint(_cfg(), "on")
        check("endpoint on while MINIMAL: restores the LM first (swap back + "
              "aux started), reports it",
              r2["restored_llm"] is True and calls["swap"] == ["big"]
              and ("llm-aux", "start") in calls["svc"]
              and ("clear", None) in calls["min"]
              and calls["ep"][-1][0] == "set")

        for c in calls.values():
            c.clear()
        SV.minimal_state = lambda: {}
        r3 = SUP.apply_endpoint(_cfg(), "off")
        check("endpoint off: clears the flag, touches nothing else",
              calls["ep"] == [("clear", None)] and calls["svc"] == []
              and r3["action"] == "off")

        try:
            SUP.apply_endpoint(_cfg(), "sideways")
            check("apply_endpoint rejects a bad action", False)
        except ValueError:
            check("apply_endpoint rejects a bad action", True)

        # minimal on while endpoint on → endpoint deliberately cleared + reported
        for c in calls.values():
            c.clear()
        SV.endpoint_state = lambda: {"on": True}
        r4 = SUP.apply_minimal(_cfg(), "on")
        check("minimal on ENDS endpoint mode (recent explicit act wins) and says so",
              r4["cleared_endpoint"] is True and ("clear", None) in calls["ep"]
              and any(c[0] == "set" for c in calls["min"]))
    finally:
        for n, f in orig.items():
            setattr(SV, n, f)

    # ── the weekly schedule is held while endpoint is on ─────────────────────
    sched0, wants0, apply0 = SV.read_schedule, SV.schedule_wants_minimal, SUP.apply_minimal
    ep0 = SV.endpoint_state
    applied = []
    SV.read_schedule = lambda: {"enabled": True, "windows": {"mon": [["09:00", "17:00"]]}}
    SV.schedule_wants_minimal = lambda sched, now: True     # a boundary is due
    SUP.apply_minimal = lambda cfg, a: applied.append(a)
    try:
        SV.endpoint_state = lambda: {"on": True}
        out = SUP._reconcile_schedule(_cfg(), False)
        check("schedule tick under endpoint mode: held — nothing applied, "
              "last_want passes through",
              applied == [] and out is False)
        SV.endpoint_state = lambda: {}
        SUP._reconcile_schedule(_cfg(), False)
        check("…and acts again once endpoint mode is off", applied == ["on"])
    finally:
        SV.read_schedule, SV.schedule_wants_minimal = sched0, wants0
        SUP.apply_minimal, SV.endpoint_state = apply0, ep0

    # ── OpsRunner refuses every job while the flag is on ─────────────────────
    with tempfile.TemporaryDirectory() as td:
        fl0 = SV.ENDPOINT_FLAG
        SV.ENDPOINT_FLAG = Path(td) / "run" / "endpoint.flag"
        try:
            runner = OPS.OpsRunner({"control_dir": td})
            SV.set_endpoint({"on": True})
            r = runner.start("distill", {})
            check("ops.start under endpoint mode: refused, names the switch, "
                  "spawns nothing",
                  r["ok"] is False and "endpoint mode" in r["error"]
                  and "endpoint off" in r["error"] and not runner.running())
            SV.clear_endpoint()
            # off again: the gate opens — the call reaches the verb validator
            # (which raises on the fake verb), instead of being answered by
            # the endpoint refusal.
            try:
                runner.start("no-such-verb", {})
                check("…and the gate opens when it's off (the verb validator "
                      "answers, not the gate)", False)
            except ValueError as e:
                check("…and the gate opens when it's off (the verb validator "
                      "answers, not the gate)", "unknown command" in str(e))
        finally:
            SV.ENDPOINT_FLAG = fl0

    # ── status surfaces ──────────────────────────────────────────────────────
    with tempfile.TemporaryDirectory() as td:
        st0, lg0, fl0 = SUP.STATE, SUP.LOGS, SV.ENDPOINT_FLAG
        SUP.STATE = Path(td) / "run" / "supervisor.json"
        SUP.LOGS = Path(td) / "log"
        SV.ENDPOINT_FLAG = Path(td) / "run" / "endpoint.flag"
        SUP.STATE.parent.mkdir(parents=True, exist_ok=True)
        SUP.LOGS.mkdir(parents=True, exist_ok=True)
        SUP.STATE.write_text(json.dumps({"supervisor": os.getpid(),
                                         "services": {"kb": os.getpid()}}))
        SV.set_endpoint({"on": True})
        try:
            d = SUP.status_data()
            check("status_data carries the endpoint flag",
                  d.get("endpoint", {}).get("on") is True)
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                SUP.cmd_status()
            check("cmd_status prints the ENDPOINT MODE banner with the off command",
                  "ENDPOINT MODE" in buf.getvalue()
                  and "endpoint off" in buf.getvalue())
            buf2 = io.StringIO()
            with contextlib.redirect_stdout(buf2):
                SUP.cmd_endpoint("status")
            check("cmd_endpoint status: ON state speaks plainly",
                  "endpoint mode ON" in buf2.getvalue())
            SV.clear_endpoint()
            buf3 = io.StringIO()
            with contextlib.redirect_stdout(buf3):
                SUP.cmd_endpoint("status")
            check("cmd_endpoint status: OFF state too",
                  "endpoint mode OFF" in buf3.getvalue())
        finally:
            SUP.STATE, SUP.LOGS, SV.ENDPOINT_FLAG = st0, lg0, fl0

    print()
    if check.failed:
        print(f"{check.failed} FAILED")
        return 1
    print("ALL PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
