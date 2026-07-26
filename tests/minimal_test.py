"""Acceptance test for minimal mode — vacate all VRAM, keep serving the KB.

  * plan_minimal (pure): on/off transitions as request-lane ops, with the
    active exclusive + non-exclusive LMs stopped, standby siblings left alone,
    and embed handled per config (cpu -> restart, off -> stop).
  * serving.minimal_state / set / clear round-trip + embed_argv device: the
    embedder drops to -ngl 0 (CPU, 0 VRAM) when the flag is on.
  * apply_minimal fires the right async requests (through the swap/service lanes).
  * status_data surfaces the flag and cmd_status prints the banner.
  * query.answer (kb_ask): with the embedder DOWN the BM25 lexical arm engages
    automatically (so kb_ask keeps answering with zero GPU); with it UP the
    read path is unchanged (no BM25).

Run:  python tests/minimal_test.py     (stdlib only)
"""
import contextlib
import io
import json
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from knowledgehost import serving as SV
from knowledgehost import supervisor as SUP


def check(label, cond):
    if cond:
        print(f"  ok  {label}")
    else:
        print(f"  FAIL  {label}")
        check.failed += 1
check.failed = 0


def _cfg(embed_enabled=True, embed_mode="cpu"):
    return {"host": "127.0.0.1", "port": 8998,
            "serving": {"llms": [{"name": "big", "exclusive": True, "default": True},
                                 {"name": "small", "exclusive": True},   # standby
                                 {"name": "aux", "exclusive": False}],   # resident
                        "embed": {"enabled": embed_enabled, "port": 11437},
                        "reranker": {"enabled": True},
                        "minimal": {"embed": embed_mode}}}


def main():
    # ── plan_minimal (pure) ──────────────────────────────────────────────────
    on = SUP.plan_minimal(_cfg(), "on", active="big")
    check("on: stops the ACTIVE exclusive + the non-exclusive LM, leaves standby",
          on["stop"] == ["llm-big", "llm-aux"] and "llm-small" not in on["stop"])
    check("on: embed=cpu -> restart (comes up on CPU), flag records device+restore",
          on["restart"] == ["embed"] and on["flag"]
          == {"on": True, "embed": "cpu", "restore": "big"} and on["swap"] is None)

    on_off = SUP.plan_minimal(_cfg(embed_mode="off"), "on", active="big")
    check("on: embed=off -> embed is STOPPED, nothing restarted",
          "embed" in on_off["stop"] and on_off["restart"] == []
          and on_off["flag"]["embed"] == "off")

    on_noembed = SUP.plan_minimal(_cfg(embed_enabled=False), "on", active="small")
    check("on: honours a swapped-in active model; no embed service to touch",
          on_noembed["stop"] == ["llm-small", "llm-aux"]
          and on_noembed["restart"] == [] and "embed" not in on_noembed["stop"])

    off = SUP.plan_minimal(_cfg(), "off", restore="big")
    check("off: swaps the model back, starts the non-exclusive LM, clears the flag",
          off["swap"] == "big" and off["start"] == ["llm-aux"]
          and off["flag"] is None and off["restart"] == ["embed"])

    off_default = SUP.plan_minimal(_cfg(), "off", restore="")
    check("off: with no remembered restore, falls back to the default LM",
          off_default["swap"] == "big")

    # ── flag round-trip + embed device ───────────────────────────────────────
    with tempfile.TemporaryDirectory() as td:
        flag0, llama0 = SV.MINIMAL_FLAG, SV._llama_server
        SV.MINIMAL_FLAG = Path(td) / "minimal.flag"
        SV._llama_server = lambda: "llama-server"
        try:
            check("minimal_state: absent flag reads as full mode", SV.minimal_state() == {})
            argv_full = SV.embed_argv(_cfg(), "m.gguf")
            i = argv_full.index("-ngl")
            check("embed_argv: full mode keeps the embedder on GPU (-ngl 99)",
                  argv_full[i + 1] == "99")
            SV.set_minimal({"on": True, "embed": "cpu", "restore": "big"})
            check("set_minimal/minimal_state round-trip",
                  SV.minimal_state().get("on") and SV.minimal_state()["restore"] == "big")
            argv_min = SV.embed_argv(_cfg(), "m.gguf")
            j = argv_min.index("-ngl")
            check("embed_argv: minimal mode drops the embedder to CPU (-ngl 0)",
                  argv_min[j + 1] == "0")
            SV.clear_minimal()
            check("clear_minimal removes the flag", SV.minimal_state() == {})
        finally:
            SV.MINIMAL_FLAG, SV._llama_server = flag0, llama0

    # ── apply_minimal fires the right async requests ─────────────────────────
    calls = {"svc": [], "swap": [], "flag": []}
    orig = {n: getattr(SV, n) for n in ("request_service", "request_swap",
                                        "set_minimal", "clear_minimal",
                                        "swap_state", "minimal_state")}
    SV.request_service = lambda s, a: calls["svc"].append((s, a))
    SV.request_swap = lambda n: calls["swap"].append(n)
    SV.set_minimal = lambda d: calls["flag"].append(("set", d))
    SV.clear_minimal = lambda: calls["flag"].append(("clear", None))
    SV.swap_state = lambda: {"active": "big"}
    SV.minimal_state = lambda: {"restore": "big"}
    try:
        r_on = SUP.apply_minimal(_cfg(), "on")
        check("apply_minimal on: flag written FIRST, then stop LMs + restart embed",
              calls["flag"] == [("set", {"on": True, "embed": "cpu", "restore": "big"})]
              and ("llm-big", "stop") in calls["svc"]
              and ("llm-aux", "stop") in calls["svc"]
              and ("embed", "restart") in calls["svc"] and r_on["action"] == "on")
        calls["svc"].clear(); calls["swap"].clear(); calls["flag"].clear()
        SUP.apply_minimal(_cfg(), "off")
        check("apply_minimal off: flag cleared, model swapped back, aux started",
              calls["flag"] == [("clear", None)] and calls["swap"] == ["big"]
              and ("llm-aux", "start") in calls["svc"]
              and ("embed", "restart") in calls["svc"])
        try:
            SUP.apply_minimal(_cfg(), "sideways")
            check("apply_minimal rejects a bad action", False)
        except ValueError:
            check("apply_minimal rejects a bad action", True)
    finally:
        for n, f in orig.items():
            setattr(SV, n, f)

    # ── status_data surfaces the flag; cmd_status prints the banner ──────────
    with tempfile.TemporaryDirectory() as td:
        st0, lg0, fl0 = SUP.STATE, SUP.LOGS, SV.MINIMAL_FLAG
        SUP.STATE = Path(td) / "run" / "supervisor.json"
        SUP.LOGS = Path(td) / "log"
        SV.MINIMAL_FLAG = Path(td) / "run" / "minimal.flag"
        SUP.STATE.parent.mkdir(parents=True, exist_ok=True)
        SUP.LOGS.mkdir(parents=True, exist_ok=True)
        SUP.STATE.write_text(json.dumps({"supervisor": os.getpid(),
                                         "services": {"kb": os.getpid()},
                                         "held": ["llm-big"]}))
        SV.set_minimal({"on": True, "embed": "cpu", "restore": "big"})
        try:
            d = SUP.status_data()
            check("status_data carries the minimal flag",
                  d.get("minimal", {}).get("on") is True
                  and d["minimal"]["restore"] == "big")
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                SUP.cmd_status()
            out = buf.getvalue()
            check("cmd_status prints the MINIMAL MODE banner naming the restore model",
                  "MINIMAL MODE" in out and "minimal off" in out and "big" in out)
        finally:
            SUP.STATE, SUP.LOGS, SV.MINIMAL_FLAG = st0, lg0, fl0

    # ── kb_ask lexical fallback (query.answer) ───────────────────────────────
    from knowledgehost import query as Q

    class FakeKB:
        def __init__(self):
            self.bm25 = 0
            self.dense = 0

        def search(self, qvec, n, empirical_only=False):
            self.dense += 1
            return []

        def search_cards_bm25(self, q, pool):
            self.bm25 += 1
            return []

        def log_gap(self, q, intent):
            pass

    class FakeEmbedder:
        def embed_one(self, text, kind):
            return [0.1] * 8

    down = FakeKB()
    Q.answer(down, None, "what is companion planting")
    check("kb_ask: embedder DOWN auto-engages the BM25 lexical arm (still answers)",
          down.bm25 == 1)
    up = FakeKB()
    Q.answer(up, FakeEmbedder(), "what is companion planting")
    check("kb_ask: embedder UP is unchanged — dense path ran, no BM25 fallback",
          up.dense >= 1 and up.bm25 == 0)

    # ── weekly schedule: pure evaluator ──────────────────────────────────────
    from datetime import datetime
    W = {"enabled": True, "windows": {"mon": [["22:00", "06:00"]],   # overnight
                                      "wed": [["09:00", "17:00"]]}}   # daytime
    mon, tue, wed = datetime(2026, 7, 20, 0, 0), datetime(2026, 7, 21, 0, 0), \
        datetime(2026, 7, 22, 0, 0)
    check("schedule: inside an overnight window -> full (wants minimal False)",
          SV.schedule_wants_minimal(W, mon.replace(hour=23)) is False)
    check("schedule: overnight window's morning tail counts on the NEXT day",
          SV.schedule_wants_minimal(W, tue.replace(hour=3)) is False
          and SV.schedule_wants_minimal(W, tue.replace(hour=7)) is True)
    check("schedule: inside/outside a daytime window",
          SV.schedule_wants_minimal(W, wed.replace(hour=12)) is False
          and SV.schedule_wants_minimal(W, wed.replace(hour=18)) is True)
    check("schedule: a day with no window is minimal all day",
          SV.schedule_wants_minimal(W, tue.replace(hour=12)) is True)
    check("schedule: disabled -> None (not governing)",
          SV.schedule_wants_minimal({"enabled": False, "windows": W["windows"]},
                                    wed.replace(hour=12)) is None)

    # ── clean_schedule: validate + normalise + drop junk ─────────────────────
    cl = SV.clean_schedule({"enabled": 1, "windows": {
        "mon": [["9:5", "17:00"], ["bad", "x"], ["1:1", "1:1"]],   # pad, drop, drop(equal)
        "zz": [["1:00", "2:00"]]}})                                # bogus day dropped
    check("clean_schedule: pads HH:MM, drops malformed + zero-length + bad days",
          cl == {"enabled": True, "windows": {"mon": [["09:05", "17:00"]]}})

    # ── _reconcile_schedule: acts only on boundary crossings ─────────────────
    rec = {"apply": []}
    o2 = {n: getattr(SV, n) for n in ("read_schedule", "schedule_wants_minimal",
                                      "minimal_state")}
    ap0 = SUP.apply_minimal
    SV.read_schedule = lambda: {"enabled": True}
    SUP.apply_minimal = lambda cfg, a: rec["apply"].append(a)
    try:
        SV.schedule_wants_minimal = lambda s, n: True     # wants minimal
        SV.minimal_state = lambda: {"on": False}          # currently full
        w1 = SUP._reconcile_schedule({}, None)            # first eval → boundary
        check("_reconcile: first eval crosses to minimal on (box was full)",
              w1 is True and rec["apply"] == ["on"])
        rec["apply"].clear()
        SV.minimal_state = lambda: {"on": True}
        w2 = SUP._reconcile_schedule({}, w1)              # same want → no action
        check("_reconcile: inside the same window, no re-fire (manual override sticks)",
              w2 is True and rec["apply"] == [])
        SV.schedule_wants_minimal = lambda s, n: False    # boundary: window opened
        w3 = SUP._reconcile_schedule({}, w2)
        check("_reconcile: boundary to full-power calls apply_minimal off",
              w3 is False and rec["apply"] == ["off"])
        rec["apply"].clear()
        SV.read_schedule = lambda: {"enabled": False}
        SV.schedule_wants_minimal = lambda s, n: None
        w4 = SUP._reconcile_schedule({}, w3)
        check("_reconcile: disabled schedule stops governing (None, no action)",
              w4 is None and rec["apply"] == [])
    finally:
        for n, f in o2.items():
            setattr(SV, n, f)
        SUP.apply_minimal = ap0

    # ── autopilot stands down during minimal (must NOT swap the model back) ───
    import threading
    import time as _time
    from knowledgehost import autopilot as AP

    class FakeOps:
        def __init__(self):
            self.started = []
        def running(self):
            return False
        def start(self, cmd, args):
            self.started.append(cmd)
            return {"ok": True}
        def result(self):
            return {}
        def status(self):
            return {"exit_code": 0}

    ap = AP.Autopilot({"serving": {"llms": [], "swap_timeout_s": 900}}, FakeOps())
    lp0, ms1 = AP.load_plan, SV.minimal_state
    AP.load_plan = lambda cfg: {"enabled": True, "idle_interval_s": 60,
                                "respect_leases": False, "auto_models": True,
                                "steps": [{"command": "distill", "enabled": True,
                                           "args": {}, "min_interval_s": 0}]}
    try:
        check("_minimal_on reflects the flag",
              (setattr(SV, "minimal_state", lambda: {"on": True}) or ap._minimal_on()) is True
              and (setattr(SV, "minimal_state", lambda: {}) or ap._minimal_on()) is False)
        SV.minimal_state = lambda: {"on": True}          # minimal engaged
        t = threading.Thread(target=ap._loop, daemon=True)
        t.start(); _time.sleep(0.3); ap.stop(); t.join(timeout=2)
        check("autopilot PAUSES during minimal — never launches a step (no model swap)",
              ap.ops.started == [] and "minimal" in ap._state["last_reason"])
    finally:
        AP.load_plan, SV.minimal_state = lp0, ms1

    print()
    if check.failed:
        print(f"{check.failed} FAILURE(S)")
        return 1
    print("ALL OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
