#!/usr/bin/env python
"""Job progress: the queue survey, the live progress channel, and 'what's next'.

A distil is the longest thing this host does and it used to say nothing about its
own size — the log counted chunks up, with no total, no current document and no
end.  This covers the three pieces that fixed that: store.distill_queue (the work
ahead, surveyed once), distill.DistillProgress (the emitted OPS_PROGRESS record),
ops.OpsRunner.progress (the panel's server-side read of it), and autopilot.next_up
(what runs after this one).  Real sqlite, no LM, no server.

    python tests/ops_progress_test.py
"""
import json
import os
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from knowledgehost import autopilot as AP                  # noqa: E402
from knowledgehost import distill as D                     # noqa: E402
from knowledgehost import ops as OPS                       # noqa: E402
from knowledgehost import store as S                       # noqa: E402
from knowledgehost.kb import KB                            # noqa: E402

PASS = FAIL = 0


def check(name, cond):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  ok  {name}")
    else:
        FAIL += 1
        print(f"FAIL  {name}")


def _rec(cid, path, text):
    return {"id": cid, "path_or_url": path, "text": text, "title": "T", "section": "",
            "source_type": "pdf", "tokens": len(text.split()), "vector": [0.1, 0.2]}


class Recorder:
    """Stands in for ops.emit_progress — keeps the records instead of printing."""

    def __init__(self):
        self.recs = []

    def __call__(self, phase, **kw):
        self.recs.append({"phase": phase, **kw})

    @property
    def last(self):
        return self.recs[-1] if self.recs else {}


def survey_side():
    """store.distill_queue: what is left, per document, before the pass starts."""
    tmp = tempfile.mkdtemp()
    cfg = {"backend": "sqlite", "db_path": str(Path(tmp) / "index.db"),
           "kb_path": str(Path(tmp) / "kb.db"), "embed_dim": 8}
    store = S.make_store(cfg)
    kb = KB(cfg)
    A, B, C = "/corpus/big.pdf", "/corpus/small.pdf", "/drops/research.md"
    store.add_chunks([_rec(f"a{i}", A, f"alpha {i}") for i in range(5)])
    store.add_chunks([_rec("b1", B, "beta one"), _rec("b2", B, "beta two")])
    store.add_chunks([_rec("c1", C, "gamma one")])
    store.set_doc_meta(C, {"bundle": "vinkona"})

    q = store.distill_queue(cfg["kb_path"])
    check("a fresh corpus is entirely pending", q["pending"] == 8 and q["docs_pending"] == 3)
    check("the queue is per-document, biggest first",
          [d["doc"] for d in q["docs"]] == [A, B, C]
          and [d["pending"] for d in q["docs"]] == [5, 2, 1])

    kb.mark_distilled("a1")
    kb.mark_distilled("a2")
    kb.mark_zone_skipped("b1", "references")
    q = store.distill_queue(cfg["kb_path"])
    check("distilled chunks leave the queue", q["pending"] == 5
          and [d["pending"] for d in q["docs"] if d["doc"] == A] == [3])
    check("furniture the distiller walks past is NOT counted as pending "
          "(40% with an empty queue reads as complete, not stuck)",
          [d["pending"] for d in q["docs"] if d["doc"] == B] == [1])

    qv = store.distill_queue(cfg["kb_path"], bundle="vinkona")
    check("a bundle-restricted pass surveys only its own lane",
          qv["pending"] == 1 and [d["doc"] for d in qv["docs"]] == [C])
    qb = store.distill_queue(cfg["kb_path"], bundle="base")
    check("documents with no bundle metadata read as 'base' (matching _chunk_bundle)",
          qb["pending"] == 4 and sorted(d["doc"] for d in qb["docs"]) == [A, B])

    q = store.distill_queue(cfg["kb_path"], limit=1)
    check("the per-document list is capped but the totals are not",
          len(q["docs"]) == 1 and q["pending"] == 5 and q["truncated"] is True)

    check("an unreadable kb.db means 'total unknown', not a crash",
          store.distill_queue("/nonexistent/dir/kb.db") == {})
    kb.close()
    store.close()
    return cfg, A, B


def progress_side(cfg, A, B):
    """DistillProgress: survey → tick → the record the panel renders."""
    store = S.make_store(cfg)
    rec = Recorder()
    p = D.DistillProgress(emit=rec, every_s=0.0)     # no throttle: assert on every tick
    lines = []
    p.survey(store, cfg, log_fn=lambda f, *a: lines.append(f % a if a else f))

    check("the survey says the size of the job out loud, in the log",
          any("queue ahead" in ln and "5 chunk(s)" in ln for ln in lines))
    check("the survey names the biggest documents (which document is this job about?)",
          any("big.pdf (3)" in ln for ln in lines))
    check("the survey emits a starting record with the total",
          rec.last.get("steps") == 5 and rec.last.get("step") == 0)

    p.tick(A, concepts=4, relations=2, cards=1)
    p.tick(A, concepts=3, relations=0, cards=0)
    r = rec.last
    check("progress counts chunks against the surveyed total",
          r["step"] == 2 and r["steps"] == 5 and r["left"] == 3)
    check("the current document is named, with its OWN position",
          r["doc"] == "big.pdf" and r["doc_step"] == 2 and r["doc_steps"] == 3)
    check("what the pass produced rides along",
          r["added"] == {"concepts": 7, "relations": 2, "cards": 1})
    check("the document path is reduced to a basename (the panel shows it verbatim)",
          "/" not in r["doc"])

    p.tick(A)
    p.tick(B)
    r = rec.last
    check("finishing a document drops it out of the remaining count",
          r["docs_left"] == 1 and r["doc"] == "small.pdf")

    p.finish()
    check("the final record is marked done and carries the last count",
          rec.last.get("done") is True and rec.last["step"] == 4)

    # ── throttling: these lines share the ops log with the readable detail ──
    rec2 = Recorder()
    q = D.DistillProgress(emit=rec2, every_s=60.0)
    q.total = 100
    for _ in range(50):
        q.tick(A)
    check("a line per chunk would flood the log — emission is throttled",
          len(rec2.recs) <= 1)
    q.finish()
    check("…but the final line is never throttled away (the bar must land)",
          rec2.last.get("done") is True and rec2.last["step"] == 50)

    # ── a backend that cannot survey: count up, never invent a total ────────
    rec3 = Recorder()
    n = D.DistillProgress(emit=rec3, every_s=0.0)

    class NoSurvey:
        pass
    n.survey(NoSurvey(), cfg, log_fn=lambda f, *a: None)
    n.tick(A)
    check("with no survey the record has counts but NO total (an honest bar)",
          rec3.last["step"] == 1 and "steps" not in rec3.last and "eta_s" not in rec3.last)

    # ── rate + eta ──────────────────────────────────────────────────────────
    rec4 = Recorder()
    e = D.DistillProgress(emit=rec4, every_s=0.0)
    e.total = 120
    past = time.time() - 60.0
    e.started = past
    e._marks = [(past, 0)]
    for _ in range(60):
        e.tick(A)
    r = rec4.last
    check("rate and ETA are derived from real elapsed time (60/min → ~60s left)",
          55 <= r.get("rate_min", 0) <= 65 and 50 <= r.get("eta_s", 0) <= 70)
    store.close()


def runner_side():
    """OpsRunner.progress: the panel reads the LAST record, from the log tail."""
    tmp = tempfile.mkdtemp()
    runner = OPS.OpsRunner({"control_dir": tmp})
    logfile = Path(tmp) / "fake.log"

    class Dead:
        def poll(self):
            return 0
    runner._job = {"proc": Dead(), "command": "distill", "argv": [],
                   "started": time.time(), "logfile": str(logfile)}

    check("no log yet → no progress (never a fabricated bar)", runner.progress() is None)
    with open(logfile, "w") as f:
        f.write("2026-08-13 distilling…\n")
        f.write(OPS.PROGRESS_PREFIX + json.dumps({"phase": "distil", "step": 1, "steps": 9}) + "\n")
        f.write("… some ordinary log line\n")
        f.write(OPS.PROGRESS_PREFIX + json.dumps({"phase": "distil", "step": 7, "steps": 9}) + "\n")
        f.write("… another ordinary line\n")
    check("the LAST progress record wins, whatever follows it in the log",
          runner.progress() == {"phase": "distil", "step": 7, "steps": 9})

    with open(logfile, "a") as f:
        f.write(OPS.PROGRESS_PREFIX + '{"phase": "distil", "step": 8,')   # torn mid-write
    check("a half-written line falls back to the previous complete one",
          (runner.progress() or {}).get("step") == 7)

    # a long run's log is megabytes — only the tail is read, and the record is there
    with open(logfile, "a") as f:
        f.write("\n" + ("x" * 200 + "\n") * 2000)
        f.write(OPS.PROGRESS_PREFIX + json.dumps({"phase": "distil", "step": 9}) + "\n")
    size = os.path.getsize(logfile)
    check("a megabyte log is read from the tail, not whole",
          size > OPS._PROGRESS_TAIL_BYTES
          and (runner.progress() or {}).get("step") == 9)

    with open(logfile, "w") as f:                     # progress scrolled out of the tail
        f.write(OPS.PROGRESS_PREFIX + json.dumps({"step": 1}) + "\n")
        f.write(("y" * 200 + "\n") * 2000)
    check("a record older than the tail window reads as unknown, not as stale truth",
          runner.progress() is None)

    # A finished job's clock must STOP — "finished in 15m" that keeps climbing all
    # afternoon is the header lying about what this host is doing.
    runner._job = {"proc": Dead(), "command": "distill", "argv": [],
                   "started": time.time() - 120.0, "logfile": str(logfile)}
    first = runner.status()
    time.sleep(0.05)
    second = runner.status()
    check("a finished job's elapsed time freezes at its exit, and stays frozen",
          first["elapsed_s"] == second["elapsed_s"] == 120
          and first["ended"] == second["ended"])

    runner._job["proc"] = type("Alive", (), {"poll": lambda self: None})()
    runner._job.pop("ended")
    check("a RUNNING job's clock still ticks", runner.status()["elapsed_s"] >= 120
          and runner.status()["ended"] is None)

    runner._job = None
    check("no job at all → no progress", runner.progress() is None)


def build_side():
    """pack._distil_progress: a clean-room build owns the bar; the distiller's chunk
    counts fold INTO it instead of overwriting it."""
    import contextlib
    import io
    from knowledgehost import pack as P

    seen = []
    prog = P._distil_progress(lambda phase, **kw: seen.append({"phase": phase, **kw}))
    prog.total = 8430
    prog.tick("/corpus/big.pdf")
    r = seen[-1]
    check("the build keeps step/steps (its own 6 phases)",
          r["phase"] == "distill" and r["step"] == 2 and r["steps"] == len(P._BUILD_STEPS))
    check("the distiller's total rides along as chunk_steps, not as the bar",
          r["chunk_steps"] == 8430 and r["chunks"] == 1 and r["doc"] == "big.pdf")

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        q = P._distil_progress(None)
        q.total = 10
        q.tick("/corpus/big.pdf")
        q.finish()
    check("a plain pack build (no reporter) stays off the channel, as before",
          "OPS_PROGRESS" not in buf.getvalue())


def autopilot_side():
    """next_up: what runs after this — from the same state the loop selects on."""
    tmp = tempfile.mkdtemp()
    cfg = {"control_dir": tmp, "_config_path": None}
    plan = {"enabled": True, "steps": [
        {"command": "distill", "args": {"bundle": "vinkona"}, "label": "distil drops",
         "min_interval_s": 0, "enabled": True},
        {"command": "distill", "label": "distil the rest", "min_interval_s": 0,
         "enabled": True},
        {"command": "link", "label": "link", "min_interval_s": 3600, "enabled": True},
    ]}
    AP.save_plan(cfg, plan)
    ap = AP.Autopilot(cfg, ops=None)

    n = ap.next_up()
    check("with nothing run yet, the highest-priority step is next and due now",
          n["label"] == "distil drops" and n["due_in_s"] == 0)

    now = time.time()
    ap._hold_until[AP.step_key(plan["steps"][0])] = now + 600
    n = ap.next_up()
    check("a step standing aside (it found no work) yields to the next one",
          n["label"] == "distil the rest")

    ap._hold_until[AP.step_key(plan["steps"][1])] = now + 600
    ap._last_run[AP.step_key(plan["steps"][2])] = now      # link just ran → 1h to go
    n = ap.next_up()
    check("when nothing is due, the SOONEST step is named with its wait",
          n["label"] in ("distil drops", "distil the rest") and 500 < n["due_in_s"] <= 600)
    check("…and it says why it isn't running", "no work" in n["reason"])

    st = ap.status()
    check("the panel gets 'next' alongside the live state", st["next"]["label"] == n["label"])

    AP.save_plan(cfg, {**plan, "enabled": False})
    check("automation off → no next step is claimed", ap.next_up() is None
          and AP.Autopilot(cfg, ops=None).status()["next"] is None)


# ── the panel side: the same record, rendered ───────────────────────────────
# viewer_js_test proves the script PARSES; this proves these functions produce
# the right text, by running the real source through node with a stub DOM.
_STUBS = """
const esc = t => String(t == null ? '' : t).replace(/&/g,'&amp;').replace(/</g,'&lt;')
  .replace(/>/g,'&gt;').replace(/"/g,'&quot;');
const fmt = n => (n || 0).toLocaleString('en-US');
const fmtCompact = v => String(v);
let EL = { className: '', innerHTML: '' };
const $ = s => EL;
"""


def _extract(js, name):
    """One top-level `function name(...) {…}` (they close with a brace in column 0)."""
    i = js.index("\nfunction " + name + "(")
    j = js.index("\n}\n", i)
    return js[i:j + 3]


def panel_side():
    import shutil
    import subprocess
    from knowledgehost.viewer import INDEX_HTML
    node = shutil.which("node")
    if not node:
        print("  SKIPPED (node not installed) — the render check needs it")
        return
    js = INDEX_HTML.split("<script>", 1)[1].rsplit("</script>", 1)[0]
    k = js.index("const PHASE_TABLE")
    src = _STUBS + js[k:js.index("};", k) + 2] + "\n"
    for fn in ("fmtDur", "progPct", "pbar", "chunkTotal", "progCount", "renderJobBar",
               "statCell", "addedCells", "nextUpLine", "opsProgressCard", "opsProgressBar"):
        src += _extract(js, fn) + "\n"

    RUN = ('{"running":true,"command":"distill","elapsed_s":930,"progress":'
           '{"phase":"distil","step":1240,"steps":8430,"chunks":1240,"left":7190,"doc":"big.pdf",'
           '"doc_step":12,"doc_steps":300,"docs_left":214,"rate_min":41.5,"eta_s":10400,'
           '"added":{"concepts":900,"cards":42}}}')
    src += f"""
const RUN = {RUN};
const out = {{}};
renderJobBar(RUN); out.bar = EL.innerHTML; out.barClass = EL.className;
EL = {{ className: '', innerHTML: '' }};
renderJobBar(null); out.idleClass = EL.className;
EL = {{ className: '', innerHTML: '' }};
renderJobBar({{running:false,command:"distill",elapsed_s:12,exit_code:1}});
out.doneBar = EL.innerHTML;
out.card = opsProgressCard(RUN.progress, {{running:true,command:"distill",elapsed_s:930}},
  {{enabled:true, last_reason:"ran distill",
    next:{{label:"link",command:"link",due_in_s:900,reason:"waiting for its interval"}}}});
out.idleCard = opsProgressCard(null, {{running:false,command:"distill",exit_code:0}},
  {{enabled:false}});
out.noTotal = opsProgressCard({{phase:"distil",step:77}},
  {{running:true,command:"distill",elapsed_s:10}}, null);
out.evil = opsProgressCard({{phase:"distil",step:1,doc:"<img src=x onerror=alert(1)>"}},
  {{running:true,command:"distill",elapsed_s:1}}, null);
// a clean-room BUILD: the bar counts phases, the chunk total rides alongside
const BUILD = {{phase:"distill",step:2,steps:6,chunks:1240,chunk_steps:8430,doc:"big.pdf"}};
out.buildCard = opsProgressCard(BUILD, {{running:true,command:"collect",elapsed_s:60}}, null);
out.buildBar = opsProgressBar(BUILD, {{running:true}});
console.log(JSON.stringify(out));
"""
    r = subprocess.run([node, "-e", src], capture_output=True, text=True)
    if r.returncode != 0:
        check("the panel script runs under node", False)
        print(r.stderr.strip()[:800])
        return
    o = json.loads(r.stdout)

    check("the header strip names the job and its progress on every tab",
          "▶ distill" in o["bar"] and "1,240 / 8,430 chunks" in o["bar"]
          and "15%" in o["bar"] and o["barClass"] == "on")
    check("the header strip says which document and how long is left",
          "in big.pdf" in o["bar"] and "2h 53m left" in o["bar"]
          and "15m 30s elapsed" in o["bar"])
    check("no job → the strip is hidden, not blank-but-present", o["idleClass"] == "")
    check("a failed job stays visible with its exit code",
          "✗ distill" in o["doneBar"] and "exit 1" in o["doneBar"])

    check("the Operations card shows the queue, the document and the ETA",
          "1,240 / 8,430" in o["card"] and "big.pdf 12/300" in o["card"]
          and "documents left" in o["card"] and "214" in o["card"]
          and "~2h 53m" in o["card"])
    check("what the pass has produced is shown as it accrues",
          "concepts" in o["card"] and "900" in o["card"])
    check("what happens NEXT is on the card when automation is on",
          "next up: <b>link</b> in 15m 00s" in o["card"])
    check("with automation off the card says so (not silence)",
          "Automation is <b>off</b>" in o["idleCard"]
          and "Nothing running" in o["idleCard"])
    check("an unsurveyable queue shows a count and NO percentage",
          "77" in o["noTotal"] and "%" not in o["noTotal"] and "chunks done" in o["noTotal"])
    check("a document name is escaped, never injected into the panel",
          "<img" not in o["evil"] and "&lt;img" in o["evil"])
    check("inside a clean-room build the bar stays in PHASE space (2 of 6)…",
          "2/6" in o["buildBar"] and "33%" in o["buildCard"])
    check("…and the chunk counts ride underneath it, not over it",
          "1,240 / 8,430 chunk(s), in big.pdf" in o["buildBar"]
          and "1,240 / 8,430" in o["buildCard"])


def main():
    print("survey (store.distill_queue):")
    cfg, A, B = survey_side()
    print("\nprogress (distill.DistillProgress):")
    progress_side(cfg, A, B)
    print("\nrunner (ops.OpsRunner.progress):")
    runner_side()
    print("\nbuild (pack._distil_progress):")
    build_side()
    print("\nnext up (autopilot.next_up):")
    autopilot_side()
    print("\npanel (the same record, rendered):")
    panel_side()
    print()
    if FAIL:
        print(f"{FAIL} FAILURE(S)")
        return 1
    print(f"ops_progress: ALL OK ({PASS} checks)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
