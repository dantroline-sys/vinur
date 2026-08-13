#!/usr/bin/env python
"""Collect at structured-corpus scale: unit WINDOWS, real parallelism, and resume.

Dan's DRB collect ran six hours and died: a structured ingest stores one chunk per
VERSE, and the distiller paid (1 generic + 2 domain-card) LM calls per 25-word verse
— ~107k calls for a 4 MB text — with node embedding serialised under the global write
lock.  This battery pins the three fixes end to end against a REAL scripture collect
(live HTTP stubs for the LM and the embedder, real sqlite scratch):

  1. windows — consecutive verses of a chapter distil as ONE chunk-sized call
     (~30x fewer LM calls), with every member marked distilled when its window lands;
  2. parallelism — collect's distill keeps distill_parallel requests in flight
     (the stub proves overlapping calls), embeds precomputed OFF the write lock;
  3. resume — a build killed mid-distill keeps its scratch, and re-running the SAME
     collect pays only for the windows still missing, then finishes the file.

    python tests/collect_resume_test.py
"""
import json
import os
import shutil
import sys
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from knowledgehost.config import load_config          # noqa: E402
from knowledgehost import distill as D                 # noqa: E402
from knowledgehost import pack as P                    # noqa: E402

PASS = FAIL = 0


def check(name, cond):
    global PASS, FAIL
    if cond:
        PASS += 1; print(f"  ok  {name}")
    else:
        FAIL += 1; print(f"FAIL  {name}")


# ── live HTTP stubs ──────────────────────────────────────────────────────────
class _StubState:
    def __init__(self):
        self.lock = threading.Lock()
        self.calls = []           # (kind, served_ok) per chat call, in arrival order
        self.in_flight = 0
        self.high_water = 0
        self.die_after = None     # poison: chat calls beyond this answer 500

    def served(self, kind):
        """Successfully answered calls of one kind (poisoned 500s don't count —
        they are retries, not work done)."""
        with self.lock:
            return sum(1 for k, ok in self.calls if k == kind and ok)


def _classify(body: dict) -> str:
    msgs = body.get("messages") or []
    sys_txt = str(msgs[0].get("content", "")) if msgs else ""
    if msgs and str(msgs[-1].get("content")) == "ok":
        return "warmup"
    if "ONE structured knowledge card" in sys_txt:
        return "typed"
    if "concept" in sys_txt.lower():
        return "generic"
    return "other"


def _mk_handler(state: _StubState):
    GENERIC = json.dumps({
        "concepts": [{"label": "divine mercy", "kind": "concept",
                      "summary": "The passage describes mercy shown to the people.",
                      "questions": ["What does the passage say about mercy?"]}],
        "relations": [], "procedures": [], "criteria": []})

    class H(BaseHTTPRequestHandler):
        def _send(self, code, payload: bytes):
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def do_GET(self):
            if self.path.startswith("/v1/models"):
                self._send(200, json.dumps({"data": [{"id": "stub-model"}]}).encode())
            else:
                self._send(200, b"{}")

        def do_POST(self):
            n = int(self.headers.get("Content-Length") or 0)
            try:
                body = json.loads(self.rfile.read(n) or b"{}")
            except ValueError:
                body = {}
            if self.path.startswith("/v1/embeddings"):
                texts = body.get("input") or []
                if isinstance(texts, str):
                    texts = [texts]
                data = [{"embedding": [0.1 + 0.01 * (i % 7)] * 8, "index": i}
                        for i in range(len(texts))]
                self._send(200, json.dumps({"data": data}).encode())
                return
            kind = _classify(body)
            with state.lock:
                total = len(state.calls) + 1
                poisoned = state.die_after is not None and total > state.die_after
                state.calls.append((kind, not poisoned))
                state.in_flight += 1
                state.high_water = max(state.high_water, state.in_flight)
            try:
                time.sleep(0.12)                       # force real request overlap
                if poisoned:
                    self._send(500, b'{"error": "stub poisoned"}')
                    return
                content = GENERIC if kind != "typed" else "{}"
                self._send(200, json.dumps({"choices": [{"message": {
                    "content": content}}]}).encode())
            finally:
                with state.lock:
                    state.in_flight -= 1

        def log_message(self, *_):
            pass

    return H


def _start(state):
    srv = ThreadingHTTPServer(("127.0.0.1", 0), _mk_handler(state))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, f"http://127.0.0.1:{srv.server_address[1]}"


# ── fixture: 3 chapters × 24 verses, KJV line format ─────────────────────────
def _scripture() -> str:
    words = ("the LORD is my strength and my song and he is become my salvation "
             "for his mercy endureth and his truth is everlasting unto all").split()
    lines = []
    for ch in (1, 2, 3):
        for v in range(1, 25):
            k = (ch * 7 + v) % len(words)
            text = " ".join(words[k:] + words[:k])[:120]
            lines.append(f"Psalms {ch}:{v} {text.capitalize()}.")
    return "\n".join(lines) + "\n"


def windows_unit_tests():
    cfg = {"distill_unit_window_tokens": 100}

    def unit(i, ch, v, tok=20):
        return {"id": f"u{i}", "source_type": "scripture", "path_or_url": "/b.txt",
                "section": f"bible:Ps.{ch}.{v}", "text": f"verse {ch}:{v} words",
                "tokens": tok, "title": "B"}

    verses = [unit(i, 1, i + 1) for i in range(8)]
    out = list(D._windowed(iter(verses), cfg))
    check("verses group into token-capped windows",
          len(out) == 2 and [len(w["_members"]) for w in out] == [5, 3])
    check("a window's section is the canonical RANGE",
          out[0]["section"] == "bible:Ps.1.1-5" and out[1]["section"] == "bible:Ps.1.6-8")
    check("each unit rides in citation-tagged form",
          "[Ps.1.2] verse 1:2 words" in out[0]["text"])
    check("the window's id is its first member's (provenance stays a real chunk id)",
          out[0]["id"] == "u0" and out[0]["_members"][0] == "u0")

    mixed = [unit(0, 1, 23), unit(1, 1, 24), unit(2, 2, 1), unit(3, 2, 2)]
    out = list(D._windowed(iter(mixed), cfg))
    check("a chapter boundary breaks the window",
          [w["section"] for w in out] == ["bible:Ps.1.23-24", "bible:Ps.2.1-2"])

    prose = {"id": "p1", "source_type": "pdf", "path_or_url": "/d.pdf",
             "section": "Intro", "text": "ordinary prose", "tokens": 50}
    out = list(D._windowed(iter([unit(0, 1, 1), prose, unit(1, 1, 2)]), cfg))
    check("prose passes through untouched and splits the stream",
          out[1] is prose and "_members" not in out[0] and "_members" not in out[2])
    check("a lone unit is passed through UNCHANGED (no synthetic wrapper)",
          out[0] == unit(0, 1, 1))

    out = list(D._windowed(iter(verses), {"distill_unit_window_tokens": 0}))
    check("window tokens 0 disables grouping entirely", len(out) == 8)

    legal = [{"id": f"l{i}", "source_type": "legal", "path_or_url": "/t17.txt",
              "section": f"usc:17/{100 + i}", "text": "x" * 40, "tokens": 10}
             for i in range(3)]
    out = list(D._windowed(iter(legal), cfg))
    check("legal sections group by title with a range key",
          len(out) == 1 and out[0]["section"] == "usc:17/100-102"
          and len(out[0]["_members"]) == 3)


def collect_e2e():
    state = _StubState()
    srv, url = _start(state)
    tmp = tempfile.mkdtemp(prefix="kb-collect-resume-")
    doc = os.path.join(tmp, "drb.txt")
    with open(doc, "w", encoding="utf-8") as f:
        f.write(_scripture())

    cfg = load_config(None)
    cfg.update({
        "backend": "sqlite",
        "db_path": os.path.join(tmp, "master-index.db"),
        "kb_path": os.path.join(tmp, "master-kb.db"),
        "pack_build_dir": os.path.join(tmp, "build"),
        "sources": [tmp], "embed_url": url, "embed_dim": 8,
        "distill_url": url, "distill_urls": [url], "extract_urls": [],
        "distill_model": "stub-model", "distill_timeout_s": 15,
        "distill_parallel": 2, "ann_search": False,
        "distill_unit_window_tokens": 120,
        "auto_reconcile": False,
    })
    target = os.path.join(tmp, "out", "bible.kdb")
    answers = {"kind": "structured"}

    # ground truth: how many windows should this fixture distil as?
    from knowledgehost import ingest as ingest_mod
    from knowledgehost.store import make_store
    pre = dict(cfg, db_path=os.path.join(tmp, "pre-index.db"))
    pstore = make_store(pre)
    prof = ingest_mod.confirm_profile(pre, doc, answers)
    n_units = ingest_mod.ingest_file(pstore, None, pre, doc, profile=prof)
    wins = [w for w in D._windowed(
        (c for c in pstore.iter_chunks()), cfg) if w.get("_members")]
    n_windows = len(wins)
    pstore.close()
    check(f"fixture sanity: 72 verses ingest as units ({n_units})", n_units == 72)
    check(f"…and window into far fewer distillation calls ({n_windows})",
          4 <= n_windows <= 20)

    # ── run 1: poisoned mid-distill — the six-hour-timeout shape ─────────────
    state.die_after = 8          # warmup + a couple of windows, then the LM "dies"
    died = None
    try:
        P.add_to_collection(cfg, doc, target, "bible", license_override="CC-BY-4.0",
                            answers=answers, log_fn=lambda *a: None)
    except D.BackendUnavailable as e:
        died = e
    check("a mid-distill endpoint death aborts the build (resumable), not silently",
          died is not None)
    build = Path(cfg["pack_build_dir"]) / "collect-bible"
    check("the scratch SURVIVES the death (kb + index still on disk)",
          (build / "kb.db").is_file() and (build / "index.db").is_file())
    import sqlite3
    con = sqlite3.connect(str(build / "kb.db"))
    done1 = con.execute("SELECT COUNT(*) FROM distilled_chunks").fetchone()[0]
    con.close()
    check(f"run 1 landed part of the work before dying ({done1} units)",
          0 < done1 < n_units)
    check("the target file does not exist yet (nothing exported half-done)",
          not os.path.exists(target))
    run1_generic = state.served("generic")

    # ── run 2: the LM is back — the SAME collect resumes and finishes ────────
    state.die_after = None
    mark = len(state.calls)
    res = P.add_to_collection(cfg, doc, target, "bible", license_override="CC-BY-4.0",
                              answers=answers, log_fn=lambda *a: None)
    run2_generic = state.served("generic") - run1_generic
    check("the resumed collect completes and creates the file",
          res["ok"] and res["created"] and res.get("complete") and os.path.exists(target))
    check("resume pays ONLY for the remainder, not the whole corpus again",
          0 < run2_generic < n_windows)
    check("across both runs the SERVED generic passes ≈ the window count, "
          "NOT one per verse",
          run1_generic + run2_generic <= n_windows + 4
          and run1_generic + run2_generic < n_units)
    check("(bookkeeping) run 2 issued new calls", len(state.calls) > mark)
    check("a finished build cleans its scratch away", not build.exists())

    con = sqlite3.connect(target)
    nodes = con.execute("SELECT COUNT(*) FROM nodes").fetchone()[0]
    srcs = [r[0] for r in con.execute("SELECT bundle FROM source_registry")]
    con.close()
    check("the .kdb holds distilled knowledge under the one bundle",
          nodes > 0 and srcs and all(b == "bible" for b in srcs))

    check(f"distill kept {cfg['distill_parallel']} requests in flight "
          f"(high water {state.high_water})", state.high_water >= 2)

    # ── idempotence: re-adding the same finished doc is answered from the FILE ─
    mark = len(state.calls)
    res2 = P.add_to_collection(cfg, doc, target, "bible", license_override="CC-BY-4.0",
                               answers=answers, log_fn=lambda *a: None)
    check("re-collecting a finished, unchanged document costs NO LM calls at all "
          "(the manifest's doc hash answers it)",
          res2["ok"] and res2.get("skipped") and len(state.calls) == mark
          and not res2["added"])
    # …and a CHANGED file goes the full route again
    with open(doc, "a", encoding="utf-8") as f:
        f.write("Psalms 3:25 A new verse appended to the scroll.\n")
    res3 = P.add_to_collection(cfg, doc, target, "bible", license_override="CC-BY-4.0",
                               answers=answers, log_fn=lambda *a: None)
    check("a changed document is NOT skipped (hash mismatch → rebuild)",
          res3["ok"] and not res3.get("skipped") and len(state.calls) > mark)

    srv.shutdown()
    shutil.rmtree(tmp, ignore_errors=True)


def furniture_completeness():
    """A document whose furniture (a copyright page) is zone-skipped must still reach
    complete=True — zone-skips are PROCESSED work, not missing work.  Before the fix
    such a doc was 'incomplete' forever: doc_hashes never recorded, scratch never
    cleaned, and the unchanged-re-collect fast path could never engage."""
    state = _StubState()
    srv, url = _start(state)
    tmp = tempfile.mkdtemp(prefix="kb-collect-furn-")
    doc = os.path.join(tmp, "essay.md")
    boiler = ("Copyright © 2020 Example House. All rights reserved.\n"
              "No part of this publication may be reproduced. ISBN 978-1-2345-6789-0.\n")
    body = ("Alpine glaciers store winter snowfall and release it as summer meltwater. "
            "Their retreat shifts the timing of river flow across the basin. ") * 12
    with open(doc, "w", encoding="utf-8") as f:
        f.write("# Copyright\n\n" + boiler + "\n# Glaciers\n\n" + body + "\n")

    from knowledgehost import zones
    cfg = load_config(None)
    cfg.update({
        "backend": "sqlite",
        "db_path": os.path.join(tmp, "master-index.db"),
        "kb_path": os.path.join(tmp, "master-kb.db"),
        "pack_build_dir": os.path.join(tmp, "build"),
        "sources": [tmp], "embed_url": url, "embed_dim": 8,
        "distill_url": url, "distill_urls": [url], "extract_urls": [],
        "distill_model": "stub-model", "distill_timeout_s": 15,
        "distill_parallel": 2, "ann_search": False,
        "auto_reconcile": False,
    })

    # fixture sanity: the ingested doc really contains a furniture chunk
    from knowledgehost import ingest as ingest_mod
    from knowledgehost.store import make_store
    pre = dict(cfg, db_path=os.path.join(tmp, "pre-index.db"))
    pstore = make_store(pre)
    ingest_mod.ingest_file(pstore, None, pre, doc)
    zs = [zones.classify(c.get("section") or "", c.get("text") or "")
          for c in pstore.iter_chunks()]
    pstore.close()
    check(f"fixture sanity: the copyright page chunks as furniture ({zs})",
          "boilerplate" in zs and "body" in zs)

    target = os.path.join(tmp, "out", "essay.kdb")
    res = P.add_to_collection(cfg, doc, target, "essay", license_override="CC-BY-4.0",
                              log_fn=lambda *a: None)
    dz = ((res.get("stats") or {}).get("distill") or {}).get("skipped_zone", 0)
    check(f"the run zone-skipped the furniture ({dz})", dz >= 1)
    check("…and still reports COMPLETE (skips are processed, not missing)",
          res["ok"] and res.get("complete"))
    build = Path(cfg["pack_build_dir"]) / "collect-essay"
    check("a complete build cleans its scratch away", not build.exists())

    mark = len(state.calls)
    res2 = P.add_to_collection(cfg, doc, target, "essay", license_override="CC-BY-4.0",
                               log_fn=lambda *a: None)
    check("re-collecting it is answered from the manifest hash — zero LM calls",
          res2.get("skipped") and len(state.calls) == mark)

    srv.shutdown()
    shutil.rmtree(tmp, ignore_errors=True)


def pipeline_unit_tests():
    """The two-tier pipeline's own mechanics (fake LMs, no HTTP): a clean run lands
    everything; an empty queue returns zeros (it used to raise a misleading 'all fast
    extractor endpoints failed'); a feeder death ABORTS the run loudly (it used to be
    swallowed — the pipeline drained what was queued and reported success with most of
    the corpus unfed); domain lenses run on the VERIFY tier, never in the writer."""
    from knowledgehost.kb import KB

    tmp = tempfile.mkdtemp(prefix="kb-pipeline-")
    cfg = load_config(None)
    cfg.update({"kb_path": os.path.join(tmp, "kb.db"), "control_dir": tmp,
                "distill_unit_window_tokens": 100, "verify_batch": 3,
                "ingest_log_every": 0, "ann_search": False})

    GEN = ([{"label": "divine mercy", "kind": "concept",
             "summary": "The passage describes mercy shown to the people.",
             "questions": ["What does the passage say about mercy?"]}], [], [], [])
    TYPED = {
        "theme": {"title": "God's mercy", "concept": "divine mercy", "theme": "mercy",
                  "statement": "Mercy endures through the psalm",
                  "support": "his mercy endureth"},
        "parallel": {"title": "Mercy echo", "concept": "Ps 1:1", "relationship": "echoes",
                     "parallels": ["Ps 100:5"], "evidence": "his mercy endureth"},
    }

    class FakeLM:
        def __init__(self, url):
            self.url = url
            self.typed_calls = 0

        def extract(self, ch, reg=None):
            return GEN

        def extract_narrative(self, ch):
            return None

        def extract_typed(self, ch, card_type):
            self.typed_calls += 1
            return TYPED.get(card_type, {})

    class FakeEmbedder:
        def embed_many(self, texts, kind="document"):
            return [[0.1] * 8 for _ in texts]

        def embed_one(self, text, kind="document"):
            return [0.1] * 8

    class FakeStore:
        def __init__(self, chunks):
            self._chunks = chunks

        def iter_chunks(self):
            yield from self._chunks

        def count(self):
            return len(self._chunks)

    def unit(i):
        return {"id": f"v{i}", "source_type": "scripture", "path_or_url": "/x/ps.txt",
                "title": "PS", "section": f"bible:Ps.1.{i + 1}", "tokens": 20,
                "text": f"verse {i + 1} for his mercy endureth and his truth is everlasting"}

    real_vb = D.verify_mod.verify_batch
    D.verify_mod.verify_batch = lambda vlm, drafts, cfg_: [
        (d["concepts"], d["relations"], d["procedures"],
         {"rejected": 0, "adjusted": 0, "failed": 0}) for d in drafts]
    try:
        # ── clean run: everything lands; lenses on the verify tier only ──────
        ex, vf = FakeLM("stub://fast"), FakeLM("stub://big")
        kb = KB({"kb_path": cfg["kb_path"]})
        chunks = [unit(i) for i in range(8)]
        n_windows = len(list(D._windowed(iter([dict(c) for c in chunks]), cfg)))
        res = D._distill_pipeline(FakeStore(chunks), kb, [ex], [vf],
                                  FakeEmbedder(), cfg)
        check(f"pipeline lands every window ({res['chunks']} of {n_windows})",
              res["chunks"] == n_windows and res["failed"] == 0)
        ncards = kb.db.execute("SELECT COUNT(*) FROM procedure_cards "
                               "WHERE card_type IN ('theme','parallel')").fetchone()[0]
        check(f"domain cards land through the pipeline ({ncards})", ncards >= 2)
        check(f"lenses ran on the VERIFY tier exactly once per window "
              f"(big {vf.typed_calls}, fast {ex.typed_calls})",
              vf.typed_calls == n_windows * 2 and ex.typed_calls == 0)

        # ── empty queue: zeros, not a fake endpoint failure ──────────────────
        res2 = D._distill_pipeline(FakeStore(chunks), kb, [FakeLM("stub://fast")],
                                   [FakeLM("stub://big")], FakeEmbedder(), cfg)
        check("an already-done corpus returns zeros instead of raising",
              res2["chunks"] == 0 and res2["skipped"] == n_windows * 0 + len(chunks))
        kb.close()

        # ── feeder death: loud, resumable — never a hollow success ───────────
        class DyingStore(FakeStore):
            def iter_chunks(self):
                yield from self._chunks[:2]
                raise RuntimeError("index backend fell over")

        cfg2 = dict(cfg, kb_path=os.path.join(tmp, "kb2.db"))
        kb2 = KB({"kb_path": cfg2["kb_path"]})
        raised = None
        try:
            D._distill_pipeline(DyingStore([unit(i) for i in range(8)]), kb2, [FakeLM("f")],
                                [FakeLM("b")], FakeEmbedder(), cfg2)
        except D.BackendUnavailable as e:
            raised = str(e)
        check("a feeder death raises (resumable), never a silent partial 'success'",
              raised is not None and "feeder died" in raised
              and "index backend fell over" in raised)
        kb2.close()
    finally:
        D.verify_mod.verify_batch = real_vb
    shutil.rmtree(tmp, ignore_errors=True)


def main():
    print("windows (distill._windowed):")
    windows_unit_tests()
    print("\ncollect end-to-end (live stubs — poison, resume, parallel):")
    collect_e2e()
    print("\ncompleteness with furniture zones:")
    furniture_completeness()
    print("\ntwo-tier pipeline mechanics (fake LMs):")
    pipeline_unit_tests()
    print()
    if FAIL:
        print(f"{FAIL} FAILURE(S)")
        return 1
    print(f"collect_resume: ALL OK ({PASS} checks)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
