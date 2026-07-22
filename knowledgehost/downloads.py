"""The download lane: model pulls run OUTSIDE the single ops slot.

A transfer needs no GPU, so it must never queue behind — or block — a
distill; that single-slot contention is exactly why a Pull click could
vanish without a trace ("a job is already running", flashed for 2.5s).
Here every download is a visible row with its own state and controls.

One child at a time actually transfers — the broker's lease file is one per
rule, and two concurrent pulls would revoke each other's lease on first
close (a refcounting lease can lift this later).  The rest wait in a
visible QUEUE.  Pause = SIGTERM, partial files stay and Range-resume;
Resume = a fresh pull child (the manifest remembers a quant pick's include
glob); Discard = stop + delete the incomplete store folder — a complete
model is never deletable from here."""
from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from .amiga_net import pull as pull_mod


class Downloads:
    def __init__(self, root: Path, config_path: str = ""):
        self.root = Path(root)
        self.config_path = config_path
        self.logdir = self.root / "var" / "log" / "downloads"
        self._live: dict[str, dict] = {}      # model -> {proc, started, include}
        self._queue: list[dict] = []
        self._done: dict[str, dict] = {}      # model -> {exit, at}

    # ── internals ────────────────────────────────────────────────────────────

    def _logfile(self, model: str) -> Path:
        return self.logdir / (model.replace("/", "--") + ".log")

    def _spawn(self, model: str, include: str, revision: str) -> None:
        self.logdir.mkdir(parents=True, exist_ok=True)
        lf = open(self._logfile(model), "wb", buffering=0)   # fresh attempt, fresh log
        cmd = [sys.executable, "-m", "knowledgehost", "pull", "--model", model]
        if revision and revision != "main":
            cmd += ["--revision", revision]
        if include:
            cmd += ["--include", include]
        if self.config_path:
            cmd += ["-c", self.config_path]
        try:
            proc = subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT,
                                    stdin=subprocess.DEVNULL, cwd=str(self.root),
                                    start_new_session=True,
                                    env={**os.environ, "PYTHONUNBUFFERED": "1"})
        finally:
            lf.close()                        # the child holds its own copy —
        self._live[model] = {"proc": proc,    # keeping ours leaks one fd per pull
                             "started": time.time(),
                             "include": include, "revision": revision}
        self._done.pop(model, None)

    def _reap(self) -> None:
        for model, j in list(self._live.items()):
            rc = j["proc"].poll()
            if rc is not None:
                self._done[model] = {"exit": rc, "at": time.time()}
                self._live.pop(model)
        if not self._live and self._queue:
            nxt = self._queue.pop(0)
            self._spawn(nxt["model"], nxt["include"], nxt["revision"])

    def _last_line(self, model: str) -> str:
        try:
            data = self._logfile(model).read_bytes()[-2048:]
            lines = [ln for ln in data.decode("utf-8", "replace").splitlines()
                     if ln.strip()]
            return lines[-1][-220:] if lines else ""
        except OSError:
            return ""

    # ── the API the endpoints use ────────────────────────────────────────────

    def start(self, model: str, include: str = "", revision: str = "main") -> dict:
        self._reap()
        model = model.strip()
        if not model:
            return {"ok": False, "error": "model required"}
        if model in self._live:
            return {"ok": False, "error": f"{model} is already downloading"}
        if any(q["model"] == model for q in self._queue):
            return {"ok": False, "error": f"{model} is already queued"}
        if not include:                       # a resume repeats the quant pick
            p = pull_mod.progress(pull_mod.store_dir(self.root, model))
            include = (p or {}).get("include", "")
        if pull_mod.pulled(self.root, model):
            return {"ok": False, "error": f"{model} is already complete in the store"}
        if self._live:
            active = next(iter(self._live))
            self._queue.append({"model": model, "include": include,
                                "revision": revision})
            return {"ok": True, "state": "queued",
                    "note": f"queued behind {active} — one transfer at a time; "
                            "it starts automatically"}
        self._spawn(model, include, revision)
        return {"ok": True, "state": "pulling",
                "note": f"pull started -> models/{model.replace('/', '--')}/ "
                        "(watch the Downloads rows)"}

    def stop(self, model: str) -> dict:
        """Pause: SIGTERM the child (partials stay, Range-resume later) and
        drop any queued request for the model."""
        self._reap()
        self._queue = [q for q in self._queue if q["model"] != model]
        j = self._live.get(model)
        if j:
            try:
                os.killpg(os.getpgid(j["proc"].pid), signal.SIGTERM)
            except (ProcessLookupError, PermissionError):
                pass
            return {"ok": True, "note": "paused — partial files stay; "
                                        "Resume continues from where it stopped"}
        return {"ok": True, "note": "nothing running for it — dequeued if queued"}

    def discard(self, model: str) -> dict:
        """Stop and delete the INCOMPLETE store folder.  Refuses on a complete
        model — this is a download control, not a model manager."""
        self.stop(model)
        j = self._live.get(model)
        if j:                                  # give SIGTERM a moment to land
            for _ in range(20):
                if j["proc"].poll() is not None:
                    break
                time.sleep(0.1)
            self._reap()
        if pull_mod.pulled(self.root, model):
            return {"ok": False,
                    "error": "that model is COMPLETE — not deletable from here"}
        d = pull_mod.store_dir(self.root, model)
        if d.is_dir():
            import shutil
            shutil.rmtree(d, ignore_errors=True)
        self._done.pop(model, None)
        return {"ok": True, "note": f"discarded — {d} removed"}

    def status(self) -> list[dict]:
        """Every download the user should see: live transfers, the queue, and
        disk truth — any incomplete manifest in the store is a download,
        whether or not this process started it."""
        self._reap()
        rows: dict[str, dict] = {}
        for model, j in self._live.items():
            rows[model] = {"model": model, "state": "pulling",
                           "include": j["include"],
                           "elapsed_s": int(time.time() - j["started"]),
                           "detail": self._last_line(model)}
        for q in self._queue:
            rows[q["model"]] = {"model": q["model"], "state": "queued",
                                "include": q["include"]}
        store = self.root / "models"
        for mf in sorted(store.glob("*/.pull.json")) if store.is_dir() else []:
            p = pull_mod.progress(mf.parent)
            if not p or not p["total"] or p["have"] >= p["total"]:
                continue
            model = p["model"] or mf.parent.name.replace("--", "/", 1)
            row = rows.get(model, {"model": model, "state": "paused",
                                   "include": p["include"]})
            row.update(have_gb=round(p["have"] / 2**30, 2),
                       total_gb=round(p["total"] / 2**30, 2),
                       pct=round(100 * p["have"] / p["total"]),
                       files=f"{p['files_done']}/{p['files_total']}")
            if row["state"] == "paused":
                fin = self._done.get(model)
                # a positive exit code is a real failure; a negative one is
                # the SIGTERM WE sent — that's a pause, not an error
                if fin and (fin["exit"] or 0) > 0:
                    row["state"] = "error"
                    row["detail"] = self._last_line(model)
            rows[model] = row
        return sorted(rows.values(), key=lambda r: r["model"])
