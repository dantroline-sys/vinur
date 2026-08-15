"""Single-slot operations runner for the web control panel.

The maintenance verbs (ingest/distill/link/refine/…) are launched as **subprocesses** of
the long-lived server — the very same `python3 -m knowledgehost <verb>` you'd run by hand.
The running job is tracked by holding its live `Popen` in memory: "is a job running?" is
answered by the **kernel** (`proc.poll()` is None while alive, the exit code once done), so
there is no lock file to go stale when something crashes mid-run.  One job at a time (they
contend on the GPU lease and the KB); a second request is refused while one is live.

Safety: only the allow-listed verbs below can be launched, and only with their typed
options — the UI sends `{limit: 20, fast: true}`, never a command string — so there is no
shell-injection surface (subprocess list form, no `shell=True`).
"""
from __future__ import annotations

import logging
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

log = logging.getLogger("knowledgehost.ops")

# verb -> {option: type}.  type ∈ int | float | bool | str | path | choice:<a,b>.
# The flag is --<option-with-dashes>.
COMMANDS: dict = {
    "ingest":     {"force": "bool", "wikipedia": "bool", "distill": "bool", "limit": "int"},
    "ingest-library": {"force": "bool"},   # index the search-only document library (library_sources)
    "rebuild-fts": {},                     # reindex FTS with the configured tokenizer (no re-parse)
    "distill":    {"limit": "int", "watch": "bool", "interval": "int", "bundle": "str"},
    "recard":     {"limit": "int", "bundle": "str",    # cards-only re-pass (see HELP)
                   "all_families": "bool", "before": "str", "since": "str"},
    # clean-room pack producer (VINUR-PACK-01 §3); encryption is CLI-only —
    # a passphrase must never ride argv where `ps` can read it
    "pack":       {"path": "path", "name": "str", "title": "str", "author": "str",
                   "pack_version": "str", "describe": "str", "license": "str",
                   "allow_unlicensed": "bool", "compress": "bool",
                   "keep_build": "bool", "force": "bool"},
    # clean-room ingest+distill of ONE document, ADDED to a shareable .kdb collection
    # under a named bundle (creates the file or merges into it — one bundle per file)
    "collect":    {"doc": "path", "to": "path", "bundle": "str", "license": "str",
                   "allow_unlicensed": "bool", "answers_file": "path"},
    # deterministic cross-reference graph over structured (scripture/legal) docs (no LM)
    "citations":  {},
    # align a Vulgate-numbered edition's Psalms onto the Hebrew frame of a reference (no LM)
    "psalms":     {"edition": "str", "reference": "path"},
    # rebuild the derived-reasoning layer + mine contradictions/sibling gaps (no LM)
    "derive":     {},
    # janitor: chunks holding text the corpus already has (no LM involved)
    "dedupe":     {"near": "bool", "threshold": "float", "apply": "bool", "bundle": "str"},
    # hub search: candidates sized + judged against this machine's memory
    "find":       {"query": "str", "limit": "int"},
    # model weights via the egress broker (policy-checked, audited, resumable)
    "pull":       {"model": "str", "revision": "str", "include": "str"},
    # legacy hub-cache snapshots -> the models/ store (no network, no re-download)
    "adopt":      {"model": "str"},
    # External-dataset bulk imports (KB-only; path defaults to <name>_path in config —
    # the option only overrides it).  Long-running; stream their progress to the log.
    "import-conceptnet": {"path": "path", "min_weight": "float", "all": "bool",
                          "exclude": "list"},
    "import-atomic":     {"path": "path", "min_count": "int", "limit": "int"},
    "import-glucose":    {"path": "path", "min_count": "int", "limit": "int"},
    "import-causenet":   {"path": "path", "min_sources": "int", "limit": "int"},
    # provenance-aware undo of one bulk import (the threshold-tuning loop:
    # import → inspect → unimport → adjust → re-import)
    "unimport":          {"dataset": "choice:conceptnet,atomic,glucose,causenet"},
    "migrate-vocab": {},   # one-shot pre-1.2 -> 1.2 neutral-vocabulary migration (idempotent)
    "link":       {"limit": "int", "fast": "bool", "top_k": "int"},
    "refine":     {"limit": "int", "force": "bool"},
    "adjudicate": {"limit": "int", "batch": "int", "fast": "bool",
                   "no_auto": "bool", "auto_only": "bool"},
    "reconcile":  {"limit": "int", "top_k": "int", "anchors": "choice:corpus,all"},
    "build-ann":  {},
    "embed-nodes": {"limit": "int"},
    "optimize":   {"vacuum": "bool"},
    "edge-audit": {"limit": "int", "apply": "bool"},
    "stats":      {},
    "split":      {"force": "bool"},    # export each bundle group to its <bundle>.kdb file
    # brains: absorb a shipped .kdb into the master / permanently remove one bundle.
    # (Runtime load/unload is NOT here — it's the instant /brain endpoint, no job needed.)
    "import-bundle": {"path": "path", "name": "str", "trust": "choice:low,keep"},
    "eject-bundle":  {"bundle": "str", "dry_run": "bool", "no_export": "bool"},
}

# Human-readable help for the panel (Operations + Prioritizer): what each verb does
# ("_") and what each option means.  This is what makes an args editor usable —
# a bare {} tells the operator nothing.
HELP: dict = {
    "ingest": {"_": "Parse new/changed documents into raw chunks",
               "force": "re-process every file, ignoring the seen-manifest",
               "wikipedia": "also ingest the configured Wikipedia ZIM",
               "distill": "distil right after ingesting (needs the big LM)",
               "limit": "max files this run"},
    "ingest-library": {"_": "Index the search-only document library",
                       "force": "re-index everything"},
    "rebuild-fts": {"_": "Rebuild the FTS index with the configured tokenizer"},
    "distill": {"_": "Distil raw chunks into cards/nodes/edges (the LM pass)",
                "bundle": "ONLY chunks from this brain/bundle — 'vinkona' = the "
                          "assistant's research drops, 'base' = untagged documents; "
                          "empty = everything. Two Prioritizer steps with different "
                          "bundles are tracked as different steps",
                "limit": "max chunks this run — bounded batches let higher-priority "
                         "steps preempt between runs",
                "watch": "keep running as a concurrent ingest adds chunks",
                "interval": "watch mode: seconds between passes"},
    "recard": {"_": "Cards-only re-pass over already-distilled chunks: harvest the "
                    "card families — procedures and criteria included — from chunks "
                    "stamped before the current sweep version.  Nodes are joined, "
                    "never re-created; relations untouched.  Dedup: an identical "
                    "re-offer corroborates by hash, a reworded one by same node+"
                    "type+title",
               "limit": "max chunks this run",
               "bundle": "ONLY chunks from this provenance bundle; empty = everything",
               "all_families": "re-ask EVERY family regardless of stamp age — the "
                               "truncation-recovery sweep (cards lost to a too-small "
                               "output budget while chunks were already marked done)",
               "before": "only chunks DISTILLED before this date/time (YYYY-MM-DD "
                         "or YYYY-MM-DDTHH:MM) — bound the recovery to what "
                         "predates the budget fix; the healthy tail is spared",
               "since": "RECOVERY mirror of before: re-open chunks DISTILLED "
                        "on/after this date/time REGARDLESS of stamp, to recover "
                        "cards truncated after a chunk was stamped current "
                        "(idempotent — the title gate folds already-carded chunks)"},
    "citations": {"_": "Build the cross-reference graph over structured scripture/legal "
                       "documents: one node per canonical unit (verse/section), a "
                       "'citation' edge per reference its text makes. Deterministic, no "
                       "LM, idempotent (safe to re-run — new documents just add edges)."},
    "psalms": {"_": "Line a Vulgate-numbered edition's Psalms (e.g. the Douay-Rheims) up "
                    "with a Hebrew-numbered reference (e.g. KJV) already ingested. The "
                    "psalm numbers and per-verse offsets differ (the Douay counts the Latin "
                    "titles as verses); the offset is RECOVERED from the two texts by "
                    "wording, not guessed — low-confidence psalms are left alone and listed. "
                    "Writes the Vulgate→Hebrew key aliases so the graph + parallel reading "
                    "converge the editions. Deterministic, no LM, idempotent.",
               "edition": "Vulgate edition id (default: every known one)",
               "reference": "path of the Hebrew reference edition (default: auto)"},
    "derive": {"_": "Rebuild the derived-reasoning layer (no LM, wipe-and-rebuild, "
                    "deterministic): taxonomic inheritance (a child inherits its "
                    "class's causal/functional relations) + causal sign composition "
                    "(A increases B, B suppresses C ⇒ A suppresses C), each written as "
                    "a QUARANTINED family='derived' status='proposed' edge carrying its "
                    "full parent chain — consumed only by kb_reason in permissive mode, "
                    "invisible to every other read path. Also mines contradictions "
                    "(opposite-polarity assertions → surface questions) and sibling-"
                    "completion gaps (→ knowledge gaps, feeding the research loop). "
                    "Safe to re-run; one DELETE reverses the whole layer."},
    "dedupe": {"_": "Janitor (no LM): find chunks holding text the corpus already "
                    "has. A chunk id is sha1(path+section+text), so the same text "
                    "arriving by another route — a research drop re-exported under "
                    "a new name, one document filed twice — would distil again. "
                    "EXACT duplicates are marked against the chunk that owns the "
                    "text and never distilled again; nothing is deleted, so search "
                    "still finds them either way. Distillation does this check "
                    "inline too (distill_dedupe); this sweep is for what's already "
                    "in the store",
               "near": "also look for near-duplicates (MinHash/Jaccard over word "
                       "shingles) — the same answer written twice in different words",
               "threshold": "near-duplicate similarity floor (default 0.9)",
               "apply": "near mode: actually mark them (default is report only — "
                        "'almost the same' can also mean 'a revision of')",
               "bundle": "ONLY chunks from this provenance bundle; empty = everything"},
    "find": {"_": "Search huggingface.co (through the egress broker) and judge each "
                  "hit against this machine: exact weight size from the hub's file "
                  "list, then fits / tight / too big for the detected VRAM (or "
                  "unified/system memory). GGUF repos expand into their individual "
                  "quantisation files. Rows are numbered — pull one by giving the "
                  "pull op that number as its model",
             "query": "search words, e.g. 'qwen3 32b fp8'",
             "limit": "max candidates to size up (default 8)"},
    "pull": {"_": "Download model weights through the egress broker into the local "
                  "model store (models/<Org--Name>/). The Search box below finds "
                  "candidates by name — sized, fit-judged for this machine, and "
                  "pullable in one click. Policy-checked against "
                  "egress.toml, audited to var/log/egress.jsonl, resumable "
                  "(aria2c-accelerated when installed, wget or a built-in stream "
                  "otherwise), sha256-verified against the hub's published digests. "
                  "Inference engines run OFFLINE and load the result from disk — "
                  "they never download, never hold the token, never phone home",
             "model": "the HF model id (org/Name), or a row number from the last find",
             "revision": "repo revision (default main)",
             "include": "only repo files matching this glob (a single GGUF quant); "
                        "find fills this in automatically for numbered rows"},
    "adopt": {"_": "Move models out of the legacy Hugging Face cache "
                   "(var/cache/huggingface) into the models/ store — the one obvious "
                   "folder, same layout a pull produces. No network, no re-download: "
                   "hardlinked when possible, copied otherwise; the cache copy is left "
                   "for you to delete once happy. Adopted models appear in the Serving "
                   "tab's pickers (the cache is no longer offered there)",
              "model": "one HF id (org/Name); empty = every complete cached snapshot"},
    "import-conceptnet": {"_": "Bulk-import the ConceptNet commonsense graph",
                          "path": "assertions.csv dump (default from config)",
                          "min_weight": "drop assertions weaker than this",
                          "all": "include weak/lexical relations",
                          "exclude": "comma-separated relations to skip"},
    "import-atomic": {"_": "Bulk-import the ATOMIC if-then graph",
                      "path": "dump file (default from config)",
                      "min_count": "drop rare events", "limit": "cap rows"},
    "import-glucose": {"_": "Bulk-import the GLUCOSE causal dataset",
                       "path": "dump file", "min_count": "drop rare rows",
                       "limit": "cap rows"},
    "import-causenet": {"_": "Bulk-import CauseNet cause→effect pairs",
                        "path": "dump file", "min_sources": "evidence floor",
                        "limit": "cap rows"},
    "unimport": {"_": "Provenance-aware undo of one bulk dataset import",
                 "dataset": "which import to remove"},
    "migrate-vocab": {"_": "One-shot pre-1.2 → 1.2 vocabulary migration"},
    "link": {"_": "Type structural edges between related cards (LM judged)",
             "limit": "max pairs this run", "fast": "use the fast 9B instead of the big LM",
             "top_k": "embedding neighbours considered per node"},
    "refine": {"_": "Rewrite cards against their own sources (grounded, in place)",
               "limit": "max cards this run", "force": "also re-refine already-refined cards"},
    "adjudicate": {"_": "Judge duplicate-node merge candidates",
                   "limit": "max pairs this run", "batch": "pairs per LM call",
                   "fast": "use the fast 9B", "no_auto": "LM judges everything",
                   "auto_only": "only the high-similarity auto-merges"},
    "reconcile": {"_": "Propose cross-source merges by embedding similarity",
                  "limit": "max clusters", "top_k": "neighbours per anchor",
                  "anchors": "corpus = card-bearing only; all = every node"},
    "build-ann": {"_": "Build the fast dense-search index over node vectors"},
    "embed-nodes": {"_": "Backfill embeddings for nodes that have none",
                    "limit": "max nodes this run"},
    "optimize": {"_": "One-time node table layout fix",
                 "vacuum": "reclaim disk afterwards"},
    "edge-audit": {"_": "Cull nonsense edges — sound-alike (labels look/sound alike but "
                        "mean unrelated things) + ungrounded over-links; LM-free, soft-retract",
                   "limit": "max edges this run", "apply": "retract flagged edges (default: report only)"},
    "stats": {"_": "Print corpus statistics"},
    "split": {"_": "Export each bundle to its own .kdb brain file",
              "force": "overwrite existing files"},
    "pack": {"_": "Build a SHAREABLE knowledge pack: clean-room ingest+distill of one "
                  "file/folder in a scratch kb (the master is never touched), "
                  "license-gated at export, manifest-stamped (authorship, licensing, "
                  "compatibility), written to packs/<slug>/<slug>-<version>.kdb with a "
                  "JSON sidecar.  Resumable: a failed build keeps its scratch.  "
                  "Encryption is CLI-only (PACK_PASSPHRASE env — never argv).",
             "path": "the document or folder to distil",
             "name": "pack slug (default: the input's name)",
             "title": "human title for the manifest",
             "author": "producer name for the manifest",
             "pack_version": "semver for the artifact + manifest (default 1.0.0)",
             "describe": "one-line description",
             "license": "SPDX id filled onto sources with NO detected license "
                        "(recorded as attested — never overrides a detected one)",
             "allow_unlicensed": "build anyway; the pack is stamped shareable:false "
                                 "(private use only; import warns)",
             "compress": "gzip the artifact (import auto-detects)",
             "keep_build": "keep the scratch workspace after success",
             "force": "replace an existing same-version artifact"},
    "collect": {"_": "ADD one document to a shareable .kdb collection under a named "
                     "bundle: clean-room ingest+distil in a scratch (the master is "
                     "never touched), then create the target file or MERGE into it if "
                     "it already exists (content-hash ids ⇒ idempotent; re-adding the "
                     "same doc is a no-op).  One bundle per file.  Build a share-file "
                     "up one document at a time; import it elsewhere with import-bundle.",
                "doc": "the document (or folder) to ingest + distil",
                "to": "the .kdb collection file to create or add to (plain .kdb)",
                "bundle": "the bundle name to file the knowledge under (one per file)",
                "license": "SPDX id attested onto sources with no detected license",
                "allow_unlicensed": "add anyway even if some sources have no license",
                "answers_file": "JSON file of structured-text confirmation answers "
                                "(scripture/legal) — the wizard writes it; a confirmed "
                                "profile ingests the doc unit-by-unit"},
    "import-bundle": {"_": "Absorb a shipped .kdb brain into the master",
                      "path": "the .kdb file on this box",
                      "name": "bundle name to import under (default: its manifest name)",
                      "trust": "low = cap the brain's trust (recommended for shipped "
                               "files); keep = trust its own values (your own files)"},
    "eject-bundle": {"_": "Export a brain to its .kdb, then remove it from the master",
                     "bundle": "which brain to eject",
                     "dry_run": "count what would go, delete nothing",
                     "no_export": "skip the safety export (not recommended)"},
}

# ── job result channel ────────────────────────────────────────────────────────
# A verb may print one final machine-readable line; the runner picks up the LAST
# one after the process exits.  The autopilot uses `did_work` to notice that a
# step found nothing to do and stand aside for lower-priority steps.
RESULT_PREFIX = "OPS_RESULT "


def emit_result(did_work: bool, **stats) -> None:
    """Called by the verbs (via __main__) at the end of a pass."""
    import json as _json
    print(RESULT_PREFIX + _json.dumps({"did_work": bool(did_work), **stats}),
          flush=True)


# ── live progress channel ─────────────────────────────────────────────────────
# A long verb may print these AS IT GOES (one per phase); the runner streams them
# into the ops log and the panel renders the LAST one as a progress bar.  Free-text
# log lines remain the human-readable detail underneath.
PROGRESS_PREFIX = "OPS_PROGRESS "
_PROGRESS_TAIL_BYTES = 65536      # how far back progress() looks for the last record


def emit_progress(phase: str, *, step: int | None = None, steps: int | None = None,
                  **extra) -> None:
    """Print a machine-readable progress line for the panel's bar.  `phase` names the
    stage; `step`/`steps` (1-based) drive the bar; `extra` carries display detail."""
    import json as _json
    rec: dict = {"phase": str(phase)}
    if step is not None:
        rec["step"] = int(step)
    if steps is not None:
        rec["steps"] = int(steps)
    rec.update(extra)
    print(PROGRESS_PREFIX + _json.dumps(rec, ensure_ascii=False), flush=True)


def _argv(command: str, args: dict) -> list:
    """Validate the typed options for `command` and render them to CLI flags.  Raises
    ValueError on an unknown verb/option/value — nothing unvalidated reaches the shell."""
    if command not in COMMANDS:
        raise ValueError(f"unknown command: {command}")
    spec = COMMANDS[command]
    out: list = []
    for key, val in (args or {}).items():
        if key not in spec:
            raise ValueError(f"{command}: unknown option {key!r}")
        typ = spec[key]
        flag = "--" + key.replace("_", "-")
        if typ == "bool":
            if val:
                out.append(flag)
        elif typ == "int":
            out += [flag, str(int(val))]               # int() rejects non-numeric
        elif typ.startswith("choice:"):
            allowed = typ.split(":", 1)[1].split(",")
            sv = str(val)
            if sv not in allowed:
                raise ValueError(f"{command}: {key} must be one of {allowed}")
            out += [flag, sv]
        elif typ == "str":
            sv = str(val).strip()
            # Constrained charset: it becomes a CLI value (list form, no shell), but keep
            # it to identifier-like tokens so it can never look like a flag or path.
            # ':' is allowed for the YYYY-MM-DDTHH:MM timestamps recard's before/since
            # advertise (HELP documented them, then this check rejected them).
            if sv and not all(ch.isalnum() or ch in "._-:" for ch in sv):
                raise ValueError(f"{command}: {key} must be alphanumeric (._-: allowed)")
            if sv:
                out += [flag, sv]
        elif typ == "float":
            out += [flag, str(float(val))]             # float() rejects non-numeric
        elif typ == "list":
            # comma-separated identifier tokens, passed as ONE CLI value
            toks = [t.strip() for t in str(val).split(",") if t.strip()]
            for t in toks:
                if not all(ch.isalnum() or ch in "._-" for ch in t):
                    raise ValueError(f"{command}: {key} items must be alphanumeric (._- allowed)")
            if toks:
                out += [flag, ",".join(toks)]
        elif typ == "path":
            sv = str(val).strip()
            # A filesystem path (list-form argv, no shell) — slashes/spaces are fine,
            # but it must never be mistakable for a flag, and no control characters.
            if sv.startswith("-"):
                raise ValueError(f"{command}: {key} must not start with '-'")
            if any(ord(ch) < 32 for ch in sv):
                raise ValueError(f"{command}: {key} contains control characters")
            if sv:
                out += [flag, sv]
    return out


class OpsRunner:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.config_path = cfg.get("_config_path")
        ctrl = cfg.get("control_dir") or str(Path(__file__).resolve().parent.parent / "var")
        self.logdir = Path(ctrl).expanduser() / "ops-logs"
        self.logdir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._job: dict | None = None              # the live job, in memory — the source of truth
        self._seq = 0                              # job identity for result(job_id) — July #16
        self._prev: dict = {}                      # last few REPLACED jobs' summaries, by id

    def _build_cmd(self, command: str, argv: list) -> list:
        cmd = [sys.executable, "-m", "knowledgehost"]
        if self.config_path:                            # same config the server runs on
            cmd += ["-c", self.config_path]
        return cmd + [command, *argv]

    def running(self) -> bool:
        j = self._job
        return bool(j and j["proc"].poll() is None)

    def start(self, command: str, args: dict | None = None) -> dict:
        with self._lock:
            if self.running():
                return {"ok": False, "error": "a job is already running", "status": self.status()}
            argv = _argv(command, args or {})          # raises on anything invalid
            # Close the previous job's log handle — it was held (never closed) in the
            # replaced dict, one leaked fd per job until the server hit EMFILE under
            # autopilot.  The child owns its own duplicated fd, so this is always safe.
            old = self._job
            if old and old.get("logfh"):
                try:
                    old["logfh"].close()
                except OSError:
                    pass
            if old:
                # The replaced job is FINISHED (running() refused otherwise) —
                # keep its summary so result(job_id) can still answer for it
                # after the slot is reused (July #16: the autopilot read whatever
                # occupied the slot, so a manual job launched in the window made
                # it mistake the wrong job's outcome for its own step's).
                self._prev[old["id"]] = {"command": old["command"],
                                         "exit_code": old["proc"].poll(),
                                         "logfile": old["logfile"]}
                while len(self._prev) > 4:
                    self._prev.pop(next(iter(self._prev)))
            self._seq += 1
            ts = time.strftime("%Y%m%d-%H%M%S")
            # The id in the name keeps two same-second launches from sharing a
            # file (the second used to TRUNCATE the first's log — and with it
            # the OPS_RESULT line the autopilot reads back).
            logfile = self.logdir / f"{command}-{ts}-{self._seq}.log"
            lf = open(logfile, "wb", buffering=0)
            cmd = self._build_cmd(command, argv)
            log.info("ops: launching %s", " ".join(cmd))
            env = {**os.environ, "PYTHONUNBUFFERED": "1"}   # stream prints to the log live
            proc = subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT,
                                    start_new_session=True,   # own group → clean tree kill
                                    env=env)
            self._job = {"proc": proc, "logfh": lf, "command": command, "argv": argv,
                         "started": time.time(), "logfile": str(logfile),
                         "id": self._seq}
            return {"ok": True, "status": self.status()}

    def stop(self) -> dict:
        with self._lock:
            if not self.running():
                return {"ok": False, "error": "no job is running"}
            proc = self._job["proc"]
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)   # whole session
            except (ProcessLookupError, PermissionError):
                proc.terminate()
        for _ in range(20):                            # up to ~2s for a graceful exit
            if proc.poll() is not None:
                break
            time.sleep(0.1)
        if proc.poll() is None:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                proc.kill()
        return {"ok": True, "status": self.status()}

    def status(self) -> dict:
        j = self._job
        if not j:
            return {"running": False, "command": None}
        rc = j["proc"].poll()
        if rc is not None and not j.get("ended"):
            # Stop the clock the first time we see it exited — otherwise "finished
            # in 15m" keeps climbing all afternoon.  (First SIGHTING, not the exact
            # exit instant: close enough at a 2.5s poll, and never wrong by more.)
            j["ended"] = time.time()
        return {"running": rc is None, "command": j["command"], "argv": j["argv"],
                "started": j["started"], "ended": j.get("ended"),
                "elapsed_s": round((j.get("ended") or time.time()) - j["started"]),
                "exit_code": rc, "logfile": j["logfile"], "id": j.get("id")}

    @staticmethod
    def _result_line(logfile) -> dict | None:
        import json as _json
        payload = None
        try:
            with open(logfile, "r", errors="replace") as f:
                for line in f:
                    if line.startswith(RESULT_PREFIX):
                        try:
                            payload = _json.loads(line[len(RESULT_PREFIX):])
                        except ValueError:
                            pass
        except OSError:
            return None
        return payload

    def result(self, job_id: int | None = None) -> dict | None:
        """The finished job's OPS_RESULT line (parsed), plus command/exit_code —
        or None while running / when the job never emitted one.  The autopilot
        reads this right after a step completes to learn whether it did work.

        With `job_id` (from status()["id"]) the answer is FOR THAT JOB even if
        the single slot has since been reused by a manual panel launch — and a
        finished identified job always answers with at least command/exit_code,
        payload or not, so the failure backoff can't be blinded by the race."""
        j = self._job
        if job_id is not None and (not j or j.get("id") != job_id):
            prev = self._prev.get(job_id)
            if not prev:
                return None
            return {"command": prev["command"], "exit_code": prev["exit_code"],
                    **(self._result_line(prev["logfile"]) or {})}
        if not j:
            return None
        rc = j["proc"].poll()
        if rc is None:
            return None
        payload = self._result_line(j["logfile"])
        if payload is None and job_id is None:
            return None                    # legacy contract (metrics collector)
        return {"command": j["command"], "exit_code": rc, **(payload or {})}

    def progress(self) -> dict | None:
        """The live job's LAST OPS_PROGRESS record, parsed — the panel's bar and the
        header status both read this.  Only the tail bytes are read: a long distil's
        log runs to megabytes and this is polled every couple of seconds."""
        import json as _json
        j = self._job
        if not j:
            return None
        try:
            with open(j["logfile"], "rb") as f:
                f.seek(0, os.SEEK_END)
                size = f.tell()
                f.seek(max(0, size - _PROGRESS_TAIL_BYTES))
                blob = f.read().decode("utf-8", "replace")
        except OSError:
            return None
        for line in reversed(blob.splitlines()):
            k = line.find(PROGRESS_PREFIX)
            if k < 0:
                continue
            try:
                rec = _json.loads(line[k + len(PROGRESS_PREFIX):])
            except ValueError:      # a torn last line (the job is mid-write) — look older
                continue
            if isinstance(rec, dict):
                return rec
        return None

    def tail(self, n: int = 300) -> str:
        j = self._job
        if not j:
            return ""
        try:
            with open(j["logfile"], "r", errors="replace") as f:
                return "".join(f.readlines()[-int(n):])
        except OSError:
            return ""

    def shutdown(self) -> None:
        """Best-effort: stop a running job when the server itself is going down, so a job
        doesn't outlive the server that was tracking it."""
        if self.running():
            log.info("ops: server shutdown — stopping the running job")
            self.stop()
