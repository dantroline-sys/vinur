"""Deferred-ingest inbox — the durable 'Needs your input' queue.

A bulk crawl never blocks on a human.  When it meets a structured/ambiguous document
(scripture / legal) it can't ingest without a confirmation, it records the question set
HERE and moves on; the document is set aside until the user answers.  Answers are
grouped **once per profile** (structure.profile_signature): a whole Title-17 folder
raises ONE request, a Bible raises another.  Answering a request confirms every
document filed under it, and the next crawl ingests them unit-by-unit.

Its own tiny SQLite db (WAL) so the crawl subprocess and the server can both touch it
safely.  Pure stdlib.
"""
from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path

_SCHEMA = """
CREATE TABLE IF NOT EXISTS requests(
  id        INTEGER PRIMARY KEY,
  signature TEXT UNIQUE,          -- structure.profile_signature (the 'once per profile' key)
  kind      TEXT,
  profile   TEXT,                 -- the proposed profile (JSON) the questions came from
  questions TEXT,                 -- structure.questions_for(profile) (JSON)
  answers   TEXT,                 -- the user's replies (JSON) once answered
  confirmed TEXT,                 -- structure.apply_answers(...) result (JSON) once answered
  state     TEXT DEFAULT 'pending',   -- pending | answered | dismissed
  created   REAL,
  updated   REAL);
CREATE TABLE IF NOT EXISTS request_docs(
  request_id INTEGER,
  path       TEXT,
  added      REAL,
  UNIQUE(request_id, path));
"""


def db_path(cfg: dict) -> str:
    explicit = cfg.get("pending_db")
    if explicit:
        return str(Path(explicit).expanduser())
    ctrl = cfg.get("control_dir") or str(Path(__file__).resolve().parent.parent / "var")
    return str(Path(ctrl).expanduser() / "run" / "pending.db")


def _loads(v):
    try:
        return json.loads(v) if v else None
    except (ValueError, TypeError):
        return None


class Pending:
    def __init__(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.executescript(_SCHEMA)
        self.db.commit()

    # ── write side (the crawl) ────────────────────────────────────────────────
    def defer(self, signature: str, kind: str, profile: dict, questions: list, path: str):
        """File `path` under the pending request for `signature`, creating that request
        (with its questions) the first time the signature is seen.  Returns (id, is_new)."""
        now = time.time()
        r = self.db.execute("SELECT id FROM requests WHERE signature=?", (signature,)).fetchone()
        if r is None:
            cur = self.db.execute(
                "INSERT INTO requests(signature,kind,profile,questions,state,created,updated)"
                " VALUES(?,?,?,?, 'pending', ?, ?)",
                (signature, kind or "", json.dumps(profile), json.dumps(questions), now, now))
            rid, is_new = cur.lastrowid, True
        else:
            rid, is_new = r["id"], False
        self.db.execute("INSERT OR IGNORE INTO request_docs(request_id,path,added) VALUES(?,?,?)",
                        (rid, path, now))
        self.db.commit()
        return rid, is_new

    # ── read side (the crawl consults these to decide what to do with a doc) ──
    def confirmed_profile(self, signature: str):
        """The confirmed profile for a signature whose request is answered, else None."""
        r = self.db.execute("SELECT state,confirmed FROM requests WHERE signature=?",
                            (signature,)).fetchone()
        return _loads(r["confirmed"]) if r and r["state"] == "answered" else None

    def answer_for_path(self, path: str):
        """The confirmed profile for a specific deferred doc whose request is now answered."""
        r = self.db.execute(
            "SELECT r.confirmed FROM request_docs d JOIN requests r ON r.id=d.request_id "
            "WHERE d.path=? AND r.state='answered'", (path,)).fetchone()
        return _loads(r["confirmed"]) if r else None

    # ── the server / inbox ────────────────────────────────────────────────────
    def answer(self, request_id: int, answers: dict, confirmed: dict) -> bool:
        now = time.time()
        cur = self.db.execute(
            "UPDATE requests SET answers=?, confirmed=?, state='answered', updated=? "
            "WHERE id=? AND state!='dismissed'",
            (json.dumps(answers), json.dumps(confirmed), now, request_id))
        self.db.commit()
        return cur.rowcount > 0

    def dismiss(self, request_id: int) -> bool:
        cur = self.db.execute("UPDATE requests SET state='dismissed', updated=? WHERE id=?",
                             (time.time(), request_id))
        self.db.commit()
        return cur.rowcount > 0

    def get(self, request_id: int):
        r = self.db.execute("SELECT * FROM requests WHERE id=?", (request_id,)).fetchone()
        return self._shape(r) if r else None

    def by_signature(self, signature: str):
        r = self.db.execute("SELECT * FROM requests WHERE signature=?", (signature,)).fetchone()
        return self._shape(r) if r else None

    def list(self, state: str | None = "pending") -> list:
        if state:
            rows = self.db.execute("SELECT * FROM requests WHERE state=? ORDER BY created", (state,))
        else:
            rows = self.db.execute("SELECT * FROM requests ORDER BY created")
        return [self._shape(r) for r in rows]

    def pending_count(self) -> int:
        return self.db.execute("SELECT COUNT(*) FROM requests WHERE state='pending'").fetchone()[0]

    def _shape(self, r: sqlite3.Row) -> dict:
        docs = [d["path"] for d in self.db.execute(
            "SELECT path FROM request_docs WHERE request_id=? ORDER BY added", (r["id"],))]
        return {"id": r["id"], "signature": r["signature"], "kind": r["kind"],
                "state": r["state"], "created": r["created"], "updated": r["updated"],
                "profile": _loads(r["profile"]) or {}, "questions": _loads(r["questions"]) or [],
                "answers": _loads(r["answers"]), "confirmed": _loads(r["confirmed"]),
                "docs": docs, "doc_count": len(docs)}

    def close(self):
        self.db.close()


def open_pending(cfg: dict) -> Pending:
    return Pending(db_path(cfg))
