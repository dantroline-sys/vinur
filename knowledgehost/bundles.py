"""Modular knowledge bundles — split one master ``kb.db`` into per-provenance
bundle files, and assemble a disposable session **working DB** from a selected
set (a *scenario*).

Why this is cheap here: every node/card/edge id is a **content hash** (see
``kb._hash``) — the same concept distilled in two places gets the *same* id.  So
combining bundles is ``INSERT OR IGNORE`` across files: shared concepts dedup,
an edge that names a node from another bundle relinks by itself, and no id
coordination / region-prefix scheme (spec §4.9) is needed at all.  Collision of
an id means collision of *content*, which is exactly the merge you want.

Two directions, one closure primitive:
- **split**  master → one ``<bundle>.kdb`` per provenance group (its closure).
- **assemble**  selected sources/bundles → one working DB the session opens.
  If pre-split bundle files exist we merge those wholesale; otherwise we extract
  the closure straight from the master.  Either way the hot read path stays
  single-file (spec §16.7: "ship granular, run consolidated").

On top of those, the **brain** operations (a brain = a shippable bundle file):
- **import_bundle**  absorb a FOREIGN ``.kdb`` — one authored elsewhere — into
  the master under a single bundle name.  New rows get their support trust
  capped so shipped knowledge lands at trust_tier 'low' no matter what the file
  claims; rows already in the master (same content hash) are never touched.
- **eject_bundle**  the inverse: export the bundle's closure to its ``.kdb``
  (nothing is ever lost), then remove its provenance from the master —
  refcount-aware, so shared rows survive with the ejected support stripped.
- **load / unload**  are not here at all: they are just the ``unloaded_bundles``
  selection filter + a working-DB reassembly, completely non-destructive.

Everything here is plain sqlite3 + json (no numpy / no GPU) so it runs anywhere
the CLI does and is unit-testable without the model stack.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import sqlite3
import time
from pathlib import Path

log = logging.getLogger("knowledgehost.bundles")

# Tables that carry *knowledge* and therefore travel with a bundle.  The
# operational tables (distilled_chunks, node_merge_candidates, knowledge_gaps)
# are authoring/pipeline state, not shippable knowledge — they are recreated
# empty in every working DB by the normal schema init and never copied.
KNOWLEDGE_TABLES = ("source_registry", "nodes", "procedure_cards", "edges",
                    "surface_questions", "surface_propositions")

BUNDLE_EXT = ".kdb"
DEFAULT_BUNDLE = "base"

# Support-trust ceiling stamped on rows a foreign brain contributes: below
# facets._TRUST_MED (0.40), so every derived trust_tier facet comes out 'low'.
# The read path (fit-gate, retrieval filters) then treats shipped knowledge
# with appropriate skepticism until the operator promotes it deliberately.
IMPORT_TRUST_CAP = 0.35


# ── low-level sqlite helpers ─────────────────────────────────────────────────
def _connect(path, *, encrypted: bool = False, key: str | None = None
             ) -> sqlite3.Connection:
    from . import dbcrypt
    c = dbcrypt.connect(str(path), encrypted=encrypted, key=key)
    c.row_factory = sqlite3.Row
    return c


def _selected_bundles(cfg: dict, master, doc_ids) -> set:
    """Bundle names covering the selected sources (for encryption decisions)."""
    return {s["bundle"] for s in list_sources(master) if s["doc_id"] in doc_ids}


def _needs_encryption(cfg: dict, bundle_names) -> bool:
    return bool(set(bundle_names) & set(cfg.get("encrypted_bundles") or []))


def _cols(conn: sqlite3.Connection, table: str) -> list[str]:
    return [r[1] for r in conn.execute(f"PRAGMA table_info({table})").fetchall()]


def _has_table(conn: sqlite3.Connection, table: str) -> bool:
    return bool(conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
        (table,)).fetchone())


def _clone_schema(src: sqlite3.Connection, dst: sqlite3.Connection) -> None:
    """Replicate the master's exact table+index DDL into a fresh DB.  Using the
    source's own ``sqlite_master`` keeps bundle files schema-identical to the
    master with zero DDL duplication (survives future migrations for free)."""
    for (sql,) in src.execute(
            "SELECT sql FROM sqlite_master WHERE sql IS NOT NULL "
            "AND type IN ('table','index')").fetchall():
        try:
            dst.execute(sql)
        except sqlite3.OperationalError as e:      # pragma: no cover - defensive
            log.debug("schema clone skipped one stmt: %s", e)


def _copy(dst: sqlite3.Connection, table: str, cols: list[str], rows: list) -> int:
    if not rows:
        return 0
    ph = ",".join("?" * len(cols))
    dst.executemany(
        f"INSERT OR IGNORE INTO {table}({','.join(cols)}) VALUES({ph})", rows)
    return len(rows)


def _support_docids(support_json) -> set:
    try:
        return {e.get("doc_id") for e in json.loads(support_json or "[]")
                if isinstance(e, dict) and e.get("doc_id")}
    except (ValueError, TypeError):
        return set()


# ── closure: pull one provenance group + everything it references ────────────
def extract_closure(src: sqlite3.Connection, dst: sqlite3.Connection,
                    doc_ids) -> dict:
    """Copy the closure of ``doc_ids`` from ``src`` into ``dst`` (which must
    already have the schema).  Closure = every source row for those docs, every
    node/card/edge whose *support* cites one of them, plus the nodes those cards
    and edges reference (so no edge dangles).  Borrowed nodes are duplicated into
    every bundle that needs them — harmless, because the id is a content hash and
    they dedup on the eventual merge.  Returns per-table copied counts."""
    doc_ids = {d for d in doc_ids if d}
    counts: dict = {}

    # 1. sources
    scols = _cols(src, "source_registry")
    srows = [tuple(r) for r in src.execute("SELECT * FROM source_registry")
             if r["doc_id"] in doc_ids]
    counts["source_registry"] = _copy(dst, "source_registry", scols, srows)

    # 2. owned nodes/cards/edges (support cites one of our docs)
    ncols = _cols(src, "nodes")
    nrows, owned_nodes = [], set()
    for r in src.execute("SELECT * FROM nodes"):
        if _support_docids(r["support"]) & doc_ids:
            nrows.append(tuple(r))
            owned_nodes.add(r["id"])

    ccols = _cols(src, "procedure_cards")
    crows, owned_cards, ref_nodes = [], set(), set()
    for r in src.execute("SELECT * FROM procedure_cards"):
        if _support_docids(r["support"]) & doc_ids:
            crows.append(tuple(r))
            owned_cards.add(r["id"])
            if r["node_id"]:
                ref_nodes.add(r["node_id"])

    ecols = _cols(src, "edges")
    erows, owned_edges = [], set()
    for r in src.execute("SELECT * FROM edges"):
        if _support_docids(r["support"]) & doc_ids:
            erows.append(tuple(r))
            owned_edges.add(r["id"])
            ref_nodes.update((r["src_id"], r["dst_id"]))

    # 3. borrowed nodes (referenced by an owned card/edge but owned elsewhere)
    need = {n for n in ref_nodes if n and n not in owned_nodes}
    if need:
        need_l = list(need)
        for i in range(0, len(need_l), 500):        # chunk the IN(...) list
            batch = need_l[i:i + 500]
            qm = ",".join("?" * len(batch))
            for r in src.execute(
                    f"SELECT * FROM nodes WHERE id IN ({qm})", batch):
                if r["id"] not in owned_nodes:
                    nrows.append(tuple(r))
                    owned_nodes.add(r["id"])

    counts["nodes"] = _copy(dst, "nodes", ncols, nrows)
    counts["procedure_cards"] = _copy(dst, "procedure_cards", ccols, crows)
    counts["edges"] = _copy(dst, "edges", ecols, erows)

    # 4. retrieval surfaces for everything we copied
    owned_ids = owned_nodes | owned_cards | owned_edges
    for table in ("surface_questions", "surface_propositions"):
        if not _has_table(src, table):
            continue
        cols = _cols(src, table)
        rows = [tuple(r) for r in src.execute(f"SELECT * FROM {table}")
                if r["target_id"] in owned_ids]
        counts[table] = _copy(dst, table, cols, rows)

    dst.commit()
    return counts


def merge_db(src: sqlite3.Connection, dst: sqlite3.Connection,
             tables=KNOWLEDGE_TABLES, *, intersect: bool = False) -> dict:
    """Wholesale ``INSERT OR IGNORE`` copy of an already-closed bundle file into
    the working DB.  Order doesn't matter and re-runs are idempotent (content-hash
    ids), so merging N bundles is just calling this N times.  ``intersect`` copies
    only the columns both sides have — for FOREIGN files whose schema may lead or
    trail ours (a missing column takes the schema default on insert)."""
    counts: dict = {}
    for t in tables:
        if not _has_table(src, t) or not _has_table(dst, t):
            continue
        cols = _cols(src, t)
        if intersect:
            dcols = set(_cols(dst, t))
            cols = [c for c in cols if c in dcols]
            if not cols:
                continue
            sel = ",".join(cols)
            rows = [tuple(r) for r in src.execute(f"SELECT {sel} FROM {t}")]
        else:
            rows = [tuple(r) for r in src.execute(f"SELECT * FROM {t}")]
        counts[t] = _copy(dst, t, cols, rows)
    dst.commit()
    return counts


# ── manifest: a .kdb file's self-description ─────────────────────────────────
def write_manifest(dst: sqlite3.Connection, bundle: str, counts: dict,
                   cfg: dict) -> None:
    """Single-row JSON manifest inside a bundle file: what it is, when it was
    exported, and which embed model produced its node vectors (an importer on a
    different model strips the vectors and re-embeds)."""
    dst.execute("CREATE TABLE IF NOT EXISTS bundle_manifest(json TEXT)")
    dst.execute("DELETE FROM bundle_manifest")
    dst.execute("INSERT INTO bundle_manifest(json) VALUES(?)", (json.dumps({
        "format": 1, "name": bundle, "created": time.time(),
        "embed_model": cfg.get("embed_model") or "",
        "counts": counts}),))


def read_manifest(conn: sqlite3.Connection) -> dict | None:
    if not _has_table(conn, "bundle_manifest"):
        return None
    row = conn.execute("SELECT json FROM bundle_manifest LIMIT 1").fetchone()
    try:
        m = json.loads(row[0]) if row else None
        return m if isinstance(m, dict) else None
    except (ValueError, TypeError):
        return None


def inspect_bundle_file(path: str) -> dict:
    """What's in a .kdb without importing it — for the panel / CLI preview."""
    p = Path(path).expanduser()
    if not p.exists():
        raise ValueError(f"no such file: {p}")
    conn = _connect(str(p))
    try:
        if not _has_table(conn, "source_registry") or not _has_table(conn, "nodes"):
            raise ValueError(f"{p.name} is not a knowledge bundle "
                             "(missing source_registry/nodes)")
        counts = {t: conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
                  for t in KNOWLEDGE_TABLES if _has_table(conn, t)}
        return {"file": str(p), "manifest": read_manifest(conn),
                "sources": list_sources(conn), "counts": counts}
    except sqlite3.DatabaseError as e:
        raise ValueError(f"{p.name} is not readable as a bundle "
                         f"(encrypted, or not sqlite): {e}") from None
    finally:
        conn.close()


# ── import: absorb a foreign brain into the master ──────────────────────────
def _cap_support_trust(support_json, cap: float):
    """Cap every support entry's trust at `cap`; None if nothing changed."""
    try:
        entries = json.loads(support_json or "[]")
    except (ValueError, TypeError):
        return None
    changed = False
    for e in entries if isinstance(entries, list) else []:
        t = e.get("trust") if isinstance(e, dict) else None
        if isinstance(t, (int, float)) and t > cap:
            e["trust"] = round(cap, 3)
            changed = True
    return json.dumps(entries) if changed else None


def _new_ids(master: sqlite3.Connection, pre: set, table: str,
             key: str = "id") -> list:
    return [r[0] for r in master.execute(f"SELECT {key} FROM {table}")
            if r[0] not in pre]


def import_bundle(cfg: dict, path: str, *, name: str | None = None,
                  trust: str = "low", log_fn=None) -> dict:
    """Absorb a foreign ``.kdb`` into the MASTER under one bundle name.

    Content-hash ids make this idempotent: re-importing the same brain is a
    no-op, and rows the master already has are never modified (so a shared
    concept keeps its local trust/provenance).  Everything NEW gets:
      * its source rows rebranded to the single bundle `name`
        (default: manifest name, else the file stem) — a shipped file that
        calls itself 'base' can't pollute the local base group;
      * support trust capped at IMPORT_TRUST_CAP unless trust='keep'
        (use 'keep' for your own brains moving between your own boxes);
      * embeddings stripped when the file's manifest names a different embed
        model than this install (embed-nodes backfills them).
    Returns counts + follow-up notes.  Load/unload afterwards is the
    ``unloaded_bundles`` filter; permanent removal is ``eject_bundle``."""
    say = log_fn or log.info
    if trust not in ("low", "keep"):
        raise ValueError(f"trust must be 'low' or 'keep', not {trust!r}")
    info = inspect_bundle_file(path)               # validates the file
    src_sources = info["sources"]
    bundle = _slug(name or (info["manifest"] or {}).get("name")
                   or Path(path).stem)
    master_path = str(Path(cfg.get("_master_kb_path")
                           or cfg["kb_path"]).expanduser())
    if not os.path.exists(master_path):
        raise ValueError(f"no master kb at {master_path} — run ingest first "
                         "(or copy the brain in as your kb.db)")

    src = _connect(str(Path(path).expanduser()))
    master = _connect(master_path)
    try:
        existing = {s["bundle"] for s in list_sources(master)}
        src_docs = {s["doc_id"] for s in src_sources}
        pre_docs = {r[0] for r in master.execute(
            "SELECT doc_id FROM source_registry")}
        if bundle in existing and not (src_docs & pre_docs):
            raise ValueError(
                f"bundle '{bundle}' already exists in the master — import "
                f"with a different --name, or eject '{bundle}' first")

        # what the master holds NOW, so we can tell which rows are the brain's
        pre = {t: {r[0] for r in master.execute(f"SELECT id FROM {t}")}
               for t in ("nodes", "procedure_cards", "edges")}

        counts = merge_db(src, master, intersect=True)

        new_docs = [d for d in src_docs if d not in pre_docs]
        newness = {t: _new_ids(master, pre[t], t)
                   for t in ("nodes", "procedure_cards", "edges")}

        # 1. rebrand: every source THIS import introduced joins bundle `name`
        master.executemany(
            "UPDATE source_registry SET bundle=? WHERE doc_id=?",
            [(bundle, d) for d in new_docs])

        # 2. trust cap on the brain's own rows (shared rows were IGNOREd)
        capped = 0
        if trust == "low":
            for t in ("nodes", "procedure_cards", "edges"):
                ids = newness[t]
                for i in range(0, len(ids), 500):
                    batch = ids[i:i + 500]
                    qm = ",".join("?" * len(batch))
                    upd = []
                    for rid, sup in master.execute(
                            f"SELECT id, support FROM {t} WHERE id IN ({qm})",
                            batch):
                        s2 = _cap_support_trust(sup, IMPORT_TRUST_CAP)
                        if s2 is not None:
                            upd.append((s2, rid))
                    master.executemany(
                        f"UPDATE {t} SET support=? WHERE id=?", upd)
                    capped += len(upd)
            master.executemany(
                "UPDATE source_registry SET trust_weight=? WHERE doc_id=? "
                "AND (trust_weight IS NULL OR trust_weight>?)",
                [(IMPORT_TRUST_CAP, d, IMPORT_TRUST_CAP) for d in new_docs])

        # 3. foreign embed model ⇒ the shipped vectors are in the wrong space
        stripped = 0
        man_model = (info["manifest"] or {}).get("embed_model") or ""
        our_model = cfg.get("embed_model") or ""
        if man_model and our_model and man_model != our_model and newness["nodes"]:
            ids = newness["nodes"]
            for i in range(0, len(ids), 500):
                batch = ids[i:i + 500]
                qm = ",".join("?" * len(batch))
                stripped += master.execute(
                    f"UPDATE nodes SET embedding=NULL WHERE id IN ({qm})",
                    batch).rowcount
            say(f"import '{bundle}': embed model mismatch "
                f"({man_model} ≠ {our_model}) — stripped {stripped} vectors; "
                "run embed-nodes to backfill")
        master.commit()

        new_counts = {t: len(v) for t, v in newness.items()}
        res = {"bundle": bundle, "file": info["file"],
               "sources_new": len(new_docs),
               "sources_shared": len(src_docs) - len(new_docs),
               "new": new_counts, "merged": counts,
               "trust": trust, "support_capped": capped,
               "embeddings_stripped": stripped,
               "next": ["facetize", "build-ann"]
                       + (["embed-nodes"] if stripped else [])}
        say(f"import '{bundle}': {len(new_docs)} new source(s), "
            f"{new_counts} new rows (trust={trust})")
        return res
    finally:
        src.close()
        master.close()


# ── eject: export-then-remove a bundle from the master ──────────────────────
def eject_bundle(cfg: dict, bundle: str, *, export_first: bool = True,
                 dry_run: bool = False, log_fn=None) -> dict:
    """Permanently remove one bundle's provenance from the MASTER — the
    destructive half of de-merge (the reversible half is just unloading it).
    Unless dry_run, the bundle's closure is first exported to its ``.kdb``
    (export_first), so an eject can always be undone by re-importing the file.
    Shared rows survive with the ejected support stripped (see unimport.py —
    same refcount rules as the dataset undo)."""
    say = log_fn or log.info
    from .kb import KB
    from .unimport import remove_docs
    master_path = str(Path(cfg.get("_master_kb_path")
                           or cfg["kb_path"]).expanduser())
    mcfg = {**cfg, "kb_path": master_path, "ann_search": False}
    kb = KB(mcfg)
    try:
        docs = {s["doc_id"] for s in list_sources(kb._raw)
                if s["bundle"] == bundle}
        if not docs:
            raise ValueError(f"no sources in bundle '{bundle}' — nothing to eject")
        exported = None
        if export_first and not dry_run:
            exported = split(mcfg, force=True, only={bundle},
                             log_fn=say).get(bundle, {}).get("file")
        st = remove_docs(kb, docs, dry_run=dry_run)
        st.update({"bundle": bundle, "sources": len(docs),
                   "exported": exported, "dry_run": dry_run})
        if not dry_run:
            say(f"eject '{bundle}': {st}")
        return st
    finally:
        kb.close()


# ── scenario resolution ──────────────────────────────────────────────────────
def bundle_of(row) -> str:
    """The bundle group a source belongs to (its ``bundle`` tag, else base)."""
    b = row["bundle"] if ("bundle" in row.keys() and row["bundle"]) else None
    return b or DEFAULT_BUNDLE


def list_sources(master: sqlite3.Connection) -> list[dict]:
    has_bundle = "bundle" in _cols(master, "source_registry")
    sel = "doc_id, title, source_type, status" + (", bundle" if has_bundle else "")
    out = []
    for r in master.execute(f"SELECT {sel} FROM source_registry"):
        d = dict(r)
        d["bundle"] = d.get("bundle") or DEFAULT_BUNDLE
        out.append(d)
    return out


def _match(token: str, row: dict) -> bool:
    """A scenario token selects a source by wildcard, bundle, doc_id, or title."""
    return token == "*" or token in (row.get("bundle"), row.get("doc_id"),
                                     row.get("title"))


def select_sources(sources: list[dict], scenario: dict,
                   unloaded=()) -> set:
    """Apply a scenario's include/exclude to the source list → set of doc_ids.
    include absent/empty ⇒ everything; exclude prunes.  Retracted sources are
    always dropped.  ``unloaded`` (the runtime load/unload toggle — see
    unloaded_set) prunes whole bundles AFTER the scenario, so 'unload the X
    brain' composes with whatever scenario is active."""
    inc = scenario.get("include") or ["*"]
    exc = scenario.get("exclude") or []
    live = [s for s in sources if s.get("status") != "retracted"]
    picked = [s for s in live if any(_match(t, s) for t in inc)]
    picked = [s for s in picked if not any(_match(t, s) for t in exc)]
    if unloaded:
        picked = [s for s in picked if s.get("bundle") not in set(unloaded)]
    return {s["doc_id"] for s in picked}


def unloaded_set(cfg: dict) -> set:
    """Bundles currently switched OFF (the runtime 'unload that brain' state).
    Persisted as a comma-separated scalar (``unloaded_bundles``) so the panel's
    config writer can round-trip it; empty ⇒ everything loads."""
    raw = cfg.get("unloaded_bundles") or ""
    toks = raw if isinstance(raw, (list, tuple, set)) else str(raw).split(",")
    return {t.strip() for t in toks if str(t).strip()}


def active_scenario_name(cfg: dict) -> str:
    return (cfg.get("active_scenario") or cfg.get("default_scenario")
            or "all").strip() or "all"


def scenario_def(cfg: dict, name: str) -> dict:
    scenarios = cfg.get("scenarios") or {}
    if name in scenarios and isinstance(scenarios[name], dict):
        return scenarios[name]
    return {}                                       # unknown / "all" ⇒ everything


def is_modular(cfg: dict) -> bool:
    """Modularity is engaged only when the operator has defined scenarios, asked
    for a non-default one, or unloaded a bundle at runtime.  Absent all three,
    everything short-circuits to the master kb.db and behaviour is byte-for-byte
    identical to before this feature."""
    if cfg.get("scenarios") or unloaded_set(cfg):
        return True
    name = active_scenario_name(cfg)
    return name not in ("", "all")


# ── working-DB assembly (the session entry point) ────────────────────────────
def _work_dir(cfg: dict) -> Path:
    d = cfg.get("bundle_work_dir") or ""
    if not d:
        d = str(Path(cfg["kb_path"]).expanduser().parent / "work")
    p = Path(d).expanduser()
    p.mkdir(parents=True, exist_ok=True)
    return p


def _bundle_dir(cfg: dict) -> Path | None:
    d = cfg.get("bundle_dir") or ""
    return Path(d).expanduser() if d else None


def _cache_key(name: str, doc_ids: set, stamp: float) -> str:
    h = hashlib.sha1()
    h.update(name.encode())
    for d in sorted(doc_ids):
        h.update(b"\x00")
        h.update(d.encode())
    h.update(f"|{int(stamp)}".encode())
    return h.hexdigest()[:12]


def assemble_working_db(cfg: dict, *, force: bool = False, log_fn=None) -> str:
    """Resolve the active scenario and return the path of a ready-to-open working
    DB.  If the scenario selects everything and no scenarios are configured, this
    is a no-op that returns the master path unchanged.  Otherwise it builds (and
    caches by content) a disposable working DB and returns *that* path.

    Assembly source: pre-split bundle files under ``bundle_dir`` if present,
    else the closure straight out of the master.  Cached on (scenario, selected
    docs, master mtime) so an unchanged selection reuses the last build."""
    say = log_fn or log.info
    master_path = str(Path(cfg["kb_path"]).expanduser())
    if not is_modular(cfg):
        return master_path
    if not os.path.exists(master_path):
        say("no master kb.db yet — nothing to assemble")
        return master_path

    name = active_scenario_name(cfg)
    scen = scenario_def(cfg, name)
    unloaded = unloaded_set(cfg)
    master = _connect(master_path)                  # base/master is always clear
    try:
        sources = list_sources(master)
        doc_ids = (select_sources(sources, scen, unloaded)
                   if (scen or name != "all" or unloaded)
                   else {s["doc_id"] for s in sources})
        if not doc_ids:
            say(f"scenario '{name}' selected 0 sources — serving empty working DB")
        # If the selection pulls in a sensitive bundle, the working DB holds that
        # content and must itself be encrypted at rest (spec §16.6).
        sel_bundles = _selected_bundles(cfg, master, doc_ids)
        encrypted = _needs_encryption(cfg, sel_bundles)
        cfg["kb_encrypted"] = encrypted             # so the serving KB opens it right
        from . import dbcrypt
        key = dbcrypt.key_for(cfg) if encrypted else None
        try:
            stamp = os.path.getmtime(master_path)
        except OSError:
            stamp = time.time()
        # If we'll assemble from on-disk bundle files, fold their mtimes in too so a
        # re-split invalidates the cached working DB (master alone may be unchanged).
        bdir = _bundle_dir(cfg)
        if bdir and bdir.exists():
            for f, _b in _bundle_files_for(bdir, master, doc_ids):
                try:
                    stamp = max(stamp, os.path.getmtime(f))
                except OSError:
                    pass
        ck = _cache_key(name, doc_ids, stamp) + ("-enc" if encrypted else "")
        work = _work_dir(cfg) / f"work-{_slug(name)}-{ck}.db"
        if work.exists() and not force:
            say(f"scenario '{name}': reusing working DB {work.name} "
                f"({len(doc_ids)} sources{', encrypted' if encrypted else ''})")
            return str(work)

        tmp = work.with_suffix(".db.tmp")
        if tmp.exists():
            tmp.unlink()
        dst = _connect(str(tmp), encrypted=encrypted, key=key)
        try:
            _clone_schema(master, dst)
            counts = _assemble_into(cfg, master, dst, doc_ids, name, say)
        finally:
            dst.close()
        os.replace(tmp, work)
        say(f"scenario '{name}': built {work.name} from {len(doc_ids)} sources "
            f"{'[encrypted] ' if encrypted else ''}→ {counts}")
        _prune_stale(_work_dir(cfg), keep=work.name)
        return str(work)
    finally:
        master.close()


def _assemble_into(cfg, master, dst, doc_ids, name, say) -> dict:
    """Merge from pre-split bundle files when they exist for the selected
    bundles, else extract the closure straight from the master.  Encrypted bundle
    files are opened with the at-rest key."""
    bdir = _bundle_dir(cfg)
    files = _bundle_files_for(bdir, master, doc_ids) if bdir else []
    if files:
        from . import dbcrypt
        key = dbcrypt.key_for(cfg)
        total: dict = {}
        for f, bundle in files:
            enc = dbcrypt.bundle_is_encrypted(cfg, bundle)
            src = _connect(str(f), encrypted=enc, key=key if enc else None)
            try:
                for t, n in merge_db(src, dst).items():
                    total[t] = total.get(t, 0) + n
            finally:
                src.close()
        say(f"scenario '{name}': merged {len(files)} bundle file(s)")
        return total
    return extract_closure(master, dst, doc_ids)


def _bundle_files_for(bdir: Path, master, doc_ids) -> list[tuple]:
    """Which on-disk bundle files cover the selected sources → [(path, bundle)].
    A file is included if any selected source is tagged into its bundle."""
    if not bdir or not bdir.exists():
        return []
    want = set()
    for s in list_sources(master):
        if s["doc_id"] in doc_ids:
            want.add(s["bundle"])
    files = []
    for b in sorted(want):
        f = bdir / f"{_slug(b)}{BUNDLE_EXT}"
        if f.exists():
            files.append((f, b))
    return files


def _slug(s: str) -> str:
    return "".join(c if (c.isalnum() or c in "-_") else "-"
                   for c in (s or "").strip()).strip("-") or "bundle"


def _prune_stale(work_dir: Path, keep: str, limit: int = 6) -> None:
    """Keep the working-DB cache from growing unbounded: drop the oldest builds,
    never the one we just returned."""
    try:
        builds = sorted((p for p in work_dir.glob("work-*.db")),
                        key=lambda p: p.stat().st_mtime, reverse=True)
        for p in builds[limit:]:
            if p.name != keep:
                p.unlink(missing_ok=True)
    except OSError:
        pass


# ── split: master → per-bundle files ─────────────────────────────────────────
def split(cfg: dict, out_dir: str | None = None, *, force: bool = False,
          only: set | None = None, log_fn=None) -> dict:
    """Export each provenance group in the master into its own ``<bundle>.kdb``
    file (the group's closure).  Returns {bundle: {file, counts}}.  Idempotent —
    re-running overwrites the files with a fresh closure.  ``only`` restricts the
    export to the named bundles (eject uses this for its safety export)."""
    say = log_fn or log.info
    master_path = str(Path(cfg["kb_path"]).expanduser())
    out = Path(out_dir or (cfg.get("bundle_dir") or
                           (Path(master_path).parent / "bundles"))).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    from . import dbcrypt
    key = dbcrypt.key_for(cfg)
    master = _connect(master_path)
    result: dict = {}
    try:
        groups: dict = {}
        for s in list_sources(master):
            groups.setdefault(s["bundle"], set()).add(s["doc_id"])
        if only is not None:
            groups = {b: d for b, d in groups.items() if b in only}
        for bundle, docs in sorted(groups.items()):
            enc = dbcrypt.bundle_is_encrypted(cfg, bundle)
            f = out / f"{_slug(bundle)}{BUNDLE_EXT}"
            if f.exists() and not force:
                say(f"bundle '{bundle}': exists ({f.name}) — skip (use --force)")
                result[bundle] = {"file": str(f), "skipped": True}
                continue
            tmp = f.with_suffix(BUNDLE_EXT + ".tmp")
            if tmp.exists():
                tmp.unlink()
            dst = _connect(str(tmp), encrypted=enc, key=key if enc else None)
            try:
                _clone_schema(master, dst)
                counts = extract_closure(master, dst, docs)
                write_manifest(dst, bundle, counts, cfg)
                dst.commit()
            finally:
                dst.close()
            os.replace(tmp, f)
            result[bundle] = {"file": str(f), "counts": counts,
                              "sources": len(docs), "encrypted": enc}
            say(f"bundle '{bundle}': {f.name}  {'[encrypted] ' if enc else ''}{counts}")
    finally:
        master.close()
    return result
