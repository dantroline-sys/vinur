"""VINUR-PACK-01 — the knowledge-pack producer (`python3 -m knowledgehost pack`).

A pack is the CLEAN-ROOM distillate of one document or folder: built against a
scratch kb (never the master), so every row is traceable to the declared
sources and their licenses.  The output is a single `.kdb` (optionally gzip'd
and/or SQLCipher-encrypted) with a format-2 manifest inside AND beside it (the
JSON sidecar), stored under `packs/<slug>/<slug>-<version>.kdb[...]`.

Stages (§3.1): scratch workspace → ingest → distill → link → LICENSE GATE →
manifest v2 + VACUUM → package → atomic move + sidecar → cleanup.  A failed or
interrupted build keeps its scratch (the distilled set is the checkpoint), so
re-running the same command resumes.

The license gate (§5) refuses unshareable sources by default; `--license`
fills EMPTY licenses only (recorded as attested), `--allow-unlicensed` builds
a `shareable: false` pack for private use.  Encryption (§7) is access control
for distribution — who can OPEN the artifact — never usage control.
"""

import gzip as gzip_mod
import hashlib
import json
import logging
import os
import re
import shutil
import subprocess
import time
from pathlib import Path

from . import bundles
from . import dbcrypt

log = logging.getLogger("knowledgehost.pack")

_SEMVER = re.compile(r"^\d+\.\d+\.\d+$")

# ── license gate (§5) ─────────────────────────────────────────────────────────
# class → how the pack may ship.  Rank orders restrictiveness; the pack's
# effective license is the most restrictive source's label.
_RANK = {"free": 0, "attribution": 1, "share-alike": 2, "nc": 3}


def license_class(label: str | None) -> str:
    """'free' | 'attribution' | 'share-alike' | 'nc' | 'refused' for one source
    license label (SPDX-ish, as licensing.detect stores them).  Unknown, empty,
    ND and proprietary all refuse — the conservative default the contract sets."""
    s = (label or "").strip().lower()
    if not s or s in ("unknown", "none", "proprietary", "all rights reserved"):
        return "refused"
    toks = set(t for t in re.split(r"[^a-z0-9]+", s) if t)
    if "nd" in toks:
        return "refused"                        # no-derivatives: a distillate IS one
    if toks & {"cc0", "unlicense", "wtfpl", "pd"} or ("public" in toks and "domain" in toks):
        return "free"
    if "nc" in toks:
        return "nc"
    if toks & {"sa", "gpl", "lgpl", "agpl", "gfdl", "copyleft"}:
        return "share-alike"
    if "by" in toks or toks & {"mit", "bsd", "apache", "isc", "zlib", "mpl", "x11"}:
        return "attribution"
    return "refused"


def gate(sources: list, *, override: str = "", allow_unlicensed: bool = False) -> dict:
    """§5 verdicts for the registry rows of a scratch kb.  `sources` rows carry
    doc_id/title/license/license_holder/license_url.  Returns ok/effective/
    shareable/per_source/attribution/refused/filled (doc_ids the override
    filled — the caller writes those back so the export carries them)."""
    per, refused, filled = [], [], []
    worst = ("free", "")                        # (class, label)
    for r in sources:
        lic, attested = (r.get("license") or "").strip(), False
        if not lic and override:
            lic, attested = override.strip(), True
            filled.append(r.get("doc_id"))
        cls = license_class(lic)
        per.append({"doc_id": r.get("doc_id"), "title": r.get("title"),
                    "license": lic or "unknown", "holder": r.get("license_holder") or "",
                    "url": r.get("license_url") or "", "class": cls,
                    "attested": attested})
        if cls == "refused":
            refused.append(per[-1])
        elif _RANK[cls] >= _RANK.get(worst[0], 0):
            worst = (cls, lic)
    ok = not refused or allow_unlicensed
    shareable = not refused
    effective = "UNSHAREABLE" if refused else (worst[1] or "CC0-1.0")
    attribution = [f"{p['title'] or p['doc_id']} — {p['holder'] or 'unattributed'} ({p['license']})"
                   for p in per if p["class"] in ("attribution", "share-alike", "nc")]
    return {"ok": ok, "shareable": shareable, "effective": effective,
            "per_source": per, "attribution": attribution, "refused": refused,
            "filled": filled}


def gate_table(res: dict) -> str:
    """The refusal's per-source verdict table, remedies included."""
    lines = ["license gate: per-source verdicts"]
    for p in res["per_source"]:
        lines.append(f"  {'REFUSE' if p['class'] == 'refused' else p['class']:<12}"
                     f" {p['license']:<20} {p['title'] or p['doc_id']}")
    if res["refused"]:
        lines.append("remedies: set a license on the source (`source --license`), "
                     "fill EMPTY ones with --license <SPDX> (attested), or build a "
                     "private pack with --allow-unlicensed (stamped shareable:false)")
    return "\n".join(lines)


# ── manifest v2 (§4) ──────────────────────────────────────────────────────────
def content_hash(conn) -> str:
    """sha256 over a canonical serialization of the knowledge tables.  Row order
    is normalized by sorting each table's serialized rows, so the hash survives
    VACUUM/rowid churn and identifies the CONTENT."""
    h = hashlib.sha256()
    for t in bundles.KNOWLEDGE_TABLES:
        try:
            cols = [r[1] for r in conn.execute(f"PRAGMA table_info({t})")]
            if not cols:
                continue
            rows = sorted(
                json.dumps([v.hex() if isinstance(v, bytes) else v for v in row],
                           ensure_ascii=False)
                for row in conn.execute(f"SELECT * FROM {t}"))
        except Exception:
            continue
        h.update(json.dumps([t, cols], ensure_ascii=False).encode())
        for r in rows:
            h.update(r.encode())
    return "sha256:" + h.hexdigest()


def vocab_hash() -> str:
    from .distill import FEATURE_VOCAB
    return "sha256:" + hashlib.sha256(
        json.dumps(FEATURE_VOCAB, sort_keys=True).encode()).hexdigest()


def _vinur_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(Path(__file__).resolve().parent.parent),
            capture_output=True, text=True, timeout=5).stdout.strip()
    except Exception:
        return ""


def _group_counts(conn, table: str, col: str) -> dict:
    try:
        return {str(r[0] or ""): r[1] for r in conn.execute(
            f"SELECT {col}, COUNT(*) FROM {table} GROUP BY {col}")}
    except Exception:
        return {}


def manifest_v2(conn, *, slug: str, version: str, cfg: dict, gate_res: dict,
                meta: dict, counts: dict) -> dict:
    from .distill import _FAMILY_VERSION, RECARD_VERSION
    from .kb import SCHEMA_VERSION
    return {
        "format": 2, "kind": "pack",
        # format-1 fields kept in place so old importers keep working unchanged
        "name": slug, "created": time.time(),
        "embed_model": cfg.get("embed_model") or "",
        "counts": counts,
        "pack": {
            "id": slug, "title": meta.get("title") or slug, "version": version,
            "created": time.time(),
            "producer": {"name": meta.get("author") or "",
                         "contact": meta.get("contact") or ""},
            "description": meta.get("describe") or "",
            "data_kind": {
                "regimes": _group_counts(conn, "nodes", "regime"),
                "card_types": _group_counts(conn, "procedure_cards", "card_type"),
            },
        },
        "content": {"counts": counts, "content_hash": content_hash(conn)},
        "licensing": {
            "effective": gate_res["effective"], "shareable": gate_res["shareable"],
            "attribution": gate_res["attribution"],
            "per_source": [{k: p[k] for k in
                            ("doc_id", "title", "license", "holder", "url", "attested")}
                           for p in gate_res["per_source"]],
            "text_included": False,
        },
        "compat": {
            "schema_version": SCHEMA_VERSION,
            "embed_model": cfg.get("embed_model") or "",
            "vocab_hash": vocab_hash(),
            "card_families": dict(_FAMILY_VERSION),
            "requires": [],                     # generic packs, by construction
        },
        "pipeline": {
            "distill_model": cfg.get("distill_model") or "",
            "distill_max_tokens": cfg.get("distill_max_tokens"),
            "recard_version": RECARD_VERSION,
            "vinur_commit": _vinur_commit(),
        },
    }


def write_manifest_v2(conn, manifest: dict) -> None:
    conn.execute("CREATE TABLE IF NOT EXISTS bundle_manifest(json TEXT)")
    conn.execute("DELETE FROM bundle_manifest")
    conn.execute("INSERT INTO bundle_manifest(json) VALUES(?)",
                 (json.dumps(manifest, ensure_ascii=False),))
    conn.commit()


# ── the clean-room pipeline (§3.1 steps 1–4) ─────────────────────────────────
def scratch_cfg(cfg: dict, build: Path, src: Path) -> dict:
    """Derived config: every data path lands in the build dir; serving/embed
    endpoints, leases and tuning are inherited; the master is never named."""
    s = dict(cfg)
    s.pop("_master_kb_path", None)
    s.update({
        "kb_path": str(build / "kb.db"),
        "db_path": str(build / "index.db"),
        "ann_path": str(build / "kb.db.ann"),
        "bundle_dir": str(build / "bundles"),
        "sources": [str(src if src.is_dir() else src.parent)],
        "research_solved_dir": "",              # NEVER pull vinkona drops into a pack
        "library_root": "", "library_sources": [],
        "unloaded_bundles": [], "kb_encrypted": False,
    })
    return s


# The phases a clean-room build moves through, in order — shared by _pipeline
# (ingest/distill/recard/link) and add_to_collection (export/done) so the panel's
# progress bar has one consistent step sequence.
_BUILD_STEPS = ("ingest", "distill", "recard", "link", "export", "done")


def _emit(report, phase: str, **extra) -> None:
    """Fire the optional progress reporter with a consistent step number.  A no-op
    when no reporter was passed (build_pack), so the OPS_PROGRESS lines only appear
    for callers that want the bar (collect)."""
    if not report:
        return
    try:
        report(phase, step=_BUILD_STEPS.index(phase) + 1, steps=len(_BUILD_STEPS), **extra)
    except Exception:                           # progress is cosmetic — never fail a build for it
        pass


def _distil_progress(report):
    """Route the distiller's live chunk counts into the BUILD's phase bar.  A build
    is a 6-phase job that happens to contain a chunk-counting one; the bar stays in
    phase space (2 of 6) and gains "1,240 / 8,430 chunks, in <doc>" underneath,
    instead of the two progress channels overwriting each other.  No reporter
    (build_pack) → the distiller stays silent on the channel, as it did before."""
    from . import distill as distill_mod

    def emit(_phase, **kw):
        kw.pop("step", None)                      # the build owns step/steps
        total = kw.pop("steps", None)
        if total is not None:
            kw["chunk_steps"] = total
        _emit(report, "distill", **kw)

    return distill_mod.DistillProgress(
        emit=emit if report else (lambda *_a, **_k: None))


def _pipeline(scfg: dict, src: Path, say, *, label: str = "pack", report=None,
              profile=None) -> dict:
    """ingest → distill → link inside the scratch.  Module-level so tests can
    patch it; raises BackendUnavailable when no distill endpoint is up (the
    scratch survives — re-running resumes).  `label` prefixes the log lines
    (so a collect build doesn't say "pack"); `report` gets per-phase progress.
    `profile` is a CONFIRMED structure profile for a single structured document —
    threaded to ingest_file so a scripture/legal file is ingested unit-by-unit."""
    from . import distill as distill_mod
    from . import embed as embed_mod
    from . import ingest as ingest_mod
    from . import link as link_mod
    from . import lm_lease
    from .kb import KB
    from .store import make_store

    store = make_store(scfg)
    kb = KB(scfg)
    embedder = embed_mod.Embedder(scfg)
    stats: dict = {}
    try:
        if not embedder.embed_one("warmup", "document"):
            say(f"{label}: embed endpoint unreachable — the file will carry no vectors "
                "(importers re-embed; retrieval inside the build is sparse-only)")
            # keep going: ingest/distill degrade the same way the normal ops do
        _emit(report, "ingest")
        say(f"{label}: ingesting {src.name} …")
        t0 = time.time()
        if src.is_file():
            n = ingest_mod.ingest_file(store, embedder, scfg, str(src), profile=profile)
            stats["ingest"] = {"docs": 1 if n else 0, "chunks": n}
            if profile and profile.get("ingest_as") == "structured":
                say(f"{label}: ingested {src.name} as structured {profile.get('kind')} "
                    f"({n} canonical unit(s))")
        else:
            stats["ingest"] = ingest_mod.crawl(store, embedder, scfg)
        chunks = stats["ingest"].get("chunks") or 0
        say(f"{label}: ingested {stats['ingest']} ({time.time() - t0:.1f}s)")

        extractors = distill_mod.fast_endpoints(scfg, log)
        verifiers = distill_mod.verify_endpoints(scfg, log)
        if not (extractors or verifiers):
            raise distill_mod.BackendUnavailable(
                f"no distill endpoint up — start one, then re-run {label} (the "
                "build resumes from its scratch)")
        # single-tier note: verify tier absent, or the SAME server+model as extract
        # (self-verification collapses to single-tier).  Lower quality, not a failure —
        # a shared build should record it so its provenance is honest.
        _eid = lambda es: {(getattr(e, "url", None), getattr(e, "model", None)) for e in es}
        stats["single_tier"] = (not verifiers) or _eid(verifiers) <= _eid(extractors)

        _emit(report, "distill", chunks=chunks)
        say(f"{label}: distilling {chunks} chunk(s) into concepts / relations / cards "
            "(the long step — progress lines follow) …")
        t0 = time.time()
        stats["distill"] = distill_mod.distill_corpus(
            store, kb, extractors or verifiers, embedder, scfg,
            verifiers=verifiers if extractors else None,
            progress=_distil_progress(report))
        say(f"{label}: distilled {stats['distill']} ({time.time() - t0:.1f}s)")

        # RECARD RECOVERY — the crux for a thrown-away scratch.  A chunk truncated at
        # max_tokens keeps its distil stamp but SKIPS its recard stamp, deferring the
        # rest of its cards to "the recard pass".  On the master that pass runs later;
        # here the scratch is deleted at export, so those cards would be lost forever.
        # Run it now (all_families re-mines the truncated chunks; a no-op when nothing
        # truncated) so the bundle is complete before it ships.
        _emit(report, "recard")
        try:
            stats["recard"] = distill_mod.recard_corpus(
                store, kb, verifiers or extractors, embedder, scfg, all_families=True)
            rec = (stats["recard"] or {}).get("chunks", 0)
            if rec:
                say(f"{label}: recovered cards for {rec} truncated chunk(s) ({stats['recard']})")
        except distill_mod.BackendUnavailable as e:
            stats["recard_incomplete"] = True
            log.warning("%s: card recovery (recard) could not finish — the bundle may be "
                        "missing cards from truncated chunks; re-run %s to recover: %s",
                        label, label, e)

        # completeness safety net (≈0 on the success path — an interruption RAISES and
        # is handled by the caller; this catches any residual undistilled chunk).
        try:
            stats["undistilled"] = max(0, store.count() - kb.counts().get("distilled_chunks", 0))
        except Exception:
            stats["undistilled"] = 0

        _emit(report, "link", **{k: stats["distill"].get(k) for k in ("concepts", "cards")})
        lm = (verifiers or extractors)[0]
        try:
            say(f"{label}: linking concepts …")
            t0 = time.time()
            stats["link"] = link_mod.link_concepts(kb, lm, scfg, lease=lm_lease.BIG)
            say(f"{label}: linked {stats['link']} ({time.time() - t0:.1f}s)")
        except Exception as e:                  # linkage is a bonus, not a gate
            say(f"{label}: link skipped: {e}")

        # cross-reference graph for a structured (scripture/legal) doc, if confirmed —
        # built here so a shared bundle ships WITH its citation/commentary edges.
        if (profile and profile.get("ingest_as") == "structured"
                and profile.get("build_citations", True)):
            try:
                from . import citations as cite_mod
                stats["citations"] = cite_mod.build(store, kb, scfg, log=log)
                say(f"{label}: cross-reference graph — {stats['citations']}")
            except Exception as e:              # the graph is a bonus, not a gate
                say(f"{label}: citations skipped: {e}")
    finally:
        kb.close()
        store.close()
    return stats


# ── packaging (§3.1 steps 5–8) ───────────────────────────────────────────────
def _copy_tables(src, dst) -> None:
    """Copy every user table (schema + rows) between two open connections —
    the transport form both encryption directions share."""
    for row in src.execute("SELECT name, sql FROM sqlite_master "
                           "WHERE type='table' AND name NOT LIKE 'sqlite_%'"):
        name, ddl = row[0], row[1]
        if not ddl:
            continue
        dst.execute(ddl)
        cols = [r[1] for r in src.execute(f"PRAGMA table_info({name})")]
        ph = ",".join("?" for _ in cols)
        dst.executemany(f"INSERT INTO {name} VALUES({ph})",
                        (tuple(r) for r in src.execute(f"SELECT * FROM {name}")))
    dst.commit()


def _encrypt_copy(plain: Path, enc: Path, key: str) -> None:
    """SQLCipher copy of a plain .kdb (tables only — transport form)."""
    if enc.exists():
        enc.unlink()
    src = bundles._connect(str(plain))
    dst = dbcrypt.connect(str(enc), encrypted=True, key=key)
    try:
        _copy_tables(src, dst)
    finally:
        src.close()
        dst.close()


def unwrap_encrypted(path: str, key: str, out_dir: str) -> str:
    """Open an encrypted pack and produce a PLAIN file in `out_dir`: either the
    sealed inner file (compressed artifacts ride a one-table blob container) or
    a plain copy of the encrypted database.  Wrong key → a named error."""
    try:
        con = dbcrypt.connect(str(path), encrypted=True, key=key)
        try:
            tables = {r[0] for r in con.execute(
                "SELECT name FROM sqlite_master WHERE type='table'")}
            if "pack_blob" in tables:            # sealed bytes (the compressed case)
                name, data = con.execute("SELECT name, data FROM pack_blob").fetchone()
                out = Path(out_dir) / name
                out.write_bytes(data)
                return str(out)
            out = Path(out_dir) / "pack-plain.kdb"
            import sqlite3 as _sq
            dst = _sq.connect(str(out))
            try:
                _copy_tables(con, dst)
            finally:
                dst.close()
            return str(out)
        finally:
            con.close()
    except ValueError:
        raise
    except Exception as e:
        raise ValueError(f"could not decrypt {Path(path).name} — wrong passphrase, "
                         f"missing SQLCipher driver, or a corrupted pack ({e})")


def build_pack(cfg: dict, path: str, *, name: str | None = None, title: str = "",
               author: str = "", contact: str = "", version: str = "1.0.0",
               describe: str = "", license_override: str = "",
               allow_unlicensed: bool = False, compress: bool = False,
               encrypt: bool = False, passphrase: str = "", keep_build: bool = False,
               force: bool = False, log_fn=None) -> dict:
    """§3 end to end.  Returns {artifact, sidecar, manifest, stats}."""
    say = log_fn or log.info
    src = Path(path).expanduser()
    if not src.exists():
        raise ValueError(f"no such file or folder: {src}")
    if not _SEMVER.match(version):
        raise ValueError(f"--pack-version must be semver (got {version!r})")
    slug = bundles._slug(name or src.stem)
    pack_dir = Path(cfg.get("pack_dir") or "packs") / slug
    ext = ".kdb" + (".gz" if compress else "") + (".enc" if encrypt else "")
    artifact = pack_dir / f"{slug}-{version}{ext}"
    clash = list(pack_dir.glob(f"{slug}-{version}.kdb*")) if pack_dir.is_dir() else []
    clash = [c for c in clash if not c.name.endswith(".manifest.json")]
    if clash and not force:
        raise ValueError(f"{clash[0].name} already exists — bump --pack-version "
                         "or pass --force to replace it")
    if encrypt:
        if not dbcrypt.available():
            raise ValueError("pack encryption needs the SQLCipher driver "
                             "(pip install sqlcipher3-binary)")
        passphrase = passphrase or os.environ.get("PACK_PASSPHRASE", "")
        if not passphrase:
            raise ValueError("--encrypt needs a passphrase (PACK_PASSPHRASE env) "
                             "— refusing an empty key")

    build = Path(cfg.get("pack_build_dir") or "var/packs/build") / slug
    build.mkdir(parents=True, exist_ok=True)
    scfg = scratch_cfg(cfg, build, src)
    say(f"pack build: clean-room at {build} (master untouched)")
    stats = _pipeline(scfg, src, say)

    # license gate over the scratch registry (§5)
    con = bundles._connect(scfg["kb_path"])
    try:
        rows = [dict(r) for r in con.execute(
            "SELECT doc_id,title,license,license_holder,license_url FROM source_registry")]
        if not rows:
            raise ValueError("nothing was ingested — no sources in the scratch kb "
                             f"(is {src} a supported format?)")
        g = gate(rows, override=license_override, allow_unlicensed=allow_unlicensed)
        if not g["ok"]:
            raise ValueError("license gate refused the pack:\n" + gate_table(g))
        if g["filled"]:
            ph = ",".join("?" for _ in g["filled"])
            con.execute(f"UPDATE source_registry SET license=? WHERE doc_id IN ({ph}) "
                        "AND (license IS NULL OR license='')",
                        (license_override, *g["filled"]))
            con.commit()
            say(f"license --{license_override} attested onto {len(g['filled'])} "
                "source(s) with no detected license")
        if not g["shareable"]:
            say("pack will be stamped shareable:false (unlicensed sources) — "
                "private use only")
    finally:
        con.close()

    # export the scratch as one .kdb, then upgrade its manifest to format 2
    exported = bundles.split(scfg, out_dir=str(build), force=True)
    if len(exported) != 1:
        raise ValueError(f"clean-room build produced {sorted(exported)} bundles — "
                         "expected exactly one; check source folder bundle mappings")
    kdb = Path(next(iter(exported.values()))["file"])
    con = bundles._connect(str(kdb))
    try:
        counts = (bundles.read_manifest(con) or {}).get("counts") or {}
        man = manifest_v2(con, slug=slug, version=version, cfg=cfg, gate_res=g,
                          meta={"title": title, "author": author,
                                "contact": contact, "describe": describe},
                          counts=counts)
        write_manifest_v2(con, man)
        con.execute("VACUUM")
        con.commit()
    finally:
        con.close()

    # package: compress → encrypt → atomic move → sidecar LAST (§3.2, §6, §7)
    work = kdb
    if compress:
        gz = kdb.with_suffix(kdb.suffix + ".gz")
        with open(kdb, "rb") as fi, gzip_mod.open(gz, "wb") as fo:
            shutil.copyfileobj(fi, fo)
        work = gz
    if encrypt:
        if compress:                             # compress-then-encrypt: wrap the gz bytes
            enc = work.with_suffix(work.suffix + ".enc")
            _seal_bytes(work, enc, passphrase)
        else:
            enc = work.with_suffix(work.suffix + ".enc")
            _encrypt_copy(work, enc, passphrase)
        work = enc
    pack_dir.mkdir(parents=True, exist_ok=True)
    tmp = pack_dir / (artifact.name + ".tmp")
    shutil.copyfile(work, tmp)
    os.replace(tmp, artifact)
    sidecar = Path(str(artifact) + ".manifest.json")
    sidecar.write_text(json.dumps(man, indent=2, ensure_ascii=False))
    say(f"pack ready: {artifact} ({artifact.stat().st_size} bytes) + sidecar")

    if not keep_build:
        shutil.rmtree(build, ignore_errors=True)
    return {"artifact": str(artifact), "sidecar": str(sidecar),
            "manifest": man, "stats": stats}


def _plain_target(target: Path) -> None:
    """A collection file is a plain, mergeable .kdb — reject compress/encrypt
    wrappers (those are for `pack`, the one-shot publish artifact)."""
    if target.suffix in (".gz", ".enc") or target.name.endswith((".gz", ".enc")):
        raise ValueError("a collection target must be a plain .kdb (it grows by "
                         "merge) — compress/encrypt when you publish it with `pack`")


def _structure_preview(cfg: dict, doc: str) -> dict:
    """Analyze a single document and, if it has canonical structure worth confirming,
    return the plain-language questions for the wizard.  Never writes; never fatal —
    any problem degrades to 'no questions' so the build can still proceed normally."""
    try:
        d = Path(doc).expanduser()
        if not d.is_file():
            return {}
        from . import ingest as ingest_mod, structure as S
        prof = ingest_mod.analyze_doc(cfg, str(d))
        qs = S.questions_for(prof)
        return {"kind": prof.get("kind"), "confidence": prof.get("confidence", 0),
                "scheme": prof.get("scheme"), "unit": prof.get("unit"),
                "warnings": prof.get("warnings", []),
                "needs_confirm": bool(qs), "questions": qs}
    except Exception as e:                          # analysis is advisory, never a gate
        return {"error": str(e)}


def collection_preview(cfg: dict, target: str, bundle: str, doc: str = "") -> dict:
    """Cheap, side-effect-free look at a collection target for the wizard: does it
    exist, what bundle does it already hold, and is that compatible with `bundle`
    (one bundle per file)?  When `doc` is given, also analyze it and attach any
    structured-text confirmation questions.  Never runs the pipeline."""
    tgt = Path(target).expanduser()
    bundle_slug = bundles._slug(bundle)
    out = {"ok": True, "target": str(tgt), "bundle": bundle, "exists": tgt.exists(),
           "compatible": True, "counts": {}, "file_bundle": ""}
    if doc:
        out["structure"] = _structure_preview(cfg, doc)
    if not bundle_slug:
        return {"ok": False, "error": "a bundle name is required"}
    try:
        _plain_target(tgt)
    except ValueError as e:
        return {"ok": False, "error": str(e)}
    if not tgt.exists():
        return out
    try:
        info = bundles.inspect_bundle_file(str(tgt))
    except ValueError as e:
        return {"ok": False, "error": f"{tgt.name} exists but is not a readable "
                f"knowledge bundle: {e}"}
    held = sorted({bundles._slug(s.get("bundle") or "") for s in info.get("sources", [])
                   if s.get("bundle")})
    file_bundle = (info.get("manifest") or {}).get("name") or (held[0] if held else "")
    out["counts"] = info.get("counts") or {}
    out["file_bundle"] = file_bundle
    # one bundle per file: any held group other than this one blocks the add
    others = [h for h in held if h and h != bundle_slug]
    if others:
        out["compatible"] = False
        out["error"] = (f"{tgt.name} already holds bundle "
                        f"'{others[0]}' — one bundle per file; choose a different "
                        "file, or match that bundle name")
    return out


def add_to_collection(cfg: dict, doc: str, target: str, bundle: str, *,
                      license_override: str = "", allow_unlicensed: bool = False,
                      force: bool = False, dry_run: bool = False, log_fn=None,
                      report=None, answers: dict | None = None) -> dict:
    """Clean-room ingest+distill of ONE document (or folder), then ADD its distilled
    closure to a shareable ``.kdb`` collection under ``bundle`` — creating the file
    or merging into it if it already exists (content-hash ids ⇒ idempotent, so
    re-adding the same doc is a no-op).  One bundle per file: a target that already
    holds a different bundle is refused.  The master kb is NEVER touched (a scratch
    build, like `pack`).  ``dry_run`` validates + previews without running anything.

    Returns {ok, target, bundle, created, added, totals, shareable, stats}."""
    say = log_fn or log.info
    src = Path(doc).expanduser()
    if not src.exists():
        raise ValueError(f"no such file or folder: {doc}")
    bundle_slug = bundles._slug(bundle)
    if not bundle_slug:
        raise ValueError("a bundle name is required (it groups the shared knowledge)")
    tgt = Path(target).expanduser()
    _plain_target(tgt)

    prev = collection_preview(cfg, str(tgt), bundle)
    if not prev.get("ok") or not prev.get("compatible"):
        raise ValueError(prev.get("error", "target is not compatible"))
    if dry_run:
        return {"ok": True, "dry_run": True, "target": str(tgt), "bundle": bundle,
                "created": not tgt.exists(), "current": prev.get("counts") or {},
                "file_bundle": prev.get("file_bundle") or ""}

    # a structured document (scripture/legal) that the user CONFIRMED is ingested one
    # canonical unit at a time; anything else takes the ordinary heading-chunk path.
    profile = None
    if src.is_file() and answers:
        try:
            from . import ingest as ingest_mod
            profile = ingest_mod.confirm_profile(cfg, str(src), answers)
            if profile.get("ingest_as") == "structured":
                say(f"collect: will ingest {src.name} as structured {profile.get('kind')} "
                    "(confirmed) — one node per canonical unit")
        except Exception as e:
            say(f"collect: could not apply the confirmation answers ({e}) — ingesting normally")
            profile = None

    build = Path(cfg.get("pack_build_dir") or "var/packs/build") / f"collect-{bundle_slug}"
    build.mkdir(parents=True, exist_ok=True)
    scfg = scratch_cfg(cfg, build, src)
    say(f"collect: clean-room build at {build} (your master KB is untouched)")
    stats = _pipeline(scfg, src, say, label="collect", report=report, profile=profile)

    # license gate over the scratch registry (same rules as `pack`), then stamp the
    # whole scratch into the one bundle so `split` emits exactly it.
    con = bundles._connect(scfg["kb_path"])
    try:
        rows = [dict(r) for r in con.execute(
            "SELECT doc_id,title,license,license_holder,license_url FROM source_registry")]
        if not rows:
            raise ValueError("nothing was ingested — no sources in the scratch kb "
                             f"(is {src} a supported format?)")
        g = gate(rows, override=license_override, allow_unlicensed=allow_unlicensed)
        if not g["ok"]:
            raise ValueError("license gate refused the collection:\n" + gate_table(g))
        if g["filled"]:
            ph = ",".join("?" for _ in g["filled"])
            con.execute(f"UPDATE source_registry SET license=? WHERE doc_id IN ({ph}) "
                        "AND (license IS NULL OR license='')",
                        (license_override, *g["filled"]))
            say(f"license --{license_override} attested onto {len(g['filled'])} source(s)")
        if not g["shareable"]:
            say("collection carries unlicensed sources — mark a --license before you share")
        con.execute("UPDATE source_registry SET bundle=?", (bundle,))
        con.commit()
    finally:
        con.close()

    _emit(report, "export")
    say(f"collect: exporting the '{bundle}' bundle closure …")
    exported = bundles.split(scfg, out_dir=str(build), force=True, only={bundle})
    entry = exported.get(bundle) or {}
    kdb = entry.get("file")
    if not kdb or "counts" not in entry:
        raise ValueError("clean-room export produced nothing to add "
                         "(the document distilled to no shareable knowledge)")
    kdb = Path(kdb)

    tgt.parent.mkdir(parents=True, exist_ok=True)
    tmp = tgt.with_suffix(tgt.suffix + ".tmp")
    if tmp.exists():
        tmp.unlink()
    created = not tgt.exists()
    if created:
        shutil.copyfile(kdb, tmp)                       # fresh file = the export itself
        dst = bundles._connect(str(tmp))
        try:
            totals = {t: dst.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
                      for t in bundles.KNOWLEDGE_TABLES if bundles._has_table(dst, t)}
            bundles.write_manifest(dst, bundle, totals, cfg)   # re-stamp name=bundle
            dst.commit()
        finally:
            dst.close()
        added = dict(totals)
    else:
        shutil.copyfile(tgt, tmp)                       # merge into a copy, then swap
        dst = bundles._connect(str(tmp))
        srcc = bundles._connect(str(kdb))
        try:
            tabs = [t for t in bundles.KNOWLEDGE_TABLES if bundles._has_table(dst, t)]
            before = {t: dst.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0] for t in tabs}
            bundles.merge_db(srcc, dst, intersect=True)   # INSERT OR IGNORE (content-hash)
            totals = {t: dst.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0] for t in tabs}
            # honest "added" = rows that were actually NEW (re-adding a doc → {} )
            added = {t: totals[t] - before[t] for t in tabs if totals[t] - before[t] > 0}
            bundles.write_manifest(dst, bundle, totals, cfg)
            dst.commit()
        finally:
            srcc.close()
            dst.close()
    os.replace(tmp, tgt)

    undistilled = stats.get("undistilled", 0)
    recard_incomplete = bool(stats.get("recard_incomplete"))
    single_tier = bool(stats.get("single_tier"))
    recovered = (stats.get("recard") or {}).get("chunks", 0)
    complete = undistilled == 0 and not recard_incomplete
    # Keep the scratch when the build couldn't finish, so re-running the SAME collect
    # resumes distil/recard from the checkpoint and the rest merges in idempotently.
    if complete:
        shutil.rmtree(build, ignore_errors=True)
    else:
        say(f"collect: build INCOMPLETE — "
            + (f"{undistilled} chunk(s) still undistilled" if undistilled else "card recovery unfinished")
            + f"; scratch kept at {build}. Re-run the same collect to finish (it resumes); "
              "the remainder merges into the file.")

    say(f"collect: {'created' if created else 'updated'} {tgt} [{bundle}] — "
        f"added {added or 'nothing new'}; file now holds {totals}"
        + (f"; recovered cards for {recovered} truncated chunk(s)" if recovered else "")
        + ("; distilled SINGLE-TIER (no independent verify — quality note)" if single_tier else ""))
    _emit(report, "done", created=created, added=added, target=str(tgt), bundle=bundle,
          complete=complete, single_tier=single_tier)
    return {"ok": True, "target": str(tgt), "bundle": bundle, "created": created,
            "added": added, "totals": totals, "shareable": g["shareable"],
            "complete": complete, "single_tier": single_tier,
            "recovered_truncated": recovered, "undistilled": undistilled, "stats": stats}


def _seal_bytes(src: Path, dst: Path, key: str) -> None:
    """Encrypt an arbitrary file's bytes into a one-table SQLCipher container
    (used for compressed artifacts, whose bytes aren't a database)."""
    if dst.exists():
        dst.unlink()
    con = dbcrypt.connect(str(dst), encrypted=True, key=key)
    try:
        con.execute("CREATE TABLE pack_blob(name TEXT, data BLOB)")
        con.execute("INSERT INTO pack_blob VALUES(?,?)", (src.name, src.read_bytes()))
        con.commit()
    finally:
        con.close()
