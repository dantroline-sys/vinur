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


def _pipeline(scfg: dict, src: Path, say) -> dict:
    """ingest → distill → link inside the scratch.  Module-level so tests can
    patch it; raises BackendUnavailable when no distill endpoint is up (the
    scratch survives — re-running resumes)."""
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
            say("embed endpoint unreachable — pack will carry no vectors "
                "(importers re-embed; retrieval inside the build is sparse-only)")
            # keep going: ingest/distill degrade the same way the normal ops do
        if src.is_file():
            n = ingest_mod.ingest_file(store, embedder, scfg, str(src))
            stats["ingest"] = {"docs": 1 if n else 0, "chunks": n}
        else:
            stats["ingest"] = ingest_mod.crawl(store, embedder, scfg)
        say(f"pack ingest: {stats['ingest']}")

        extractors = distill_mod.fast_endpoints(scfg, log)
        verifiers = distill_mod.verify_endpoints(scfg, log)
        if not (extractors or verifiers):
            raise distill_mod.BackendUnavailable(
                "no distill endpoint up — start one, then re-run pack (the "
                "build resumes from its scratch)")
        stats["distill"] = distill_mod.distill_corpus(
            store, kb, extractors or verifiers, embedder, scfg,
            verifiers=verifiers if extractors else None)
        say(f"pack distill: {stats['distill']}")

        lm = (verifiers or extractors)[0]
        try:
            stats["link"] = link_mod.link_concepts(kb, lm, scfg, lease=lm_lease.BIG)
            say(f"pack link: {stats['link']}")
        except Exception as e:                  # linkage is a bonus, not a gate
            say(f"pack link skipped: {e}")
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
