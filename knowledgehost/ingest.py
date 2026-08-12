"""Offline ingestion pipeline — heavy, batch, run on demand / monthly.

A **manifest** (path, content_hash, mtime, version, status) makes every run
**incremental**: only new/changed files are (re)processed.  Per source we
extract ``(section, text)`` blocks, chunk them by heading (~200-400 tokens),
embed the chunks via the nomic endpoint (``search_document:`` prefix), and
upsert into the store under a stable id (idempotent re-ingest).

Security: filenames are treated as **opaque data** — never interpolated into a
shell or an LM prompt (the file-scraper injection surface).  All extracted text
is sanitized before storage.  Parse runs need no network.
"""
from __future__ import annotations

import hashlib
import logging
import os
import shutil
import time

from . import research, sanitize
from .chunk import chunk_blocks, chunk_id
from .sources import extractor_for, MissingDependency

log = logging.getLogger("knowledgehost.ingest")


class EmbedUnavailable(Exception):
    """The embed endpoint dropped mid-ingest.  On the lance backend (no sparse-only
    fallback) we abort rather than silently drop chunks and mark the doc done."""


def _content_hash(path: str) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as fh:
        for blk in iter(lambda: fh.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def _store_records(store, embedder, cfg, *, source_type, title, path_or_url, items, version):
    """Embed (in batches) and upsert one row per item, NO re-chunking.  `items` yields
    (section, text, tokens) already at the granularity we want stored — heading-chunks
    for ordinary docs, one addressable unit (a verse / a section) for structured ones.
    Returns the row count."""
    title = sanitize.clean(title, 300)
    batch_text, batch_rec = [], []
    n = 0
    # When embedding, the batch is bounded by the embed endpoint / GPU memory.  When NOT
    # embedding (the lexical library), commit in big transactions instead — far fewer fsyncs.
    flush_at = cfg["embed_batch"] if embedder else int(cfg.get("ingest_write_batch", 1000))

    def flush():
        nonlocal batch_text, batch_rec
        if not batch_rec:
            return
        vecs = embedder.embed_many(batch_text, "document") if embedder else None
        if embedder and vecs is None:
            # Transport failure mid-ingest — usually the embed server being
            # restarted after a leak (Vinkona's watchdog).  Wait it out and
            # retry, so this batch keeps its vectors instead of silently going
            # sparse-only (sqlite) or aborting the document (lance).
            wait_s = float(cfg.get("embed_recover_wait_s", 300) or 0)
            if wait_s and getattr(embedder, "ever_ok", False) and embedder.wait_ready(wait_s):
                vecs = embedder.embed_many(batch_text, "document")
        # lance stores nothing without a vector; a whole-batch None means the
        # endpoint stayed down — abort (resumable) so we don't drop chunks AND
        # mark the doc done.  sqlite keeps them sparse-only, so it can proceed.
        if embedder and vecs is None and cfg.get("backend") == "lance":
            raise EmbedUnavailable("embed endpoint dropped mid-ingest")
        for rec, vec in zip(batch_rec, vecs or [None] * len(batch_rec)):
            rec["vector"] = vec
        store.add_chunks(batch_rec)
        batch_text, batch_rec = [], []

    for section, text, tokens in items:
        text = sanitize.clean(text)
        if not text:
            continue
        section = sanitize.clean(section, 300)
        rec = {
            "id": chunk_id(path_or_url, section, text),
            "source_type": source_type, "title": title, "section": section,
            "path_or_url": path_or_url, "text": text,
            "tokens": tokens, "version": version, "ingested_at": time.time(),
        }
        batch_rec.append(rec)
        batch_text.append(text)
        n += 1
        if len(batch_rec) >= flush_at:
            flush()
    flush()
    return n


def _embed_and_store(store, embedder, cfg, *, source_type, title,
                     path_or_url, blocks, version):
    """Chunk (by heading, ~200-400 tokens) -> embed -> upsert.  Returns chunk count."""
    items = ((ch["section"], ch["text"], ch["tokens"]) for ch in chunk_blocks(blocks, cfg))
    return _store_records(store, embedder, cfg, source_type=source_type, title=title,
                          path_or_url=path_or_url, items=items, version=version)


def ingest_file(store, embedder, cfg, path: str, *, force=False, collection=None,
                profile=None) -> int:
    """Ingest one document if new/changed.  Returns chunks added (0 if skipped).
    `collection` (library ingest) tags the chunks' source_type with a topical bucket
    (science/fiction/…) instead of the format, so search can filter by it.  `profile`
    is a CONFIRMED structure profile (structure.apply_answers) — when present and
    `ingest_as=='structured'` the doc is ingested one canonical unit (verse/section)
    per chunk instead of by heading; `ingest_as=='plain'` just takes the normal path."""
    ext = os.path.splitext(path)[1].lower()
    vinkona = (collection is None) and research.is_research_doc(path)   # not for library docs
    fn = None if vinkona else extractor_for(path)
    if fn is None and not vinkona:
        return 0
    try:
        st = os.stat(path)
    except OSError:
        return 0

    prev = store.manifest.get(path)
    version = int(store.manifest.meta_get("version", "1"))
    # A file skipped for a missing parser dependency is RETRIED every run — the
    # dependency may have been installed since (./install.sh --pdf) and the file
    # itself hasn't changed, so the unchanged-skips would otherwise bury it
    # forever.  Retrying is free while the dep is still absent (the extractor
    # raises on import, before the file is even opened), so the content hash is
    # deferred until something actually parses.
    retry_dep = bool(prev) and prev["status"] == "missing_dep" and not force
    chash = None
    if not retry_dep:
        if prev and not force and abs(prev["mtime"] - st.st_mtime) < 1e-6:
            return 0                               # unchanged by mtime — cheap skip
        chash = _content_hash(path)
        if prev and not force and prev["content_hash"] == chash:
            store.manifest.set(path, chash, st.st_mtime, version, "ok")
            return 0                               # mtime moved but bytes identical

    if vinkona:
        if chash is None:
            chash = _content_hash(path)
        return _ingest_research_doc(store, embedder, cfg, path, version, chash, st)

    if profile and profile.get("confirmed") and profile.get("ingest_as") == "structured":
        if chash is None:
            chash = _content_hash(path)
        n = _ingest_structured_doc(store, embedder, cfg, path, version, chash, st, profile)
        if n is not None:
            return n
        # 0 units parsed under the confirmed profile — fall through to normal ingest so
        # the document isn't silently lost (a mis-confirmation shouldn't drop content).

    try:
        title, blocks = fn(path, cfg)
    except MissingDependency as e:
        if retry_dep:
            return 0            # still missing — the manifest row already says so
        log.warning("skip %s — %s", os.path.basename(path), e)
        store.manifest.set(path, chash, st.st_mtime, version, "missing_dep")
        return 0
    except Exception as e:
        log.warning("failed to parse %s: %s", os.path.basename(path), e)
        if chash is None:
            chash = _content_hash(path)
        store.manifest.set(path, chash, st.st_mtime, version, "error")
        return 0

    if retry_dep:
        log.info("previously-skipped %s parses now (dependency installed)",
                 os.path.basename(path))
    if chash is None:
        chash = _content_hash(path)
    store.delete_by_path(path)                     # re-ingest cleanly if changed
    source_type = collection or {".pdf": "pdf", ".epub": "epub", ".html": "html",
                                 ".htm": "html"}.get(ext, "text")
    n = _embed_and_store(store, embedder, cfg, source_type=source_type,
                         title=title, path_or_url=path, blocks=blocks,
                         version=version)
    store.manifest.set(path, chash, st.st_mtime, version, "ok")
    return n


def collection_for(cfg, root: str, path: str, default: str | None = None) -> str:
    """Topical collection for a library doc: an explicit `library_collections` folder map
    (bare key matches a path SEGMENT, glob matches the whole path; first match wins) wins;
    else `default` (set when the crawl root is itself a chosen collection folder — the whole
    subtree is one collection); else the doc's top folder under its library root; else 'library'."""
    p = str(path).replace("\\", "/")
    mapping = cfg.get("library_collections") or {}
    if isinstance(mapping, dict):
        pl = p.lower()
        segs = {s for s in pl.split("/") if s}
        import fnmatch as _fn
        for key, coll in mapping.items():
            k = str(key).lower()
            if k in segs or _fn.fnmatch(pl, k):
                return str(coll)
    if default:
        return default
    try:
        rel = os.path.relpath(path, root)
        top = rel.replace("\\", "/").split("/")[0]
        if top and top not in (".", ".."):
            return top.lower()
    except ValueError:
        pass
    return "library"


def _parse_job(path, cfg, collection, prev_chash):
    """Worker-process job: content-hash + parse ONE file to (title, blocks).  Touches NO
    database — the store write happens back on the single main thread — so it's safe to fan
    out across a process pool.  Returns a small status dict the writer applies."""
    try:
        st = os.stat(path)
        chash = _content_hash(path)
    except OSError:
        return {"path": path, "status": "gone"}
    base = {"path": path, "collection": collection, "chash": chash, "mtime": st.st_mtime}
    if prev_chash and chash == prev_chash:
        return {**base, "status": "unchanged"}             # bytes identical → skip parse+write
    fn = extractor_for(path)
    if fn is None:
        return {**base, "status": "unsupported"}
    try:
        title, blocks = fn(path, cfg)
    except MissingDependency as e:
        return {**base, "status": "missing_dep", "err": str(e)}
    except Exception as e:                                  # pragma: no cover - per-file guard
        return {**base, "status": "error", "err": str(e)}
    return {**base, "status": "ok", "title": title, "blocks": blocks}


def _apply_parsed(store, embedder, cfg, res, version) -> int:
    """Main-thread writer for one parsed file: set its manifest status and upsert its chunks.
    The single point that touches the store, so the SQLite single-writer rule is preserved."""
    path, status = res["path"], res["status"]
    if status == "gone":
        return 0
    if status == "unchanged":                              # mtime moved but bytes identical
        store.manifest.set(path, res["chash"], res["mtime"], version, "ok")
        return 0
    if status == "unsupported":
        return 0
    if status in ("missing_dep", "error"):
        log.warning("%s %s: %s",
                    "skip" if status == "missing_dep" else "failed to parse",
                    os.path.basename(path), res.get("err"))
        store.manifest.set(path, res["chash"], res["mtime"], version, status)
        return 0
    store.delete_by_path(path)                             # clean re-ingest if changed
    n = _embed_and_store(store, embedder, cfg, source_type=res["collection"],
                         title=res["title"], path_or_url=path, blocks=res["blocks"],
                         version=version)
    store.manifest.set(path, res["chash"], res["mtime"], version, "ok")
    return n


def _skip_dirs(cfg) -> set:
    """Realpath'd folders the crawl must never descend into.  Currently the
    quarantine dir: `clear-queue` typically MOVES cleared files into a
    `quarantined/` subfolder INSIDE a source root, so the crawl has to skip it or
    it would just re-ingest everything it was asked to revert."""
    q = str(cfg.get("quarantine_dir") or "").strip()
    return {os.path.realpath(os.path.expanduser(q))} if q else set()


def _walk_pruned(root, skip: set):
    """os.walk(root) that never descends into any realpath in `skip` (mutating
    the dirs list in place, the way os.walk expects, before it recurses)."""
    for dirpath, dirs, files in os.walk(root):
        if skip:
            dirs[:] = [d for d in dirs
                       if os.path.realpath(os.path.join(dirpath, d)) not in skip]
        yield dirpath, dirs, files


def crawl_library(store, embedder, cfg, *, force=False) -> dict:
    """Index the search-only library (library_sources) into its OWN store — lexical FTS
    by default (embedder passed only when library_dense), NOT distilled.  Each doc is
    tagged with its topical `collection`.  Parsing fans out across a process pool
    (``ingest_workers``) while the DB is written from this one thread; the FTS index is
    optimised at the end.  The cheap tier that feeds Vinkona's research loop a local 'google'."""
    exts = set(cfg["extensions"])
    every = cfg["ingest_log_every"]
    version = int(store.manifest.meta_get("version", "1"))
    lib_root = os.path.realpath(cfg["library_root"]) if cfg.get("library_root") else ""
    skip = _skip_dirs(cfg)                                 # never re-crawl the quarantine dir

    # 1. enumerate work: cheap mtime-skip here; the byte-identical skip is done in the worker
    #    (which is reading the file to hash it anyway).
    jobs = []                                              # (path, collection, prev_content_hash)
    for root in cfg.get("library_sources") or []:
        if not os.path.isdir(root):
            log.info("library root missing, skipping: %s", root)
            continue
        # When the crawl root is a direct subfolder of the trusted library_root (the web
        # Library panel writes exactly these), the whole subtree is ONE collection named for
        # that subfolder — so nested structure doesn't splinter the corpus firewall.
        base_coll = None
        if lib_root and os.path.realpath(os.path.dirname(os.path.normpath(root))) == lib_root:
            base_coll = os.path.basename(os.path.normpath(root)).lower()
        for dirpath, _dirs, files in _walk_pruned(root, skip):
            for name in files:
                if os.path.splitext(name)[1].lower() not in exts:
                    continue
                path = os.path.join(dirpath, name)
                try:
                    st = os.stat(path)
                except OSError:
                    continue
                prev = store.manifest.get(path)
                # missing_dep rows re-parse every run — the dependency may have
                # been installed since (see ingest_file for the reasoning).
                retry_dep = bool(prev) and prev["status"] == "missing_dep"
                if prev and not force and not retry_dep \
                        and abs(prev["mtime"] - st.st_mtime) < 1e-6:
                    continue                               # unchanged by mtime — cheap skip
                coll = collection_for(cfg, root, path, default=base_coll)
                prev_chash = None if (force or retry_dep) \
                    else (prev["content_hash"] if prev else None)
                jobs.append((path, coll, prev_chash))

    docs = chunks = 0
    by_collection: dict = {}

    def _handle(res):
        nonlocal docs, chunks
        added = _apply_parsed(store, embedder, cfg, res, version)
        if added:
            docs += 1
            chunks += added
            c = res.get("collection") or "library"
            by_collection[c] = by_collection.get(c, 0) + 1
        if docs and every and docs % every == 0:
            log.info("library … %d docs / %d chunks", docs, chunks)

    # 2. parse in parallel (I/O + CPU + any OCR), write serially (SQLite is single-writer).
    workers = int(cfg.get("ingest_workers", 0) or 0) or (os.cpu_count() or 1)
    if workers > 1 and len(jobs) > 1:
        import multiprocessing as _mp
        from concurrent.futures import ProcessPoolExecutor, as_completed
        # 'fork' (not 3.14's default forkserver): the ingest CLI is single-threaded here, so
        # fork is safe AND avoids re-importing the whole package per worker.  Workers never
        # touch the DB, so the parent's inherited sqlite connection is harmless.
        try:
            ctx = _mp.get_context("fork")
        except ValueError:                                # non-posix — take the platform default
            ctx = _mp.get_context()
        log.info("library: parsing %d file(s) across %d worker(s)", len(jobs), workers)
        with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as ex:
            futs = [ex.submit(_parse_job, p, cfg, c, h) for p, c, h in jobs]
            for fut in as_completed(futs):
                _handle(fut.result())
    else:
        for p, c, h in jobs:
            _handle(_parse_job(p, cfg, c, h))

    if hasattr(store, "optimize_fts"):
        store.optimize_fts()                              # merge FTS segments after the bulk load
    if hasattr(store, "build_stoplist"):
        store.build_stoplist()                            # LEARN over-reporting terms from the corpus
    return {"docs": docs, "chunks": chunks, "collections": by_collection}


def _ingest_research_doc(store, embedder, cfg, path, version, chash, st) -> int:
    """Ingest one of Vinkona's solved research drops (research_loop_spec §6): the
    ``## Sources`` blocks become the distillable chunks (source_type='vinkona'), and the
    doc's ``# Question`` / ``kb_query`` / provenance are stashed in doc_meta so the
    distiller frames extraction on the question and can close the gap that opened it."""
    try:
        question, blocks, meta = research.parse_research_doc(path)
    except Exception as e:
        log.warning("failed to parse research doc %s: %s", os.path.basename(path), e)
        store.manifest.set(path, chash, st.st_mtime, version, "error")
        return 0
    store.delete_by_path(path)
    title = question or os.path.splitext(os.path.basename(path))[0]
    n = _embed_and_store(store, embedder, cfg, source_type="vinkona",
                         title=title, path_or_url=path, blocks=blocks, version=version)
    # The per-doc frame the distiller reads back (merged into each chunk by iter_chunks).
    store.set_doc_meta(path, {
        "provenance": "vinkona", "bundle": "vinkona",
        "trust": float(cfg.get("vinkona_trust", 0.25)),
        "question": question or None,
        "kb_query": meta.get("kb_query"),
        "kind": meta.get("kind"),
        # card hints (brains): the shape Vinkona says her answer wants to be — the
        # distiller runs the matching typed extractor for the drop and seeds the
        # card's discriminators from the features.  A nudge, never authority.
        "card_type": meta.get("card_type"),
        "context_features": meta.get("context_features"),
    })
    store.manifest.set(path, chash, st.st_mtime, version, "ok")
    log.info("research drop ingested: %s (%d chunk(s))", os.path.basename(path), n)
    return n


def _structured_text(path, cfg) -> str:
    """Raw text for structure parsing.  Scripture/legal editions are line-oriented, so
    for text formats read the file directly (preserving line structure); otherwise fall
    back to the format extractor's blocks, joined (best-effort)."""
    ext = os.path.splitext(path)[1].lower()
    if ext in {".txt", ".text", ".md", ".markdown"}:
        try:
            with open(path, encoding="utf-8", errors="replace") as f:
                return f.read()
        except OSError:
            return ""
    fn = extractor_for(path)
    if fn is None:
        return ""
    try:
        _title, blocks = fn(path, cfg)
    except Exception:
        return ""
    return "\n".join(b.get("text", "") for b in blocks)


def _profile_maps(cfg, profile):
    """Reference maps for a structured ingest: the config's shipped maps plus the
    document's own ad-hoc, answer-derived aliases (structure.apply_answers)."""
    from . import structure
    return structure.load_reference_maps(cfg.get("reference_maps") or [],
                                         extra=(profile or {}).get("reference_map"))


def analyze_doc(cfg, path: str, *, kind_hint=None) -> dict:
    """Read a document and PROPOSE a structure profile (structure.analyze) using the
    config's reference maps — the shared entry point for the collect wizard's preview
    and the bulk-ingest confirm gate.  Read-only; never writes."""
    from . import structure
    text = _structured_text(path, cfg)
    maps = structure.load_reference_maps(cfg.get("reference_maps") or [])
    return structure.analyze(text, kind_hint=kind_hint, maps=maps)


def confirm_profile(cfg, path: str, answers: dict) -> dict:
    """analyze_doc → apply the user's confirmation answers → a CONFIRMED profile ready
    to pass to ingest_file(profile=…)."""
    from . import structure
    return structure.apply_answers(analyze_doc(cfg, path), answers or {})


_EDITION_SHORT = {"douay-rheims": "DRB"}


def _translation_label(profile, path) -> str:
    """A short label naming this edition/translation (KJV, DRB, …), so aligned verses are
    attributable.  From the confirmed edition when known, else the file's stem."""
    ed = profile.get("edition")
    eid = ed.get("id") if isinstance(ed, dict) else ed
    if eid:
        return _EDITION_SHORT.get(eid, str(eid).upper())
    stem = os.path.splitext(os.path.basename(path))[0]
    return (stem.strip() or "text").upper()[:24]     # translation sigla are conventionally caps


def _ingest_structured_doc(store, embedder, cfg, path, version, chash, st, profile):
    """Ingest a CONFIRMED scripture/legal document one canonical unit at a time: each
    verse / section becomes a single chunk whose `section` is its canonical key
    (bible:John.3.16 / usc:17/106), tagged source_type='scripture'|'legal'.  That key is
    the stable node identity a later citation-graph pass builds edges on, so units from
    DIFFERENT documents that name the same passage converge.  Returns the unit count, or
    None if nothing parsed (caller falls back to normal ingest — content is never lost)."""
    from . import structure
    text = _structured_text(path, cfg)
    if not text.strip():
        return None
    maps = _profile_maps(cfg, profile)
    kind = profile.get("kind")
    source_type = "scripture" if kind == "scripture" else "legal" if kind == "legal" else "text"
    units = list(structure.parse_units(text, profile, maps=maps))
    if not units:
        log.warning("structured ingest of %s parsed 0 units under the confirmed profile "
                    "— falling back to normal ingest", os.path.basename(path))
        return None
    store.delete_by_path(path)
    # the translation label rides the chunk title, so multiple editions of the same verse
    # (keyed identically) line up side by side, each named (KJV / DRB / …).
    title = _translation_label(profile, path)
    items = ((r.key, t, max(1, len(t.split()))) for r, t in units)
    n = _store_records(store, embedder, cfg, source_type=source_type, title=title,
                       path_or_url=path, items=items, version=version)

    # commentary layer (decided at the confirm step): store each interleaved note as a
    # chunk keyed by the verse it annotates (source_type='commentary'), so the citation
    # graph can link it — captured HERE, in one parse, not by re-reading the doc later.
    n_notes = 0
    if profile.get("layer_commentary") and kind == "scripture":
        notes = structure.parse_annotations(text, profile, maps=maps)
        if notes:
            note_items = ((anchor.key, note, max(1, len(note.split()))) for anchor, note in notes)
            n_notes = _store_records(store, embedder, cfg, source_type="commentary",
                                     title=title, path_or_url=path, items=note_items,
                                     version=version)

    store.set_doc_meta(path, {
        "structured": True, "kind": kind, "scheme": profile.get("scheme"),
        "translation": title, "edition": profile.get("edition"),
        "work": profile.get("work"), "book_order": profile.get("book_order"),
        "reference_map": profile.get("reference_map") or {},
        "extra_books": profile.get("extra_books") or [],
        "build_citations": profile.get("build_citations", True),
        "layer_commentary": bool(profile.get("layer_commentary")),
    })
    store.manifest.set(path, chash, st.st_mtime, version, "ok")
    log.info("structured %s ingest: %s (%d unit(s)%s, keyed canonically)",
             kind, os.path.basename(path), n,
             f", {n_notes} note(s)" if n_notes else "")
    return n


def _ingest_or_defer(store, embedder, cfg, path, *, force, confirm_exts, pend):
    """Bulk-crawl gate for a structured/ambiguous document.  Confirm-eligible files
    (plain text) are ANALYZED first: real scripture/legal structure is set aside into
    the 'Needs your input' inbox (once per profile) instead of being ingested on a
    guess; a standing answer for its profile ingests it unit-by-unit.  Everything else
    takes the ordinary path.  Never blocks — deferral is immediate.  Returns
    (chunks_added, wants_graph) — wants_graph is True when a confirmed STRUCTURED doc
    asked for the cross-reference graph."""
    def _structured(confirmed):
        added = ingest_file(store, embedder, cfg, path, force=True, profile=confirmed)
        return added, bool(confirmed.get("build_citations", True))

    if pend is None or os.path.splitext(path)[1].lower() not in confirm_exts:
        return ingest_file(store, embedder, cfg, path, force=force), False
    from . import structure
    try:
        st = os.stat(path)
    except OSError:
        return 0, False
    prev = store.manifest.get(path)
    unchanged = bool(prev) and not force and abs(prev["mtime"] - st.st_mtime) < 1e-6
    if unchanged and prev["status"] == "ok":
        return 0, False                            # already ingested, nothing changed
    if unchanged and prev["status"] == "deferred":
        confirmed = pend.answer_for_path(path)     # answered since it was set aside?
        if confirmed is None:
            return 0, False                        # still awaiting the user — no re-analysis
        return _structured(confirmed)
    # new or changed → analyze and decide
    prof = analyze_doc(cfg, path)
    if not structure.should_confirm(prof):
        return ingest_file(store, embedder, cfg, path, force=force), False
    sig = structure.profile_signature(prof)
    confirmed = pend.confirmed_profile(sig)        # this profile already confirmed?
    if confirmed is not None:
        return _structured(confirmed)
    # defer: file the questions once per profile, mark the doc awaiting input
    _rid, is_new = pend.defer(sig, prof.get("kind"), prof, structure.questions_for(prof), path)
    version = int(store.manifest.meta_get("version", "1"))
    store.manifest.set(path, _content_hash(path), st.st_mtime, version, "deferred")
    log.info("%s %s needs confirmation (%s) — set aside for your input%s",
             "structured document" if is_new else "another", os.path.basename(path),
             prof.get("kind"), " [new request]" if is_new else "")
    return 0, False


def crawl(store, embedder, cfg, *, force=False) -> dict:
    """Walk every configured source root and ingest supported files.  Vinkona's research
    outbox (research_solved_dir) is crawled too — its .md drops route to the vinkona path.
    Structured/ambiguous plain-text docs are set aside for confirmation (the 'Needs your
    input' inbox) rather than ingested on a guess — see `structured_confirm_exts`."""
    exts = set(cfg["extensions"])
    docs = chunks = 0
    every = cfg["ingest_log_every"]
    roots = list(cfg["sources"])
    solved = cfg.get("research_solved_dir")
    if solved and solved not in roots:
        roots.append(solved)                       # low-trust vinkona bundle (research §6)
    skip = _skip_dirs(cfg)                          # never re-crawl the quarantine dir

    confirm_exts = {str(e).lower() for e in (cfg.get("structured_confirm_exts") or [])}
    pend = None
    if confirm_exts:
        try:
            from . import pending as pending_mod
            pend = pending_mod.open_pending(cfg)
        except Exception as e:                     # the gate is best-effort, never fatal
            log.warning("deferred-ingest inbox unavailable (%s) — ingesting without the "
                        "confirm gate", e)
            pend = None
    deferred = 0
    want_graph = False
    try:
        for root in roots:
            if not os.path.isdir(root):
                log.info("source root missing, skipping: %s", root)
                continue
            for dirpath, _dirs, files in _walk_pruned(root, skip):
                for name in files:
                    if os.path.splitext(name)[1].lower() not in exts:
                        continue
                    path = os.path.join(dirpath, name)
                    added, wants = _ingest_or_defer(store, embedder, cfg, path,
                                                    force=force, confirm_exts=confirm_exts, pend=pend)
                    want_graph = want_graph or wants
                    if added:
                        docs += 1
                        chunks += added
                    if docs and every and docs % every == 0:
                        log.info("… %d docs / %d chunks", docs, chunks)
        deferred = pend.pending_count() if pend else 0
    finally:
        if pend:
            pend.close()
    out = {"docs": docs, "chunks": chunks}
    # a structured doc was ingested and asked for the graph → build it now (idempotent)
    if want_graph and cfg.get("auto_citations", True):
        try:
            from . import citations as citations_mod
            from .kb import KB
            kb = KB(cfg)
            try:
                out["citations"] = citations_mod.build(store, kb, cfg, log=log)
                log.info("cross-reference graph built: %s", out["citations"])
            finally:
                kb.close()
        except Exception as e:                     # the graph is a bonus, never fails the crawl
            log.warning("citations pass skipped: %s", e)
    if deferred:
        out["needs_confirm"] = deferred
        log.info("%d document group(s) need your confirmation before ingest — "
                 "answer them in the 'Needs your input' panel", deferred)
    return out


def _quarantine_roots(cfg) -> list:
    """The source roots a queued file may live under (expanded + realpath'd)."""
    roots = []
    for r in (cfg.get("sources") or []):
        rp = os.path.realpath(os.path.expanduser(str(r)))
        if rp not in roots:
            roots.append(rp)
    return roots


def _unique_dest(dest: str) -> str:
    """A non-clobbering variant of `dest` — adds ' (2)', ' (3)', … before the ext."""
    if not os.path.exists(dest):
        return dest
    base, ext = os.path.splitext(dest)
    i = 2
    while os.path.exists(f"{base} ({i}){ext}"):
        i += 1
    return f"{base} ({i}){ext}"


def _suggested_quarantine_dir(cfg) -> str:
    """The conventional home for cleared files: a ``quarantined/`` folder inside the
    first source root (``~/Documents`` → ``~/Documents/quarantined``).  The crawl
    skips it (see _skip_dirs), so moved files stay reverted."""
    roots = _quarantine_roots(cfg)
    return os.path.join(roots[0], "quarantined") if roots else ""


def quarantine_docs(cfg, doc_ids, *, dry_run: bool = False) -> dict:
    """MOVE the source files of ``doc_ids`` out of the ingest queue into
    ``cfg['quarantine_dir']``, preserving each file's path relative to its source
    root, so an accidental over-ingest can be reverted without the files being
    re-crawled.  The conventional home is a ``quarantined/`` subfolder INSIDE a
    source root (``~/Documents/Science/x.pdf`` → ``~/Documents/quarantined/Science/
    x.pdf``); the crawl always skips the quarantine dir, so files moved there don't
    come back.

    Only real files under a configured ``sources`` root move; URL/virtual docs
    (``wikipedia:``, ``zim://``, ``http(s)://``), files outside every root, and
    files already living under the quarantine dir are left in place (their DB rows
    are cleared by clear_queue regardless).  With more than one source root the
    per-root basename prefixes the mirrored path so same-named trees don't merge; a
    file already present at the destination is never overwritten (a numeric suffix
    is added).  ``dry_run`` counts without moving.  Returns
    {moved, skipped_nonfile, skipped_outside, errors, dest_root, samples} — or
    {error: …, needs_quarantine_dir, suggested_dir} if the dir is unset, or {error:
    …} if the dir contains/equals a source root (which would leave nothing to
    crawl)."""
    raw_q = str(cfg.get("quarantine_dir") or "").strip()
    if not raw_q:
        # No silent default — the user must choose where cleared files land, or
        # they'll never find them.  Refuse (the orchestrator makes no DB changes),
        # but hand back the conventional suggestion so the UI/CLI can pre-fill it.
        return {"moved": 0, "skipped_nonfile": 0, "skipped_outside": 0, "errors": 0,
                "dest_root": "", "samples": [], "needs_quarantine_dir": True,
                "suggested_dir": _suggested_quarantine_dir(cfg),
                "error": "quarantine_dir is not set — choose a folder (Settings › Paths, "
                         "or set quarantine_dir in config) so cleared files have a known home"}
    roots = _quarantine_roots(cfg)
    multi = len(roots) > 1
    qroot = os.path.realpath(os.path.expanduser(raw_q))
    res = {"moved": 0, "skipped_nonfile": 0, "skipped_outside": 0, "errors": 0,
           "dest_root": qroot, "samples": []}
    # A quarantine dir strictly INSIDE a source root is the intended layout (the
    # crawl skips it).  Only refuse the degenerate cases that would swallow a whole
    # source: the dir IS a root, or it CONTAINS one (root nested under quarantine).
    for r in roots:
        if qroot == r or r.startswith(qroot + os.sep):
            return {**res, "error": f"quarantine_dir {qroot} equals or contains source root {r} "
                    "— set it to a subfolder like <source root>/quarantined instead"}
    for doc in doc_ids or []:
        if not doc or "://" in str(doc):               # URL/virtual source, not a file
            res["skipped_nonfile"] += 1
            continue
        p = os.path.realpath(os.path.expanduser(str(doc)))
        if not os.path.isfile(p):
            res["skipped_nonfile"] += 1
            continue
        if p == qroot or p.startswith(qroot + os.sep):  # already in quarantine — leave it
            res["skipped_outside"] += 1
            continue
        root = next((r for r in roots if p == r or p.startswith(r + os.sep)), None)
        if root is None:
            res["skipped_outside"] += 1
            continue
        rel = os.path.relpath(p, root)
        dest = (os.path.join(qroot, os.path.basename(root.rstrip(os.sep)) or "root", rel)
                if multi else os.path.join(qroot, rel))
        if len(res["samples"]) < 5:
            res["samples"].append({"from": p, "to": dest})
        if dry_run:
            res["moved"] += 1
            continue
        try:
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            shutil.move(p, _unique_dest(dest))
            res["moved"] += 1
        except OSError as e:
            res["errors"] += 1
            log.warning("quarantine: could not move %s -> %s: %s", p, dest, e)
    return res


def clear_ingest_queue(cfg, store, kb_path, *, include_partial: bool = False,
                       quarantine: bool = True, dry_run: bool = False) -> dict:
    """Revert the distillation queue — the one entry point the CLI and the panel
    both call.  Optionally MOVE untouched docs' source files to the quarantine dir
    (so they don't re-ingest), then drop their chunks/doc_meta/manifest from the
    store; with include_partial, also trim partially-distilled docs' pending
    chunks.  Files move BEFORE the DB clear; a file that fails to move (and its
    now-cleared chunks) simply re-ingests on the next crawl — reported via
    quarantine.errors, self-correcting rather than silently lost.  dry_run previews
    counts (docs/chunks to drop, files to quarantine) and writes nothing."""
    scan = store.clear_queue(kb_path, include_partial=include_partial, dry_run=True)
    if scan.get("error"):
        return {"ok": False, **{k: v for k, v in scan.items() if k != "queued_doc_ids"}}
    doc_ids = scan.get("queued_doc_ids") or []
    q = (quarantine_docs(cfg, doc_ids, dry_run=True) if quarantine
         else {"moved": 0, "skipped_nonfile": 0, "skipped_outside": 0, "errors": 0})
    out = {k: v for k, v in scan.items() if k != "queued_doc_ids"}   # never leak the list
    out["quarantine"] = q
    if q.get("error"):
        return {"ok": False, "error": q["error"],
                "needs_quarantine_dir": bool(q.get("needs_quarantine_dir")),
                "suggested_quarantine_dir": q.get("suggested_dir", ""), **out}
    if dry_run:
        return {"ok": True, "dry_run": True, **out}
    if quarantine:
        out["quarantine"] = quarantine_docs(cfg, doc_ids, dry_run=False)
    res = store.clear_queue(kb_path, include_partial=include_partial, dry_run=False)
    final = {k: v for k, v in res.items() if k != "queued_doc_ids"}
    final["quarantine"] = out["quarantine"]
    final["ok"] = not res.get("error")
    return final


def ingest_wikipedia(store, embedder, cfg, *, limit: int | None = None,
                     force: bool = False) -> dict:
    """Ingest a Kiwix Wikipedia ZIM (pre-rendered HTML articles).

    **Resumable**: each article is checkpointed in the manifest by its
    ``zim://<url>`` key, so a stop-and-restart skips everything already done and
    picks up where it left off — essential for a multi-hour full-Wikipedia run.

    **Duplicate-proof**: the lance backend appends (no upsert), so an article is
    marked ``pending`` before embedding and ``ok`` after; on resume the one
    article interrupted mid-write is the only ``pending`` one, and its partial
    rows are cleared with ``delete_by_path`` before redo.  First-run articles are
    unseen, so no (millions of) no-op deletes are issued on the happy path.
    """
    zim = cfg.get("zim_path")
    if not zim or not os.path.isfile(zim):
        log.info("no zim_path configured/found, skipping Wikipedia")
        return {"articles": 0, "chunks": 0, "skipped": 0}
    from .sources import wikipedia
    version = int(store.manifest.meta_get("version", "1"))
    arts = chunks = skipped = 0
    every = cfg["ingest_log_every"]
    for url, title, blocks in wikipedia.iter_articles(zim):
        key = f"zim://{url}"
        prev = store.manifest.get(key)
        if prev and not force and prev["status"] == "ok" and prev["version"] == version:
            skipped += 1
            continue                                   # already ingested — resume past it
        if prev:                                       # a prior attempt left rows — clear them
            store.delete_by_path(key)
        store.manifest.set(key, "zim", 0.0, version, "pending")
        n = _embed_and_store(store, embedder, cfg, source_type="wikipedia",
                             title=title, path_or_url=key, blocks=blocks, version=version)
        store.manifest.set(key, "zim", 0.0, version, "ok")
        arts += 1
        chunks += n
        if every and arts % every == 0:
            log.info("… %d new articles / %d chunks (%d already done)",
                     arts, chunks, skipped)
        if limit and arts >= limit:
            break
    store.manifest.set(f"zim://{os.path.basename(zim)}",
                       _content_hash(zim) if os.path.getsize(zim) < (1 << 30) else "big",
                       os.path.getmtime(zim), version, "ok")
    return {"articles": arts, "chunks": chunks, "skipped": skipped}
