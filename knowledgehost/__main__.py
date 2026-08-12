"""Entry point.

  python3 -m knowledgehost [-c config.toml]            # serve the query tool
  python3 -m knowledgehost ingest [--force] [--wikipedia] [--limit N]
  python3 -m knowledgehost stats                       # index stats, then exit
  python3 -m knowledgehost bump-version                # version++ (monthly swap)
"""
from __future__ import annotations

import argparse
import logging
import os
import shutil
import sys
import time

from . import distill as distill_mod
from . import ingest as ingest_mod
from . import ops as ops_mod
from . import server
from .config import load_config
from .distill import BackendUnavailable
from .embed import Embedder
from .kb import KB
from .store import make_store
from .tools import Tools


def _build(cfg):
    store = make_store(cfg)
    embedder = Embedder(cfg)
    return store, embedder


def _remove_path(path: str) -> list:
    """Delete a db file (with its -wal/-shm sidecars and any ANN index files) or a
    directory.  Returns the paths actually removed.  The ANN sidecars matter: a reset
    that leaves them behind serves a stale index after re-ingest — every ANN hit is a
    dead id, so search silently returns nothing."""
    removed = []
    if os.path.isdir(path):
        shutil.rmtree(path)
        removed.append(path)
    else:
        for p in (path, path + "-wal", path + "-shm",
                  path + ".ann.usearch", path + ".ann.ids.json"):
            if os.path.exists(p):
                os.remove(p)
                removed.append(p)
    return removed


def _cmd_reset(cfg, args, log) -> int:
    """Clear the import so you can start from scratch.  Scopes: default = raw +
    KB; --kb = only the distilled KB; --raw = only the raw chunk store."""
    do_raw = bool(args.raw) or not args.kb
    do_kb = bool(args.kb) or not args.raw

    targets = []
    if do_raw:
        targets.append(("raw store + manifest", cfg["db_path"]))
        targets.append(("raw lance tables", cfg["lance_dir"]))
    if do_kb:
        targets.append(("distilled KB (nodes/edges/cards)", cfg["kb_path"]))
    existing = [(lbl, p) for lbl, p in targets if os.path.exists(p)]
    if not existing:
        log.info("nothing to clear — already empty.")
        return 0

    # Best-effort summary of what's about to be lost.
    try:
        if do_raw and os.path.exists(cfg["db_path"]):
            store = make_store(cfg)
            print(f"  raw chunks: {store.count():,}")
            store.close()
        if do_kb and os.path.exists(cfg["kb_path"]):
            kb = KB(cfg)
            print(f"  kb: {kb.counts()}")
            kb.close()
    except Exception as e:                       # never let reporting block a reset
        log.warning("could not read current contents (%s)", e)

    print("This will PERMANENTLY delete:")
    for lbl, p in existing:
        print(f"  - {lbl}: {p}")
    if not args.yes:
        try:
            ans = input('Type "yes" to confirm: ').strip().lower()
        except (EOFError, KeyboardInterrupt):
            ans = ""
        if ans != "yes":
            print("aborted — nothing deleted.")
            return 1

    removed = []
    for _lbl, p in existing:
        removed += _remove_path(p)
    log.info("cleared %d path(s): %s", len(removed), ", ".join(removed))
    return 0


def _run_distill(cfg, embedder, log, *, limit=None, watch=False, interval=30, bundle=None) -> int:
    """Distil raw chunks into the structured KB (owns its own raw-store handle so
    `--watch` can reopen for a fresh snapshot).  Returns a process exit code.

    In watch mode it loops, distilling new chunks as a concurrent ingest adds them
    (the distilled set is the checkpoint, so each pass only does fresh chunks),
    reopens the store each idle cycle so the lance backend sees new commits, and
    re-probes the big-LM endpoints so a 'sometimes available' duplicate is picked
    up (in parallel) when it appears and dropped when it goes."""
    # Say what THIS process actually read — "which config is it using?" must
    # be answerable from the log, not by reasoning about launch paths.
    log.info("config: %s — distill_parallel=%r extract_urls=%s verify_urls=%s",
             cfg.get("_config_path") or "(defaults + env only)",
             cfg.get("distill_parallel"),
             cfg.get("extract_urls") or "(none)",
             cfg.get("verify_urls") or cfg.get("distill_urls") or "(none)")
    if not embedder.embed_one("warmup", "document"):
        log.error("embed endpoint unreachable — distillation needs vectors.")
        return 1
    kb = KB(cfg)
    store = make_store(cfg)
    rc = 0
    prev_up = None
    verify_on = cfg.get("verify", True)
    try:
        while True:
            # Two tiers, re-probed each pass: fast extractors + big verifiers.  Whichever
            # is up shapes the run; a 'sometimes available' endpoint joins/leaves live.
            fast = distill_mod.fast_endpoints(cfg) if verify_on else []
            big = distill_mod.verify_endpoints(cfg)
            if fast and big:
                extractors, verifiers, mode = fast, big, "two-tier (fast→verify)"
            elif big:
                extractors, verifiers, mode = big, None, "single-tier (big LM only)"
            elif fast:                                # fast up but no verifier — unverified
                extractors, verifiers, mode = fast, None, "UNVERIFIED (fast LM, no verifier)"
                log.warning("verifier endpoints down — running fast LM WITHOUT verification")
            else:
                extractors, verifiers, mode = [], None, "none"
            up = (tuple(e.url for e in extractors), tuple(v.url for v in (verifiers or [])))
            if up != prev_up:
                log.info("distill: %s — extract=%s verify=%s", mode,
                         ", ".join(up[0]) or "(none)", ", ".join(up[1]) or "(none)")
                prev_up = up
            if not extractors:
                if watch and not limit:
                    log.warning("no distill endpoint up — waiting %ds…", interval)
                    time.sleep(interval)
                    continue
                ups = []
                try:                       # never claim "nothing up" while
                    from .serving import up_llm_urls   # something is serving
                    ups = up_llm_urls(cfg)
                except Exception:
                    pass
                hint = ("" if not ups else
                        "  Serving RIGHT NOW: "
                        + ", ".join(f"{n} at {u}" for n, u in ups)
                        + " — point extract_urls / verify_urls there "
                          "(Settings), or swap the expected model back in.")
                log.error("no distill endpoint up (extract=%s verify=%s) — start one first.%s",
                          ", ".join(cfg.get("extract_urls") or []),
                          ", ".join(cfg.get("verify_urls") or cfg.get("distill_urls") or []),
                          hint)
                rc = 1
                break
            stats = distill_mod.distill_corpus(store, kb, extractors, embedder, cfg,
                                               limit=limit, verifiers=verifiers, bundle=bundle)
            log.info("distilled: %s", stats)
            if not stats.get("chunks"):
                # a wall of zero-and-skipped stats reads as failure — say the
                # true thing: this lane has no pending work
                log.info("nothing to distill%s: %s chunk(s) already done, "
                         "%s outside the bundle, %s furniture-zone, %s duplicate "
                         "— this lane is COMPLETE, not failing",
                         f" in bundle '{bundle}'" if bundle else "",
                         stats.get("skipped", 0), stats.get("outside_bundle", 0),
                         stats.get("skipped_zone", 0), stats.get("skipped_dupe", 0))
            log.info("kb: %s", kb.counts())
            ops_mod.emit_result(stats.get("chunks", 0) > 0, **stats)
            if not watch or limit:
                break
            log.info("watch: waiting %ds for ingest to add more chunks (Ctrl-C to stop)…",
                     interval)
            time.sleep(interval)
            store.close()
            store = make_store(cfg)            # fresh snapshot (lance sees new versions)
    except BackendUnavailable as e:
        log.error("aborted (resumable): %s", e)
        rc = 1
    except KeyboardInterrupt:
        log.info("watch stopped")
    finally:
        kb.close()
        try:
            store.close()
        except Exception:
            pass
    return rc


def _run_link(cfg, log, *, limit=None, top_k=None, fast=False) -> int:
    """Phase 1 graph linkage: type structural edges (is_a/requires/part_of/alternative/
    related) between card-bearing concepts and their embedding-neighbours, via an LM.
    KB-only + one LM endpoint; resumable (judged pairs are checkpointed) and lease-aware
    (yields the matching GPU while Vinkona is using it).  --fast routes to the extractor LM —
    relation-typing is well within its range and it clears the sweep far quicker than the
    big verifier LM.  Build the ANN index first for speed at scale."""
    from . import distill as distill_mod
    from . import link as link_mod
    from . import lm_lease
    endpoints = distill_mod.fast_endpoints if fast else distill_mod.verify_endpoints
    lease = lm_lease.FAST if fast else lm_lease.BIG
    tier = "fast extractor LM" if fast else "big verifier LM"
    urls_hint = (cfg.get("extract_urls") if fast
                 else (cfg.get("verify_urls") or cfg.get("distill_urls"))) or []
    live = endpoints(cfg, log)
    if not live:
        log.error("no %s answered (%s) — %s", tier, ", ".join(urls_hint),
                  distill_mod.endpoint_down_hint(cfg))
        return 1
    log.info("link: using the %s", tier)
    kb = KB(cfg)
    try:
        stats = link_mod.link_concepts(kb, live[0], cfg, limit=limit, top_k=top_k, lease=lease)
        log.info("link: %s", stats)
        log.info("kb: %s", kb.counts())
        ops_mod.emit_result(stats.get("judged", 0) > 0, **{
            k: v for k, v in stats.items() if not isinstance(v, (list, dict))})
    except BackendUnavailable as e:
        log.error("aborted (resumable): %s", e)
        return 1
    except KeyboardInterrupt:
        log.info("link stopped (resumable)")
    finally:
        kb.close()
    return 0


def _run_refine(cfg, store, embedder, log, *, limit=None, force=False) -> int:
    """Phase 2 card refinement: re-read each card's source document and rewrite the card in
    place into the ideal 'what do I do now' form, grounded in that source.  Big-LM work (needs
    the 64k context); demand-weighted by hit_count, resumable (refined cards are skipped unless
    --force), lease-aware.  Needs the raw chunk store (source) + the big LM endpoint."""
    from . import distill as distill_mod
    from . import refine as refine_mod
    from . import lm_lease
    if not embedder.embed_one("warmup", "document"):
        log.error("embed endpoint unreachable — refinement re-embeds each card.")
        store.close()
        return 1
    big = distill_mod.verify_endpoints(cfg, log)
    if not big:
        log.error("no big verifier LM answered (%s) — %s",
                  ", ".join(cfg.get("verify_urls") or cfg.get("distill_urls") or []),
                  distill_mod.endpoint_down_hint(cfg))
        store.close()
        return 1
    lm = big[0]
    # A 46k-token source + a reasoning model is slow; the default distill timeout (~120s) gives
    # up mid-prompt on big documents.  Use a generous refine-specific budget so the client
    # doesn't hang up while llama.cpp is still processing.
    lm.timeout = int(cfg.get("refine_timeout_s", 600))
    log.info("refine: per-call timeout %ds, source budget %d tokens",
             lm.timeout, int(cfg.get("refine_source_tokens", 46000)))
    kb = KB(cfg)
    try:
        stats = refine_mod.refine_cards(kb, store, embedder, lm, cfg,
                                        limit=limit, force=force, lease=lm_lease.BIG)
        log.info("refine: %s", stats)
        log.info("kb: %s", kb.counts())
        ops_mod.emit_result(stats.get("candidates", 0) > 0, **{
            k: v for k, v in stats.items() if not isinstance(v, (list, dict))})
    except BackendUnavailable as e:
        log.error("aborted (resumable): %s", e)
        return 1
    except KeyboardInterrupt:
        log.info("refine stopped (resumable)")
    finally:
        kb.close()
        store.close()
    return 0


def _run_dedupe(cfg, store, log, *, near=False, threshold=0.9, apply=False,
                bundle=None) -> int:
    """The janitor: find chunks that hold text the corpus already has.

    EXACT (always): identical once normalised — the same document re-exported
    under a new name, or filed in two places.  These are marked against the
    chunk that owns the text, so they never cost another LM call.  Nothing is
    deleted: the row and its FTS entry stay, so search still finds it either way.

    NEAR (--near): MinHash/Jaccard over word shingles, for the same answer
    written twice in slightly different words.  Reported by default and only
    acted on with --apply, because "almost the same" can also mean "a revision
    of" — which you usually want to keep and distil."""
    from . import dedupe as dd
    kb = KB(cfg)
    exact = near_found = 0
    try:
        texts = {}
        for ch in store.iter_chunks():
            if bundle is not None and distill_mod_bundle(ch) != bundle:
                continue
            texts[ch["id"]] = ch.get("text") or ""
        log.info("dedupe: scanning %d chunks", len(texts))

        owners: dict = {}
        for cid, text in texts.items():
            th = dd.text_hash(text)
            owner = kb.claim_text(th, cid)
            owners[cid] = owner
            if owner != cid:
                exact += 1
                kb.record_dupe(cid, owner, th, kind="exact", similarity=1.0)
                if not kb.is_distilled(cid):
                    kb.mark_distilled(cid)     # the owner's distillation covers it
        kb.db.commit()
        log.info("dedupe: %d exact duplicate(s) — marked against the chunk that owns "
                 "the text (never distilled again)", exact)

        if near:
            # Compare only what isn't already an exact duplicate of something.
            live = [(cid, t) for cid, t in texts.items() if owners.get(cid) == cid]
            pairs = list(dd.near_pairs(live, threshold=threshold))
            near_found = len(pairs)
            for a, b, sim in pairs[:200]:
                log.info("  near %.3f  %s  ~  %s", sim, a, b)
            if len(pairs) > 200:
                log.info("  … and %d more", len(pairs) - 200)
            if apply:
                for a, b, sim in pairs:
                    if kb.dupe_of(b):
                        continue
                    kb.record_dupe(b, a, "", kind="near", similarity=sim)
                    if not kb.is_distilled(b):
                        kb.mark_distilled(b)
                kb.db.commit()
                log.info("dedupe: %d near-duplicate(s) marked (--apply)", near_found)
            else:
                log.info("dedupe: %d near-duplicate(s) found at >= %.2f — nothing marked "
                         "(re-run with --apply to act on them)", near_found, threshold)
        log.info("dedupe totals: %s", kb.dupe_stats())
        ops_mod.emit_result(exact > 0 or near_found > 0, exact=exact, near=near_found,
                            applied=bool(apply and near))
    finally:
        kb.close()
        store.close()
    return 0


def distill_mod_bundle(ch):
    from . import distill as distill_mod
    return distill_mod._chunk_bundle(ch)


def _run_pack(cfg, log, args) -> int:
    """VINUR-PACK-01 §3: clean-room build of a shareable pack from one file or
    folder.  The master kb is never opened for writing; a failed build keeps
    its scratch (re-running the same command resumes)."""
    from . import pack as pack_mod
    from .distill import BackendUnavailable as _BU
    path = (args.args[0] if args.args else None) or args.path
    if not path:
        log.error("pack needs an input: pack <file|folder> [--name N] "
                  "[--pack-version 1.0.0] [--license SPDX] [--compress] …")
        return 1
    try:
        res = pack_mod.build_pack(
            cfg, path, name=args.name, title=args.title or "",
            author=args.author or "", contact=args.contact or "",
            version=args.pack_version, describe=args.describe or "",
            license_override=args.license or "",
            allow_unlicensed=args.allow_unlicensed,
            compress=args.compress, encrypt=args.encrypt,
            passphrase=os.environ.get("PACK_PASSPHRASE", ""),
            keep_build=args.keep_build, force=args.force, log_fn=log.info)
    except ValueError as e:
        log.error("%s", e)
        ops_mod.emit_result(False, error=str(e))
        return 1
    except _BU as e:
        log.error("aborted (scratch kept — re-run to resume): %s", e)
        ops_mod.emit_result(False, error=str(e))
        return 1
    counts = (res["manifest"].get("content") or {}).get("counts") or {}
    log.info("pack ready: %s", res["artifact"])
    ops_mod.emit_result(True, artifact=res["artifact"], **counts)
    return 0


def _run_analyze(cfg, log, args) -> int:
    """`analyze <doc>`: read a plain-text scripture/legal document and PROPOSE a
    structure profile (kind, addressing scheme, books/sections found, how
    cross-references are written, samples, warnings) for the user to CONFIRM before
    a structured ingest.  Read-only; nothing is written unless --save is given."""
    import json as _json
    from . import structure as S
    doc = (args.args[0] if args.args else None) or args.doc
    if not doc:
        log.error("analyze needs a document: analyze <file> [--kind scripture|legal] [--save profile.json]")
        return 1
    try:
        text = open(os.path.expanduser(doc), encoding="utf-8", errors="replace").read()
    except OSError as e:
        log.error("cannot read %s: %s", doc, e)
        return 1
    maps = S.load_reference_maps(cfg.get("reference_maps"))
    prof = S.analyze(text, kind_hint=args.kind, maps=maps)
    if maps.stats["book_aliases"] or maps.stats["key_aliases"]:
        print(f"  refmaps  : {maps.stats['book_aliases']} book alias(es), "
              f"{maps.stats['key_aliases']} key alias(es) loaded")
    for bad in maps.stats["unmapped_targets"]:
        log.warning("reference map targets an unknown canonical book: %r (ignored)", bad)

    print(f"\n  document : {doc}")
    print(f"  kind     : {prof['kind']}   (confidence {prof.get('confidence', 0):.2f})")
    print(f"  scheme   : {prof.get('scheme')}   — one unit per {prof.get('unit')}")
    if prof["kind"] == "scripture":
        bl = prof.get("books") or []
        shown = ", ".join(f"{b['canonical']} ({b['verses_seen']})" for b in bl[:12])
        print(f"  books    : {len(bl)} detected — {shown}{' …' if len(bl) > 12 else ''}")
    elif prof["kind"] == "legal":
        cs = prof.get("citation_style", {})
        print(f"  citation : marker {cs.get('section_marker')} · subsections {cs.get('subsection')} · "
              f"full-form {cs.get('full_citation')}")
    print(f"  stats    : {prof.get('stats')}")
    if prof.get("work"):
        print(f"  work     : {prof['work']}   (the namespace local refs resolve under)")
    if prof.get("samples"):
        print("  samples  :  canonical-key  [citation]  text")
        for s in prof["samples"]:
            print(f"     {s.get('key', '?')}   [{s['citation']}]  {s['text']}")
    for w in prof.get("warnings", []):
        print(f"  ⚠ {w}")

    qs = S.questions_for(prof)
    if qs:
        print("\n  I need you to confirm a few things before a structured ingest "
              "(answer these in an --answers-file JSON, id → value):")
        for q in qs:
            print(f"\n   • [{q['id']}] {q['prompt']}")
            if q.get("detail"):
                print(f"       {q['detail']}")
            for o in q.get("options", []):
                mark = "default" if o["value"] == q.get("default") else "      "
                print(f"       ({mark}) {o['value']}: {o['label']}")
        example = _json.dumps({q["id"]: q.get("default", "") for q in qs}, ensure_ascii=False)
        print(f"\n     e.g.  echo '{example}' > answers.json  &&  "
              "collect <doc> --to out.kdb --bundle mybundle --answers-file answers.json")
    print("\n  → If this looks right, re-run with --save <profile.json> to keep the profile, "
          "or answer the questions above for a structured ingest. If a book/scheme is wrong, "
          "tell me what it should be.\n")

    if args.save:
        try:
            with open(os.path.expanduser(args.save), "w", encoding="utf-8") as f:
                _json.dump(prof, f, indent=2, ensure_ascii=False)
            log.info("profile written to %s (%d sample(s), %d warning(s))",
                     args.save, len(prof.get("samples", [])), len(prof.get("warnings", [])))
        except OSError as e:
            log.error("could not write %s: %s", args.save, e)
            return 1
    return 0


def _run_collect(cfg, log, args) -> int:
    """`collect <doc> --to <file.kdb> --bundle <name>`: clean-room ingest+distil of one
    document, ADDED to a shareable .kdb collection (created or merged into).  The master
    kb is never touched.  --dry-run validates + previews without running anything."""
    import json as _json
    from . import pack as pack_mod
    from .distill import BackendUnavailable as _BU
    doc = (args.args[0] if args.args else None) or args.doc
    target, bundle = args.to, args.bundle
    if not (doc and target and bundle):
        log.error("collect needs: collect <doc> --to <file.kdb> --bundle <name> "
                  "[--license SPDX] [--allow-unlicensed] [--answers-file a.json] [--dry-run]")
        return 1
    answers = None
    if args.answers_file:
        try:
            with open(os.path.expanduser(args.answers_file), encoding="utf-8") as f:
                answers = _json.load(f)
            if not isinstance(answers, dict):
                raise ValueError("answers file must be a JSON object of id → value")
        except (OSError, ValueError) as e:
            log.error("cannot read --answers-file %s: %s", args.answers_file, e)
            return 1
    try:
        res = pack_mod.add_to_collection(
            cfg, doc, target, bundle, license_override=args.license or "",
            allow_unlicensed=args.allow_unlicensed, force=args.force,
            dry_run=args.dry_run, log_fn=log.info, answers=answers,
            report=ops_mod.emit_progress)      # live phase bar in the Operations panel
    except ValueError as e:
        log.error("%s", e)
        ops_mod.emit_result(False, error=str(e))
        return 1
    except _BU as e:
        log.error("aborted (no distil endpoint up — start one and re-run): %s", e)
        ops_mod.emit_result(False, error=str(e))
        return 1
    if res.get("dry_run"):
        verb = "create" if res["created"] else "add to"
        log.info("collect (dry-run): would %s %s under bundle '%s'%s — nothing written.",
                 verb, res["target"], res["bundle"],
                 (f" (file currently holds {res['current']})" if res.get("current") else ""))
        return 0
    added = res.get("added") or {}
    log.info("collect: %s %s [%s] — added %s; file now holds %s",
             "created" if res["created"] else "updated", res["target"], res["bundle"],
             added, res.get("totals"))
    if not res.get("complete", True):
        log.warning("collect: build is INCOMPLETE — re-run the same collect to finish "
                    "distilling / recovering cards (it resumes; the rest merges in).")
    if res.get("single_tier"):
        log.info("collect: distilled single-tier (no independent verify tier was up) — "
                 "a quality note for a shared bundle, not an error.")
    if not res.get("shareable", True):
        log.warning("collection has unlicensed sources — set a --license before sharing")
    ops_mod.emit_result(True, target=res["target"], bundle=res["bundle"],
                        created=res["created"], complete=bool(res.get("complete", True)),
                        single_tier=bool(res.get("single_tier")),
                        recovered_truncated=res.get("recovered_truncated", 0),
                        **{f"added_{k}": v for k, v in added.items()})
    return 0


def _run_recard(cfg, store, embedder, log, *, limit=None, bundle=None,
                all_families=False, before=None, since=None) -> int:
    """Cards-only sweep over already-distilled chunks: harvest the card families
    (procedures and criteria included from v3, plus branch/troubleshooting/
    expectation/misconception/enumeration) from corpus stamped before the current
    RECARD_VERSION.  --all-families re-asks EVERY family regardless of stamp age
    (truncation recovery).  --before bounds the sweep to chunks distilled before a
    date; --since is its recovery mirror — it re-opens the recent window
    REGARDLESS of stamp, to recover chunks truncated after they were stamped
    current.  Joins existing concept nodes and never re-emits nodes or relations,
    so the adjudication queue stays quiet.  Resumable (the recarded set is the
    checkpoint); big-LM work, fanned out like distill."""
    from . import distill as distill_mod
    from datetime import datetime

    def _epoch(val, flag):
        try:
            return datetime.fromisoformat(str(val)).timestamp(), None
        except ValueError:
            return None, ("%s: %r is not a date/time (YYYY-MM-DD or "
                          "YYYY-MM-DDTHH:MM)." % (flag, val))
    cutoff = floor = None
    if before:
        cutoff, err = _epoch(before, "--before")
        if err:
            log.error(err)
            return 1
        log.info("recard: bounded to chunks distilled before %s", before)
    if since:
        floor, err = _epoch(since, "--since")
        if err:
            log.error(err)
            return 1
        log.info("recard: RECOVERY of chunks distilled since %s (re-opens the "
                 "window regardless of stamp)", since)
    if not embedder.embed_one("warmup", "document"):
        log.error("embed endpoint unreachable — recard needs vectors for the new cards.")
        return 1
    big = distill_mod.verify_endpoints(cfg)
    if not big:
        log.error("no big verifier LM answered (%s) — %s",
                  ", ".join(cfg.get("verify_urls") or cfg.get("distill_urls") or []),
                  distill_mod.endpoint_down_hint(cfg))
        return 1
    # verify_endpoints builds clients with the VERDICT budget (verify_max_tokens,
    # 1024) — this sweep emits procedure/criteria payloads, so re-stamp the
    # extraction knobs or the recovery pass would truncate its own harvest.
    for lm in big:
        lm.max_tokens = cfg.get("distill_max_tokens", 3072)
        lm.timeout = cfg.get("distill_timeout_s", lm.timeout)
    kb = KB(cfg)
    try:
        stats = distill_mod.recard_corpus(store, kb, big, embedder, cfg,
                                          limit=limit, bundle=bundle,
                                          all_families=all_families,
                                          before=cutoff, since=floor)
        log.info("recard: %s", stats)
        log.info("kb: %s", kb.counts())
        ops_mod.emit_result(stats.get("chunks", 0) > 0 or stats.get("no_menu", 0) > 0,
                            **stats)
    except BackendUnavailable as e:
        log.error("aborted (resumable): %s", e)
        return 1
    except KeyboardInterrupt:
        log.info("recard stopped (resumable)")
    finally:
        kb.close()
        store.close()
    return 0


def _run_adjudicate(cfg, log, *, limit=None, batch=8, watch=False, interval=30,
                    auto=True, auto_only=False, fast=False) -> int:
    """Drain the node-merge queue.  A deterministic pre-pass (auto_resolve) first clears
    the lexically-obvious pairs without any LM — merging plural/exact duplicates, adding
    is_a for token-subsets, and deferring the weak tail — so the LM only judges the thin
    ambiguous band that remains.  --fast routes that residual to the fast extractor LM instead
    of the big verifier LM: same/distinct on synonyms is well within its range and it clears
    the queue far quicker.  Resumable and lease-aware (yielding the matching GPU)."""
    from . import adjudicate as adj
    from . import distill as distill_mod
    from . import lm_lease
    endpoints = distill_mod.fast_endpoints if fast else distill_mod.verify_endpoints
    lease = lm_lease.FAST if fast else lm_lease.BIG
    tier = "fast extractor LM" if fast else "big verifier LM"
    urls_hint = (cfg.get("extract_urls") if fast
                 else (cfg.get("verify_urls") or cfg.get("distill_urls"))) or []
    kb = KB(cfg)
    rc = 0
    try:
        if auto:
            astats = adj.auto_resolve(kb, cfg, limit=limit)
            log.info("auto-resolve: %s", astats)
            if auto_only or astats.get("open_remaining", 0) == 0:
                if auto_only:
                    log.info("--auto-only: skipping the LM pass (%d still open for it).",
                             astats.get("open_remaining", 0))
                return 0
        while True:
            lms = endpoints(cfg)
            if not lms:
                if watch and not limit:
                    log.warning("no %s endpoint up for adjudication — waiting %ds…",
                                tier, interval)
                    time.sleep(interval)
                    continue
                log.error("no %s answered (%s) — %s", tier, ", ".join(urls_hint),
                          distill_mod.endpoint_down_hint(cfg))
                rc = 1
                break
            log.info("adjudicating residual with the %s", tier)
            stats = adj.adjudicate_queue(kb, lms[0], cfg, limit=limit, batch=batch, lease=lease)
            log.info("adjudicated: %s", stats)
            # adjudicate_queue reports 'judged' ('seen' belongs to the auto_resolve pass)
            ops_mod.emit_result(stats.get("judged", 0) > 0, **stats)
            if not watch or limit or stats.get("open_remaining", 0) == 0:
                break
            log.info("watch: %ds until the next pass (Ctrl-C to stop)…", interval)
            time.sleep(interval)
    except KeyboardInterrupt:
        log.info("adjudication stopped (resumable)")
    finally:
        kb.close()
    return rc


def _run_import_conceptnet(cfg, log, *, path=None, min_weight=None,
                           include_lexical=False, exclude=None, limit=None) -> int:
    """Stream the ConceptNet dump into the KB (regime=conventional, low trust,
    has_reference=0).  Resumable/idempotent — re-run to add only what is new."""
    from . import conceptnet as cn
    src = path or cfg.get("conceptnet_path") or ""
    if not src:
        log.error("no ConceptNet path — pass --path PATH or set conceptnet_path in config.")
        return 1
    src = os.path.expanduser(src)
    if not os.path.exists(src):
        log.error("ConceptNet dump not found: %s", src)
        return 1
    kb = KB(cfg)
    try:
        stats = cn.import_conceptnet(
            kb, src,
            min_weight=cfg["conceptnet_min_weight"] if min_weight is None else min_weight,
            trust=cfg["conceptnet_trust"],
            include_lexical=include_lexical or cfg["conceptnet_include_lexical"],
            exclude=(exclude if exclude is not None else cfg["conceptnet_exclude"]),
            limit=limit)
        log.info("conceptnet import: %s", stats)
        log.info("kb: %s", kb.counts())
    except KeyboardInterrupt:
        log.info("interrupted — partial import is committed (re-run to resume).")
    finally:
        kb.close()
    return 0


def _run_edge_audit(cfg, log, *, apply=False, limit=None) -> int:
    """Cull nonsense graph associations without an LM: flag relations whose labels are
    orthographically alike but semantically distant (sound-alike links), or that are
    distant + uncited + never co-occur in the corpus (ungrounded over-links).  Reports
    by default; --apply soft-retracts (reversible via edge_audit.restore)."""
    from .kb import KB
    from . import edge_audit
    kb = KB(cfg)
    try:
        rep = edge_audit.audit_edges(kb, cfg, apply=apply, limit=limit)
    finally:
        kb.close()
    log.info("edge-audit: scanned %d active edge(s), flagged %d %s — %s",
             rep["scanned"], rep["flagged"], rep["by_verdict"] or "{}",
             f"retracted {rep['applied']}" if apply
             else "report only (pass --apply to retract)")
    if rep["cooc_note"]:
        log.warning("edge-audit: %s", rep["cooc_note"])
    for f in rep["sample"]:
        log.info("  [%s] %r --%s--> %r  (orth %.2f sem %s cooc %s) — %s",
                 f["verdict"], f["src"], f["type"], f["dst"],
                 f["orth"], f["sem"], f["cooc"], f["reason"])
    return 0


def _run_migrate_vocab(cfg, log) -> int:
    """One-shot: migrate a pre-1.2 KB to the CONF-01/1.2 domain-neutral vocabulary
    (relation 'incompatible', table 'uses', acts_via roles) and rename the distilled
    'contraindicated_in' edge type.  Idempotent — safe to re-run."""
    from .kb import KB
    from . import conflict
    kb = KB(cfg)
    try:
        stats = conflict.migrate_vocab(kb.db)   # _LockedConn proxies execute/script/commit
        n = kb.db.execute("UPDATE edges SET type='incompatible_with' "
                          "WHERE type='contraindicated_in'").rowcount
        kb.db.commit()
        log.info("migrate-vocab: %s; distilled edges renamed: %d", stats, n)
    finally:
        kb.close()
    return 0


def _run_unimport(cfg, log, *, dataset=None) -> int:
    """Provenance-aware undo of one bulk dataset import — the tuning loop's
    middle step (import -> inspect -> UNIMPORT -> adjust thresholds -> re-import)."""
    from . import unimport as un
    if not dataset or dataset not in un.DATASETS:
        log.error("unimport needs --dataset {%s}", ",".join(un.DATASETS))
        return 1
    doc_id = un.DATASETS[dataset]
    kb = KB(cfg)
    try:
        row = kb.db.execute("SELECT 1 FROM source_registry WHERE doc_id=?", (doc_id,)).fetchall()
        if not row:
            log.info("%s (%s) is not imported — nothing to do.", dataset, doc_id)
            return 0
        stats = un.unimport(kb, doc_id)
        log.info("unimport %s: %s", dataset, stats)
        log.info("kb: %s", kb.counts(fresh=True))
        log.info("note: run build-ann to refresh the dense node index.")
    finally:
        kb.close()
    return 0


def _run_clear_queue(cfg, store, log, *, include_partial=False, quarantine=True,
                     dry_run=False, yes=False) -> int:
    """Bulk-revert the distillation queue (undo an accidental over-ingest): move
    untouched docs' source files to the quarantine dir, then drop their chunks;
    with include_partial also trim partially-distilled docs' pending chunks."""
    from . import ingest as ingest_mod
    kb_path = cfg["kb_path"]
    # quarantine_dir has no silent default — prompt for it (or refuse) so the user
    # knows where cleared files went.
    if quarantine and not str(cfg.get("quarantine_dir") or "").strip():
        if yes:
            log.error("quarantine_dir is not set — set it in config (Settings › Paths) or "
                      "pass --no-quarantine to only drop DB rows.")
            return 1
        suggested = ingest_mod._suggested_quarantine_dir(cfg)   # usually <root>/quarantined
        try:
            ans = input("Quarantine folder is not set — cleared files must be moved somewhere "
                        "you can find.\n"
                        + (f"Enter an absolute path [{suggested}] (Enter = accept, blank+Enter "
                           "twice = cancel): " if suggested
                           else "Enter an absolute path (blank = cancel): ")).strip()
        except EOFError:
            ans = ""
        if not ans and suggested:
            ans = suggested
        ans = os.path.expanduser(ans)
        if not ans or not os.path.isabs(ans):
            log.info("no quarantine folder set — aborted. Set quarantine_dir in config, or "
                     "re-run with --no-quarantine.")
            return 1
        cfg = {**cfg, "quarantine_dir": ans}
        log.info("using quarantine_dir=%s for this run — add it to config to persist.", ans)
    prev = ingest_mod.clear_ingest_queue(cfg, store, kb_path, include_partial=include_partial,
                                         quarantine=quarantine, dry_run=True)
    if not prev.get("ok"):
        log.error("clear-queue: %s", prev.get("error", "scan failed"))
        return 1
    q = prev.get("quarantine") or {}
    log.info("queue: %d untouched doc(s) / %d chunk(s)%s", prev["queued_docs"],
             prev["queued_chunks"],
             ("; %d partial doc(s), %d pending chunk(s)"
              % (prev["partial_docs"], prev["partial_chunks"])) if prev.get("partial_docs") else "")
    if quarantine:
        log.info("quarantine → %s: %d file(s) to move (%d non-file, %d outside a source root)",
                 q.get("dest_root", "?"), q.get("moved", 0),
                 q.get("skipped_nonfile", 0), q.get("skipped_outside", 0))
    would = prev["queued_chunks"] + (prev["partial_chunks"] if include_partial else 0)
    if dry_run:
        log.info("dry-run: would remove %d chunk(s)%s and move %d file(s) — nothing written.",
                 would, " (incl. partial trim)" if include_partial else "",
                 q.get("moved", 0) if quarantine else 0)
        return 0
    if would == 0 and (not quarantine or q.get("moved", 0) == 0):
        log.info("queue already empty — nothing to do.")
        return 0
    if not yes:
        msg = (f"Remove {prev['queued_chunks']} chunk(s) from {prev['queued_docs']} untouched doc(s)"
               + (f", trim {prev['partial_chunks']} pending chunk(s) from {prev['partial_docs']} "
                  "partial doc(s)" if include_partial else "")
               + (f", and MOVE {q.get('moved', 0)} file(s) to {q.get('dest_root')}"
                  if quarantine else "")
               + "? [y/N] ")
        try:
            ans = input(msg)
        except EOFError:
            ans = ""
        if ans.strip().lower() not in ("y", "yes"):
            log.info("aborted — nothing changed.")
            return 1
    res = ingest_mod.clear_ingest_queue(cfg, store, kb_path, include_partial=include_partial,
                                        quarantine=quarantine, dry_run=False)
    if not res.get("ok"):
        log.error("clear-queue: %s", res.get("error", "failed"))
        return 1
    q = res.get("quarantine") or {}
    log.info("clear-queue: removed %d chunk(s) from %d untouched doc(s)%s; quarantined %d file(s)%s (%.1fs)",
             res["chunks_removed"], res["queued_docs"],
             " + partial trim" if include_partial else "", q.get("moved", 0),
             (", %d move error(s)" % q["errors"]) if q.get("errors") else "",
             res.get("elapsed_s", 0.0))
    ops_mod.emit_result(res["chunks_removed"] > 0 or q.get("moved", 0) > 0,
                        chunks_removed=res["chunks_removed"], docs=res["queued_docs"],
                        quarantined=q.get("moved", 0), move_errors=q.get("errors", 0))
    return 0


def _run_import_atomic(cfg, log, *, path=None, min_count=None, limit=None) -> int:
    """Stream the ATOMIC if-then graph into the KB (regime=conventional, low trust,
    has_reference=0).  Idempotent — re-run to add only what is new."""
    from . import atomic as at
    src = path or cfg.get("atomic_path") or ""
    if not src:
        log.error("no ATOMIC path — pass --path PATH or set atomic_path in config "
                  "(use the aggregated v4_atomic_all_agg.csv).")
        return 1
    src = os.path.expanduser(src)
    if not os.path.exists(src):
        log.error("ATOMIC dump not found: %s", src)
        return 1
    kb = KB(cfg)
    try:
        stats = at.import_atomic(kb, src, trust=cfg["atomic_trust"],
                                 min_count=cfg["atomic_min_count"] if min_count is None else min_count,
                                 limit=limit)
        log.info("atomic import: %s", stats)
        log.info("kb: %s", kb.counts())
    except KeyboardInterrupt:
        log.info("interrupted — partial import is committed (re-run to resume).")
    finally:
        kb.close()
    return 0


def _run_import_glucose(cfg, log, *, path=None, min_count=None, limit=None) -> int:
    """Stream GLUCOSE general causal rules into the KB (regime=conventional, low trust,
    has_reference=0).  Idempotent — re-run to add only what is new."""
    from . import glucose as gl
    src = path or cfg.get("glucose_path") or ""
    if not src:
        log.error("no GLUCOSE path — pass --path PATH or set glucose_path in config.")
        return 1
    src = os.path.expanduser(src)
    if not os.path.exists(src):
        log.error("GLUCOSE dump not found: %s", src)
        return 1
    kb = KB(cfg)
    try:
        stats = gl.import_glucose(kb, src, trust=cfg["glucose_trust"],
                                  min_count=cfg["glucose_min_count"] if min_count is None else min_count,
                                  limit=limit)
        log.info("glucose import: %s", stats)
        log.info("kb: %s", kb.counts())
    except KeyboardInterrupt:
        log.info("interrupted — partial import is committed (re-run to resume).")
    finally:
        kb.close()
    return 0


def _run_import_causenet(cfg, log, *, path=None, min_sources=None, limit=None) -> int:
    """Stream CauseNet-precision into the KB (causal edges, grounded, has_reference=1).
    Idempotent — re-run to add only what is new."""
    from . import causenet as cnet
    src = path or cfg.get("causenet_path") or ""
    if not src:
        log.error("no CauseNet path — pass --path PATH or set causenet_path in config.")
        return 1
    src = os.path.expanduser(src)
    if not os.path.exists(src):
        log.error("CauseNet dump not found: %s", src)
        return 1
    kb = KB(cfg)
    try:
        stats = cnet.import_causenet(kb, src, trust=cfg["causenet_trust"],
                                     regime=cfg["causenet_regime"],
                                     min_sources=cfg["causenet_min_sources"] if min_sources is None
                                     else min_sources,
                                     limit=limit)
        log.info("causenet import: %s", stats)
        log.info("kb: %s", kb.counts())
    except KeyboardInterrupt:
        log.info("interrupted — partial import is committed (re-run to resume).")
    finally:
        kb.close()
    return 0


def _run_optimize(cfg, log, *, vacuum=False) -> int:
    """One-time layout fix: move the `embedding` column last so `ask` hydration doesn't
    read past the inline blob on every row lookup.  --vacuum also reclaims disk after."""
    kb = KB(cfg)
    try:
        import time as _t
        t0 = _t.perf_counter()
        changed = kb.migrate_node_layout()
        if changed:
            log.info("nodes layout migrated in %.1fs", _t.perf_counter() - t0)
            if vacuum:
                log.info("VACUUM (reclaiming disk; needs free space ≈ db size)…")
                kb._raw.execute("VACUUM")
                log.info("VACUUM done")
        else:
            log.info("nodes layout already optimal (embedding is last) — nothing to do.")
    finally:
        kb.close()
    return 0


def _run_bundles(cfg, args, log) -> int:
    """Modular-knowledge CLI (§16), all against the MASTER kb.db:
      bundles                       list provenance groups + source counts
      source <doc_id> [--title T] [--bundle B]   rename / regroup a source
      scenario [name]               show scenarios (or resolve one → sources)
      split [dir] [--force]         export each bundle group to <bundle>.kdb
    """
    import json as _json
    from . import bundles as B
    cmd = args.command
    if cmd == "split":
        out = (args.args[0] if args.args else None) or args.out
        res = B.split(cfg, out, force=args.force, log_fn=log.info)
        print(_json.dumps({k: v.get("counts", v) for k, v in res.items()}, indent=2))
        return 0

    kb = KB(cfg)
    try:
        if cmd == "bundles" and args.args and args.args[0] == "licenses":
            from . import licensing
            for s in kb.list_sources(500):
                fl = licensing.flags_for(s.get("license"))
                perms = "".join(k[0].upper() if fl.get(k) is True else
                                ("·" if fl.get(k) is None else "-")
                                for k in ("redistribute", "derivatives", "commercial"))
                print(f"{perms}  {(s.get('license') or 'unknown'):18s} "
                      f"{(s.get('license_holder') or ''):28s} {s.get('doc_id')}")
            print("\n(RDC = Redistribute/Derivatives/Commercial — CAP=yes ·=unknown -=no)")
            return 0
        if cmd == "bundles":
            summ = kb.bundle_summary()
            for b in summ:
                print(f"{b['bundle']:20s} {b['sources']:4d} source(s)")
            if not summ:
                print("(no sources registered yet)")
            return 0
        if cmd == "source":
            if not args.args:
                # no doc_id → list sources with their bundle tag + licence
                for s in kb.list_sources(500):
                    lic = s.get("license") or "unknown"
                    who = f" © {s['license_holder']}" if s.get("license_holder") else ""
                    print(f"{s.get('doc_id')}\t[{s.get('bundle') or 'base'}]\t{lic}{who}\t{s.get('title')}")
                return 0
            doc_id = args.args[0]
            edits = (args.title, args.bundle, args.license, args.license_holder, args.license_url)
            if all(e is None for e in edits):
                s = kb.get_source(doc_id)
                if s:
                    s["shippable_flags"] = kb.license_of(doc_id)["flags"]
                print(_json.dumps(s, indent=2) if s else f"no such source: {doc_id}")
                return 0 if s else 1
            row = kb.set_source(doc_id, title=args.title, bundle=args.bundle,
                                license=args.license, license_holder=args.license_holder,
                                license_url=args.license_url)
            if not row:
                log.error("no such source: %s", doc_id)
                return 1
            log.info("source %s → title=%r bundle=%r license=%r holder=%r", doc_id,
                     row.get("title"), row.get("bundle") or "base",
                     row.get("license"), row.get("license_holder"))
            return 0
        if cmd == "scenario":
            scenarios = cfg.get("scenarios") or {}
            active = B.active_scenario_name(cfg)
            if not args.args:
                print(f"active: {active}")
                print(f"available: {', '.join(scenarios) or '(none — serving all)'}")
                return 0
            name = args.args[0]
            scen = B.scenario_def(cfg, name)
            srcs = B.list_sources(kb._raw)
            unloaded = B.unloaded_set(cfg)
            picked = (B.select_sources(srcs, scen, unloaded)
                      if (scen or name != "all" or unloaded)
                      else {s['doc_id'] for s in srcs})
            if unloaded:
                print(f"(unloaded brains: {', '.join(sorted(unloaded))})")
            print(f"scenario '{name}' → {len(picked)} source(s):")
            for s in srcs:
                if s["doc_id"] in picked:
                    print(f"  {s['doc_id']}  [{s['bundle']}]  {s.get('title')}")
            return 0
    finally:
        kb.close()
    return 0


def _library_store_cfg(cfg) -> dict:
    """The library rides its OWN sqlite index (separate file from the graph corpus)."""
    return {**cfg, "backend": "sqlite", "db_path": cfg["library_db"]}


def _library_store(cfg, log=None):
    """Open the search-only library store if it's been built; else None (tool stays hidden)."""
    db = cfg.get("library_db")
    if not (db and os.path.exists(db)):
        return None
    try:
        from .store import make_store
        return make_store(_library_store_cfg(cfg))
    except Exception as e:                     # pragma: no cover - defensive
        if log:
            log.warning("library store unavailable (%s)", e)
        return None


def _run_rebuild_fts(cfg, log) -> int:
    """Rebuild the full-text index with the configured tokenizer (fts_tokenizer) — reindexes
    from the stored chunk text, NO source re-parse.  Run once after changing the tokenizer.
    Covers the library store and, when the graph runs on sqlite, its chunk store too."""
    from .store import make_store
    targets = []
    lib_db = cfg.get("library_db")
    if lib_db and os.path.exists(lib_db):
        targets.append(("library", _library_store_cfg(cfg)))
    if cfg.get("backend") == "sqlite":                    # lance has no FTS to rebuild
        targets.append(("graph", cfg))
    done = 0
    for name, scfg in targets:
        store = make_store(scfg)
        try:
            if not hasattr(store, "rebuild_fts"):
                continue
            n = store.rebuild_fts()
            store.optimize_fts()
            if hasattr(store, "build_stoplist"):
                store.build_stoplist()                    # re-learn the stoplist post-reindex
            done += 1
            log.info("rebuilt FTS for %s store: %d chunk(s) reindexed", name, n)
        finally:
            store.close()
    if not done:
        log.info("no sqlite FTS store found to rebuild (nothing to do)")
    return 0


def _run_ingest_library(cfg, log, *, force=False) -> int:
    """Index the library folder tree into its own FTS store (lexical by default; embeds too
    only when library_dense).  NOT distilled — this is the cheap search-only tier."""
    from .store import make_store
    from .embed import Embedder
    if not (cfg.get("library_sources")):
        log.error("no library_sources configured (see config: [library_sources]) — nothing to index")
        return 1
    store = make_store(_library_store_cfg(cfg))
    embedder = Embedder(cfg) if cfg.get("library_dense") else None   # lexical-first: no embed
    try:
        stats = ingest_mod.crawl_library(store, embedder, cfg, force=force)
        log.info("library indexed: %s", stats)
        import json as _json
        print(_json.dumps(stats, indent=2))
    finally:
        store.close()
    return 0


def _run_eval(cfg, log, *, gold=None, retriever="current_path", trace=False) -> int:
    """Retrieval eval harness (contract §8): score a retriever over the graded gold set
    and print the abstention-quality report.  Needs the KB + embedder (the current-path
    retriever embeds each query); no LM / raw store."""
    from . import evalharness
    from .embed import Embedder
    gold_path = gold or cfg.get("eval_gold_path") or "eval/gold.jsonl"
    if not os.path.exists(gold_path):
        log.error("gold set not found: %s (see eval/README.md)", gold_path)
        return 1
    embedder = Embedder(cfg)
    kb = KB(cfg)
    try:
        metrics = evalharness.run(kb, embedder, cfg, gold_path=gold_path,
                                  retriever=retriever, trace=trace)
    finally:
        kb.close()
    return 0 if metrics.get("accept", {}).get("passed") else 2


def _run_build_ann(cfg, log) -> int:
    """Build the HNSW ANN index over node embeddings (read-path speed).  Re-run after big
    imports/`embed-nodes` so it reflects the current node set."""
    from . import ann as ann_mod
    if not ann_mod.available():
        log.error("usearch not installed — `pip install usearch` to build the ANN index.")
        return 1
    kb = KB(cfg)
    try:
        path = cfg.get("ann_path") or (cfg["kb_path"] + ".ann")
        path = os.path.expanduser(path)
        stats = ann_mod.build_from_kb(
            kb, path,
            connectivity=cfg["ann_connectivity"],
            expansion_add=cfg["ann_expansion_add"],
            expansion_search=cfg["ann_expansion_search"],
            dtype=cfg["ann_dtype"],
            min_nodes=cfg["ann_min_nodes"])
        log.info("build-ann: %s", stats)
    finally:
        kb.close()
    return 0


def _run_citations(cfg, log) -> int:
    """`citations`: build the deterministic cross-reference graph over structured
    (scripture/legal) documents — one node per canonical unit (verse/section), a
    'citation' edge per reference its text makes.  No LM; idempotent (safe to re-run)."""
    from . import citations as cite_mod
    store = make_store(cfg)
    kb = KB(cfg)
    try:
        stats = cite_mod.build(store, kb, cfg, log=log)
        log.info("citations: %s", stats)
        log.info("kb: %s", kb.counts())
        if not stats.get("units"):
            log.info("citations: no structured (scripture/legal) documents found — "
                     "ingest one first (collect or the confirm gate), then re-run.")
        ops_mod.emit_result(stats.get("units", 0) > 0, **stats)
    finally:
        kb.close()
        store.close()
    return 0


def _run_reconcile(cfg, log, *, anchors="corpus", limit=None, top_k=None) -> int:
    """Queue merge candidates between your existing nodes and the imported commonsense
    sets, then point the user at `adjudicate` to resolve them with the big LM."""
    from . import reconcile_import as rec
    kb = KB(cfg)
    try:
        stats = rec.reconcile_imports(kb, cfg, anchors=anchors, limit=limit, top_k=top_k)
        log.info("reconcile: %s", stats)
        if stats.get("nodes_without_vectors"):
            log.warning("%d active nodes have NO embedding and were invisible to the "
                        "scan — run `embed-nodes` first to reconcile them too.",
                        stats["nodes_without_vectors"])
        if stats.get("queued"):
            log.info("next: run `adjudicate --watch` to judge the %d queued pair(s) with "
                     "the big LM.", stats["queued"])
    except RuntimeError as e:
        log.error("%s", e)
        return 1
    finally:
        kb.close()
    return 0


def _run_embed_nodes(cfg, embedder, log, *, limit=None) -> int:
    """Backfill embeddings for NULL-embedding nodes (e.g. bulk-imported ConceptNet
    terms) so they surface in dense search.  Resumable; safe to re-run."""
    if not embedder.embed_one("warmup", "document"):
        log.error("embed endpoint unreachable — start the nomic embed server first.")
        return 1
    from . import conceptnet as cn
    kb = KB(cfg)
    try:
        stats = cn.embed_nodes(kb, embedder, cfg, limit=limit)
        log.info("embed-nodes: %s", stats)
        log.info("kb: %s", kb.counts())
    except KeyboardInterrupt:
        log.info("interrupted — embedded nodes are committed (re-run to resume).")
    finally:
        kb.close()
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(prog="knowledgehost",
                                 description="Vinur — a local general-knowledge tool host.")
    ap.add_argument("command", nargs="?", default="serve",
                    choices=["serve", "ingest", "distill", "recard", "pack", "dedupe", "find", "pull", "adopt", "adjudicate", "reconcile",
                             "link", "refine", "import-conceptnet", "import-atomic",
                             "import-glucose", "import-causenet", "unimport", "embed-nodes", "build-ann",
                             "optimize", "edge-audit", "stats", "reset", "bump-version", "migrate-vocab",
                             "bundles", "split", "source", "scenario", "eval", "facetize",
                             "ingest-library", "rebuild-fts", "import-bundle", "eject-bundle",
                             "collect", "analyze", "citations", "clear-queue"])
    # positional args for the modular-bundle verbs:
    #   source <doc_id> [--title ..] [--bundle ..]   scenario [name]   split [dir]
    #   import-bundle <file.kdb>     eject-bundle <bundle>
    ap.add_argument("args", nargs="*", help="positional args for bundles/source/scenario/split")
    ap.add_argument("-c", "--config", help="path to a TOML config file")
    ap.add_argument("--host"); ap.add_argument("--port", type=int)
    ap.add_argument("--backend", choices=["sqlite", "lance"])
    ap.add_argument("--force", action="store_true",
                    help="ingest: re-process every file, ignoring the manifest")
    ap.add_argument("--wikipedia", action="store_true",
                    help="ingest: also ingest the configured Wikipedia ZIM")
    ap.add_argument("--distill", action="store_true",
                    help="ingest: distil the newly-ingested chunks right after (needs the big LM)")
    ap.add_argument("--watch", action="store_true",
                    help="distill: keep distilling as a concurrent ingest adds chunks")
    ap.add_argument("--interval", type=int, default=30,
                    help="distill --watch: seconds to wait between passes (default 30)")
    ap.add_argument("--limit", type=int,
                    help="ingest/distill/recard/adjudicate: cap items processed (testing)")
    ap.add_argument("--batch", type=int, default=8,
                    help="adjudicate: merge-candidate pairs per big-LM call (default 8)")
    ap.add_argument("--no-auto", action="store_true",
                    help="adjudicate: skip the deterministic pre-pass (LM-judge everything)")
    ap.add_argument("--auto-only", action="store_true",
                    help="adjudicate: run ONLY the deterministic pre-pass, no LM")
    ap.add_argument("--fast", action="store_true",
                    help="adjudicate/link: use the fast extractor LM instead of the big verifier LM")
    ap.add_argument("--path", help="import-conceptnet/import-bundle: path to the input file")
    ap.add_argument("--name", help="import-bundle: bundle name to absorb the file under "
                                   "(default: its manifest name, else the file stem)")
    ap.add_argument("--trust", choices=["low", "keep"], default="low",
                    help="import-bundle: cap the brain's support trust to 'low' (default; "
                         "shipped knowledge earns promotion) or 'keep' its own values "
                         "(your own brains moving between your own boxes)")
    ap.add_argument("--dry-run", action="store_true", dest="dry_run",
                    help="eject-bundle / clear-queue: scan and count, change nothing")
    ap.add_argument("--no-export", action="store_true", dest="no_export",
                    help="eject-bundle: skip the safety export to <bundle>.kdb first")
    ap.add_argument("--include-partial", action="store_true", dest="include_partial",
                    help="clear-queue: also trim the pending chunks of partially-distilled docs")
    ap.add_argument("--no-quarantine", action="store_true", dest="no_quarantine",
                    help="clear-queue: only drop DB rows; do NOT move source files aside "
                         "(they will re-ingest on the next crawl unless you remove them)")
    ap.add_argument("--min-weight", type=float, dest="min_weight",
                    help="import-conceptnet: drop assertions below this weight")
    ap.add_argument("--all", action="store_true",
                    help="import-conceptnet: also import the lexical/etymological bulk")
    ap.add_argument("--dataset", choices=["conceptnet", "atomic", "glucose", "causenet"],
                    help="unimport: which bulk import to remove (provenance-aware undo)")
    ap.add_argument("--exclude",
                    help="import-conceptnet: comma-separated relation names to ALWAYS skip "
                         "(e.g. FormOf,DerivedFrom,EtymologicallyRelatedTo — see conceptnet._REL)")
    ap.add_argument("--min-count", type=int, dest="min_count",
                    help="import-atomic/glucose: annotator/worker agreement floor (overrides config)")
    ap.add_argument("--min-sources", type=int, dest="min_sources",
                    help="import-causenet: DISTINCT supporting sources required (overrides config)")
    ap.add_argument("--anchors", choices=["corpus", "all"], default="corpus",
                    help="reconcile: anchor on your distilled corpus (default) or every node")
    ap.add_argument("--top-k", type=int, dest="top_k",
                    help="reconcile: nearest neighbours to queue per anchor")
    ap.add_argument("--vacuum", action="store_true",
                    help="optimize: VACUUM after migrating to reclaim disk space")
    ap.add_argument("--title", help="source: new display title for the source (rename)")
    ap.add_argument("--bundle",
                    help="source: assign the source to this bundle group | "
                         "distill/recard: only chunks from this provenance bundle (e.g. 'vinkona' — "
                         "distil Vinkona's research drops ahead of the big corpus)")
    ap.add_argument("--all-families", action="store_true", dest="all_families",
                    help="recard: re-ask EVERY card family regardless of each chunk's "
                         "stamp age (truncation recovery — cards lost to a too-small "
                         "output budget while the chunk was already marked done)")
    ap.add_argument("--before",
                    help="recard: only chunks DISTILLED before this date/time "
                         "(YYYY-MM-DD or YYYY-MM-DDTHH:MM, local) — bound the "
                         "recovery sweep to what predates the budget fix; the "
                         "healthy tail distilled since is spared")
    ap.add_argument("--since",
                    help="recard: RECOVERY mirror of --before — re-open chunks "
                         "DISTILLED on/after this date/time (YYYY-MM-DD or "
                         "YYYY-MM-DDTHH:MM, local) REGARDLESS of their recard "
                         "stamp, to recover cards truncated after a chunk was "
                         "already stamped current (idempotent: the title gate "
                         "folds chunks that already have their cards)")
    ap.add_argument("--author", help="pack: producer name for the manifest")
    ap.add_argument("--contact", help="pack: producer contact for the manifest")
    ap.add_argument("--pack-version", default="1.0.0", dest="pack_version",
                    help="pack: semver for the artifact name + manifest (default 1.0.0)")
    ap.add_argument("--describe", help="pack: one-line description for the manifest")
    ap.add_argument("--doc", help="collect: the document (or folder) to ingest + distil "
                                  "(or pass it positionally)")
    ap.add_argument("--to", dest="to",
                    help="collect: the .kdb collection file to create or add to")
    ap.add_argument("--kind", choices=["scripture", "legal"],
                    help="analyze: force the corpus kind instead of auto-detecting")
    ap.add_argument("--save", dest="save",
                    help="analyze: write the proposed profile to this JSON file "
                         "(after you've eyeballed it) for a later structured ingest")
    ap.add_argument("--answers-file", dest="answers_file",
                    help="collect: JSON file of structured-text confirmation answers "
                         "(id → value, from `analyze`/the wizard) — ingests the doc "
                         "unit-by-unit under the confirmed profile")
    ap.add_argument("--allow-unlicensed", action="store_true", dest="allow_unlicensed",
                    help="pack: build despite unlicensed sources — stamped "
                         "shareable:false (private use only; import warns)")
    ap.add_argument("--compress", action="store_true",
                    help="pack: gzip the artifact (.kdb.gz — import auto-detects)")
    ap.add_argument("--encrypt", action="store_true",
                    help="pack: SQLCipher-encrypt the artifact (.enc) with the "
                         "PACK_PASSPHRASE env var — access control in transit, "
                         "NOT usage control after import")
    ap.add_argument("--keep-build", action="store_true", dest="keep_build",
                    help="pack: keep the scratch workspace after a successful build")
    ap.add_argument("--passphrase", default="",
                    help="import-bundle: passphrase for an encrypted pack "
                         "(or the PACK_PASSPHRASE env var)")
    ap.add_argument("--verify", action="store_true",
                    help="import-bundle: recompute the pack's content hash and "
                         "refuse on mismatch (tamper check)")
    ap.add_argument("--near", action="store_true",
                    help="dedupe: also find near-duplicates (same text, different wording)")
    ap.add_argument("--threshold", type=float, default=0.9,
                    help="dedupe --near: similarity floor (default 0.9)")
    ap.add_argument("--apply", action="store_true",
                    help="dedupe --near / edge-audit: apply changes (default reports only)")
    ap.add_argument("--model", help="pull: the HF model id to fetch (org/Name), "
                                    "or a row number from the last find")
    ap.add_argument("--revision", default="main", help="pull: repo revision (default main)")
    ap.add_argument("--include", help="pull: only repo files matching this glob "
                                      "(how a single GGUF quant is pulled)")
    ap.add_argument("--query", help="find: search words (the CLI also takes them "
                                    "positionally: ./vinur.sh find qwen3 32b)")
    ap.add_argument("--out", help="split: output directory for bundle files")
    ap.add_argument("--license", help="source: SPDX licence id (e.g. CC-BY-NC-4.0, proprietary)")
    ap.add_argument("--license-holder", dest="license_holder",
                    help="source: to whom the licence/copyright belongs")
    ap.add_argument("--license-url", dest="license_url", help="source: licence terms URL")
    ap.add_argument("--gold", help="eval: path to the JSONL gold set (default eval_gold_path)")
    ap.add_argument("--retriever", default="current_path",
                    help="eval: which registered retriever to score (default current_path)")
    ap.add_argument("--trace", action="store_true",
                    help="eval: per-query debug logging (top card id, tier)")
    ap.add_argument("--kb", action="store_true",
                    help="reset: clear only the distilled KB (keep raw chunks)")
    ap.add_argument("--raw", action="store_true",
                    help="reset: clear only the raw chunk store (keep the KB)")
    ap.add_argument("-y", "--yes", action="store_true",
                    help="reset / clear-queue: skip the confirmation prompt")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    # remember the config file so the server's ops-runner launches subprocesses (and the
    # Settings panel writes) against the SAME config the server itself is using.
    cfg["_config_path"] = args.config or os.environ.get("KNOWLEDGEHOST_CONFIG")
    for k in ("host", "port", "backend"):
        if getattr(args, k):
            cfg[k] = getattr(args, k)

    logging.basicConfig(
        level=getattr(logging, cfg["log_level"].upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    log = logging.getLogger("knowledgehost")

    if args.command == "reset":            # before _build, so we don't recreate files
        return _cmd_reset(cfg, args, log)

    if args.command == "import-conceptnet":   # KB-only; no raw store / embed endpoint
        return _run_import_conceptnet(
            cfg, log, path=args.path, min_weight=args.min_weight,
            include_lexical=args.all,
            exclude=([s.strip() for s in args.exclude.split(",") if s.strip()]
                     if args.exclude is not None else None),
            limit=args.limit)

    if args.command == "import-atomic":       # KB-only; no raw store / embed endpoint
        return _run_import_atomic(cfg, log, path=args.path,
                                  min_count=args.min_count, limit=args.limit)

    if args.command == "import-glucose":      # KB-only; no raw store / embed endpoint
        return _run_import_glucose(cfg, log, path=args.path,
                                   min_count=args.min_count, limit=args.limit)

    if args.command == "import-causenet":     # KB-only; no raw store / embed endpoint
        return _run_import_causenet(cfg, log, path=args.path,
                                    min_sources=args.min_sources, limit=args.limit)

    if args.command == "unimport":            # KB-only; provenance-aware undo of a bulk import
        return _run_unimport(cfg, log, dataset=args.dataset)

    if args.command == "migrate-vocab":       # KB-only; pre-1.2 -> 1.2 neutral vocabulary
        return _run_migrate_vocab(cfg, log)

    if args.command == "reconcile":           # KB-only; vectorised, no LM/embed endpoint
        return _run_reconcile(cfg, log, anchors=args.anchors, limit=args.limit,
                              top_k=args.top_k)

    if args.command == "link":                # KB + an LM; no raw store / embed endpoint
        return _run_link(cfg, log, limit=args.limit, top_k=args.top_k, fast=args.fast)

    if args.command == "build-ann":           # KB-only; builds the HNSW node index
        return _run_build_ann(cfg, log)

    if args.command == "optimize":            # KB-only; one-time node layout fix
        return _run_optimize(cfg, log, vacuum=args.vacuum)

    if args.command == "edge-audit":          # KB-only; LM-free graph-hygiene pass
        return _run_edge_audit(cfg, log, apply=args.apply, limit=args.limit)

    if args.command in ("bundles", "split", "source", "scenario"):   # modular §16, KB-only
        return _run_bundles(cfg, args, log)
    if args.command == "pack":                # clean-room pack producer (PACK-01 §3)
        return _run_pack(cfg, log, args)      # clean-room: opens its own scratch store
    if args.command == "collect":             # add one doc to a shareable .kdb collection
        return _run_collect(cfg, log, args)   # clean-room: opens its own scratch store
    if args.command == "analyze":             # propose a structure profile (no ingest)
        return _run_analyze(cfg, log, args)
    if args.command == "citations":           # deterministic cross-reference graph (no LM)
        return _run_citations(cfg, log)

    if args.command == "import-bundle":       # absorb a shipped brain into the master
        import json as _json
        from . import bundles as B
        path = (args.args[0] if args.args else None) or args.path
        if not path:
            log.error("import-bundle needs the file: import-bundle <file.kdb> "
                      "[--name N] [--trust low|keep]")
            return 1
        try:
            res = B.import_bundle(cfg, path, name=args.name, trust=args.trust,
                                  log_fn=log.info, verify=args.verify,
                                  passphrase=(args.passphrase
                                              or os.environ.get("PACK_PASSPHRASE", "")))
        except ValueError as e:
            log.error("%s", e)
            return 1
        kb = KB({**cfg, "ann_search": False})   # facet the new rows right away
        try:
            res["facetized"] = kb.facetize()
        finally:
            kb.close()
        print(_json.dumps(res, indent=2))
        return 0
    if args.command == "eject-bundle":        # export, then permanently remove one bundle
        import json as _json
        from . import bundles as B
        bundle = (args.args[0] if args.args else None) or args.bundle
        if not bundle:
            log.error("eject-bundle needs the bundle name: eject-bundle <bundle> "
                      "[--dry-run] [--no-export]")
            return 1
        try:
            st = B.eject_bundle(cfg, bundle, export_first=not args.no_export,
                                dry_run=args.dry_run, log_fn=log.info)
        except ValueError as e:
            log.error("%s", e)
            return 1
        print(_json.dumps(st, indent=2))
        return 0

    if args.command == "eval":                # retrieval eval harness (KB + embedder)
        return _run_eval(cfg, log, gold=args.gold, retriever=args.retriever, trace=args.trace)

    if args.command == "ingest-library":      # index the search-only document library
        return _run_ingest_library(cfg, log, force=args.force)
    if args.command == "rebuild-fts":         # migrate FTS to the configured tokenizer
        return _run_rebuild_fts(cfg, log)

    if args.command == "facetize":            # backfill the multi-axis facet layer (KB-only)
        kb = KB(cfg)
        try:
            import time as _t
            t0 = _t.perf_counter()
            counts = kb.facetize(limit=args.limit)
            log.info("facetized %s in %.1fs", counts, _t.perf_counter() - t0)
            import json as _json
            print(_json.dumps({"facetized": counts, "facet_counts": kb.facet_counts()}, indent=2))
        finally:
            kb.close()
        return 0

    store, embedder = _build(cfg)

    if args.command == "bump-version":
        v = int(store.manifest.meta_get("version", "1")) + 1
        store.manifest.meta_set("version", v)
        print(f"version -> {v}")
        store.close()
        return 0

    if args.command == "stats":
        from . import ann as ann_mod
        kb = KB(cfg)
        ann_path = os.path.expanduser(cfg.get("ann_path") or (cfg["kb_path"] + ".ann"))
        ann_status = ("built" if ann_mod.index_exists(ann_path)
                      else ("absent (run build-ann)" if ann_mod.available()
                            else "absent (usearch not installed)"))
        print({"backend": store.backend, "chunks": store.count(),
               "dense": store.has_vectors(),
               "version": store.manifest.meta_get("version", "1"),
               "kb": kb.counts(), "ann_index": ann_status})
        kb.close(); store.close()
        return 0

    if args.command == "distill":
        store.close()                          # _run_distill owns its own store handle
        return _run_distill(cfg, embedder, log, limit=args.limit,
                            watch=args.watch, interval=args.interval,
                            bundle=getattr(args, "bundle", None))

    if args.command == "refine":               # raw store (source) + KB + big LM
        return _run_refine(cfg, store, embedder, log, limit=args.limit, force=args.force)

    if args.command == "recard":               # raw store + KB + big LM, cards only
        return _run_recard(cfg, store, embedder, log, limit=args.limit,
                           bundle=getattr(args, "bundle", None),
                           all_families=getattr(args, "all_families", False),
                           before=getattr(args, "before", None),
                           since=getattr(args, "since", None))

    if args.command == "find":                 # hub search + fits-this-machine verdicts
        store.close()
        from . import modelfind
        from .amiga_net import broker as _broker
        from .serving import proxy_env
        os.environ.update(proxy_env(cfg))      # the broker honours the proxy too
        query = (args.query or " ".join(args.args)).strip()
        if not query:
            log.error("find needs a search query:  ./vinur.sh find <words>")
            return 2
        try:
            n = modelfind.find(query, limit=args.limit or 8,
                               say=lambda m: log.info("%s", m))
            ops_mod.emit_result(True, query=query, results=n)
            return 0
        except _broker.EgressDenied as e:
            log.error("%s", e)
        except Exception as e:
            log.error("find failed: %s", e)
        ops_mod.emit_result(False, query=query)
        return 1

    if args.command == "adopt":                # legacy hub cache -> models/ store
        store.close()
        from .serving import adopt_cached
        n = adopt_cached(args.model or "", say=lambda m: log.info("%s", m))
        ops_mod.emit_result(True, adopted=n)
        return 0

    if args.command == "pull":                 # model weights via the egress broker
        store.close()
        if not args.model:
            log.error("pull needs --model org/Name (or a row number from find)")
            return 2
        from .amiga_net import broker as _broker
        from .amiga_net import pull as _pull
        from .serving import proxy_env
        os.environ.update(proxy_env(cfg))      # the broker honours the proxy too
        model, include = args.model, args.include or ""
        if model.isdigit():                    # a row number from the last find
            from . import modelfind
            sel = modelfind.pick(int(model))
            if not sel:
                log.error("no row %s in the last find — run ./vinur.sh find "
                          "<query> first, or give the org/Name id", model)
                return 2
            model, saved_include = sel
            include = include or saved_include
            log.info("pull row %s -> %s%s", args.model, model,
                     f" (only {include})" if include else "")
        try:
            _pull.pull(model, revision=args.revision, include=include,
                       say=lambda m: log.info("%s", m))
            ops_mod.emit_result(True, model=model)
            return 0
        except _broker.EgressDenied as e:
            log.error("%s", e)
        except KeyboardInterrupt:
            log.info("pull interrupted — partial files are kept; re-run to resume")
        except Exception as e:
            log.error("pull failed: %s — partial files are kept; re-run to resume", e)
        ops_mod.emit_result(False, model=model)
        return 1

    if args.command == "dedupe":               # janitor: duplicate text, no LM
        return _run_dedupe(cfg, store, log, near=getattr(args, "near", False),
                           threshold=getattr(args, "threshold", 0.9),
                           apply=getattr(args, "apply", False),
                           bundle=getattr(args, "bundle", None))

    if args.command == "adjudicate":
        store.close()
        return _run_adjudicate(cfg, log, limit=args.limit, batch=args.batch,
                               watch=args.watch, interval=args.interval,
                               auto=not args.no_auto, auto_only=args.auto_only,
                               fast=args.fast)

    if args.command == "embed-nodes":
        store.close()                          # KB + embed endpoint only
        return _run_embed_nodes(cfg, embedder, log, limit=args.limit)

    if args.command == "clear-queue":          # revert the distillation queue (+ quarantine files)
        rc = _run_clear_queue(cfg, store, log, include_partial=args.include_partial,
                              quarantine=not args.no_quarantine,
                              dry_run=args.dry_run, yes=args.yes)
        store.close()
        return rc

    if args.command == "ingest":
        if not embedder.embed_one("warmup", "document"):
            if cfg["backend"] == "lance":
                log.error("embed endpoint unreachable — the lance backend can't store "
                          "chunks without vectors. Start the embed server and retry.")
                store.close()
                return 1
            log.warning("embed endpoint unreachable — ingesting WITHOUT vectors "
                        "(sparse FTS only; re-run after it's up to add dense).")
        try:
            stats = ingest_mod.crawl(store, embedder, cfg, force=args.force)
            log.info("documents: %s", stats)
            ops_mod.emit_result(stats.get("docs", 0) > 0 or stats.get("chunks", 0) > 0,
                                **stats)
            if args.wikipedia:
                wstats = ingest_mod.ingest_wikipedia(store, embedder, cfg,
                                                     limit=args.limit, force=args.force)
                log.info("wikipedia: %s", wstats)
        except ingest_mod.EmbedUnavailable as e:
            log.error("embed endpoint dropped mid-ingest (%s) — aborted; nothing lost, "
                      "re-run to resume.", e)
            store.close()
            return 1
        if cfg["backend"] == "lance" and hasattr(store, "maybe_build_ann"):
            store.maybe_build_ann()
        rc = 0
        if args.distill:                       # opt-in: distil right after ingest
            store.close()                      # _run_distill opens its own handle
            return _run_distill(cfg, embedder, log)
        store.close()
        return rc

    # default: serve.  Modular knowledge (§16): if a non-'all' scenario / bundles
    # are configured, assemble a disposable working DB from the selected sources and
    # serve THAT — the master kb.db stays the untouched authoring source of truth
    # (maintenance verbs still edit the master via the config's kb_path).
    from . import bundles
    cfg["_master_kb_path"] = cfg["kb_path"]
    try:
        work = bundles.assemble_working_db(cfg, log_fn=log.info)
    except Exception as e:                          # never let assembly wedge startup
        log.error("working-DB assembly failed (%s) — serving master kb.db", e)
        work = cfg["kb_path"]
    if work != cfg["kb_path"]:
        cfg["kb_path"] = work
        cfg["ann_path"] = ""                        # derive "<working>.ann", not the master's
        log.info("modular: serving scenario '%s' from %s",
                 bundles.active_scenario_name(cfg), work)
    kb = KB(cfg)
    lib = _library_store(cfg, log)              # search-only document library (if built)
    if lib is not None:
        log.info("document library loaded (%d chunks) — library_search tool enabled",
                 lib.count())
    tools = Tools(store, embedder, cfg, kb, library_store=lib)
    server.serve(cfg, store, tools, kb)
    return 0


if __name__ == "__main__":
    sys.exit(main())
