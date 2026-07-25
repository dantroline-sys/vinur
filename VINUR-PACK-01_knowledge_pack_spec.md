# VINUR-PACK-01 — Knowledge Packs (clean-room export / import of shareable distillates)

**Status:** Draft for review · **Doc version:** 1.0 · **Date:** 2026-07-25
**Component:** `python3 -m knowledgehost pack` (producer) + `import-brain` extensions (consumer)
**Builds on:** the `.kdb` bundle machinery (`bundles.py`, manifest format 1) — packs are `.kdb`
files with a richer manifest and a defined production method; every existing import guarantee
(content-hash idempotence, source rebranding, trust capping, embed-mismatch strip) is inherited.

RFC 2119 keywords (MUST, MUST NOT, SHOULD, MAY) are normative.

---

## 1. Scope and non-goals

A **pack** is a single-file, self-contained knowledge artifact — the distillate of one document
or one folder — produced in a **clean-room build** (an empty scratch kb, so every row is
traceable to the declared sources and their licenses), stamped with authorship, dates, data
kind, licensing, and a compatibility block, and stored under a well-known folder for sharing
through any file channel (git repo, HTTP directory, USB stick).

In scope:

1. The `pack` verb: point vinur at a file or folder; it builds the distillate in isolation and
   emits `packs/<slug>/<slug>-<version>.kdb[.gz]` plus a JSON sidecar.
2. Manifest **format 2**: identity, authorship, data kind, licensing, compatibility, pipeline
   provenance.
3. The **license gate** at export.
4. Optional gzip compression and optional passphrase encryption of the artifact.
5. Import-side extensions: compressed/encrypted artifacts, sidecar hash verification, and the
   compatibility decision table with named remedies.

Out of scope (v1 — see Appendix A):

- **Raw source text in packs.** Exports ship the six knowledge tables only (distillate + ≤25-word
  evidence spans). A `--with-text` variant is deferred and MUST NOT ship before the license gate
  can prove redistribution rights per source.
- **Usage-control DRM.** Encryption (§7) controls *who can open* a pack in transit and at rest.
  Once decrypted and imported, its content is ordinary kb rows on the consumer's box. This spec
  makes no stronger claim and implementations MUST NOT advertise one.
- **Cross-pack node-ID canonicalization.** Packs join the consumer's graph by label + embedding
  at import (existing `link_to_node` / adjudication machinery), never by pre-agreed IDs.
- **Signing / trust webs, delta packs, marketplace metadata.**

## 2. Definitions

- **Pack** — a `.kdb` file (optionally compressed/encrypted) whose manifest has `format: 2` and
  `kind: "pack"`, plus its JSON **sidecar**.
- **Clean-room build** — production against a scratch kb + store initialized empty, so the pack
  contains only rows derived from the declared inputs.
- **Sidecar** — `<artifact>.manifest.json`, a byte-identical copy of the manifest written beside
  the artifact. It exists so repositories can be browsed — and encrypted packs previewed —
  without opening SQLite. The in-file manifest is authoritative when both are readable.
- **Effective license** — the most restrictive license among the pack's sources (§5).
- **Generic pack** — `compat.requires == []` (the default; clean-room builds are generic by
  construction). **Contract pack** — a pack that names prerequisites (e.g. a CONF-01 overlay);
  these are produced by other toolchains, not by the `pack` verb.
- **Hard / soft compatibility** — a hard mismatch refuses the import and names the remedy; a
  soft mismatch proceeds with a stated degradation or an automatic repair.

## 3. The `pack` verb (producer)

```
python3 -m knowledgehost pack --path <file|folder>
    [--name <slug>] [--title <text>] [--author <name>] [--contact <text>]
    [--pack-version <semver>] [--describe <text>]
    [--license <SPDX>] [--allow-unlicensed]
    [--compress] [--encrypt] [--keep-build] [--force]
```

Also an ops-registry verb (typed args → panel Ops tab and the Bundles tab's "Build pack…").

### 3.1 Pipeline (normative order)

1. **Scratch workspace** at `var/packs/build/<slug>/`: its own `kb.db` and chunk store. The
   derived config MUST override every data-path key and MUST inherit the live serving/embed
   endpoints, lease settings, and tuning. The master kb MUST NOT be opened for writing at any
   stage.
2. **Ingest** `--path` (same formats, zones, furniture rules as normal ingest).
3. **Distill** — the standard pipeline, unmodified (two-tier or collapsed, fan-out, worker
   resilience, `distill_max_tokens` budget). Inline family extraction stamps recard-current.
4. **Link** — one internal pass so the pack ships with its own connectivity.
5. **License gate** (§5). A refusal here stops the build with the per-source verdict table.
6. **Manifest** (§4) written into the file; `VACUUM`.
7. **Package**: optional gzip (§6), optional encryption (§7); artifact moved atomically to
   `packs/<slug>/`; sidecar written last.
8. **Cleanup**: scratch removed on success (`--keep-build` retains it). On failure or interrupt
   the scratch MUST be retained — the distilled set is the checkpoint, so re-running the same
   `pack` command resumes rather than restarts.

### 3.2 Output location and naming

- Folder: `packs/<slug>/` under the vinur tree (`pack_dir` config key; default `packs/`).
- Artifact: `<slug>-<version>.kdb`, `.kdb.gz` when compressed, `.kdb.enc` when encrypted
  (`.kdb.gz.enc` when both — compress before encrypt, always).
- `<slug>` from `--name`, else the input's stem, slugged by the existing `_slug` rules.
- An existing artifact of the same slug+version MUST NOT be overwritten without `--force`;
  the refusal names the bump (`--pack-version`).

### 3.3 One job at a time

`pack` runs under the single-slot ops runner like every other job and respects GPU leases; a
long build is interruptible and resumable (§3.1.8).

## 4. Manifest format 2

Stored as the single row of `bundle_manifest` (as today) and mirrored to the sidecar.
`read_manifest` MUST keep accepting format 1; format 2 adds:

```jsonc
{
  "format": 2, "kind": "pack",
  "pack": {
    "id": "<slug>", "title": "...", "version": "1.0.0",
    "created": 1690000000.0,
    "producer": {"name": "...", "contact": "..."},        // --author / --contact
    "description": "...",                                  // --describe
    "data_kind": {                                         // derived, not asserted
      "regimes": {"empirical": 812, "historical": 40},     // node counts per regime
      "domains": ["..."],                                  // facet domains present
      "card_types": {"procedure": 61, "criteria": 34}
    }
  },
  "content": {
    "counts": { /* per exported table, as format 1 */ },
    "content_hash": "sha256:..."                           // canonical row serialization
  },
  "licensing": {
    "effective": "CC-BY-SA-4.0",
    "shareable": true,                                     // false ⇒ gate was overridden
    "attribution": ["Author, Title, source URL", "..."],
    "per_source": [{"doc_id": "...", "title": "...", "license": "...",
                    "holder": "...", "url": "...", "attested": false}],
    "text_included": false                                 // always false in v1
  },
  "compat": {
    "schema_version": 12,                                  // kb migration level
    "embed_model": "...", "embed_dim": 1024,
    "vocab_hash": "sha256:...",                            // FEATURE_VOCAB, canonical JSON
    "card_families": {"branches": 1, "enumerations": 2, "procedures": 3, "criteria": 3},
    "requires": []                                         // generic packs: always empty
  },
  "pipeline": {                                            // reproducibility, not compat
    "distill_model": "...", "distill_max_tokens": 8192,
    "recard_version": 3, "vinur_commit": "..."
  }
}
```

MUST: `format`, `kind`, `pack.id/version/created`, `content`, `licensing.effective/shareable`,
`compat.schema_version/embed_model/card_families`. SHOULD: everything else. Unknown fields MUST
be preserved by tools that rewrite manifests.

## 5. License gate

Source licenses come from the existing registry fields (detected at ingest — SPDX tag / CC URL /
copyright line — or set via the `source` verb). At export every source is classified:

| Class (examples)                          | Verdict                                                    |
| ----------------------------------------- | ---------------------------------------------------------- |
| Public domain, CC0                         | export                                                     |
| CC-BY                                      | export; attribution roster MUST list the source            |
| CC-BY-SA (and compatible copyleft)         | export; pack `effective` becomes the share-alike license   |
| Any NC variant                             | export; `effective` carries the NC term (stays visible)    |
| ND variants, proprietary, unknown, empty   | **REFUSE**                                                 |

- `effective` = the most restrictive verdict present; one refused source refuses the pack, and
  the refusal prints the per-source table with each remedy (`source --license`, or the flags
  below).
- `--license <SPDX>` fills EMPTY licenses only (never overrides a detected one) and marks those
  sources `attested: true` — the producer's claim, on the record.
- `--allow-unlicensed` builds anyway but stamps `shareable: false`; import of such a pack MUST
  print a provenance warning. This is the private-use escape hatch, not a sharing path.
- Rationale note for the record: packs contain distillate plus evidence spans capped at 25 words
  by the extraction schema — the gate governs derivative-work terms, not text redistribution
  (v1 ships no text; see non-goals).

## 6. Compression

`--compress` → gzip (stdlib), `.kdb.gz`. Import and `inspect` MUST sniff magic bytes
(`1f 8b`) and decompress to a temp file transparently; the sidecar always carries the manifest
regardless. SHOULD be suggested in the CLI hint when the raw artifact exceeds 64 MB. No other
codecs in v1 (zero-dependency discipline).

## 7. Encryption

Plain statement of what this is: **access control for distribution** — only key/passphrase
holders can open the artifact. It is not usage control; after decrypt+import the knowledge is
ordinary rows in the consumer's kb.

- **v1 — passphrase (symmetric):** `--encrypt` prompts for (or `PACK_PASSPHRASE` supplies) a
  passphrase; the artifact is a SQLCipher database via the existing `dbcrypt` driver lane
  (`sqlcipher3` / `pysqlcipher3`, optional dependency). Absent driver ⇒ refuse with the
  existing install-remedy message; MUST NOT fall back to plaintext. Import accepts
  `--passphrase`; `inspect` on an encrypted pack serves the sidecar only.
- **v2 — recipients (public-key), deferred:** `age`-style recipient encryption so a pack can be
  published readable only by named key holders. MUST arrive as an optional external tool or
  optional import, never a new hard dependency (dependency-ratchet discipline). Appendix A.

Compression composes as compress-then-encrypt (§3.2).

## 8. Import (consumer)

`import-brain` (CLI + panel) is extended, changing nothing for plain format-1 `.kdb` files:

1. **Unwrap**: sniff gzip → decompress to temp; encrypted → require passphrase, decrypt to temp.
2. **Verify**: if a sidecar sits beside the artifact, its `content_hash` MUST match the
   manifest's; recompute of the content hash SHOULD be offered (`--verify`) and any mismatch
   refuses with "artifact does not match its manifest".
3. **Compatibility decision table** (evaluated before any write):

| Check                     | Kind | On mismatch                                                                 |
| ------------------------- | ---- | --------------------------------------------------------------------------- |
| `schema_version` > host   | hard | refuse: "pack needs kb schema N, host has M — update vinur"                 |
| unknown `card_families`   | hard | refuse naming the family: "host doesn't know 'X' cards — update vinur"      |
| `embed_model`/`dim`       | soft | existing behavior: strip vectors, note names `embed-nodes` backfill + a time estimate at the measured embed rate |
| `vocab_hash`              | soft | warn: "feature vocabularies differ — fit-gating may be weaker for this pack" |
| `requires` non-empty      | soft | warn naming each missing pack and the import order (v1 does not enforce)    |
| `shareable: false`        | soft | provenance warning (§5)                                                     |

4. **Absorb** exactly as today: content-hash idempotent, sources rebranded to the pack id,
   support trust capped unless `trust='keep'`. The pack's manifest never sets its own trust.
5. **Follow-up note** printed on success: run `link` then `adjudicate` — a pack is a seed, not
   mature tissue; connectivity to the local graph grows in the first cycle.

## 9. Panel surface

Bundles tab gains: **Build pack…** (path picker + metadata form + live gate verdict per
source), a pack list over `packs/` rendered from sidecars (title, version, license, counts,
shareable flag), and import accepting `.kdb.gz`/`.kdb.enc`. Ops tab shows `pack` like any other
job. Help entries updated. Nothing else moves.

## 10. Acceptance checks (normative)

1. Clean-room isolation: a `pack` build on a box with a populated master leaves the master's
   file bytes untouched (mtime/hash compare in the test).
2. Resume: interrupt a build after partial distillation; re-run completes without re-distilling
   done chunks.
3. Gate matrix: one fixture source per row of §5's table produces exactly the specified
   verdict; the refusal output names every unlicensed source and both remedies.
4. `--license` fills only empty licenses and marks `attested`; detected licenses are never
   overwritten.
5. Round-trips: plain, `--compress`, `--encrypt` (skipped with a note when no SQLCipher
   driver), and compress+encrypt all import back to identical row counts; re-import is a no-op.
6. Compat table: each hard row refuses with its named remedy; each soft row proceeds and
   prints its stated warning; format-1 `.kdb` files still import unchanged.
7. Sidecar tamper: altering the artifact after sidecar write is caught by `--verify`.
8. Manifest v2 survives `inspect` → reserialize with unknown fields intact.

---

## Appendix A — deferred, in likely order

- **`--with-text` packs** (consumer-side re-distill/recard becomes possible): requires the gate
  to prove per-source redistribution rights; ships only for PD/CC0/CC-BY/CC-BY-SA source sets.
- **Recipient encryption** (`age` or equivalent, optional tooling only).
- **`requires` enforcement** at import (v1 warns), once contract packs exist in the wild.
- **Pack update flow** (eject old version → import new) as a one-click panel action; the
  mechanics already exist as two operations.
- **Signing.** Content hash + trusted transport covers v1; signatures only matter with an
  untrusted repository ecosystem, which does not exist yet.
