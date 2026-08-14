# Vinur — a local knowledge host

A standalone service that maintains a large, **local, offline** knowledge base
and answers over it with **cited passages** — a **Wikipedia snapshot** plus your
own **PDFs, books, journals and documents**, distilled into typed, cited
knowledge cards. Vinur is built to sit behind a front-end: there is no end-user
chat surface. What it does carry is a full web **control panel** (seven
two-level tabs — Ask, Distilled, Curation, Operations, Serving, Stats,
Settings) for operating the box: running maintenance jobs with live progress,
browsing what was distilled, managing and tuning served models, steering the
egress broker — no SSH needed.

**Vinur pairs with [Vinkona](https://github.com/dantroline-sys/vinkona)**, the
local voice assistant it was originally built for (*vinur* and *vinkona* are
Icelandic for a friend). The two lived in one repository until 2026-07-13 and
were split so each can be licensed and developed on its own terms: Vinur stays
**Apache 2.0**; Vinkona (the user-facing front-end) continues under PolyForm
Noncommercial. Nothing here depends on Vinkona: the seam is one small HTTP
contract (`GET /tools` + `POST /call`, see
[MAC_TOOLS.md](https://github.com/dantroline-sys/vinkona/blob/main/assistant/MAC_TOOLS.md)),
so any client that speaks it — Vinkona, a script, `curl` — can call `kb_search`
like any other tool. The design rationale lives in
[KNOWLEDGE.md](https://github.com/dantroline-sys/vinkona/blob/main/assistant/KNOWLEDGE.md).

It is a **separate store from Vinkona's `memories`**: bulk, low-trust, reference-
only, with its own ANN/FTS index. It returns **data, never instructions** —
every passage is sanitized and cited before any LM reads it.

## Two halves

- **Query service** (`serve`) — light, fast, always up. The tool host Vinkona calls.
- **Ingestion + distillation pipeline** — heavy, batch, run on demand or by the
  built-in **Prioritizer** when idle: `ingest` (crawl → sanitize → chunk →
  embed) → `distill` (raw chunks → typed nodes/edges/cards via the big LM,
  two-tier extract/verify) → `recard` / `link` / `refine` / `reconcile` /
  `adjudicate` / `facetize` re-passes, plus janitors (`dedupe`, `edge-audit`).
  Some 45 CLI verbs share one dispatch (`python3 -m knowledgehost <verb>`); the
  maintenance set is runnable from the panel's Operations tab with typed,
  documented arguments and a progress strip that follows you onto every tab.

## Two store backends (one interface)

| backend  | sparse | dense | needs | use for |
|----------|--------|-------|-------|---------|
| `sqlite` *(default)* | FTS5 | brute-force (numpy or pure-python) | **nothing** beyond the stdlib | the PDF collection / Phase 1; fine to ~1M chunks |
| `lance`  | LanceDB FTS | IVF-PQ, on-disk/mmap | `./install.sh --lance` | a **full Wikipedia** snapshot (10–40M chunks) |

Retrieval is **hybrid** on either: a dense (embedding) arm and a sparse (BM25/
FTS) arm, fused by **Reciprocal Rank Fusion**, then reranked. FTS carries exact
terms (proper names, IDs) where embeddings are weakest; fusion is the biggest
quality lever at encyclopedia scale.

## Quick start

```bash
# 0) install — builds .venv (uv, bootstrapped in-tree; deps pinned by uv.lock),
#    seeds config.toml, runs the offline smoke test
./install.sh                             # stdlib+numpy (sqlite backend)
./install.sh --all                       # + every parser (pdf/epub/html/zim) + lance
./install.sh --pdf --epub                # or pick the formats your collection has

# edit config.toml: sources, backend, embed_url …

# 1) ingest your documents (incremental; only new/changed files are processed)
./ingest.sh                              # crawls config's `sources`
./ingest.sh --wikipedia                  # also a configured Kiwix ZIM
./ingest.sh --wikipedia --limit 500      # cap ZIM articles (smoke)

# 2) serve the query tool
./run.sh                                 # http://127.0.0.1:8771

# later: ./install.sh status | uninstall [--purge]
```

**Filesystem guarantee:** everything the host writes stays inside this folder —
indexes and the knowledge base in `var/`, caches and temp in `var/cache` +
`var/tmp`, the reranker model in `models/`, your settings in `config.toml`
(see `env.sh`). Reads (`sources`, `zim_path`) can point anywhere. Relative
paths in config resolve against this folder, absolute paths are honoured.
`./install.sh uninstall` removes the software; `--purge` also removes the
knowledge base; deleting the folder removes every trace.

The embed endpoint (nomic at `127.0.0.1:11437`, shared with Vinkona's memory
store) is **optional**: if it's down, ingestion and search run **sparse-only**
(FTS) and log it; re-run `ingest` once it's up to add the dense vectors.

## Verify it (zero installs)

```bash
bash tests/make_fixtures.sh /tmp/kb-fixtures   # tiny txt/md/html corpus
python3 tests/smoke.py /tmp/kb-fixtures        # ingest -> kb_search -> cited passages
```

```bash
curl -s localhost:8771/health
curl -s localhost:8771/tools
curl -s -X POST localhost:8771/call -H 'Content-Type: application/json' \
     -d '{"name":"kb_search","arguments":{"query":"who discovered the Krebs cycle","k":3}}'
```

`kb_search` returns a JSON object (string-encoded per the contract):

```json
{ "passages": [ {"text","title","section","path_or_url","source_type","score"}, … ],
  "confidence": 0.62, "low_confidence": false, "dense_used": true }
```

`confidence` is the top rerank score; **`low_confidence`** is the signal for
Vinkona to fall back to web search instead of answering from a weak passage.

## Ingestion: what's supported

| format | extractor | dependency (lazy) |
|--------|-----------|-------------------|
| `.txt` / `.md` | section split on Markdown headings | stdlib |
| `.html` / `.htm` | `trafilatura` else a stdlib `html.parser` sectioner | `trafilatura` (optional) |
| `.pdf`  | PyMuPDF text layer + TOC sections; **OCR fallback** for scanned pages | `pymupdf`; `ocrmypdf`/`tesseract` on PATH |
| `.epub` | `ebooklib`, chapters through the HTML sectioner | `ebooklib` |
| Wikipedia | Kiwix **ZIM** (pre-rendered HTML) via `libzim`, split on `<h2>/<h3>` | `libzim` |

A **manifest** (path, content_hash, mtime, version) makes every run incremental;
chunk ids are `sha1(path+section+text)` so re-ingest is idempotent. A monthly
Wikipedia refresh: drop in the new ZIM, `bump-version`, re-ingest.

### Canonical texts (scripture & legal)

Structured corpora get their own lane: `analyze` proposes a structure profile
(scripture or legal — books/chapters/verses, articles/sections), the collect
wizard **confirms before ingesting** (deferred questions land in the panel's
"Needs your input" inbox), and reference maps plus first-class **editions** do
the rest — deuterocanon handled, multi-edition verse alignment with parallel
reading (`read`), a deterministic cross-reference graph (`citations`),
commentary layering, and Vulgate↔Hebrew Psalm reconciliation (`psalms`).
Domain card lenses (themes/parallels for scripture; definitions/obligations/
exceptions for legal) ride the normal distill pass.

### Duplicate text

That id covers the same document arriving twice by the same route, but not the
same *text* arriving by a different one — a research drop re-exported under a
new name, one PDF filed in two folders. Distillation therefore also claims each
chunk's **normalised-text hash** (`distill_dedupe`, on by default): whoever
claims it first gets distilled, and anything else holding that text is
checkpointed against the winner instead of costing a second set of LM calls.
Nothing is deleted — the row and its FTS entry stay, so search finds it by
either path, and `chunk_dupes` records where else the text lives.

For a store that already has duplicates in it, the janitor sweeps:

```bash
python3 -m knowledgehost dedupe                    # exact — marks them, no LM
python3 -m knowledgehost dedupe --near             # + report near-duplicates
python3 -m knowledgehost dedupe --near --apply     # …and act on them
```

Near-duplicate detection is MinHash/Jaccard over word shingles — for the same
answer written twice in different words. It reports by default rather than
marking, because "almost the same" also describes a revision, which you usually
want to keep and distil.

## Wiring into Vinkona

Already wired on the Vinkona side (see its `assistant/config.py` `knowledge`
block and the `MultiHost` build in `cascade_server.py`):

```toml
knowledge = { enabled = true, tool_url = "http://127.0.0.1:8771" }
```

On the research path, prefer `kb_search` **before** the web (local-first); use
web for recency or when `low_confidence` is set.

`kb_search` is one of **seven tools** on the catalogue (`GET /tools`), all
callable through the same `POST /call` contract:

- `kb_ask` — structured answer from the distilled cards: fit-gated, with a
  confidence band, provenance and surfaced contradictions; it abstains rather
  than guesses.
- `kb_reason` — deterministic graph reasoning, no generation: compare, paths,
  about, effects, siblings, contradictions, verify.
- `kb_investigate` — a multi-hop agentic graph walk with a synthesis
  fact-checked against the graph.
- `library_search` — the search-only lexical document library.
- `kb_brain` — list/load/unload knowledge bundles at runtime.
- `ops_annotate` — the external-oracle annotation surface (VINUR-OPS-01).

## Knowledge packs, bundles & brains

Every source group is a **bundle**, exportable to its own `.kdb` file and
shippable. `pack` produces a **knowledge pack** — a clean-room ingest+distill
of one file or folder in a scratch kb, license-gated at export,
manifest-stamped (authorship, licensing, compatibility), optionally gzip'd or
passphrase-encrypted. `collect` is its sibling for growing a shareable
collection document-by-document, with a completeness manifest so an unchanged
re-collect is a fast no-op. `import-bundle` absorbs a shipped `.kdb` at capped
trust; **brains** are bundles you can load/unload live (`/brain`, `kb_brain`)
without touching the master. A Vinkona machine that never runs a host can
still consume packs **in-process** through its `LocalKB` tier — vinur's read
path imported as a zero-hard-dependency library. Contract:
[`VINUR-PACK-01_knowledge_pack_spec.md`](VINUR-PACK-01_knowledge_pack_spec.md).

## Running Vinur on its own machine (with its own LMs)

Vinur never serves a chat LM itself — distillation and verification are just
OpenAI-compatible endpoints in config (`distill_urls` / `extract_urls` /
`verify_urls`). On a box of its own (say, the big-VRAM machine, with Vinkona
elsewhere), declare what this machine serves in config.toml's `[serving]`
table and let `./vinur.sh` supervise all of it — the kb, the vLLM chat
model(s), the nomic embed endpoint, and the CPU reranker:

```bash
./install.sh --serving --llama  # vLLM venv (big chat models) + an in-tree
                                #   llama-server build (embed + reranker)
$EDITOR config.toml           # [[serving.llms]] entries + embed/reranker — see the example
./vinur.sh start              # everything up, watched, logged to var/log/
./vinur.sh status             # dead services show the reason line from their log
./vinur.sh logs [svc]         # follow;  restart [svc] / stop as expected
```

Which weights to get (file formats per engine, download commands, where they
land, a recommended 96 GB pairing) is covered in
[`serving/README.md`](serving/README.md) — including **exclusive swap mode**
for models too big to co-reside: mark entries `exclusive = true` and one runs
at a time, swapped via `./vinur.sh swap <name>`, `POST /serving/swap`, or a
Prioritizer step's `"model"` key (so distill batches under one model, then
verify batches under the other).

Model acquisition is built in and **brokered**: `./vinur.sh find <words>`
searches the hub with every hit sized and judged against this machine
(fits / tight / too big), `./vinur.sh pull` downloads through the egress
broker (policy-checked, audited, resumable — engines then run offline from
the local store), and `adopt` absorbs legacy HF-cache snapshots without
re-downloading. `./vinur.sh minimal on` vacates all VRAM while the KB keeps
serving (Serving › Schedule drives it on a weekly timer);
`./vinur.sh service install` registers a systemd/launchd start-at-login unit;
`status --json` is the machine seam for scripts and shells; `./vinur.sh net`
prints the broker's window — policy, open leases, recent audited egress.

With `[serving]` empty (the default), `./vinur.sh` simply supervises the kb —
a one-machine Vinkona setup keeps using Vinkona's own tiers as before.

To let a remote Vinkona reach this host, set `host = "0.0.0.0"` **and**
`auth_token` (the server refuses a LAN bind without a token, because `/ops`
runs maintenance jobs). On the Vinkona side point `knowledge.tool_url`,
`knowledge_host.url`, and — for the research hand-off — the exporter at this
box: with `research.export.folder = "http://this-box:8771"` (plus `token`),
solved-research drops POST to this host's `/drop` route and land in
`research_solved_dir` exactly as if the folder were shared. The web control
panel (`http://this-box:8771/`) works over the same connection, so
maintenance needs no SSH.

## Security

- All ingested content is **UNTRUSTED**; the tool returns data, fenced as low-
  trust by Vinkona before any LM reads it. Passages can colour an answer, never
  issue commands.
- Filenames are treated as **opaque data** — never shelled or prompt-interpolated.
- Service binds **localhost by default**; a non-loopback bind *requires*
  `auth_token` — the server refuses to start otherwise. Every POST route and
  every control/status GET carries the Bearer gate. The deliberate exceptions
  are the read-only query routes (`/search`, `/ask`, `/library`, `/tools`,
  `/health` and the viewer's browse endpoints), which stay public so a LAN
  client can ask questions — they can read the knowledge, never change it,
  and `/ask`'s gap logging is capped so an anonymous client cannot grow the
  research queue unboundedly.
- **Egress is deny-by-default.** Every byte leaving the box goes through the
  `amiga_net` broker under lease-only policy rules (`egress.toml`), one JSON
  audit line per decision; inference engines launch with `HF_HUB_OFFLINE=1`
  and never see the Hugging Face token. The panel's Settings › Network tab
  runs a **leak check** — binds, policy, running-engine environment, token
  file permissions — graded with traffic lights and a one-line fix per
  finding. The scanned inventory lives in
  [`docs/net-inventory.md`](docs/net-inventory.md).
- Settings editable over HTTP are **allowlisted and fail closed** on sensitive
  names — no endpoint, path, token or bind key is reachable from the panel.
- Keep parsers (PyMuPDF/Tesseract) patched; parsing needs no network.

## Layout

The package holds ~55 modules; these are the load-bearing ones:

```
knowledgehost/
  config.py     defaults < TOML < env (KNOWLEDGEHOST_*); HTTP-editable keys allowlisted
  store.py      SqliteStore + LanceStore behind make_store(); shared SQLite manifest
  ingest.py     incremental crawl + Wikipedia ZIM; sanitize -> chunk -> embed -> upsert
  structure.py  canonical-text analyze/confirm (scripture/legal); refmaps/ + editions
  distill.py    raw chunks -> typed nodes/edges/cards (two-tier extract/verify, fan-out)
  kb.py         the structured KB: nodes/edges/cards, facets, gaps, provenance
  retrieval.py  hybrid dense+FTS read path -> RRF -> rerank (with rerank.py)
  tools.py      the seven-tool catalogue (kb_search/kb_ask/kb_reason/kb_investigate/
                library_search/kb_brain/ops_annotate)
  reason.py     deterministic graph reasoning + the derive layer; investigate.py walks
  server.py     stdlib HTTP: query routes, /call, ops/serving/net control routes
  viewer.py     the whole control panel — one file, seven two-level tabs
  ops.py        the panel's job runner: 32 maintenance verbs, typed args, progress
  autopilot.py  the Prioritizer — ordered idle-work plan + scheduling windows
  bundles.py    provenance bundles, .kdb export/import, brains; pack.py packs+collect
  serving.py    [serving] engines (vllm | llama.cpp), tune.toml, exclusive swap
  supervisor.py ./vinur.sh's engine — the kb + serving services, watched; minimal mode
  amiga_net/    the egress broker: policy, leases, pull, audit; posture.py leak check
  sources/      pdf, epub, html, text, wikipedia extractors (heavy deps lazy)
tests/          make_fixtures.sh + smoke.py + the scripts/gates.sh battery
```

## Disclaimer

This software is provided as-is, for research and reference purposes, without
warranty, and is not validated or intended for production or safety-critical
use.

## License

Apache License 2.0 — see [LICENSE](LICENSE). Vinur was split out of the
[Vinkona monorepo](https://github.com/dantroline-sys/vinkona) on 2026-07-13 and
continues under Apache 2.0 as its own project.
