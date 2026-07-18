# Vinur — a local knowledge host

A standalone service that maintains a large, **local, offline** knowledge base
and answers over it with **cited passages** — a **Wikipedia snapshot** plus your
own **PDFs, books, journals and documents**, distilled into typed, cited
knowledge cards. Vinur is a more or less **headless API**: apart from its web
control panel (config, operations, browsing) there is no end-user interface —
it is built to sit behind a front-end.

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

- **Query service** (`serve`) — light, fast, always up. The tool Vinkona calls.
- **Ingestion pipeline** (`ingest`) — heavy, batch, run on demand / monthly.

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

## Wiring into Vinkona

Already wired on the Vinkona side (see its `assistant/config.py` `knowledge`
block and the `MultiHost` build in `cascade_server.py`):

```toml
knowledge = { enabled = true, tool_url = "http://127.0.0.1:8771" }
```

On the research path, prefer `kb_search` **before** the web (local-first); use
web for recency or when `low_confidence` is set.

## Running Vinur on its own machine (with its own LMs)

Vinur never serves a chat LM itself — distillation and verification are just
OpenAI-compatible endpoints in config (`distill_urls` / `extract_urls` /
`verify_urls`). On a box of its own (say, the big-VRAM machine, with Vinkona
elsewhere), declare what this machine serves in config.toml's `[serving]`
table and let `./vinur.sh` supervise all of it — the kb, the vLLM chat
model(s), the nomic embed endpoint, and the CPU reranker:

```bash
./install.sh --serving        # + vLLM in its own serving/.venv (GPU box; big download)
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
  `auth_token` (Bearer on `/call` and every control route) — the server
  refuses to start otherwise.
- Keep parsers (PyMuPDF/Tesseract) patched; parsing needs no network.

## Layout

```
knowledgehost/
  config.py     defaults < TOML < env (KNOWLEDGEHOST_*)
  embed.py      nomic /v1/embeddings client (stdlib urllib; search_query/document prefixes)
  chunk.py      section-aware chunking + stable idempotent ids
  store.py      SqliteStore + LanceStore behind make_store(); shared SQLite manifest
  rerank.py     RRF fusion + intent-conditioned heuristic reranker (cross-encoder = drop-in)
  ingest.py     incremental crawl + Wikipedia ZIM; sanitize -> chunk -> embed -> upsert
  tools.py      the kb_search tool (embed+FTS -> fuse -> rerank -> cited passages + confidence)
  server.py     stdlib HTTP: /health /tools /call (+ /drop, control panel)
  supervisor.py ./vinur.sh's engine — the kb + [serving] services, watched
  serving.py    exec one declared LM/embed service (vllm | llama.cpp)
  sources/      pdf, epub, html, text, wikipedia extractors (heavy deps lazy)
tests/          make_fixtures.sh + smoke.py (zero-install end-to-end)
```

## Disclaimer

This software is provided as-is, for research and reference purposes, without
warranty, and is not validated or intended for production or safety-critical
use.

## License

Apache License 2.0 — see [LICENSE](LICENSE). Vinur was split out of the
[Vinkona monorepo](https://github.com/dantroline-sys/vinkona) on 2026-07-13 and
continues under Apache 2.0 as its own project.
