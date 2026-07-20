# Getting the weights (GPU box)

What to download, where it lands, and which file format each engine expects.
Everything stays inside this repo tree (see `env.sh`): vLLM weights under
`var/cache/huggingface/`, GGUFs under `models/`.

## TL;DR — the recommended 96 GB pair

Nothing to download by hand: put the HF repo ids in `config.toml` and the
weights fetch themselves on first `./vinur.sh start` (with progress in
`var/log/llm-<name>.log`). Budget ~60 GB of disk and a while on first run.

```toml
[[serving.llms]]
name   = "primary"                       # card extraction / distill battery
engine = "vllm"
model  = "Qwen/Qwen3-32B-FP8"            # ~33 GB
port   = 11438
max_model_len          = 16384
gpu_memory_utilization = 0.55
kv_cache_dtype         = "fp8"

[[serving.llms]]
name   = "secondary"                     # different LAB on purpose: the
engine = "vllm"                          # two-model disagreement gate only
                                         # works across training lineages
model  = "RedHatAI/Mistral-Small-3.2-24B-Instruct-2506-FP8"   # ~24 GB
port   = 11435
max_model_len          = 16384
gpu_memory_utilization = 0.30
kv_cache_dtype         = "fp8"

[serving.embed]
enabled = true                           # nomic GGUF auto-downloads (~260 MB)

[serving.reranker]
enabled = true                           # bge GGUF auto-downloads (~600 MB)
```

The two `gpu_memory_utilization` fractions must leave headroom (≤ ~0.9
total) — the embed server and CUDA fragmentation live in the remainder.

## Tuning how vLLM loads and runs each model

The load/run-critical knobs are first-class TOML keys on every
`[[serving.llms]]` entry — set only what you mean; vLLM's own default
applies otherwise. `config.example.toml` documents each inline. The mapping:

| TOML key | `vllm serve` flag | what it tunes |
|---|---|---|
| `quantization` | `--quantization` | checkpoint format when auto-detection needs help: `modelopt` (NVFP4), `fp8`, `awq`, `gptq`, … |
| `kv_cache_dtype` | `--kv-cache-dtype` | KV cache precision (`auto` / `fp8`) |
| `dtype` | `--dtype` | compute dtype for unquantized weights |
| `max_model_len` | `--max-model-len` | served context window — the KV budget |
| `gpu_memory_utilization` | `--gpu-memory-utilization` | VRAM fraction this server may claim |
| `max_num_seqs` | `--max-num-seqs` | concurrent-sequence cap |
| `tensor_parallel` | `--tensor-parallel-size` | GPUs to shard across |
| `cpu_offload_gb` | `--cpu-offload-gb` | GB of weights spilled to RAM |
| `swap_space` | `--swap-space` | GB of CPU swap for preempted sequences |
| `served_model_name` | `--served-model-name` | the `model` name clients send |
| `enforce_eager` | `--enforce-eager` | disable CUDA graphs (less VRAM, slower) |
| `trust_remote_code` | `--trust-remote-code` | repos that ship custom code |

Plus two escape hatches on every entry: `env = { ... }` (extra process
environment, e.g. `VLLM_ATTENTION_BACKEND`) and `args = [...]` — **any**
other `vllm serve` flag, appended last so it also overrides the keys above.
`engine = "llama"` entries take `ctx_size` (`-c`) and `n_gpu_layers`
(`-ngl`, default 99) the same way.

**Quantization pairing rule of thumb:** match the KV cache to the
checkpoint. **NVFP4** and **FP8** checkpoints want `kv_cache_dtype = "fp8"`
— it halves KV memory (double the context/batch for the same VRAM) and
keeps the whole attention path on the 8-bit tensor cores instead of
bouncing through 16-bit KV. NVFP4 repos usually auto-detect; if not, set
`quantization = "modelopt"`. For AWQ/GPTQ on older GPUs, fp8 KV still saves
memory but verify output quality — those stacks are less exercised with it.

### Worked example: a 73 GB model on a 96 GB card

`torch.OutOfMemoryError` at startup does **not** mean the model is too big.
vLLM pre-allocates KV cache up to `gpu_memory_utilization × VRAM` and sizes
its startup profiling for the model's *declared* context (new models
declare 200K+) at serving-fleet concurrency — untuned, that overshoots any
card the weights merely fit on. Cap what you'll actually use:

```toml
max_model_len          = 16384   # the real KV budget — not the native 262144
max_num_seqs           = 8       # a distill/verify box, not a serving fleet
gpu_memory_utilization = 0.92    # ~7 GB headroom for CUDA ctx/graphs/JIT workspace
kv_cache_dtype         = "fp8"   # NVFP4/FP8 checkpoint → double the KV per GB
```

That's ~73 GB weights + ~14 GB KV + overhead, comfortably inside 95 GB —
tens of thousands of fp8 KV tokens, plenty for batched distillation. Still
OOM within a whisker of fitting? `env = { PYTORCH_CUDA_ALLOC_CONF =
"expandable_segments:True" }` (fragmentation), then `enforce_eager = true`
(frees CUDA-graph memory, costs some speed). If instead you want *big*
batch throughput, that's what the smaller pair (~45–60 GB) in swap mode is
for — 35+ GB of KV instead of 14.

## When two models don't fit: exclusive swap mode

Newer large models can make co-residency impossible — e.g. a ~70 GB 4-bit
Qwen3.5-122B-A10B, a ~60 GB Mistral Small 4, or a ~45 GB Qwen3-Coder-Next:
any two of those overflow 96 GB. Mark such entries `exclusive = true` and
they form one **GPU group of which exactly one runs**; the others sit
standby and get swapped in on request (stop the resident one, spawn the
other, wait for `/health` — minutes for big weights, budget in
`serving.swap_timeout_s`, default 900):

```toml
[[serving.llms]]
name      = "primary"                 # boots (default = true)
engine    = "vllm"
model     = "Qwen/Qwen3.5-122B-A10B-NVFP4"    # ~70 GB — general/research distill
port      = 11438
exclusive = true
default   = true
kv_cache_dtype         = "fp8"        # NVFP4 checkpoint → 8-bit KV (see Tuning)
max_model_len          = 16384
gpu_memory_utilization = 0.90         # lone resident — it can take the card

[[serving.llms]]
name      = "secondary"               # standby until swapped in
engine    = "vllm"
model     = "mistralai/Mistral-Small-4..."    # ~60 GB — verify / second opinion
port      = 11435
exclusive = true
kv_cache_dtype         = "fp8"
max_model_len          = 16384
gpu_memory_utilization = 0.90
```

For `engine = "container"` entries the supervisor stops the **container**
(`podman/docker stop vinur-llm-<name>`, TERM→KILL inside the container),
never just the attached client — killing the client would orphan the model
with its VRAM still held, and the incoming entry would then die with
`Free memory on device … is less than desired GPU memory utilization`.
start/swap/stop also sweep orphaned `vinur-llm-*` containers left behind by
a crashed client or a dead supervisor run, so a restart self-heals that
state.

Three ways to trigger a swap:

```bash
./vinur.sh swap secondary             # CLI (waits until ready)
```

```
POST /serving/swap {"name": "secondary"}    # authed; returns immediately —
GET  /serving/swap                          # poll until status=ready
```

and — the one that makes **batched distillation** work unattended — the
autopilot, which routes models **automatically**: with the plan's
"Automatic model swapping" on (the default), each Prioritizer step swaps in
the exclusive entry its verb's LM lane points at before running —
distill/refine follow `distill_urls`, link/adjudicate follow their `fast`
flag to `extract_urls`, ingest only when it distils inline; embed-only
verbs derive nothing. So a stock plan on a swap-mode box just works: no
per-step annotation. The step's optional `"model"` field (visible in the
Prioritizer table whenever exclusive models exist; the greyed value shows
what auto would pick) pins a specific entry instead:

```json
{ "steps": [
  {"command": "distill", "label": "distill backlog"},
  {"command": "refine", "model": "secondary", "label": "refine under the 2nd model"}
]}
```

A swap costs minutes, so never interleave models per item — the priority
order doubles as the phase order, and consecutive steps sharing a model
swap once, not per run. Also note the alternative: pick one model
small enough to co-reside with the big one (e.g. a ~45 GB coder primary
plus a ~24 GB FP8 secondary fits in 96 GB) and skip swapping entirely —
resident pairs are strictly simpler when the quality trade-off is
acceptable.

### Parallel distillation — saturating the batch engine

Serving a big card with vLLM and then sending it **one request at a time**
wastes the card: continuous batching is vLLM's whole trick, and a lone
sequence leaves most of the GPU idle. The distiller therefore fans out
automatically — when an LM-lane URL (distill/extract/verify) resolves to a
`[[serving.llms]]` entry with `engine = "vllm"` or `"container"`, it keeps
**8 requests in flight** against that endpoint (capped by the entry's
`max_num_seqs`), which vLLM folds into one GPU batch. llama.cpp endpoints
(one slot by default) and URLs not in `[serving]` stay sequential, exactly
as before. The top-level `distill_parallel` knob (also in the panel's
Settings tab) overrides auto: set it explicitly when the LM lives on a
**remote** vLLM box this config doesn't serve — auto can't see a foreign
engine — or set `1` to force the old sequential behaviour. Expect a
per-run log line `distill fan-out: 8 concurrent requests -> …` when it
engages. Batching raises throughput, not single-request speed: each
request's latency grows with the batch, so keep `distill_timeout_s`
comfortable.

## engine = "container" — the recommended route on bleeding-edge distros

Bare-metal vLLM needs the host's CUDA toolkit and compiler to be inside
NVIDIA's support matrix, and on fast-moving distros (Fedora) they usually
aren't — that's the whole `Could not find nvcc` / `unsupported GNU version`
saga. `engine = "container"` ends it: the supervisor runs the **official
vLLM image** as its child (attached `podman run --rm`, so start/stop/
watchdog/swap behave exactly like any other service), and the image carries
the matched toolkit + compiler. The host needs **only the NVIDIA driver**.

One-time host setup:

```bash
sudo dnf install podman nvidia-container-toolkit
sudo nvidia-ctk cdi generate --output=/etc/cdi/nvidia.yaml   # GPU wiring (CDI)
# smoke test — CDI injects nvidia-smi into any image:
podman run --rm --device nvidia.com/gpu=all ubuntu nvidia-smi
```

(Using docker instead? Its `--gpus all` needs the daemon configured once:
`sudo nvidia-ctk runtime configure --runtime=docker && sudo systemctl
restart docker`. podman needs no daemon step.)

Then an entry is the `vllm` engine with two extra keys:

```toml
[[serving.llms]]
name    = "primary"
engine  = "container"
model   = "org/Model-NVFP4"
port    = 11438
image   = "docker.io/vllm/vllm-openai:v0.11.0"   # pin a tag; :latest is the default
kv_cache_dtype         = "fp8"                   # same first-class keys, same flags
max_model_len          = 16384
gpu_memory_utilization = 0.90
```

Details worth knowing:

- **Weights land in the same place** — `var/cache/huggingface` is mounted
  into the image (SELinux `:z` label applied), so the in-tree guarantee,
  the Serving tab's weights chips, and pre-downloads all work unchanged.
- `env = { ... }` goes **into the container** (`-e`); `args` append to the
  image's `vllm serve` entrypoint after the mapped keys, same as bare metal.
- First start pulls the image (~10+ GB, one-time; stored in podman's own
  storage, not this tree) before the model load — budget that into the
  first `swap_timeout_s` window, or `podman pull` the image beforehand.
- podman over docker, deliberately: it's daemonless, so the container is a
  real child of the supervisor and TERM/KILL semantics just work. Docker is
  accepted (`runtime = "docker"`, uses `--gpus all`) with the caveat that
  the docker *client* dying doesn't always stop the server-side container.
- JIT compiles (the FlashInfer sm120 MoE module) happen **inside** the
  image with its own toolchain — no host nvcc, no gcc ceiling, cached in
  the mounted HF cache's sibling dirs per image version.
- **Which engine actually ran?** The first line of `var/log/llm-<name>.log`
  is the exact command (`exec: podman run …` vs `exec: …/serving/.venv/bin/
  vllm serve …`). If a "container" entry still shows host-toolchain errors
  (`unsupported GNU version`, `Could not find nvcc`), that log line will
  show it never launched as one — the usual causes are editing the wrong
  config file (`KNOWLEDGEHOST_CONFIG` wins over `./config.toml`), editing a
  different entry than the one that boots (the `default = true` exclusive
  entry), or a TOML slip: `engine = "container"` must sit inside its own
  `[[serving.llms]]` block *above* any `[serving.llms.env]` sub-table
  header, or it silently lands in `env`.

## engine = "vllm" — what kind of files

vLLM loads **HF-format safetensors repos** (a directory of `*.safetensors` +
`config.json` + tokenizer). It does **not** load GGUF — those are for the
`llama` engine below.

Picking a repo:

- **Hopper/Blackwell GPUs** (H100, RTX PRO 6000, …): prefer an **FP8** repo —
  native speed, near-full quality. Look for `-FP8` in the name, from the
  model's own org (`Qwen/Qwen3-32B-FP8`) or a reputable quantizer
  (`RedHatAI/...-FP8`).
- **Older GPUs** (Ada/Ampere, e.g. RTX 4090/3090): use an **AWQ** or
  **GPTQ-Int4** repo instead (`...-AWQ`); FP8 execution isn't native there.
- Unquantized BF16 repos also work but are ~2× the size for no quality win
  at this scale — only bother when no good quant exists.

Three ways to get them, all equivalent to vLLM:

1. **Automatic** (easiest): set `model` to the repo id and start. Downloads
   go to `var/cache/huggingface/` because `./vinur.sh` sources `env.sh`
   (`HF_HOME`); nothing lands in `~/.cache`.
2. **Pre-download** (nicer on a slow link — resumable, shows progress):
   ```bash
   source ./env.sh
   serving/.venv/bin/hf download Qwen/Qwen3-32B-FP8
   ```
   Same cache location; the first start then finds everything offline.
3. **Local directory**: point `model` at a path
   (`model = "/big/disk/Qwen3-32B-FP8"`) containing the repo snapshot.
   Use this when weights live on another volume — or symlink the cache:
   `ln -s /big/disk/hf var/cache/huggingface`.

### Where the downloaded weights actually are

`var/cache/huggingface/hub/` — the panel's **Serving** tab prints this path in
its footer (with repo count, total GB, and any reclaimable partials), and each
model's row shows its own folder. Inside it:

```
hub/models--Qwen--Qwen3-32B-FP8/
├── blobs/                     the real files, named by content hash
│   └── *.incomplete           partial downloads (litter after a failed fetch)
├── snapshots/<revision>/      readable tree — symlinks into blobs/
└── refs/main                  which revision that is
```

Inspect a model as it appears to vLLM with `ls -lL
var/cache/huggingface/hub/models--*/snapshots/*/` (`-L` follows the symlinks,
so you see the real sizes). Deleting a whole `models--…` folder un-downloads
that model and nothing else; deleting stray `*.incomplete` blobs is always
safe — a resumed fetch writes under a fresh name.

### Behind a proxy

**Nothing in this stack reads your OS or desktop proxy settings.** vLLM does no
proxying of its own; the downloads happen in `huggingface_hub` (via `requests`)
and, for Xet transfers, in a Rust `reqwest` client. On Linux both read the
environment and nothing else — no GNOME/KDE settings, no `/etc` config. So the
proxy has to be handed over explicitly:

```toml
http_proxy  = "http://proxy.example.internal:3128"
https_proxy = "http://proxy.example.internal:3128"
no_proxy    = "corp.internal"          # loopback is added for you
```

Setting it in the shell works too, but config.toml is the reliable place:

- the supervisor's children inherit the environment the supervisor was
  started with, and a systemd unit or cron start has no login shell;
- **`engine = "container"` gets no host environment at all** — proxy vars must
  ride in as `-e` flags, which is exactly what these keys do.

`no_proxy` always gains `localhost,127.0.0.1,::1` plus every declared serving
host: the knowledge host calls its own LMs over loopback, and Python's urllib
has no built-in localhost bypass, so a proxy that swallows those requests looks
like every model on the box going down at once. `./vinur.sh start` warns if
your shell has a proxy set without that exemption. Credentials embedded in a
proxy URL are redacted from the `exec:` log line.

If a download fails only under Xet, set `HF_HUB_DISABLE_XET=1` to fall back to
plain HTTPS through `requests` — a proxy that inspects TLS sometimes upsets the
Xet client's connection reuse.

**Gated repos** (Llama, some Mistral originals): accept the license on
huggingface.co once, create a read token there, and export it before
starting: `export HF_TOKEN=hf_...`. The recommended pair above is not gated.

## engine = "llama" — what kind of files

llama.cpp loads a **single `.gguf` file**. Get the instruct model's GGUF from
a reputable converter (e.g. `bartowski/<model>-GGUF` on HF) — **Q4_K_M** is
the sane default quant; Q8_0 if VRAM is plentiful.

Place it in `models/` (any path works; relative paths resolve against the
repo root):

```bash
source ./env.sh
serving/.venv/bin/hf download bartowski/Qwen2.5-7B-Instruct-GGUF \
    Qwen2.5-7B-Instruct-Q4_K_M.gguf --local-dir models
```

```toml
[[serving.llms]]
name   = "small"
engine = "llama"                # needs llama-server on $PATH (or $LLAMA_SERVER)
model  = "models/Qwen2.5-7B-Instruct-Q4_K_M.gguf"
port   = 11438
```

Use this engine for small boxes or single-model setups; vLLM wins on batch
throughput, which is what distillation runs are.

## Embed + reranker — llama-server, not vLLM

Both run on llama.cpp *by design*: at 137M/568M parameters a vLLM instance's
per-process overhead would dwarf them, and they must stay up while the big
exclusive models swap — the CPU reranker and the tiny embedder live happily
in the margin the `gpu_memory_utilization` fractions leave free.

Their GGUFs auto-download into `models/` on first start
(`nomic-embed-text-v1.5.f16.gguf` ~260 MB, `bge-reranker-v2-m3-Q8_0.gguf`
~600 MB). What they DO need is a **`llama-server` binary** — a standalone
Vinur box builds its own, in-tree:

```bash
./install.sh --llama     # → bin/llama-server (CUDA when nvcc is present, else CPU)
```

Resolution order everywhere (serving.py, run-reranker.sh):
`$LLAMA_SERVER` → `bin/llama-server` → `$PATH` → a sibling Vinkona
checkout's `../vinkona/assistant/bin/llama-server`. `./vinur.sh start` warns
upfront when a declared service needs the binary and none is found.

## Wire the ports, then verify

The kb calls whatever answers at the configured URLs — keep them consistent
with the ports above (these are the defaults, so usually: touch nothing):

```toml
distill_urls = ["http://127.0.0.1:11438"]   # primary
extract_urls = ["http://127.0.0.1:11435"]   # secondary
verify_urls  = []                           # empty => distill_urls
embed_url    = "http://127.0.0.1:11437"
rerank_url   = "http://127.0.0.1:11439"
```

```bash
./vinur.sh start
./vinur.sh status                     # every service 'up' (first start: models downloading —
                                      #   follow with ./vinur.sh logs llm-primary)
curl -s localhost:11438/v1/models     # vLLM answers with the loaded model id
curl -s localhost:8771/health         # the kb itself
```

**Match the model NAMES, not just the ports.** llama-server ignores the
`model` field in requests; **vLLM validates it** and answers
`404 The model ... does not exist` on a mismatch — moving a client from
llama-server to vLLM surfaces every stale name. Give each entry a stable
role alias and send exactly that from every client:

```toml
[[serving.llms]]                      # this box
served_model_name = "big"
```

- the kb's own write path: `distill_model = "big"` (and `extract_model` /
  `verify_model` if those point at vLLM entries — their defaults are GGUF
  *filenames* from the llama-server days);
- a remote Vinkona using this box as its background LM:
  `"big_lm": {"remote": true, "url": "http://this-box:11438", "model": "big"}`.

`curl -s localhost:11438/v1/models` shows the names the server will accept.

Both sides also self-heal the common case: the kb's LM client reacts to a
404 by asking the server what it serves and, when there is exactly one
model, adopts that name for the rest of the run (logged — fix the config to
make it permanent); Vinkona's remote tiers do the same reconciliation at
startup. A server hosting *several* names is never guessed at — the error
lists them instead.

Disk cleanup: `./install.sh uninstall` removes `serving/.venv`; the weight
caches are plain directories — delete `var/cache/huggingface/` or files in
`models/` whenever you drop a model from the config.

## Troubleshooting

### Start here: read that service's log

Every service logs to `var/log/<service>.log` (`llm-<name>`, `embed`,
`reranker`, `kb`). Three ways in, all the same file:

```bash
./vinur.sh logs llm-secondary       # follow it live
./vinur.sh status                   # one line per service + why a dead one died
```

…or the **Log** button on the panel's Serving tab, which tails it in place.

**A dying process's last line is almost never the cause.** vLLM signs off with
`For further information visit https://errors.pydantic.dev/…` — the message
that matters (`Value error, <field> …`) is a few lines *above* it, and says
which config key this vLLM version rejected. The panel extracts that line into
the note column; from a shell, read upward:

```bash
grep -B8 "errors.pydantic.dev" var/log/llm-secondary.log | head -40
```

### Starting and stopping one service

```bash
./vinur.sh stop llm-secondary       # HELD: the watchdog will not revive it
./vinur.sh start llm-secondary      # …until you say so — also clears a
./vinur.sh restart llm-secondary    #    "gave up after 5 restarts" verdict
```

The Serving tab's Start / Stop / Restart buttons post the same requests. A
`start` on an **exclusive** model becomes a swap, because its sibling is
holding the VRAM.

### A download that stopped

The weights chip reads **stalled** when partial files exist but nothing has
been written for over two minutes — the fetch is stuck, not slow. Common
causes, all visible in the log:

| In the log | What it is |
|---|---|
| `429` / `Too Many Requests` | HF rate-limited an anonymous download — set `hf_token` |
| `401` / `GatedRepoError` | licence not accepted, or no token |
| `403` | the token's account lacks access to that repo |
| `No space left on device` | the cache disk is full (tens of GB per model) |
| `ReadTimeoutError` | network died mid-transfer |

Restarting the service resumes from the partial blobs in every case — nothing
re-downloads from zero.

### `RuntimeError: Could not find nvcc and default cuda_home='/usr/local/cuda' doesn't exist`

The engine crashed inside a **JIT-compiled kernel path**: FlashInfer
compiles some kernels at first use, which needs the CUDA **toolkit**
(`nvcc`) — the driver alone isn't enough. Read the traceback to see *which*
consumer wants it; the two seen in practice:

- `gen_cutlass_fused_moe_sm120_module` via
  `quantization/modelopt.py` — the **NVFP4/FP8 fused-MoE module** on
  consumer Blackwell (sm120). This is the path MoE models like
  Qwen3.5-A*B take; there is **no prebuilt escape hatch in the wheel** —
  attention-backend env vars do NOT help here.
- an attention backend JIT — for this one alone,
  `env = { VLLM_ATTENTION_BACKEND = "FLASH_ATTN" }` (or `TRITON_ATTN`)
  sidesteps the JIT at some throughput cost.

Fixes, in order of preference:

1. **Install the CUDA toolkit — the real fix.** Blackwell (sm120) needs
   **CUDA ≥ 12.8**, and for the NVFP4-MoE case this is the ONLY reliable
   fix: on sm120 the FlashInfer JIT module is effectively the sole NVFP4-MoE
   implementation, so no env flag routes around it.  Install the TOOLKIT
   ONLY — never let an installer replace the working driver:
   - Fedora (NVIDIA's repo lags Fedora releases; the runfile is the
     dependable route):
     ```bash
     wget https://developer.download.nvidia.com/compute/cuda/13.0.0/local_installers/cuda_13.0.0_580.65.06_linux.run
     sudo sh cuda_13.0.0_*_linux.run --toolkit --silent      # --toolkit = no driver change
     ```
     (any current 12.8+/13.x runfile works — check developer.nvidia.com/cuda-downloads)
   - Ubuntu: `sudo apt install cuda-toolkit-12-9` (NVIDIA repo — the
     `cuda-toolkit-*` packages never touch the driver; plain `cuda` does).
   `/usr/local/cuda` appears and the launcher finds it (it also probes
   `/usr/local/cuda*`, `/opt/cuda`, `/usr/lib/cuda` and nvcc on `$PATH`,
   and logs `CUDA_HOME not set — using the toolkit at …`; somewhere odder →
   `env = { CUDA_HOME = "/where/cuda/lives" }`). The **first start then
   JIT-compiles for several minutes** (one-time; cached in
   `var/cache/flashinfer` via env.sh) — that's what
   `serving.swap_timeout_s = 900` is sized for; raise it if a first-boot
   swap reports a timeout while the log shows compilation running. Bonus:
   `nvcc` upgrades `./install.sh --llama` to a CUDA build too.
2. **No system installs:** FlashInfer publishes ahead-of-time artifacts —
   `flashinfer-cubin`, and `flashinfer-jit-cache` built per CUDA version —
   that remove the nvcc need. Install into the serving venv from
   FlashInfer's own index (pick the cuXXX matching the venv's torch):
   ```bash
   source ./env.sh
   vk_uv pip install --python serving/.venv/bin/python3 flashinfer-cubin
   vk_uv pip install --python serving/.venv/bin/python3 \
       flashinfer-jit-cache --extra-index-url https://flashinfer.ai/whl/cu129/
   ```
   (Package/index names move with FlashInfer releases — check its docs if
   the resolve fails.)
3. **For the MoE case, worth one free try:** `env =
   { VLLM_USE_FLASHINFER_MOE_FP4 = "0" }` asks vLLM for its built-in
   NVFP4-MoE path instead. Whether the wheel actually ships one for sm120
   varies by version — if the model then fails differently or slows down,
   revert and use fix 1 or 2.

Either way the Serving panel tab shows the crash line in its note column —
if a model's service is dead and its weights chip says ready, the reason is
an engine error like this one, not a download.

### `#error -- unsupported GNU version! gcc versions later than N are not supported!`

The sequel to the nvcc error on bleeding-edge distros: the toolkit is
installed, but the **system gcc is newer than nvcc's supported ceiling**
(the same reason the runfile installer needed `--override`). nvcc reads
`NVCC_APPEND_FLAGS` from the environment, so fix it per model entry:

```toml
env = { NVCC_APPEND_FLAGS = "-allow-unsupported-compiler" }   # quick; NVIDIA's
                                                              # "at your own risk"
```

or, to stay inside the support matrix, install a versioned compat compiler
and hand it to nvcc instead:

```bash
sudo dnf install gcc14-c++          # Fedora ships versioned gcc packages
```
```toml
env = { NVCC_APPEND_FLAGS = "-ccbin /usr/bin/g++-14" }
```

One-time cost either way: after the first successful JIT compile the module
is cached (`var/cache/flashinfer`) and nvcc isn't invoked again. The same
mismatch hits `./install.sh --llama`'s CUDA build — there the escape hatch
is `VINUR_LLAMA_CMAKE_EXTRA='-DCMAKE_CUDA_FLAGS=-allow-unsupported-compiler'`
(or `-DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++-14`).

### Benign startup noise (not bugs)

Seen in healthy logs — none of these stop the engine:

- **`warning: '_POSIX_C_SOURCE' redefined`** walls, from
  `__triton_launcher.c` — Triton JIT-compiling its C launcher stubs. uv's
  standalone CPython was built against an older glibc baseline than a new
  distro's headers; gcc warns and compiles anyway. Cosmetic.
- **`Module vllm.third_party.deep_gemm was found but failed to import`**
  (with the nvcc AssertionError) — deep_gemm is an *optional* fast-GEMM
  module; vLLM probes it and falls back. Installing the CUDA toolkit makes
  it importable (and may add some GEMM throughput); without it this is a
  no-op warning.
- **`Directly load ... from the cache`** lines — the torch.compile /
  AOT caches under `var/cache/vllm` doing their job across restarts.
- **`Unknown vLLM environment variable detected: VLLM_BUILD_URL` /
  `VLLM_IMAGE_TAG` / `VLLM_BUILD_PIPELINE` / `VLLM_BUILD_COMMIT`** — the
  official image bakes its build provenance in as ENV, and vLLM's scanner
  warns on any `VLLM_`-prefixed variable it doesn't recognise, so the image
  trips its own warning.  Under podman we strip these at run
  (`--unsetenv`); docker has no unset flag, so there the four lines remain.
  Harmless either way.
- **`` The `use_fast` parameter is deprecated ``** (transformers) — vLLM's
  own image-processor plumbing calling a transformers API that moved.
  Upstream noise, nothing in your config drives it.
- **`FutureWarning: The HF_HUB_ENABLE_HF_TRANSFER environment variable is
  deprecated`** — from checkouts before the Xet switch: modern
  huggingface_hub transfers via Xet, so vinur now sets
  `HF_XET_HIGH_PERFORMANCE` instead (serving.hf_env).  If you still see
  this, either update vinur or you exported the legacy variable yourself.

The line that actually matters is `ERROR ... EngineCore failed to start` —
diagnose from the traceback directly above it.
