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

Three ways to trigger a swap:

```bash
./vinur.sh swap secondary             # CLI (waits until ready)
```

```
POST /serving/swap {"name": "secondary"}    # authed; returns immediately —
GET  /serving/swap                          # poll until status=ready
```

and — the one that makes **batched distillation** work unattended — the
autopilot: each Prioritizer step takes an optional `"model"` key, and the
step's verb only launches once that model is resident. Order the plan so
whole phases share a model and each cycle swaps twice, not per document:

```json
{ "steps": [
  {"command": "distill", "model": "primary",   "label": "distill backlog"},
  {"command": "verify",  "model": "secondary", "label": "verify under 2nd model"}
]}
```

A swap costs minutes, so never interleave models per item — batch each
phase over the whole backlog. Also note the alternative: pick one model
small enough to co-reside with the big one (e.g. a ~45 GB coder primary
plus a ~24 GB FP8 secondary fits in 96 GB) and skip swapping entirely —
resident pairs are strictly simpler when the quality trade-off is
acceptable.

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

## Embed + reranker — nothing to fetch

Both auto-download their GGUFs into `models/` on first start:
`nomic-embed-text-v1.5.f16.gguf` (~260 MB) and
`bge-reranker-v2-m3-Q8_0.gguf` (~600 MB). They only need a `llama-server`
binary on `$PATH` (or `LLAMA_SERVER=/path/to/it`).

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

Disk cleanup: `./install.sh uninstall` removes `serving/.venv`; the weight
caches are plain directories — delete `var/cache/huggingface/` or files in
`models/` whenever you drop a model from the config.
