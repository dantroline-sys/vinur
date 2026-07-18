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
args   = ["--max-model-len", "16384", "--gpu-memory-utilization", "0.55",
          "--kv-cache-dtype", "fp8"]

[[serving.llms]]
name   = "secondary"                     # different LAB on purpose: the
engine = "vllm"                          # two-model disagreement gate only
                                         # works across training lineages
model  = "RedHatAI/Mistral-Small-3.2-24B-Instruct-2506-FP8"   # ~24 GB
port   = 11435
args   = ["--max-model-len", "16384", "--gpu-memory-utilization", "0.30",
          "--kv-cache-dtype", "fp8"]

[serving.embed]
enabled = true                           # nomic GGUF auto-downloads (~260 MB)

[serving.reranker]
enabled = true                           # bge GGUF auto-downloads (~600 MB)
```

The two `gpu-memory-utilization` fractions must leave headroom (≤ ~0.9
total) — the embed server and CUDA fragmentation live in the remainder.

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
