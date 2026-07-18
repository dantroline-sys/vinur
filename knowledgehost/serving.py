"""Standalone LM serving — the services `./vinur.sh` manages beside the kb.

When Vinur runs on its own machine (no Vinkona to borrow LMs from), the
`serving` config table declares what this box serves: chat LMs (vLLM or
llama.cpp), the nomic embed endpoint, and the CPU reranker.  This module
turns one declared service into the exec'd server process:

    python3 -m knowledgehost.serving <name>     # an llms[] entry's name
    python3 -m knowledgehost.serving embed      # llama-server --embedding

It resolves the config, builds the engine's argv, and `exec`s it — so the
supervisor's child IS the server (signals and exit status pass straight
through, nothing to reap in between).  Argv building is pure (`llm_argv`,
`embed_argv`) and unit-tested; only `main` touches the OS.

Engines:
  vllm   serving/.venv/bin/vllm (install with ./install.sh --serving).
         `model` is a HF id (weights land in var/cache/huggingface via
         env.sh's HF_HOME) or a local path.
  llama  llama-server on $PATH or $LLAMA_SERVER.  `model` is a GGUF path
         (relative paths anchor to the repo root, like config paths do).
"""
from __future__ import annotations

import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

EMBED_MODEL_FILE = "nomic-embed-text-v1.5.f16.gguf"
EMBED_MODEL_URL = ("https://huggingface.co/nomic-ai/nomic-embed-text-v1.5-GGUF"
                   f"/resolve/main/{EMBED_MODEL_FILE}")


def find_llama_server(root: Path = ROOT) -> str | None:
    """Resolve the llama.cpp server binary, or None.  Order: $LLAMA_SERVER,
    the in-tree build (./install.sh --llama -> bin/), PATH, then a sibling
    Vinkona checkout's build (one box often has both repos)."""
    exe = os.environ.get("LLAMA_SERVER", "").strip()
    if exe:
        return exe
    own = root / "bin" / "llama-server"
    if own.is_file() and os.access(own, os.X_OK):
        return str(own)
    from shutil import which
    found = which("llama-server")
    if found:
        return found
    sibling = root.parent / "vinkona" / "assistant" / "bin" / "llama-server"
    if sibling.is_file() and os.access(sibling, os.X_OK):
        return str(sibling)
    return None


def _llama_server() -> str:
    found = find_llama_server()
    if not found:
        raise FileNotFoundError(
            "llama-server not found — build it with ./install.sh --llama "
            "(in-tree), or set LLAMA_SERVER=/path/to/llama-server")
    return found


def _anchored(path: str) -> str:
    """Relative model paths anchor to the repo root (same rule as config paths)."""
    return str(ROOT / Path(path).expanduser())


# First-class vLLM tuning keys — one TOML key per load/run knob people actually
# set per model (fit, quantization, KV cache), mapped straight onto `vllm serve`
# flags.  A key is emitted only when the entry SETS it (vLLM's own default wins
# otherwise); anything not listed here goes in `args`, which is appended LAST
# so it also overrides these.  (config.example.toml documents each.)
_VLLM_KEYS = [
    # (toml key, cli flag, kind: value | flag)
    ("quantization",           "--quantization",           "value"),
    ("kv_cache_dtype",         "--kv-cache-dtype",         "value"),
    ("dtype",                  "--dtype",                  "value"),
    ("max_model_len",          "--max-model-len",          "value"),
    ("gpu_memory_utilization", "--gpu-memory-utilization", "value"),
    ("max_num_seqs",           "--max-num-seqs",           "value"),
    ("tensor_parallel",        "--tensor-parallel-size",   "value"),
    ("cpu_offload_gb",         "--cpu-offload-gb",         "value"),
    ("swap_space",             "--swap-space",             "value"),
    ("served_model_name",      "--served-model-name",      "value"),
    ("enforce_eager",          "--enforce-eager",          "flag"),
    ("trust_remote_code",      "--trust-remote-code",      "flag"),
]


def _mapped_flags(entry: dict, table: list) -> list[str]:
    out: list[str] = []
    for key, flag, kind in table:
        if key not in entry:
            continue
        v = entry[key]
        if kind == "flag":
            if v:
                out.append(flag)
        elif v is not None and str(v) != "":
            out += [flag, str(v)]
    return out


def llm_argv(entry: dict, root: Path = ROOT) -> list[str]:
    """argv for one llms[] entry ({name, engine, model, port, host, exclusive,
    default, env, args} + the first-class tuning keys in _VLLM_KEYS /
    ctx_size + n_gpu_layers for llama)."""
    for k in ("name", "engine", "model", "port"):
        if not entry.get(k):
            raise ValueError(f"serving.llms entry needs '{k}': {entry}")
    host = str(entry.get("host") or "127.0.0.1")
    port = str(int(entry["port"]))
    args = [str(a) for a in (entry.get("args") or [])]
    engine = entry["engine"]
    if engine == "vllm":
        vllm = root / "serving" / ".venv" / "bin" / "vllm"
        if not vllm.exists():
            raise FileNotFoundError(
                f"{vllm} missing — run ./install.sh --serving first")
        return ([str(vllm), "serve", str(entry["model"]),
                 "--host", host, "--port", port]
                + _mapped_flags(entry, _VLLM_KEYS) + args)
    if engine == "llama":
        model = _anchored(str(entry["model"]))
        if not os.path.isfile(model):
            raise FileNotFoundError(f"GGUF not found: {model}")
        # -ngl 99 default (a box serving LMs wants them on GPU); ctx_size /
        # n_gpu_layers are the first-class knobs, args can override anything
        # because llama-server takes the LAST occurrence of a repeated flag.
        argv = [_llama_server(), "-m", model, "--host", host, "--port", port,
                "-ngl", str(entry.get("n_gpu_layers", 99))]
        if "ctx_size" in entry:
            argv += ["-c", str(entry["ctx_size"])]
        return argv + args
    raise ValueError(f"unknown serving engine '{engine}' (vllm | llama)")


def embed_argv(cfg: dict, model_path: str) -> list[str]:
    """argv for the nomic embed endpoint (llama-server --embedding).

    -ub/-b ≥ the per-sequence window so a full-length input never overflows
    the physical batch (the embed-server gotcha run-reranker.sh notes).
    """
    scfg = cfg["serving"]["embed"]
    host = str(scfg.get("host") or "127.0.0.1")
    port = str(int(scfg.get("port") or 11437))
    args = [str(a) for a in (scfg.get("args") or [])]
    return [_llama_server(), "--embedding", "-m", model_path,
            "--host", host, "--port", port,
            "-ngl", "99", "-c", "2048", "-b", "2048", "-ub", "2048",
            "--pooling", "mean"] + args


def ensure_embed_model(root: Path = ROOT) -> str:
    """Download the nomic GGUF into models/ once (resumable curl-less stdlib fetch)."""
    dest = root / "models" / EMBED_MODEL_FILE
    if dest.is_file():
        return str(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".part")
    print(f"downloading {EMBED_MODEL_FILE} (~260 MB) -> {dest}", flush=True)
    with urllib.request.urlopen(EMBED_MODEL_URL, timeout=60) as r, open(tmp, "wb") as f:
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
    os.replace(tmp, dest)
    return str(dest)


# ── exclusive-group swap protocol ────────────────────────────────────────────
# When the declared models cannot co-reside in VRAM, entries marked
# `exclusive = true` form one GPU group: the supervisor keeps exactly ONE of
# them running and swaps on request.  The handshake is two files under
# var/run (same cross-process idiom as lm_lease):
#     swap.req    {"name": ...}          written by any requester
#     swap.state  {"active", "status": ready|swapping|error, ...}
#                                        written ONLY by the supervisor
# Requesters are the CLI (./vinur.sh swap NAME), the kb server's authed
# POST /serving/swap, and the autopilot (a step's "model" key) — so batched
# phases run distill under one model, then verify under the other.

SWAP_REQ = ROOT / "var" / "run" / "swap.req"
SWAP_STATE = ROOT / "var" / "run" / "swap.state"


def swap_state() -> dict:
    try:
        return json.loads(SWAP_STATE.read_text())
    except (OSError, ValueError):
        return {}


def request_swap(name: str) -> None:
    SWAP_REQ.parent.mkdir(parents=True, exist_ok=True)
    tmp = SWAP_REQ.with_suffix(".tmp")
    tmp.write_text(json.dumps({"name": name, "at": time.time()}))
    os.replace(tmp, SWAP_REQ)


def ensure_active(name: str, timeout_s: float = 900.0, poll_s: float = 1.0,
                  progress=None) -> dict:
    """Client side: make `name` the resident exclusive model, waiting for the
    supervisor to finish the swap.  A no-op when it is already active.  Raises
    RuntimeError on a supervisor-reported error (or no supervisor at all) and
    TimeoutError past timeout_s (big weights legitimately take minutes)."""
    st = swap_state()
    if not st:
        raise RuntimeError("no swap state — is the supervisor running? (./vinur.sh start)")
    if st.get("active") == name and st.get("status") == "ready":
        return st
    request_swap(name)
    deadline = time.time() + timeout_s
    nudge = time.time() + 15.0                   # re-request if ours got overwritten
    last = None
    while time.time() < deadline:
        st = swap_state()
        if st != last and progress:
            progress(st)
            last = st
        if st.get("status") == "ready" and st.get("active") == name:
            return st
        if st.get("status") == "error" and st.get("request") == name:
            raise RuntimeError(st.get("error") or "swap failed")
        if st.get("status") == "ready" and time.time() > nudge:
            request_swap(name)                   # lost race with another requester
            nudge = time.time() + 15.0
        time.sleep(poll_s)
    raise TimeoutError(f"swap to '{name}' not ready after {int(timeout_s)}s")


def probe_ready(host: str, port: int, timeout_s: float = 1.5) -> bool:
    """One readiness poke: /health answers 200 on both vLLM and llama-server
    (llama returns 503 while the model is still loading)."""
    try:
        with urllib.request.urlopen(f"http://{host}:{port}/health", timeout=timeout_s) as r:
            return r.status == 200
    except urllib.error.HTTPError:
        return False                             # up but still loading (503)
    except OSError:
        return False                             # not listening yet


def cuda_home_probe(environ: dict = None, prefixes: tuple = ("/usr/local", "/opt", "/usr/lib")) -> str | None:
    """vLLM's JIT paths (FlashInfer — its default attention backend on newer
    GPUs) need the CUDA *toolkit* at runtime and die with "Could not find nvcc
    and default cuda_home='/usr/local/cuda' doesn't exist" when $CUDA_HOME is
    unset and that symlink is missing.  Probe the usual spots so a toolkit
    that IS installed — just not at the default path — gets found.  Returns
    the toolkit root to use, or None ($CUDA_HOME already set, or none found —
    see serving/README.md Troubleshooting for the no-toolkit options)."""
    env = os.environ if environ is None else environ
    if env.get("CUDA_HOME") or env.get("CUDA_PATH"):
        return None
    from shutil import which
    nv = which("nvcc")
    if nv:
        return str(Path(nv).resolve().parent.parent)
    candidates: list[Path] = []
    for pre in prefixes:
        p = Path(pre)
        candidates.append(p / "cuda")
        candidates += sorted(p.glob("cuda-*"), reverse=True)   # newest first
    for c in candidates:
        if (c / "bin" / "nvcc").is_file():
            return str(c)
    return None


# Known engine-failure signatures → actionable hints.  The Serving tab and
# status put these next to the dead service so nobody has to re-diagnose a
# failure mode we've already seen in the wild.
_FAILURE_HINTS = [
    ("Could not find nvcc",
     "CUDA toolkit missing — vLLM's JIT kernels (NVFP4/FP8 MoE on consumer "
     "Blackwell needs them) can't build. serving/README.md → Troubleshooting "
     "has the toolkit-only install."),
    ("CUDA out of memory",
     "VRAM overflow — lower gpu_memory_utilization / max_model_len, check "
     "what else is resident (exclusive swap mode exists for this)."),
    ("401 Client Error",
     "gated HF repo — accept its license on huggingface.co and export "
     "HF_TOKEN before starting."),
    ("GatedRepoError",
     "gated HF repo — accept its license on huggingface.co and export "
     "HF_TOKEN before starting."),
    ("No space left on device",
     "disk full — weights live in var/cache/huggingface; prune models you "
     "dropped from the config."),
]


def failure_hint(text: str) -> str | None:
    """Map a service's dying words to the fix, if it's a failure we know."""
    for needle, hint in _FAILURE_HINTS:
        if needle in (text or ""):
            return hint
    return None


def toolkit_warning(cfg: dict, toolkit_present: bool | None = None) -> str | None:
    """Start-time preflight: vllm entries declared but no CUDA toolkit on the
    box.  Loud when a model looks NVFP4/modelopt (on consumer Blackwell that
    WILL die in the FlashInfer JIT); gentle otherwise (JIT paths MAY need it).
    toolkit_present injects the detection for tests."""
    vllm_entries = [e for e in cfg["serving"]["llms"] if e.get("engine") == "vllm"]
    if not vllm_entries:
        return None
    if toolkit_present is None:
        toolkit_present = bool(os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")
                               or cuda_home_probe({}))
    if toolkit_present:
        return None
    fp4 = [str(e.get("name")) for e in vllm_entries
           if "fp4" in str(e.get("model", "")).lower()
           or str(e.get("quantization", "")).lower() == "modelopt"]
    if fp4:
        return (f"no CUDA toolkit (nvcc) found, and {', '.join(fp4)} looks "
                "NVFP4/modelopt-quantized — on consumer Blackwell that model WILL "
                "fail in the FlashInfer JIT ('Could not find nvcc').  Install the "
                "toolkit first: serving/README.md → Troubleshooting.")
    return ("no CUDA toolkit (nvcc) found — vLLM runs, but JIT kernel paths "
            "(FlashInfer MoE, some attention backends) will fail if a model "
            "needs them.  serving/README.md → Troubleshooting.")


# ── panel status: is this box hosting models, and are the weights here? ─────

def _tree_size(p: Path) -> int:
    total = 0
    try:
        for f in p.rglob("*"):
            try:
                if f.is_file() and not f.is_symlink():
                    total += f.stat().st_size
            except OSError:
                pass
    except OSError:
        pass
    return total


def weights_status(engine: str, model: str) -> dict:
    """Where the weights for one declared model stand ON DISK:
    ready | incomplete (mid-download or an interrupted/failed fetch) | missing.
    The service can be 'up' while vLLM is still downloading — this is the
    signal that distinguishes 'loading' from 'the fetch died'."""
    if engine == "llama":
        p = Path(_anchored(model))
        if p.is_file():
            return {"status": "ready", "path": str(p),
                    "size_gb": round(p.stat().st_size / 2**30, 2)}
        return {"status": "missing", "path": str(p),
                "detail": "GGUF not found — see serving/README.md"}
    if engine == "vllm":
        mp = Path(model).expanduser()
        if mp.is_dir():                                # local snapshot directory
            if any(mp.glob("*.safetensors")):
                return {"status": "ready", "path": str(mp),
                        "size_gb": round(_tree_size(mp) / 2**30, 1)}
            return {"status": "incomplete", "path": str(mp),
                    "detail": "local dir has no *.safetensors"}
        hub = Path(os.environ.get("HF_HOME")
                   or (ROOT / "var" / "cache" / "huggingface")) / "hub"
        d = hub / ("models--" + model.replace("/", "--"))
        if not d.is_dir():
            return {"status": "missing", "path": str(d),
                    "detail": ("downloads on first start — or pre-fetch with "
                               f"serving/.venv/bin/hf download {model}")}
        blobs = d / "blobs"
        partial = len(list(blobs.glob("*.incomplete"))) if blobs.is_dir() else 0
        snap_ok = False
        snaps = d / "snapshots"
        if snaps.is_dir():
            for s in snaps.iterdir():
                if (s / "config.json").exists() and any(s.glob("*.safetensors")):
                    snap_ok = True
                    break
        out = {"path": str(d), "size_gb": round(_tree_size(d) / 2**30, 1)}
        if partial:
            out.update(status="incomplete",
                       detail=f"{partial} file(s) mid-download — in progress, or an "
                              "interrupted fetch (restarting the service resumes it)")
        elif snap_ok:
            out.update(status="ready")
        else:
            out.update(status="incomplete",
                       detail="cache present but no complete snapshot — fetch "
                              "interrupted? (restarting the service resumes it)")
        return out
    return {"status": "unknown", "detail": f"unknown engine '{engine}'"}


def serving_status(cfg: dict) -> dict:
    """Everything the panel's Serving tab shows: declared models, their
    supervisor state (up/standby/dead/failed + last log line), weights-on-disk
    status, and the live swap state."""
    from . import supervisor as sup
    st = sup.read_state()
    sup_alive = sup.alive(st.get("supervisor", 0))
    services = st.get("services") or {}
    standby = st.get("standby") or {}
    failed = st.get("failed") or {}

    def svc_state(svc_name: str, entry: str = "") -> dict:
        if not sup_alive:
            return {"service": "supervisor-down"}
        if svc_name in failed:
            return {"service": "failed", "reason": failed[svc_name]}
        pid = services.get(svc_name)
        d: dict = {}
        if pid and sup.alive(pid):
            d = {"service": "up", "pid": pid}
        elif entry and entry in standby:
            d = {"service": "standby"}
        else:
            d = {"service": "dead" if pid else "off"}
        if d["service"] in ("up", "dead"):
            line = sup.last_log_line(svc_name)
            if line:
                d["last_log"] = line[-240:]
        if d["service"] in ("dead", "failed"):
            # The signature line is rarely LAST (NCCL/teardown noise follows a
            # crash) — scan the log tail for failure modes we know the fix for.
            try:
                tail = (sup.LOGS / f"{svc_name}.log").read_bytes()[-8192:] \
                    .decode("utf-8", "replace")
            except OSError:
                tail = ""
            hint = failure_hint(d.get("reason", "") + " " + tail)
            if hint:
                d["hint"] = hint
        return d

    llms = []
    for e in cfg["serving"]["llms"]:
        name = str(e.get("name") or "")
        item = {"name": name, "engine": str(e.get("engine") or ""),
                "model": str(e.get("model") or ""), "port": e.get("port"),
                "exclusive": bool(e.get("exclusive")), "default": bool(e.get("default")),
                "weights": weights_status(str(e.get("engine") or ""),
                                          str(e.get("model") or ""))}
        item.update(svc_state(f"llm-{name}", name))
        llms.append(item)

    emb_cfg = cfg["serving"]["embed"]
    embed = {"enabled": bool(emb_cfg.get("enabled")), "port": emb_cfg.get("port", 11437)}
    if embed["enabled"]:
        p = ROOT / "models" / EMBED_MODEL_FILE
        embed["weights"] = ({"status": "ready", "path": str(p),
                             "size_gb": round(p.stat().st_size / 2**30, 2)}
                            if p.is_file() else
                            {"status": "missing", "path": str(p),
                             "detail": "auto-downloads on first start (~260 MB)"})
        embed.update(svc_state("embed"))

    rr_cfg = cfg["serving"]["reranker"]
    reranker = {"enabled": bool(rr_cfg.get("enabled"))}
    if reranker["enabled"]:
        p = ROOT / "models" / str(cfg.get("rerank_model") or "bge-reranker-v2-m3-Q8_0.gguf")
        reranker["weights"] = ({"status": "ready", "path": str(p),
                                "size_gb": round(p.stat().st_size / 2**30, 2)}
                               if p.is_file() else
                               {"status": "missing", "path": str(p),
                                "detail": "auto-downloads on first start (~600 MB)"})
        reranker.update(svc_state("reranker"))

    return {"hosting": bool(llms or embed["enabled"] or reranker["enabled"]),
            "supervisor": {"running": sup_alive,
                           "pid": st.get("supervisor") if sup_alive else None},
            "swap": swap_state(), "llms": llms, "embed": embed, "reranker": reranker}


def main(argv: list[str] | None = None) -> None:
    import argparse
    ap = argparse.ArgumentParser(description="exec one declared serving service")
    ap.add_argument("name", help="an llms[] entry's name, or 'embed'")
    ap.add_argument("-c", "--config", default=None,
                    help="config.toml (default: ./config.toml when present)")
    ns = ap.parse_args(argv)

    from .config import load_config
    cfg_path = ns.config
    # Same resolution as the supervisor: -c wins, then $KNOWLEDGEHOST_CONFIG
    # (load_config reads it when path is None), then the repo's config.toml.
    if cfg_path is None and not os.environ.get("KNOWLEDGEHOST_CONFIG") \
            and (ROOT / "config.toml").exists():
        cfg_path = str(ROOT / "config.toml")
    cfg = load_config(cfg_path)

    if ns.name == "embed":
        _llama_server()          # resolve the binary BEFORE the ~260 MB download
        cmd = embed_argv(cfg, ensure_embed_model())
    else:
        entries = {str(e.get("name")): e for e in cfg["serving"]["llms"]}
        if ns.name not in entries:
            sys.exit(f"no serving.llms entry named '{ns.name}' "
                     f"(have: {', '.join(entries) or 'none'})")
        entry = entries[ns.name]
        cmd = llm_argv(entry)
        # Per-model environment (env = { VLLM_ATTENTION_BACKEND = "...", ... }):
        # applied here, at exec time, so the supervisor path and a manual
        # `python -m knowledgehost.serving <name>` behave identically.
        env = entry.get("env")
        if isinstance(env, dict):
            os.environ.update({str(k): str(v) for k, v in env.items()})
        if entry.get("engine") == "vllm":
            home = cuda_home_probe()
            if home:
                os.environ["CUDA_HOME"] = home
                os.environ["PATH"] = f"{home}/bin:" + os.environ.get("PATH", "")
                print(f"CUDA_HOME not set — using the toolkit at {home}", flush=True)
    print("exec:", " ".join(cmd), flush=True)
    os.execvp(cmd[0], cmd)


if __name__ == "__main__":
    main()
