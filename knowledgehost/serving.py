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
import re
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


# The container engine's default image — pin a tag in config for anything you
# care about; :latest is only the out-of-box default.
DEFAULT_VLLM_IMAGE = "docker.io/vllm/vllm-openai:latest"

# Build-provenance ENV the official image bakes in (Buildkite metadata).
# vLLM's env scanner warns on ANY unrecognised VLLM_-prefixed variable, so the
# image trips its own "Unknown vLLM environment variable detected" at every
# boot.  Podman strips them at run (--unsetenv); docker has no unset flag —
# there the warnings remain (documented benign noise in serving/README.md).
_IMAGE_NOISE_ENV = ("VLLM_BUILD_URL", "VLLM_IMAGE_TAG",
                    "VLLM_BUILD_PIPELINE", "VLLM_BUILD_COMMIT")


def _container_runtime(entry: dict) -> str:
    """podman first (daemonless — the supervisor's signal/process ownership
    works exactly like a plain child), docker accepted.  entry['runtime']
    overrides detection."""
    rt = str(entry.get("runtime") or "").strip()
    if rt:
        return rt
    from shutil import which
    for cand in ("podman", "docker"):
        if which(cand):
            return cand
    raise FileNotFoundError(
        "no container runtime found — install podman (dnf install podman "
        "nvidia-container-toolkit), or set runtime=/path on the entry")


_HF_ID = re.compile(r"^[\w.-]+/[\w.-]+$")


def resolve_model(entry: dict, root: Path = ROOT) -> dict:
    """A vllm/container entry whose `model` is a HF id resolves to the local
    model store (models/<Org--Name>/, filled by `pull`) when the snapshot is
    completely there — the engine then loads from disk and never needs the
    hub.  A local-path model, or an id still living in the legacy hub cache,
    passes through unchanged (the offline env reads the cache fine)."""
    model = str(entry.get("model") or "")
    if entry.get("engine") not in ("vllm", "container") or not _HF_ID.match(model):
        return entry
    if Path(model).expanduser().exists():
        return entry
    from .amiga_net import pull as pull_mod
    local = pull_mod.pulled(root, model)
    return {**entry, "model": str(local)} if local else entry


def hf_env(cfg: dict, engine: str, root: Path = ROOT) -> dict:
    """The OFFLINE block for an inference engine (AMIGA-OPS-01 B-14).

    Engines no longer download anything: model acquisition happens through the
    egress broker (`python3 -m knowledgehost pull`) into the local model
    store, and the engine is handed a filesystem path.  So the engine gets no
    token (the broker holds it), no transfer-accelerator flags (the broker's
    engines do that), and an environment that says so out loud — an engine
    that tries to reach the hub or phone usage stats home fails fast instead
    of quietly succeeding."""
    if engine not in ("vllm", "container"):
        return {}
    return {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
            "VLLM_NO_USAGE_STATS": "1", "DO_NOT_TRACK": "1"}


_PROXY_KEYS = ("http_proxy", "https_proxy", "all_proxy")
# Never proxy the box's own services: the kb talks to its LMs over loopback,
# and a proxy that swallows those is a very confusing outage.
_LOCAL_HOSTS = ("localhost", "127.0.0.1", "::1")


def _local_no_proxy(cfg: dict, current: str) -> str:
    hosts = [h.strip() for h in (current or "").split(",") if h.strip()]
    extra = list(_LOCAL_HOSTS)
    for e in (cfg.get("serving") or {}).get("llms") or []:
        h = str(e.get("host") or "").strip()
        if h and h not in ("0.0.0.0", "::"):
            extra.append(h)
    for h in extra:
        if h not in hosts:
            hosts.append(h)
    return ",".join(hosts)


def proxy_env(cfg: dict) -> dict:
    """Outbound-proxy environment for a serving engine.

    Nothing in the stack reads OS proxy settings: vLLM does no proxying of its
    own, huggingface_hub goes through requests (env vars only on Linux — no
    GNOME/KDE settings) and the Xet backend through reqwest (env vars again).
    So a proxied network must be declared, either in the shell or as
    http_proxy/https_proxy/no_proxy in config.toml — and for engine="container"
    it MUST be declared here, because host env doesn't cross into a container.

    no_proxy always gains loopback and the declared serving hosts."""
    out: dict = {}
    for key in _PROXY_KEYS:
        val = str(cfg.get(key) or os.environ.get(key)
                  or os.environ.get(key.upper()) or "").strip()
        if val:
            out[key] = out[key.upper()] = val
    if not out:
        return {}                                  # no proxy: touch nothing
    no_p = _local_no_proxy(cfg, str(cfg.get("no_proxy")
                                    or os.environ.get("no_proxy")
                                    or os.environ.get("NO_PROXY") or ""))
    out["no_proxy"] = out["NO_PROXY"] = no_p
    return out


def proxy_warning(cfg: dict) -> str | None:
    """Start-time preflight: a proxy is set in the SHELL but loopback isn't
    exempt, so the host's own LM/embed calls would be sent to the proxy."""
    if any(os.environ.get(k) or os.environ.get(k.upper()) for k in _PROXY_KEYS):
        no_p = (os.environ.get("no_proxy") or os.environ.get("NO_PROXY") or "")
        have = {h.strip() for h in no_p.split(",")}
        if not have & set(_LOCAL_HOSTS):
            return ("a proxy is set in the environment but no_proxy doesn't "
                    "exempt loopback — this box's own calls to its LMs "
                    "(127.0.0.1:…) would be sent to the proxy and fail. "
                    "Add: export no_proxy=localhost,127.0.0.1,::1  (env.sh does "
                    "this for services it starts; this warning is about your shell).")
    return None


def engine_env(cfg: dict, engine: str, root: Path = ROOT) -> dict:
    """Everything a serving engine needs in its environment that isn't its
    argv: Hugging Face auth/transfer, and proxy settings when the network
    needs them."""
    return {**hf_env(cfg, engine, root), **proxy_env(cfg)}


# Secrets must never reach a log file: the exec: line prints the full argv,
# and a container's env rides IN the argv as `-e KEY=value` pairs.
_SECRET_ENV = re.compile(
    r"^([A-Za-z0-9_]*(?:TOKEN|SECRET|PASSWORD|API_KEY)[A-Za-z0-9_]*)=(.+)$")


# A proxy URL routinely carries credentials in its userinfo (http://u:pw@host).
_URL_USERINFO = re.compile(r"(://)[^/@\s]+:[^/@\s]+@")


def redact_argv(cmd: list) -> list:
    return [_URL_USERINFO.sub(r"\1***:***@", _SECRET_ENV.sub(r"\1=***", str(a)))
            for a in cmd]


def container_name(entry_name: str) -> str:
    """The deterministic container name llm_argv assigns an engine="container"
    entry — the handle the supervisor stops the WORKLOAD by.  The attached
    client process is only a window onto the container (conmon/containerd owns
    it), so signalling the client can never be the authoritative stop."""
    return f"vinur-llm-{entry_name}"


def container_ref(cfg: dict, entry_name: str) -> tuple | None:
    """(runtime, container_name) for an engine="container" llms entry, or None
    when the entry isn't containerised — or no runtime is installed (nothing to
    stop through a runtime that isn't there)."""
    for e in (cfg.get("serving") or {}).get("llms") or []:
        if str(e.get("name")) == str(entry_name) and e.get("engine") == "container":
            try:
                return _container_runtime(e), container_name(str(entry_name))
            except FileNotFoundError:
                return None
    return None


def llm_argv(entry: dict, root: Path = ROOT) -> list[str]:
    """argv for one llms[] entry ({name, engine, model, port, host, exclusive,
    default, env, args} + the first-class tuning keys in _VLLM_KEYS /
    ctx_size + n_gpu_layers for llama / image + runtime for container)."""
    for k in ("name", "engine", "model", "port"):
        if not entry.get(k):
            raise ValueError(f"serving.llms entry needs '{k}': {entry}")
    host = str(entry.get("host") or "127.0.0.1")
    port = str(int(entry["port"]))
    args = [str(a) for a in (entry.get("args") or [])]
    engine = entry["engine"]
    if engine == "container":
        # The official vLLM image via podman/docker: the image carries the
        # matched CUDA toolkit + compiler the wheels were built against, so
        # the host needs ONLY the driver — this is the engine that ends the
        # bleeding-edge-distro toolchain fight (nvcc, gcc ceilings, JIT).
        rt = _container_runtime(entry)
        is_podman = "podman" in os.path.basename(rt)
        hf = root / "var" / "cache" / "huggingface"
        argv = [rt, "run", "--rm", "--name", container_name(str(entry["name"]))]
        if is_podman:
            # --replace clears a stale same-name container after an unclean
            # stop; CDI is the toolkit's GPU wiring (nvidia-ctk cdi generate).
            argv += ["--replace", "--device", "nvidia.com/gpu=all"]
            for k in _IMAGE_NOISE_ENV:      # an explicit entry env -e still wins
                argv += ["--unsetenv", k]
        else:
            argv += ["--gpus", "all"]
        # :z — SELinux shared label; without it a Fedora host denies the
        # container access to the mounted cache.  --ipc=host per vLLM's docs
        # (shared memory for its worker processes).
        argv += ["--ipc=host",
                 "-p", f"{host}:{port}:8000",
                 "-v", f"{hf}:/root/.cache/huggingface:z"]
        # A model from the local store (resolve_model) is a directory on the
        # host: mount it read-only and hand the engine the in-container path.
        model_arg = str(entry["model"])
        if os.path.isdir(model_arg):
            argv += ["-v", f"{os.path.abspath(model_arg)}:/model:ro,z"]
            model_arg = "/model"
        for k, v in (entry.get("env") or {}).items():
            argv += ["-e", f"{k}={v}"]
        # The image's entrypoint IS `vllm serve` — model is positional, the
        # same first-class keys map to the same flags.  The inner server
        # listens on 0.0.0.0:8000; -p above binds it to the host port.
        return (argv + [str(entry.get("image") or DEFAULT_VLLM_IMAGE),
                        model_arg]
                + _mapped_flags(entry, _VLLM_KEYS) + args)
    if engine == "vllm":
        # Bare-metal venv.  On bleeding-edge distros whose gcc/glibc outrun
        # NVIDIA's support matrix, prefer engine = "container" instead.
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
    raise ValueError(f"unknown serving engine '{engine}' (vllm | llama | container)")


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
    """Fetch the nomic GGUF into models/ once — through the egress broker
    (policy-checked, audited, resumable), like every other download."""
    dest = root / "models" / EMBED_MODEL_FILE
    if dest.is_file():
        return str(dest)
    from .amiga_net import broker
    print(f"downloading {EMBED_MODEL_FILE} (~260 MB) -> {dest}", flush=True)
    with broker.lease("embed model (first start)", rule_name="huggingface"):
        broker.download("embed model (first start)", EMBED_MODEL_URL, dest)
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


def entry_for_url(cfg: dict, url: str, exclusive_only: bool = False) -> dict | None:
    """The [[serving.llms]] entry that answers at `url`.  Ports must match;
    hosts match when equal, or when both are local (loopback/0.0.0.0).  None
    for foreign hosts and unparseable urls.  This is how the distiller learns
    what ENGINE sits behind an LM-lane URL (vLLM batches; llama.cpp doesn't)
    and how the autopilot maps a URL to a swappable model."""
    from urllib.parse import urlparse
    try:
        p = urlparse(url if "//" in str(url) else f"http://{url}")
        uhost, uport = (p.hostname or "").lower(), int(p.port or 0)
    except (ValueError, TypeError, AttributeError):
        return None
    if not uport:
        return None
    local = {"127.0.0.1", "localhost", "::1", "0.0.0.0", ""}
    for e in (cfg.get("serving") or {}).get("llms") or []:
        if exclusive_only and not e.get("exclusive"):
            continue
        if int(e.get("port") or 0) != uport:
            continue
        ehost = str(e.get("host") or "127.0.0.1").lower()
        if ehost == uhost or (ehost in local and uhost in local):
            return e
    return None


def exclusive_entry_for_url(cfg: dict, url: str) -> str | None:
    """Which EXCLUSIVE [[serving.llms]] entry answers at `url`?  None for
    non-exclusive entries (always resident — no swap needed).  This is what
    lets the autopilot derive a step's model from the LM-lane URL its verb
    drives (auto_models)."""
    e = entry_for_url(cfg, url, exclusive_only=True)
    return str(e.get("name")) if e else None


# Per-service control (Serving tab's start/stop/restart), one request file per
# service so two buttons pressed together can't overwrite each other — unlike
# the swap lane, where a single request file is the point (one GPU, one winner).
SVC_REQ_DIR = ROOT / "var" / "run" / "svcreq"
SVC_ACTIONS = ("start", "stop", "restart")


def _safe_service(name: str) -> str:
    """A service name is a filename here — keep it one path component."""
    s = str(name or "").strip()
    if not s or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", s):
        raise ValueError(f"bad service name: {name!r}")
    return s


def request_service(service: str, action: str) -> None:
    """Ask the supervisor to start/stop/restart ONE service.  Async like the
    swap lane: the watchdog acts within a tick and the panel re-polls."""
    s = _safe_service(service)
    if action not in SVC_ACTIONS:
        raise ValueError(f"action must be one of {'/'.join(SVC_ACTIONS)}")
    SVC_REQ_DIR.mkdir(parents=True, exist_ok=True)
    tmp = SVC_REQ_DIR / f".{s}.tmp"
    tmp.write_text(json.dumps({"service": s, "action": action, "at": time.time()}))
    os.replace(tmp, SVC_REQ_DIR / f"{s}.req")


def take_service_requests() -> list[dict]:
    """Supervisor side: consume every pending request (each file read then
    unlinked, so a request survives at most one tick)."""
    out = []
    try:
        files = sorted(SVC_REQ_DIR.glob("*.req"))
    except OSError:
        return out
    for f in files:
        try:
            d = json.loads(f.read_text())
        except (OSError, ValueError):
            d = {}
        try:
            f.unlink()
        except OSError:
            pass
        if d.get("service") and d.get("action") in SVC_ACTIONS:
            out.append(d)
    return out


def log_tail(service: str, lines: int = 200, max_bytes: int = 262144) -> dict:
    """The tail of one service's log, for the panel.  Reading the last N lines
    IS the diagnosis for most serving problems (a download that stopped, a
    gated repo, a rate limit) — the single last line the table shows is rarely
    the one that says why."""
    from . import supervisor as sup
    s = _safe_service(service)
    p = sup.LOGS / f"{s}.log"
    try:
        size = p.stat().st_size
        with p.open("rb") as fh:
            if size > max_bytes:
                fh.seek(size - max_bytes)
                fh.readline()                     # drop the partial first line
            text = fh.read().decode("utf-8", "replace")
    except OSError as e:
        return {"service": s, "path": str(p), "exists": False, "text": "",
                "detail": f"no log yet ({type(e).__name__})"}
    keep = [ln for ln in text.splitlines() if ln.strip()][-max(1, int(lines)):]
    return {"service": s, "path": str(p), "exists": True, "size": size,
            "mtime": p.stat().st_mtime, "text": "\n".join(keep)}


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
     "Blackwell needs them) can't build. Switch the entry to engine="
     "\"container\", or install the toolkit: serving/README.md → "
     "Troubleshooting."),
    ("unsupported GNU version",
     "system gcc is newer than the CUDA toolkit supports — switch the entry "
     "to engine=\"container\" (ends the toolchain fight), or add env = "
     '{ NVCC_APPEND_FLAGS = "-allow-unsupported-compiler" }. '
     "This error is HOST-toolchain only: if you meant to run the container, "
     "the entry isn't — check the 'exec:' line at the top of this log. "
     "serving/README.md → Troubleshooting."),
    ("errors.pydantic.dev",
     "vLLM REJECTED ITS OWN CONFIG before loading anything — one of the "
     "entry's keys/args isn't valid for this vLLM version (a flag that was "
     "renamed or removed, a value out of range, a bad JSON blob in "
     "speculative_config/override args). The pydantic 'Value error, …' line a "
     "few lines above names the field — open the Log button here and read "
     "upward from the URL; the last line is only the sign-off."),
    ("unresolvable CDI devices",
     "podman can't wire the GPU — the CDI spec is missing or stale. Run "
     "'sudo nvidia-ctk cdi generate --output=/etc/cdi/nvidia.yaml' "
     "(again after every driver update)."),
    ("could not select device driver",
     "docker's daemon doesn't know the NVIDIA runtime — run 'sudo nvidia-ctk "
     "runtime configure --runtime=docker && sudo systemctl restart docker', "
     "or use podman (CDI, no daemon config)."),
    ("CUDA out of memory",
     "VRAM overflow — cap max_model_len (models declare huge native "
     "contexts), lower max_num_seqs / gpu_memory_utilization, use "
     "kv_cache_dtype=\"fp8\", and check what else is resident (exclusive "
     "swap mode exists for this). serving/README.md has a worked example."),
    ("401 Client Error",
     "gated HF repo — accept its license on huggingface.co and set "
     "hf_token in config.toml (or export HF_TOKEN) before starting."),
    # Download failures: the weights never arrive and the engine dies far from
    # the real cause, so name the cause.
    ("429 Client Error",
     "Hugging Face rate-limited this box (HTTP 429) — anonymous downloads are "
     "throttled first. Set hf_token in Settings (a free read token lifts the "
     "limit), then Restart the service; the partial download resumes."),
    ("Too Many Requests",
     "Hugging Face rate-limited this box (HTTP 429) — anonymous downloads are "
     "throttled first. Set hf_token in Settings (a free read token lifts the "
     "limit), then Restart the service; the partial download resumes."),
    ("403 Client Error",
     "Hugging Face refused the download (HTTP 403) — the token is missing the "
     "repo's permission, or the licence was never accepted with THIS account. "
     "Check the repo page while signed in as the token's owner."),
    ("RepositoryNotFoundError",
     "no such HF repo (or it's private to another account) — check the `model` "
     "id for a typo; a private repo also needs hf_token."),
    ("No space left on device",
     "the disk holding var/cache/huggingface is full — weights are tens of GB "
     "each. Free space or move the cache (Settings › Paths shows where it is), "
     "then Restart; the partial download resumes."),
    ("ReadTimeoutError",
     "the download timed out mid-transfer — usually the network, occasionally "
     "an HF incident. Restart the service: the fetch resumes from the partial "
     "blobs rather than starting over."),
    ("Consistency check failed",
     "a downloaded file didn't match its expected size/hash — a corrupted or "
     "truncated transfer. Delete that repo's folder under "
     "var/cache/huggingface/hub and Restart to fetch it cleanly."),
    ("GatedRepoError",
     "gated HF repo — accept its license on huggingface.co and set "
     "hf_token in config.toml (or export HF_TOKEN) before starting."),
    ("No space left on device",
     "disk full — weights live in var/cache/huggingface; prune models you "
     "dropped from the config."),
]


# A dying process's LAST line is almost never the informative one — vLLM signs
# off with a pydantic docs URL, NCCL teardown noise, or a bare traceback tail.
# These markers pick the line that actually says what went wrong.
_CAUSE_MARKS = ("Value error,", "validation error", "ValueError:", "TypeError:",
                "KeyError:", "RuntimeError:", "OSError:", "AssertionError",
                "is not a valid", "Error: ", "error: ", "Errno", "No such file",
                "Traceback (most recent")
_CAUSE_NOISE = ("For further information visit", "https://errors.pydantic.dev",
                "During handling of the above", "The above exception was")


def cause_lines(tail: str, limit: int = 3) -> list[str]:
    """The lines from a service's log tail that name the failure, newest last.
    Shown beside the last-log line because 'exited rc=1 — <docs URL>' tells an
    operator nothing they can act on."""
    hits = []
    for ln in (tail or "").splitlines():
        s = ln.strip()
        if not s or any(n in s for n in _CAUSE_NOISE):
            continue
        if any(m in s for m in _CAUSE_MARKS):
            hits.append(s[-400:])
    return hits[-max(1, int(limit)):]


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
    toolkit_present injects the detection for tests.  engine="container"
    entries are exempt — the image carries its own toolkit."""
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
                "fail in the FlashInfer JIT ('Could not find nvcc').  Switch the "
                "entry to engine = \"container\", or install the toolkit: "
                "serving/README.md → Troubleshooting.")
    return ("no CUDA toolkit (nvcc) found — vLLM runs, but JIT kernel paths "
            "(FlashInfer MoE, some attention backends) will fail if a model "
            "needs them.  engine = \"container\" avoids this entirely; "
            "serving/README.md → Troubleshooting.")


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


def _snapshot_complete(s) -> bool:
    """A HF-cache snapshot is complete when config.json is present and every
    weight shard the snapshot NAMES resolves to a real blob — a snapshot
    entry is a symlink into blobs/, and a broken link means that shard is
    still mid-download (Path.exists() follows the link).  Sharded models are
    checked against model.safetensors.index.json, so a fetch that died before
    creating the last shard's link reads incomplete, not ready."""
    s = Path(s)
    if not (s / "config.json").exists():
        return False
    idx = s / "model.safetensors.index.json"
    if idx.exists():
        try:
            names = set(json.loads(idx.read_text()).get("weight_map", {}).values())
        except (OSError, ValueError):
            names = set()
        if names:
            return all((s / n).exists() for n in names)
    shards = list(s.glob("*.safetensors"))
    return bool(shards) and all(p.exists() for p in shards)


def _ago(seconds: float) -> str:
    s = int(max(0, seconds))
    if s < 90:
        return f"{s}s"
    if s < 5400:
        return f"{s // 60}m"
    return f"{s // 3600}h {(s % 3600) // 60}m"


def hf_cache_dir() -> Path:
    """The Hugging Face hub cache THIS box downloads weights into.  env.sh
    pins HF_HOME inside the repo (var/cache/huggingface) so nothing lands in
    ~/.cache and the container mount is the same tree; a caller who overrides
    HF_HOME wins.  Layout below it: hub/models--Org--Name/{blobs,snapshots,refs}
    — snapshots/<rev>/ is the readable tree of symlinks into the flat blobs."""
    return Path(os.environ.get("HF_HOME")
                or (ROOT / "var" / "cache" / "huggingface")) / "hub"


def hf_cache_status() -> dict:
    """Where downloaded weights live, for anyone asking 'where did the 200 GB
    go?' — the path to open, what's in it, and the stale-partial litter that a
    completed retry leaves behind (safe to delete, so it's worth naming)."""
    hub = hf_cache_dir()
    out = {"path": str(hub), "exists": hub.is_dir(), "repos": 0,
           "size_gb": 0.0, "incomplete_gb": 0.0,
           "env": "HF_HOME" if os.environ.get("HF_HOME") else "default (var/cache/huggingface)"}
    if not out["exists"]:
        return out
    repos = [d for d in hub.iterdir() if d.is_dir() and d.name.startswith("models--")]
    out["repos"] = len(repos)
    out["size_gb"] = round(_tree_size(hub) / 2**30, 1)
    partial = 0
    for d in repos:
        blobs = d / "blobs"
        if blobs.is_dir():
            for f in blobs.glob("*.incomplete"):
                try:
                    partial += f.stat().st_size
                except OSError:
                    pass
    out["incomplete_gb"] = round(partial / 2**30, 1)
    return out


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
    if engine in ("vllm", "container"):
        # container mounts the same in-tree HF cache, so one check serves both
        mp = Path(model).expanduser()
        if mp.is_dir():                                # local snapshot directory
            if any(mp.glob("*.safetensors")):
                return {"status": "ready", "path": str(mp),
                        "size_gb": round(_tree_size(mp) / 2**30, 1)}
            return {"status": "incomplete", "path": str(mp),
                    "detail": "local dir has no *.safetensors"}
        # the model store (broker-pulled snapshots) outranks the legacy hub cache
        from .amiga_net import pull as pull_mod
        is_id = bool(_HF_ID.match(model))
        if is_id:
            local = pull_mod.pulled(ROOT, model)
            if local:
                return {"status": "ready", "path": str(local),
                        "size_gb": round(_tree_size(local) / 2**30, 1)}
            # a pull mid-flight (or interrupted): the Serving tab must show a
            # LIVE download, not "missing", while the broker is filling the store
            sd = pull_mod.store_dir(ROOT, model)
            if sd.is_dir():
                parts = sorted(p for p in sd.rglob("*.part") if p.is_file())
                done = [p for p in sd.rglob("*")
                        if p.is_file() and not p.name.endswith(".part")
                        and p.name != ".pull.json"]
                if parts or done:
                    out = {"path": str(sd),
                           "size_gb": round(_tree_size(sd) / 2**30, 1)}
                    if parts:
                        try:
                            age = time.time() - max(f.stat().st_mtime for f in parts)
                        except (OSError, ValueError):
                            age = None
                        out["idle_s"] = round(age) if age is not None else None
                        if age is not None and age > 120:
                            out.update(status="stalled",
                                       detail=f"a pull left {len(parts)} file(s) "
                                              "mid-download and NOTHING has been written "
                                              f"for {_ago(age)} — re-run it (Ops › pull); "
                                              "partial files resume, nothing restarts "
                                              "from zero")
                        else:
                            out.update(status="incomplete",
                                       detail=f"downloading now — {len(done)} file(s) "
                                              f"done, {len(parts)} in flight, last "
                                              "written "
                                              f"{_ago(age) if age is not None else 'just now'}"
                                              " ago")
                    else:
                        out.update(status="incomplete",
                                   detail=f"{len(done)} file(s) here but the pull never "
                                          "finished — re-run it to fetch the rest "
                                          "(resumable)")
                    return out
        d = hf_cache_dir() / ("models--" + model.replace("/", "--"))
        if not d.is_dir():
            return {"status": "missing",
                    "path": str(pull_mod.store_dir(ROOT, model) if is_id else d),
                    "detail": ("not downloaded — engines run offline; fetch "
                               "through the egress broker: python3 -m "
                               f"knowledgehost pull --model {model} "
                               "(or the panel: Ops › pull)")}
        blobs = d / "blobs"
        parts = sorted(blobs.glob("*.incomplete")) if blobs.is_dir() else []
        partial = len(parts)
        snap_ok = False
        snaps = d / "snapshots"
        if snaps.is_dir():
            snap_ok = any(_snapshot_complete(s) for s in snaps.iterdir())
        out = {"path": str(d), "size_gb": round(_tree_size(d) / 2**30, 1)}
        # A COMPLETE snapshot wins: an interrupted first fetch leaves a stale
        # *.incomplete blob behind, and the successful retry downloads to a
        # fresh temp name — so the litter outlives the completed download and
        # must not flip a ready model back to "incomplete".
        if snap_ok:
            out.update(status="ready")
            if partial:
                out["detail"] = (f"{partial} stale .incomplete file(s) in the blob "
                                 "cache from an earlier interrupted fetch — harmless; "
                                 "delete them to reclaim disk")
        elif partial:
            # "Downloading" and "wedged" look identical in a size figure — the
            # question is whether the partial files are still GROWING.  The
            # newest write's age answers it without sampling over time.
            age = None
            try:
                age = time.time() - max(f.stat().st_mtime for f in parts)
            except (OSError, ValueError):
                pass
            out["idle_s"] = round(age) if age is not None else None
            if age is not None and age > 120:
                out.update(status="stalled",
                           detail=f"{partial} file(s) mid-download but NOTHING has been "
                                  f"written for {_ago(age)} — the fetch is stuck or dead, "
                                  "not slow. Check the log (Log button): rate limit (429), "
                                  "auth (401/403), disk, or network. Re-run the pull — "
                                  "partial files are reused, nothing restarts from zero.")
            else:
                out.update(status="incomplete",
                           detail=f"{partial} file(s) mid-download, last written "
                                  f"{_ago(age) if age is not None else 'just now'} ago "
                                  "— downloading now")
        else:
            out.update(status="incomplete",
                       detail="cache present but no complete snapshot — fetch "
                              "interrupted? (restarting the service resumes it)")
        return out
    return {"status": "unknown", "detail": f"unknown engine '{engine}'"}


def eligible_models(engine: str, root: Path = ROOT, cfg: dict | None = None) -> list[dict]:
    """Every model ON THIS DISK that `engine` could serve — the Serving tab's
    picker.  vllm/container: complete broker-store snapshots holding
    safetensors, plus complete legacy hub-cache snapshots.  llama: every GGUF
    under models/ (embed/reranker support files excluded; only the first part
    of a split GGUF — llama-server finds the siblings itself).  Disk only:
    nothing here goes near the network."""
    from .amiga_net import pull as pull_mod
    out: list[dict] = []
    seen: set[str] = set()

    def add(model: str, size_gb: float, via: str) -> None:
        if model and model not in seen:
            seen.add(model)
            out.append({"model": model, "size_gb": size_gb, "via": via})

    store = root / "models"
    if engine in ("vllm", "container"):
        for mf in sorted(store.glob("*/.pull.json")):
            try:
                man = json.loads(mf.read_text())
            except (OSError, ValueError):
                continue
            files = man.get("files") or {}
            if not any(k.endswith(".safetensors") for k in files):
                continue                      # a GGUF store dir: llama's lane
            mid = str(man.get("model") or "")
            if mid and pull_mod.pulled(root, mid):
                gb = round(sum(int(v.get("size") or 0) for v in files.values()) / 2**30, 1)
                add(mid, gb, "store")
        hub = hf_cache_dir()
        if hub.is_dir():
            for d in sorted(hub.glob("models--*")):
                snaps = d / "snapshots"
                if snaps.is_dir() and any(
                        _snapshot_complete(s) and any(s.glob("*.safetensors"))
                        for s in snaps.iterdir()):
                    add(d.name[len("models--"):].replace("--", "/", 1),
                        round(_tree_size(d) / 2**30, 1), "hub cache")
    elif engine == "llama":
        skip = {EMBED_MODEL_FILE,
                str((cfg or {}).get("rerank_model") or "bge-reranker-v2-m3-Q8_0.gguf")}
        for p in sorted(store.rglob("*.gguf")) if store.is_dir() else []:
            if p.name in skip or not p.is_file():
                continue
            m = re.search(r"-(\d{5})-of-\d{5}\.gguf$", p.name)
            if m and m.group(1) != "00001":
                continue                      # later split parts aren't loadable alone
            add(str(p.relative_to(root)), round(p.stat().st_size / 2**30, 2), "models/")
    return out


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
    held = set(st.get("held") or [])

    def svc_state(svc_name: str, entry: str = "") -> dict:
        if not sup_alive:
            return {"service": "supervisor-down", "service_name": svc_name}
        if svc_name in held:                      # stopped BY REQUEST, not by dying:
            return {"service": "stopped",         # the watchdog leaves it alone
                    "service_name": svc_name}
        if svc_name in failed:
            return {"service": "failed", "reason": failed[svc_name],
                    "service_name": svc_name}
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
            cause = cause_lines(tail)
            if cause:
                d["cause"] = cause
        d["service_name"] = svc_name
        return d

    llms = []
    choices_by_engine: dict = {}               # one disk scan per engine, not per entry
    for e in cfg["serving"]["llms"]:
        name = str(e.get("name") or "")
        engine = str(e.get("engine") or "")
        if engine not in choices_by_engine:
            choices_by_engine[engine] = eligible_models(engine, cfg=cfg)
        item = {"name": name, "engine": engine,
                "model": str(e.get("model") or ""), "port": e.get("port"),
                "exclusive": bool(e.get("exclusive")), "default": bool(e.get("default")),
                "choices": choices_by_engine[engine],
                "weights": weights_status(engine, str(e.get("model") or ""))}
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
            "swap": swap_state(), "llms": llms, "embed": embed, "reranker": reranker,
            "cache": hf_cache_status()}


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
        entry = resolve_model(entry)           # HF id -> local store path when pulled
        ws = weights_status(str(entry.get("engine") or ""), str(entry.get("model") or ""))
        if ws.get("status") == "missing" and entry.get("engine") in ("vllm", "container"):
            # Engines run OFFLINE now — failing here, with the fix named, beats
            # letting vLLM discover it can't reach the hub minutes into startup.
            sys.exit(f"weights for '{entry.get('model')}' are not on this machine "
                     f"({ws.get('path')}) — download them through the egress broker:\n"
                     f"    python3 -m knowledgehost pull --model {entry.get('model')}\n"
                     f"(or the panel: Ops › pull)")
        hf = engine_env(cfg, str(entry.get("engine") or ""))
        if entry.get("engine") == "container":
            # HF auth/transfer + proxy env must ride INTO the container as -e
            # flags (host env doesn't cross); an explicit entry env still wins.
            entry = {**entry, "env": {**hf, **dict(entry.get("env") or {})}}
        cmd = llm_argv(entry)
        if entry.get("engine") == "container":
            # env went into the argv as -e flags; just guarantee the mounted
            # cache exists so the runtime doesn't invent it with odd labels.
            (ROOT / "var" / "cache" / "huggingface").mkdir(parents=True, exist_ok=True)
        else:
            # Per-model environment (env = { NVCC_APPEND_FLAGS = "...", ... }):
            # applied here, at exec time, so the supervisor path and a manual
            # `python -m knowledgehost.serving <name>` behave identically.
            # HF/proxy env first — an explicit entry env overrides it.
            os.environ.update(hf)
            env = entry.get("env")
            if isinstance(env, dict):
                os.environ.update({str(k): str(v) for k, v in env.items()})
        if entry.get("engine") == "vllm":
            home = cuda_home_probe()
            if home:
                os.environ["CUDA_HOME"] = home
                os.environ["PATH"] = f"{home}/bin:" + os.environ.get("PATH", "")
                print(f"CUDA_HOME not set — using the toolkit at {home}", flush=True)
    print("exec:", " ".join(redact_argv(cmd)), flush=True)
    os.execvp(cmd[0], cmd)


if __name__ == "__main__":
    main()
