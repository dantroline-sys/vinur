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


def _llama_server() -> str:
    """The llama.cpp server binary: $LLAMA_SERVER wins, else PATH lookup."""
    exe = os.environ.get("LLAMA_SERVER", "").strip()
    if exe:
        return exe
    from shutil import which
    found = which("llama-server")
    if not found:
        raise FileNotFoundError(
            "llama-server not found — install llama.cpp or set LLAMA_SERVER=/path/to/llama-server")
    return found


def _anchored(path: str) -> str:
    """Relative model paths anchor to the repo root (same rule as config paths)."""
    return str(ROOT / Path(path).expanduser())


def llm_argv(entry: dict, root: Path = ROOT) -> list[str]:
    """argv for one llms[] entry ({name, engine, model, port, args, host})."""
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
        return [str(vllm), "serve", str(entry["model"]),
                "--host", host, "--port", port] + args
    if engine == "llama":
        model = _anchored(str(entry["model"]))
        if not os.path.isfile(model):
            raise FileNotFoundError(f"GGUF not found: {model}")
        # -ngl 99 default (a box serving LMs wants them on GPU); args can override
        # because llama-server takes the LAST occurrence of a repeated flag.
        return [_llama_server(), "-m", model, "--host", host, "--port", port,
                "-ngl", "99"] + args
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
        cmd = embed_argv(cfg, ensure_embed_model())
    else:
        entries = {str(e.get("name")): e for e in cfg["serving"]["llms"]}
        if ns.name not in entries:
            sys.exit(f"no serving.llms entry named '{ns.name}' "
                     f"(have: {', '.join(entries) or 'none'})")
        cmd = llm_argv(entries[ns.name])
    print("exec:", " ".join(cmd), flush=True)
    os.execvp(cmd[0], cmd)


if __name__ == "__main__":
    main()
