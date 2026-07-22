"""Model acquisition through the broker (B-15) — the replacement for letting
vLLM/huggingface_hub download weights themselves.

    python3 -m knowledgehost pull --model org/Name [--revision main]

Resolves the repo's file list via the HF tree API (which publishes each LFS
file's sha256), then downloads every file through broker.download() under ONE
lease — resumable, segmented when aria2c is installed, verified against the
published digests — into the model store:

    models/<Org--Name>/config.json, *.safetensors, tokenizer…, .pull.json

Engines are then launched OFFLINE with this local path: they never talk to
the hub, never hold the token, never phone home.  Plain HTTPS end to end — no
huggingface_hub, no Xet side-channel connections.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

from . import broker

HF = "https://huggingface.co"

# Repo files worth having besides the weights: config, tokenizer, generation
# defaults.  Skipped: *.bin when safetensors exist (legacy pickles), README,
# original/ subfolders (raw checkpoints some repos carry alongside).
_SKIP_PREFIX = ("original/", "onnx/", "coreml/", ".git")
_SKIP_SUFFIX = (".md", ".msgpack", ".h5", ".pt", ".png", ".jpg", ".gitattributes")


def store_dir(root: Path, model_id: str) -> Path:
    return root / "models" / model_id.replace("/", "--")


def _tree(model_id: str, revision: str) -> list[dict]:
    """The repo's file list, recursively — path, size, and (for LFS files)
    the sha256 the hub publishes."""
    out, cursor = [], ""
    while True:
        url = (f"{HF}/api/models/{model_id}/tree/{revision}?recursive=true"
               + (f"&cursor={cursor}" if cursor else ""))
        body = broker.request(f"list files of {model_id}", url, timeout=60)
        batch = json.loads(body)
        if not isinstance(batch, list) or not batch:
            break
        out.extend(batch)
        if len(batch) < 1000:                 # the API pages at 1000 entries
            break
        cursor = batch[-1].get("path", "")
        if not cursor:
            break
    return [e for e in out if e.get("type") == "file"]


def _wanted(files: list[dict]) -> list[dict]:
    have_st = any(f["path"].endswith(".safetensors") for f in files)
    keep = []
    for f in files:
        p = f["path"]
        if p.startswith(_SKIP_PREFIX) or p.endswith(_SKIP_SUFFIX):
            continue
        if have_st and (p.endswith(".bin") or p.endswith(".pth")):
            continue                          # legacy pickle weights: never fetch
        keep.append(f)
    return keep


def pull(model_id: str, revision: str = "main", root: Path | None = None,
         say=print) -> Path:
    """Fetch one model snapshot into the store.  Idempotent and resumable:
    complete files are skipped by size+digest, partial ones resume."""
    root = root or Path(__file__).resolve().parent.parent.parent
    dest = store_dir(root, model_id)
    dest.mkdir(parents=True, exist_ok=True)
    purpose = f"model weights: {model_id}"

    with broker.lease(purpose, rule_name="huggingface"):
        files = _wanted(_tree(model_id, revision))
        if not files:
            raise RuntimeError(f"{model_id}@{revision}: the tree API returned no "
                               "files — wrong id, private repo without a token, "
                               "or a licence not yet accepted on huggingface.co")
        total = sum(int(f.get("size") or 0) for f in files)
        say(f"pull {model_id}@{revision}: {len(files)} file(s), "
            f"~{total / 2**30:.1f} GB -> {dest}")
        manifest = {"model": model_id, "revision": revision, "files": {},
                    "pulled_at": time.time()}
        for i, f in enumerate(files, 1):
            rel_path = f["path"]
            sha = str((f.get("lfs") or {}).get("oid") or "")
            size = int(f.get("size") or 0)
            out = dest / rel_path
            out.parent.mkdir(parents=True, exist_ok=True)
            if out.exists() and out.stat().st_size == size:
                say(f"  [{i}/{len(files)}] {rel_path} — already here")
            else:
                say(f"  [{i}/{len(files)}] {rel_path} ({size / 2**20:.0f} MB)")
                broker.download(purpose, f"{HF}/{model_id}/resolve/{revision}/{rel_path}",
                                out, sha256=sha)
            manifest["files"][rel_path] = {"size": size, "sha256": sha}
        (dest / ".pull.json").write_text(json.dumps(manifest, indent=1))
    say(f"done — point the serving entry's model at '{model_id}' as before; "
        f"it now resolves to {dest} and the engine runs offline")
    return dest


def pulled(root: Path, model_id: str) -> Path | None:
    """The store path when this model is COMPLETELY here (every manifest file
    at its full size), else None — a half-pulled model must not silence the
    'not downloaded' hint."""
    d = store_dir(root, model_id)
    mf = d / ".pull.json"
    if not mf.exists():
        return None
    try:
        manifest = json.loads(mf.read_text())
        for rel_path, meta in (manifest.get("files") or {}).items():
            p = d / rel_path
            if not p.exists() or p.stat().st_size != int(meta.get("size") or 0):
                return None
    except (OSError, ValueError):
        return None
    return d
