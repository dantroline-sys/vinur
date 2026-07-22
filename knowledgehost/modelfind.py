"""Find models on the hub and judge whether they fit THIS machine.

    ./vinur.sh find qwen3 32b fp8        # search, sized, verdicts
    ./vinur.sh pull 2                    # pull row 2 of the last find

One leased broker operation: search huggingface.co's catalogue, fetch each
candidate's file list (the tree API publishes exact sizes), and say plainly
whether the weights fit the detected VRAM (or unified/system memory when
there is no discrete GPU).  GGUF repositories expand into their individual
quantisation files — each one a selectable row, and pulling it fetches only
that file.

Every row is numbered; the numbering is saved to var/run/find.json so `pull`
can take a number instead of an id.  Nothing here touches the network except
through amiga_net — same lease, same audit trail as pull itself.
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
import time
import urllib.error
import urllib.parse
from pathlib import Path

from .amiga_net import broker
from .amiga_net import pull as pullmod

GiB = 2 ** 30
_ROOT = Path(__file__).resolve().parent.parent

# multi-part GGUFs: "…-Q4_K_M-00001-of-00002.gguf" group under "…-Q4_K_M.gguf"
_SPLIT_GGUF = re.compile(r"-\d{5}-of-\d{5}\.gguf$", re.IGNORECASE)
# the quant token inside a GGUF filename, for a short row label
_QUANT = re.compile(r"(?i)\b(i?q\d[\w]*|f16|bf16|f32)\b")
# weight format hints, cosmetic only (first match wins, most specific first)
_FMT = ("fp8", "nvfp4", "awq", "gptq", "int4", "int8", "gguf", "bf16", "fp16")


def _human(n: int) -> str:
    for cut, suffix in ((1_000_000, "M"), (1_000, "k")):
        if n >= cut:
            return f"{n / cut:.1f}{suffix}".replace(".0", "")
    return str(n)


def _fmt_hint(cand: dict) -> str:
    hay = (cand["id"] + " " + " ".join(cand.get("tags") or [])).lower()
    return next((f for f in _FMT if f in hay), "")


# ── hardware ─────────────────────────────────────────────────────────────────

def budget() -> tuple[int, str]:
    """(bytes, label) of the memory the weights must fit in: total VRAM when
    nvidia-smi answers, unified memory on a Mac, system RAM as the honest
    fallback (CPU inference is real, just slow)."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.total,name",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5)
        if out.returncode == 0:
            mib, name = 0, ""
            for line in out.stdout.splitlines():
                parts = [p.strip() for p in line.split(",", 1)]
                try:
                    mib += int(float(parts[0]))
                except ValueError:
                    continue
                name = name or (parts[1] if len(parts) > 1 else "")
            if mib:
                return mib * 2 ** 20, f"{mib / 1024:.0f} GB VRAM ({name})"
    except (OSError, subprocess.SubprocessError):
        pass
    if sys.platform == "darwin":
        try:
            out = subprocess.run(["sysctl", "-n", "hw.memsize"],
                                 capture_output=True, text=True, timeout=5)
            b = int(out.stdout.strip())
            return b, f"{b / GiB:.0f} GB unified memory"
        except (OSError, ValueError, subprocess.SubprocessError):
            pass
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    kb = int(line.split()[1])
                    return kb * 1024, (f"{kb / 2 ** 20:.0f} GB system RAM "
                                       "(no GPU detected)")
    except (OSError, ValueError):
        pass
    return 0, "unknown hardware"


def fit(weight_bytes: int, budget_bytes: int) -> tuple[str, str]:
    """('fits'|'tight'|'too big'|'?', why).  Rough but honest: the runtime
    needs the weights plus ~10% for activations/CUDA graphs plus ~2 GB of KV
    cache, and vLLM won't touch the last ~10% of VRAM anyway."""
    if not budget_bytes:
        return "?", "no memory detected — can't judge"
    need = weight_bytes * 1.10 + 2 * GiB
    n, b = need / GiB, budget_bytes / GiB
    if need <= budget_bytes * 0.90:
        return "fits", f"~{n:.0f} of {b:.0f} GB"
    if need <= budget_bytes:
        return "tight", f"~{n:.0f} of {b:.0f} GB — little KV-cache headroom"
    return "too big", f"needs ~{n:.0f} GB, this machine has {b:.0f}"


# ── the hub ──────────────────────────────────────────────────────────────────

def search(query: str, limit: int = 8) -> list[dict]:
    """Top catalogue hits for the query, most-downloaded first."""
    url = (f"{pullmod.HF}/api/models?search={urllib.parse.quote(query)}"
           f"&sort=downloads&direction=-1&limit={max(limit * 3, 20)}")
    rows = json.loads(broker.request(f"model search: {query}", url, timeout=30))
    out = []
    for r in rows if isinstance(rows, list) else []:
        mid = str(r.get("modelId") or r.get("id") or "")
        if not mid or r.get("private"):
            continue
        out.append({"id": mid, "downloads": int(r.get("downloads") or 0),
                    "gated": bool(r.get("gated")), "tags": r.get("tags") or []})
        if len(out) >= limit:
            break
    return out


def _sized(cand: dict, revision: str = "main") -> dict:
    """Attach sizes from the tree API: total wanted bytes, or per-quant rows
    ('quants') when the repo is GGUF-only.  A 401/403 marks it blocked (gated
    repo, licence not accepted / no token) instead of failing the whole find."""
    try:
        files = pullmod._wanted(pullmod._tree(cand["id"], revision))
    except urllib.error.HTTPError as e:
        if e.code in (401, 403):
            cand["blocked"] = True
            return cand
        raise
    ggufs = [f for f in files if f["path"].lower().endswith(".gguf")]
    if ggufs and not any(f["path"].endswith(".safetensors") for f in files):
        groups: dict[str, int] = {}
        for f in ggufs:
            key = _SPLIT_GGUF.sub(".gguf", f["path"])
            groups[key] = groups.get(key, 0) + int(f.get("size") or 0)
        cand["quants"] = sorted(groups.items(), key=lambda kv: -kv[1])
    else:
        cand["bytes"] = sum(int(f.get("size") or 0) for f in files)
    return cand


# ── the numbered result list ─────────────────────────────────────────────────

def _picks_path(root: Path) -> Path:
    return root / "var" / "run" / "find.json"


def pick(n: int, root: Path | None = None) -> tuple[str, str] | None:
    """Row n of the last find -> (model_id, include_glob) — or None."""
    try:
        saved = json.loads(_picks_path(root or _ROOT).read_text())
        row = saved["picks"][n - 1]
        return str(row["id"]), str(row.get("include") or "")
    except (OSError, ValueError, LookupError, TypeError):
        return None


def gather(query: str, root: Path | None = None, limit: int = 8,
           budget_bytes: int | None = None, budget_label: str = "",
           engines: set | None = None, fit_only: bool = False) -> dict:
    """Search + size + judge, structured — the panel's pick-list and find()'s
    data.  `engines` (config vocabulary: vllm/container/llama) keeps only rows
    something in this box's [serving] table can actually run; `fit_only` drops
    what can't fit the memory budget.  Hidden rows are counted, never silent.
    Saves the numbered picks, so pull-by-number matches what was shown."""
    root = root or _ROOT
    if budget_bytes is None:
        budget_bytes, budget_label = budget()
    want_vllm = not engines or bool({"vllm", "container"} & engines)
    want_llama = not engines or "llama" in engines
    with broker.lease(f"model search: {query}", rule_name="huggingface"):
        cands = search(query, limit=limit)
        for c in cands:
            _sized(c)

    rows: list[dict] = []
    hidden = {"engine": 0, "fit": 0}
    for c in cands:
        hint = _fmt_hint(c)
        # which engine serves this: GGUF files are the llama engine's food,
        # safetensors repos feed the vllm/container engines
        gguf = bool(c.get("quants")) or hint == "gguf"
        engine = "llama" if gguf else "vllm"
        if (engine == "llama" and not want_llama) or (engine == "vllm" and not want_vllm):
            hidden["engine"] += 1
            continue
        base = {"id": c["id"], "engine": engine, "downloads": c["downloads"],
                "format": "" if gguf else hint}
        if c.get("blocked"):
            rows.append({**base, "include": "", "label": c["id"], "size_gb": None,
                         "verdict": "gated",
                         "why": "accept its licence on huggingface.co "
                                "(and set hf_token), then pull"})
        elif c.get("quants"):
            fitting = [q for q in c["quants"]
                       if not budget_bytes or fit(q[1], budget_bytes)[0] != "too big"]
            shown = fitting[:4] if fitting else \
                ([] if fit_only else c["quants"][-1:])    # nothing fits -> smallest
            hidden["fit"] += len(c["quants"]) - len(shown)
            for key, size in shown:
                m = _QUANT.search(Path(key).stem)
                verdict, why = fit(size, budget_bytes)
                rows.append({**base, "include": key[:-len(".gguf")] + "*",
                             "label": m.group(1) if m else Path(key).stem,
                             "size_gb": round(size / GiB, 1),
                             "verdict": verdict, "why": why})
        else:
            size = int(c.get("bytes") or 0)
            verdict, why = fit(size, budget_bytes)
            if fit_only and verdict == "too big":
                hidden["fit"] += 1
                continue
            rows.append({**base, "include": "", "label": c["id"],
                         "size_gb": round(size / GiB, 1),
                         "verdict": verdict, "why": why})

    picks = [{"id": r["id"], "include": r["include"], "engine": r["engine"]}
             for r in rows]
    path = _picks_path(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"query": query, "at": time.time(), "picks": picks},
                               indent=1))
    return {"query": query, "budget_label": budget_label, "rows": rows,
            "hidden": hidden}


def find(query: str, root: Path | None = None, limit: int = 8, say=print,
         budget_bytes: int | None = None, budget_label: str = "") -> int:
    """The CLI face of gather(): print the numbered list, return the row
    count.  No filtering here — the terminal reader sees everything, tagged."""
    g = gather(query, root=root, limit=limit,
               budget_bytes=budget_bytes, budget_label=budget_label)
    rows = g["rows"]
    if not rows:
        say(f"find '{query}': the hub returned nothing — try fewer words")
        return 0
    say(f"find '{query}' — judging against {g['budget_label']}:")
    last_repo = None
    for n, r in enumerate(rows, 1):
        pulls = f"{_human(r['downloads'])} pulls"
        eng = "llama.cpp" if r["engine"] == "llama" else "vllm"
        tail = "[" + " · ".join(x for x in (eng, r["format"], pulls) if x) + "]"
        if r["include"]:                              # a GGUF quant row
            if r["id"] != last_repo:
                say(f"     {r['id']} — GGUF repo, pick a file:  {tail}")
            say(f"{n:3d}    {r['label']:<12} {r['size_gb']:6.1f} GB  "
                f"{r['verdict']:<8} {r['why']}")
        elif r["verdict"] == "gated":
            say(f"{n:3d}  {r['id']:<44} gated — {r['why']}  {tail}")
        else:
            say(f"{n:3d}  {r['id']:<44} {r['size_gb']:6.1f} GB  "
                f"{r['verdict']:<8} {r['why']}  {tail}")
        last_repo = r["id"]
    if g["hidden"]["fit"]:
        say(f"       … {g['hidden']['fit']} more quantisation(s) "
            "not shown (won't fit / smaller than needed)")
    say("pull one:  ./vinur.sh pull <row number>     (or ./vinur.sh pull org/Name)")
    say("engine tags: [vllm] = safetensors for the vllm/container engines; "
        "[llama.cpp] = GGUF files for engine = \"llama\" entries")
    return len(rows)
