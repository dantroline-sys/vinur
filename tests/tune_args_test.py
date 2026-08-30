"""The tuned-value-wins contract (the VRAM-share bug): a knob saved through
the Tune editor (tune.toml, or set first-class on the entry) must reach the
engine even when the entry's raw `args` list carries the same flag — args is
appended last and the engines take the last occurrence, so a stale duplicate
(e.g. --gpu-memory-utilization 0.9 from an old fit recipe) used to beat the
tuned value silently: the user sets 0.6, vLLM still claims 0.9.

  * _strip_conflicts: value-pair, --k=v and boolean spellings, the --no-
    variant of tri-state flags; non-conflicting args survive in order.
  * _conflict_flags: vLLM keys (with --no- variants), llama's presence-gated
    ctx_size / n_gpu_layers.
  * llm_argv (container lane): tune.toml's 0.6 is the ONLY
    --gpu-memory-utilization in the argv; unrelated args survive.
  * llm_argv (llama lane): a tuned ctx_size beats a -c duplicate in args; an
    UN-tuned -ngl in args still wins over the built-in default (the
    documented args-override behaviour is only narrowed, not removed).
  * args_superseded names the shadowed flags for the Serving tab's note.

Run:  python tests/tune_args_test.py     (stdlib only)
"""
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from knowledgehost import serving as SV


def check(label, cond):
    if cond:
        print(f"  ok  {label}")
    else:
        print(f"  FAIL  {label}")
        check.failed += 1
check.failed = 0


def main():
    # ── _strip_conflicts spellings ───────────────────────────────────────────
    kept, dropped = SV._strip_conflicts(
        ["--gpu-memory-utilization", "0.9", "--seed", "1"],
        {"--gpu-memory-utilization"})
    check("value-pair form: flag AND its value token dropped, rest kept in order",
          kept == ["--seed", "1"] and dropped == ["--gpu-memory-utilization"])

    kept, dropped = SV._strip_conflicts(
        ["--gpu-memory-utilization=0.9", "--seed", "1"],
        {"--gpu-memory-utilization"})
    check("--k=v form: one token dropped", kept == ["--seed", "1"] and len(dropped) == 1)

    kept, dropped = SV._strip_conflicts(
        ["--enforce-eager", "--seed", "1"], {"--enforce-eager"})
    check("boolean flag followed by another flag: only itself dropped",
          kept == ["--seed", "1"])

    kept, dropped = SV._strip_conflicts(
        ["--no-enable-prefix-caching", "-x", "y"],
        {"--enable-prefix-caching", "--no-enable-prefix-caching"})
    check("the --no- spelling of a tri-state flag is caught too",
          kept == ["-x", "y"] and dropped == ["--no-enable-prefix-caching"])

    kept, dropped = SV._strip_conflicts(["--seed", "1"], {"--gpu-memory-utilization"})
    check("nothing to strip: args untouched", kept == ["--seed", "1"] and dropped == [])

    # ── _conflict_flags ──────────────────────────────────────────────────────
    fl = SV._conflict_flags({"gpu_memory_utilization": 0.6,
                             "enable_prefix_caching": True}, "container")
    check("vllm keys map to their flags, tri-state adds the --no- variant",
          fl == {"--gpu-memory-utilization", "--enable-prefix-caching",
                 "--no-enable-prefix-caching"})
    check("llama: only knobs the entry actually SETS conflict",
          SV._conflict_flags({}, "llama") == set()
          and "-c" in SV._conflict_flags({"ctx_size": 4096}, "llama")
          and "-ngl" in SV._conflict_flags({"n_gpu_layers": 20}, "llama"))

    # ── llm_argv, container lane: tune.toml wins over an args duplicate ──────
    with tempfile.TemporaryDirectory() as td:
        model_dir = Path(td) / "weights"
        model_dir.mkdir()
        (model_dir / "tune.toml").write_text("gpu_memory_utilization = 0.6\n")
        rt0 = SV._container_runtime
        SV._container_runtime = lambda e: "podman"
        try:
            entry = {"name": "big", "engine": "container", "model": str(model_dir),
                     "port": 8010,
                     "args": ["--gpu-memory-utilization", "0.9", "--seed", "7"]}
            argv = SV.llm_argv(entry, root=Path(td))
            occurrences = [i for i, a in enumerate(argv)
                           if a == "--gpu-memory-utilization"]
            check("exactly ONE --gpu-memory-utilization in the final command",
                  len(occurrences) == 1)
            check("…and it carries the TUNED value (0.6), not the stale args 0.9",
                  argv[occurrences[0] + 1] == "0.6" and "0.9" not in argv)
            check("unrelated args survive", "--seed" in argv and "7" in argv)
            check("args_superseded names the shadowed flag for the note column",
                  SV.args_superseded(entry, root=Path(td))
                  == ["--gpu-memory-utilization"])
        finally:
            SV._container_runtime = rt0

    # ── llm_argv, llama lane ─────────────────────────────────────────────────
    with tempfile.TemporaryDirectory() as td:
        gguf = Path(td) / "m.gguf"
        gguf.write_bytes(b"GGUF")
        ls0 = SV._llama_server
        SV._llama_server = lambda: "llama-server"
        try:
            e1 = {"name": "l", "engine": "llama", "model": str(gguf), "port": 8011,
                  "ctx_size": 4096, "args": ["-c", "999"]}
            argv = SV.llm_argv(e1, root=Path(td))
            check("llama: a tuned ctx_size beats the -c duplicate in args",
                  argv.count("-c") == 1
                  and argv[argv.index("-c") + 1] == "4096" and "999" not in argv)
            e2 = {"name": "l", "engine": "llama", "model": str(gguf), "port": 8011,
                  "args": ["-ngl", "40"]}
            argv2 = SV.llm_argv(e2, root=Path(td))
            check("llama: an UN-tuned -ngl in args still overrides the built-in "
                  "default (last occurrence wins)",
                  argv2.count("-ngl") == 2
                  and argv2[len(argv2) - 1 - argv2[::-1].index("-ngl") + 1] == "40")
        finally:
            SV._llama_server = ls0

    print()
    if check.failed:
        print(f"{check.failed} FAILED")
        return 1
    print("ALL PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
