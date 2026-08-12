#!/bin/bash
# AMIGA-OPS-01 §4 — the gates, one entry point.  Every check prints PASS,
# FAIL, or SKIPPED(tool absent); a SKIP is loud and honest, never a silent
# pass (B-21).  Exit 0 only when nothing FAILED and nothing REQUIRED was
# skipped.  Run it before pushing; CI (or the local equivalent) runs it on
# every commit to main.
#
#   scripts/gates.sh              # everything available
#   GATES_ALLOW_SKIP=1 …          # tolerate missing G-1..G-4 tools (sandboxes)
#
# G-1 ruff format --check   G-2 ruff check     G-3 deptry     G-4 uv lock --check
# (install the tools:  uv sync --group dev)
# Always-on stdlib gates: compile sweep, dependency ratchet (deps_test), the
# broker's own test battery, and the G-8 broker size cap.
set -u
cd "$(dirname "$0")/.."
FAIL=0; SKIP=0

note() { printf '  %-28s %s\n' "$1" "$2"; }
run()  {  # name, required-tool, cmd...
    local name="$1" tool="$2"; shift 2
    if [ -n "$tool" ] && ! command -v "$tool" >/dev/null 2>&1 \
        && ! [ -x ".venv/bin/$tool" ]; then
        note "$name" "SKIPPED ($tool not installed — uv sync --group dev)"
        SKIP=$((SKIP+1)); return
    fi
    if "$@" >/tmp/gate.$$ 2>&1; then note "$name" "PASS"
    else note "$name" "FAIL"; sed 's/^/      /' /tmp/gate.$$ | head -30; FAIL=$((FAIL+1)); fi
    rm -f /tmp/gate.$$
}
tool() {  # prefer the venv's copy
    if [ -x ".venv/bin/$1" ]; then echo ".venv/bin/$1"; else echo "$1"; fi
}

PY="python3"; [ -x .venv/bin/python3 ] && PY=".venv/bin/python3"

echo "gates ($(git rev-parse --short HEAD 2>/dev/null || echo '?')):"
run "G-1 format"        ruff   "$(tool ruff)" format --check knowledgehost tests scripts
run "G-2 lint"          ruff   "$(tool ruff)" check knowledgehost tests scripts
run "G-3 deps declared" deptry "$(tool deptry)" knowledgehost
run "G-4 lockfile"      uv     uv lock --check

# ── always-on, stdlib, no excuses ────────────────────────────────────────────
run "compile sweep"     ""     "$PY" -W error::SyntaxWarning -m py_compile \
                                   knowledgehost/*.py knowledgehost/amiga_net/*.py \
                                   tests/*.py scripts/*.py
# --help builds the WHOLE argparse parser (catches duplicate flags etc. that the
# compile sweep can't — a runtime error at startup, not a syntax error).
run "cli parser builds" ""     "$PY" -m knowledgehost --help
# node-checks the RUNTIME viewer script (the inline JS after Python renders the
# triple-quoted page) — catches a \n eaten into a real newline inside a quoted JS
# string, which serves 200 but runs no script.  Skips when node is absent.
run "viewer JS parses"  node   "$PY" tests/viewer_js_test.py
# every `path == "..."` handler in _do_POST must be in its allow-list, or the route
# 404s "not found" before the handler runs.
run "post routes wired" ""     "$PY" tests/post_routes_test.py
run "dependency ratchet" ""    "$PY" tests/deps_test.py
run "broker battery"    ""     "$PY" tests/amiga_net_test.py
run "model finder"      ""     "$PY" tests/modelfind_test.py
run "posture scan"      ""     "$PY" tests/posture_test.py
run "pack battery"      ""     "$PY" tests/pack_test.py
run "collect battery"   ""     "$PY" tests/collect_test.py
run "fs-browse battery" ""     "$PY" tests/fs_browse_test.py
run "structure battery" ""     "$PY" tests/structure_test.py
run "structured-ingest"  ""     "$PY" tests/structured_ingest_test.py
run "pending inbox"      ""     "$PY" tests/pending_test.py
run "citation graph"     ""     "$PY" tests/citations_test.py
run "ship battery"      ""     "$PY" tests/ship_test.py
run "minimal battery"   ""     "$PY" tests/minimal_test.py
run "G-8 broker size"   ""     "$PY" - <<'EOF'
import pathlib, sys
n = sum(len(p.read_text().splitlines())
        for p in pathlib.Path("knowledgehost/amiga_net").glob("*.py"))
print(f"amiga_net: {n} lines (cap 1000)")
sys.exit(0 if n < 1000 else 1)
EOF

echo
if [ "$FAIL" -gt 0 ]; then echo "gates: $FAIL FAILED"; exit 1; fi
if [ "$SKIP" -gt 0 ] && [ "${GATES_ALLOW_SKIP:-0}" != 1 ]; then
    echo "gates: $SKIP skipped and GATES_ALLOW_SKIP is not set — a skipped gate"
    echo "is not a passed gate.  Install the tools: uv sync --group dev"
    exit 1
fi
echo "gates: all green${SKIP:+ ($SKIP skipped, allowed)}"
