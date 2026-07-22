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
run "dependency ratchet" ""    "$PY" tests/deps_test.py
run "broker battery"    ""     "$PY" tests/amiga_net_test.py
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
