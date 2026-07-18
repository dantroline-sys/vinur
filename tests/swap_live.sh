#!/usr/bin/env bash
# Live exclusive-swap smoke: two stub llama-servers under the REAL supervisor —
# start boots only the default model, `swap` stops it, spawns the other, waits
# for /health, and status/state files track the whole dance.
#
# Run from the repo root:  bash tests/swap_live.sh
set -euo pipefail
cd "$(dirname "$0")/.."
TD="$(mktemp -d)"
cleanup() {
    KNOWLEDGEHOST_CONFIG="$TD/c.toml" ./vinur.sh stop >/dev/null 2>&1 || true
    rm -rf "$TD"
}
trap cleanup EXIT

touch "$TD/a.gguf" "$TD/b.gguf"
cat > "$TD/c.toml" <<EOF
[[serving.llms]]
name = "primary"
engine = "llama"
model = "$TD/a.gguf"
port = 21438
exclusive = true
default = true

[[serving.llms]]
name = "secondary"
engine = "llama"
model = "$TD/b.gguf"
port = 21435
exclusive = true
EOF

export KNOWLEDGEHOST_CONFIG="$TD/c.toml"
export LLAMA_SERVER="$PWD/tests/stub_llama.py"
export STUB_DELAY="${STUB_DELAY:-2}"

echo "== start (primary is default)"
./vinur.sh start
sleep 4
./vinur.sh status | grep -Eq "llm-primary +up"       || { echo "FAIL: primary not up"; exit 1; }
./vinur.sh status | grep -Eq "llm-secondary +standby"  || { echo "FAIL: secondary not standby"; exit 1; }
curl -sf -m 2 localhost:21438/health >/dev/null      || { echo "FAIL: primary /health"; exit 1; }

echo "== swap secondary"
./vinur.sh swap secondary
./vinur.sh status | grep -Eq "llm-secondary +up"       || { echo "FAIL: secondary not up"; exit 1; }
./vinur.sh status | grep -Eq "llm-primary +standby"  || { echo "FAIL: primary not standby"; exit 1; }
curl -sf -m 2 localhost:21435/health >/dev/null      || { echo "FAIL: secondary /health"; exit 1; }
curl -sf -m 1 localhost:21438/health >/dev/null      && { echo "FAIL: primary still answering"; exit 1; }

echo "== swap secondary again (no-op)"
./vinur.sh swap secondary | grep -q "secondary ready" || { echo "FAIL: no-op swap"; exit 1; }

echo "== swap back"
./vinur.sh swap primary
curl -sf -m 2 localhost:21438/health >/dev/null      || { echo "FAIL: primary /health after swap-back"; exit 1; }

./vinur.sh stop
echo "swap_live: ALL OK"
