#!/bin/bash
# Vinur — run the knowledge host (and, when configured, its own LMs) on THIS
# machine, without Vinkona.  One process supervisor owns everything declared
# in config.toml's [serving] table plus the kb server itself:
#
#   ./vinur.sh start            # start the kb + declared LMs/embed/reranker
#   ./vinur.sh stop             # stop them
#   ./vinur.sh restart [svc]    # restart everything, or one service
#   ./vinur.sh status           # what's up (dead services show their log reason)
#   ./vinur.sh swap <llm>       # exclusive models: load this one in place of the
#                               #   resident one (waits for /health; minutes for big weights)
#   ./vinur.sh logs [svc]       # follow logs (Ctrl-C detaches)
#   ./vinur.sh find <words>     # search the hub for models — each hit sized and
#                               #   judged (fits / tight / too big) for THIS machine
#   ./vinur.sh pull <id | row#> # download one via the egress broker (resumable)
#   ./vinur.sh net              # the broker's window: policy, leases, activity
#
# With no [serving] entries this is simply a supervised ./run.sh.  To serve
# another machine (Vinkona elsewhere), set host = "0.0.0.0" AND auth_token in
# config.toml — the server refuses a LAN bind without a token.
#
# Uses the .venv interpreter built by ./install.sh (config parsing needs
# Python >= 3.11).
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
source ./env.sh

PY="python3"
[ -x "$ROOT/.venv/bin/python3" ] && PY="$ROOT/.venv/bin/python3"
if ! "$PY" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 11) else 1)' 2>/dev/null; then
    echo "error: Python >= 3.11 required (run ./install.sh to build .venv)" >&2
    exit 1
fi

case "${1:-}" in
    -h|--help|help|"") sed -n '2,/^set /p' "$0" | sed -n 's/^#\{1,\} \{0,1\}//p' ;;
    find)   # search the hub, sized + judged against this machine:  ./vinur.sh find qwen3 32b
        shift
        exec "$PY" -m knowledgehost find "$@" ;;
    pull)   # model weights via the egress broker:  ./vinur.sh pull <org/Name | row# from find>
        shift
        M="${1:?usage: ./vinur.sh pull <org/Name | row number from find> [revision] [--include glob]}"
        shift
        REV=""
        if [ $# -ge 1 ] && [ "${1#--}" = "$1" ]; then REV="$1"; shift; fi
        exec "$PY" -m knowledgehost pull --model "$M" ${REV:+--revision "$REV"} "$@" ;;
    adopt)  # legacy HF-cache snapshots -> models/ store (no re-download)
        shift
        exec "$PY" -m knowledgehost adopt ${1:+--model "$1"} ;;
    net)    # the egress broker's window: policy, live leases, recent activity
        exec "$PY" -m knowledgehost.amiga_net.status ;;
    *) exec "$PY" -m knowledgehost.supervisor "$@" ;;
esac
