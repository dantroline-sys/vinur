#!/usr/bin/env bash
# Vinur — production installer.
#
# Builds a self-contained Python virtualenv (.venv) and installs exactly the
# dependencies your deployment needs.  The query service and the default
# `sqlite` backend require NOTHING beyond the Python stdlib (FTS5 ships with
# sqlite3); numpy is the recommended dense fast-path; every document parser and
# the Wikipedia-scale `lance` backend are optional and lazy-imported, so a host
# only installs what its corpus actually contains.
#
# Usage:
#   ./install.sh                 # venv + numpy  (sqlite backend, query service)
#   ./install.sh --all           # + every parser (pdf/epub/html/zim) + lance
#   ./install.sh --pdf --epub    # pick formats your collection has
#   ./install.sh --wikipedia     # libzim, for a Kiwix Wikipedia ZIM
#   ./install.sh --lance         # LanceDB IVF-PQ backend (Wikipedia scale)
#   ./install.sh --serving       # + vLLM in its own serving/.venv (a standalone
#                                #   GPU box serving its own LMs — see vinur.sh)
#   ./install.sh --llama         # build llama-server into ./bin (the embed
#                                #   endpoint + reranker run on it; CUDA when
#                                #   nvcc is present, else CPU)
#   ./install.sh --minimal       # stdlib only, not even numpy
#   ./install.sh --no-venv       # install into the active/system interpreter (pip)
#   ./install.sh --python 3.12 --venv /opt/kb/venv   # version, name, or path
#   ./install.sh --no-test       # skip the post-install smoke verification
#
# Dependencies are managed by uv (bootstrapped in-tree on first run — see
# env.sh): pyproject.toml declares them, uv.lock pins the exact working set on
# every platform. Re-running is safe and incremental: an existing venv is
# reused and new groups just add to it. config.toml is seeded once and never
# overwritten.
#
# Maintenance commands:
#   ./install.sh status          # what's installed and how big the data is
#   ./install.sh uninstall       # remove the venv, caches and reranker model
#                 --purge        #   ALSO delete the knowledge base (var/) and
#                                #   config.toml — asks for confirmation first
#
# Everything this script and the host write lives INSIDE this folder (see
# env.sh): .venv, var/ (indexes, kb.db, caches, tmp), models/, config.toml.
set -euo pipefail
cd "$(dirname "$0")"
source ./env.sh          # in-tree var/ caches + tmp — see env.sh

# ── maintenance commands (before flag parsing) ──────────────────────────────
if [ "${1:-}" = "status" ]; then
  echo "Knowledge host @ $(pwd)"
  [ -d .venv ]  && echo "  venv     .venv ($(du -sh .venv 2>/dev/null | cut -f1))" || echo "  venv     not installed"
  [ -d serving/.venv ] && echo "  serving  serving/.venv ($(du -sh serving/.venv 2>/dev/null | cut -f1)) — vLLM"
  [ -x bin/llama-server ] && echo "  binary   bin/llama-server (embed + reranker)"
  [ -d var ]    && echo "  data     var/ ($(du -sh var 2>/dev/null | cut -f1)) — indexes, kb.db, caches" || echo "  data     none yet"
  [ -d models ] && echo "  models   models/ ($(du -sh models 2>/dev/null | cut -f1))"
  [ -f config.toml ] && echo "  config   config.toml (yours, kept on uninstall)" || echo "  config   not seeded yet"
  echo "  All of the above lives inside this folder — nothing is written elsewhere."
  exit 0
fi
if [ "${1:-}" = "uninstall" ]; then
  purge=0; [ "${2:-}" = "--purge" ] && purge=1
  rm -rf .venv serving/.venv models var/cache var/tmp var/uv var/build bin
  find . -name __pycache__ -type d -exec rm -rf {} + 2>/dev/null || true
  echo "removed: .venv, serving/.venv, models/, bin/, var/cache, var/tmp, var/uv, var/build (software + caches)"
  if [ "$purge" -eq 1 ]; then
    echo ""
    echo "WARNING: --purge deletes the KNOWLEDGE BASE itself (var/: index.db, kb.db,"
    echo "the distilled cards — potentially days of ingestion) and config.toml."
    printf "Type 'purge' to confirm: "
    read -r answer
    if [ "$answer" = "purge" ]; then
      rm -rf var config.toml
      echo "knowledge base and config purged"
    else
      echo "skipped purge — var/ and config.toml kept"
    fi
  else
    echo "kept: var/ (your knowledge base) and config.toml — './install.sh uninstall --purge' removes them too"
  fi
  exit 0
fi

# ── defaults ────────────────────────────────────────────────────────────────
PYTHON="${PYTHON:-python3}"
VENV_DIR=".venv"
USE_VENV=1
WANT_NUMPY=1
RUN_TEST=1
WITH_PDF=0; WITH_EPUB=0; WITH_HTML=0; WITH_WIKI=0; WITH_LANCE=0; WITH_SERVING=0; WITH_LLAMA=0

# Print the leading comment block (everything after the shebang up to the first
# non-comment line), with the leading "# " stripped.
usage() { sed -n '2,/^[^#]/p' "$0" | sed -n 's/^#\{1,\} \{0,1\}//p'; exit "${1:-0}"; }

while [ $# -gt 0 ]; do
  case "$1" in
    --all)       WITH_PDF=1; WITH_EPUB=1; WITH_HTML=1; WITH_WIKI=1; WITH_LANCE=1 ;;
    --pdf)       WITH_PDF=1 ;;
    --epub)      WITH_EPUB=1 ;;
    --html)      WITH_HTML=1 ;;
    --wikipedia|--zim) WITH_WIKI=1 ;;
    --lance)     WITH_LANCE=1 ;;
    --serving)   WITH_SERVING=1 ;;   # vLLM venv (serving/.venv) — GPU box, big download
    --llama)     WITH_LLAMA=1 ;;     # build llama-server into ./bin (embed + reranker)

    --minimal|--no-numpy) WANT_NUMPY=0 ;;
    --no-venv)   USE_VENV=0 ;;
    --venv)      VENV_DIR="${2:?--venv needs a path}"; shift ;;
    --python)    PYTHON="${2:?--python needs an interpreter}"; shift ;;
    --no-test)   RUN_TEST=0 ;;
    -h|--help)   usage 0 ;;
    *) echo "unknown option: $1" >&2; usage 1 ;;
  esac
  shift
done

say()  { printf '\033[1;36m==>\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33mwarning:\033[0m %s\n' "$*" >&2; }
die()  { printf '\033[1;31merror:\033[0m %s\n' "$*" >&2; exit 1; }

# ── 1./2./3. environment + dependency set (uv sync, per-flag groups) ────────
if [ "$USE_VENV" -eq 1 ]; then
  # uv builds the venv from pyproject.toml + uv.lock — the pinned set, same on
  # every platform — and fetches a matching CPython itself if the system one
  # doesn't satisfy requires-python (>= 3.11; config.py uses stdlib tomllib).
  # --inexact keeps re-runs incremental: an existing venv is reused and new
  # groups just add to it, exactly like the old pip behaviour.
  UV_ARGS=(sync --inexact)
  [ "$WANT_NUMPY" -eq 1 ] && UV_ARGS+=(--group recommended)
  [ "$WITH_PDF"   -eq 1 ] && UV_ARGS+=(--group pdf)
  [ "$WITH_EPUB"  -eq 1 ] && UV_ARGS+=(--group epub)
  [ "$WITH_HTML"  -eq 1 ] && UV_ARGS+=(--group html)
  [ "$WITH_WIKI"  -eq 1 ] && UV_ARGS+=(--group wikipedia)
  [ "$WITH_LANCE" -eq 1 ] && UV_ARGS+=(--group lance)
  [ "$PYTHON" != "python3" ] && UV_ARGS+=(--python "$PYTHON")
  case "$VENV_DIR" in /*) VENV_ABS="$VENV_DIR" ;; *) VENV_ABS="$PWD/$VENV_DIR" ;; esac
  say "syncing $VENV_DIR from pyproject.toml + uv.lock"
  UV_PROJECT_ENVIRONMENT="$VENV_ABS" vk_uv "${UV_ARGS[@]}" || die "uv sync failed — see above"
  PY="$VENV_DIR/bin/python"
else
  # --no-venv escape hatch: install into the active/system interpreter with
  # plain pip (uv sync only manages venvs it owns). Loose ranges, not uv.lock.
  command -v "$PYTHON" >/dev/null 2>&1 || die "interpreter not found: $PYTHON (try --python python3.11)"
  "$PYTHON" - <<'PY' || die "Python >= 3.11 is required (config.py uses the stdlib tomllib)."
import sys
sys.exit(0 if sys.version_info[:2] >= (3, 11) else 1)
PY
  warn "installing into the active interpreter (no venv): $(command -v "$PYTHON")"
  PY="$PYTHON"
  PKGS=()
  [ "$WANT_NUMPY" -eq 1 ] && PKGS+=("numpy" "usearch")   # usearch: optional ANN (build-ann)
  [ "$WITH_PDF"   -eq 1 ] && PKGS+=("pymupdf")
  [ "$WITH_EPUB"  -eq 1 ] && PKGS+=("ebooklib")
  [ "$WITH_HTML"  -eq 1 ] && PKGS+=("trafilatura")
  [ "$WITH_WIKI"  -eq 1 ] && PKGS+=("libzim")
  [ "$WITH_LANCE" -eq 1 ] && PKGS+=("lancedb" "pyarrow" "pylance")
  if [ "${#PKGS[@]}" -gt 0 ]; then
    say "installing: ${PKGS[*]}"
    # via uv (A-7: no bare pip) — same resolver and cache as the lockfile sync
    vk_uv pip install --python "$PY" --upgrade "${PKGS[@]}"
  else
    say "stdlib-only install (no pip packages requested)"
  fi
fi

# ── 3a. llama-server (embed + reranker + engine="llama" entries) ────────────
# A standalone Vinur box needs its own llama.cpp: the embed endpoint and the
# CPU reranker run on llama-server (see serving/README.md — vLLM is only for
# the big chat models).  Same recipe Vinkona's installer proved out:
# LLAMA_CURL=OFF (the URL-download path is never used; it drags curl/OpenSSL
# probing in), Darwin adds GGML_OPENMP=OFF (Apple clang has no libomp; Metal +
# Accelerate make it pointless), CUDA auto-enables only when nvcc is present.
# VINUR_LLAMA_CMAKE_EXTRA appends ad-hoc cmake flags without edits.
if [ "$WITH_LLAMA" -eq 1 ]; then
  if [ -x bin/llama-server ]; then
    say "bin/llama-server already built — delete it to force a rebuild"
  else
    platform_flags=""
    if [ "$(uname -s)" = Darwin ]; then
      platform_flags="-DGGML_OPENMP=OFF"
      vk_require_tools git cmake || die "building llama.cpp needs git + cmake (see above)"
    else
      vk_require_tools git cmake gcc "g++:gcc-c++|g++" make \
        || die "building llama.cpp needs the C++ toolchain + cmake (see above)"
    fi
    cuda_flag="-DGGML_CUDA=OFF"
    if command -v nvcc >/dev/null 2>&1; then
      cuda_flag="-DGGML_CUDA=ON"
      say "llama: nvcc found — building with CUDA (GPU embeddings)"
    elif [ "$(uname -s)" != Darwin ]; then
      warn "no nvcc — building CPU-only (fine for the reranker; embeddings run"
      warn "slower). Install the CUDA toolkit and re-run for a GPU build, or"
      warn "append flags via VINUR_LLAMA_CMAKE_EXTRA."
    fi
    src="var/build/llama.cpp"
    say "llama: cloning/updating llama.cpp into $src"
    if [ ! -d "$src/.git" ]; then
      mkdir -p var/build
      git clone --depth 1 https://github.com/ggml-org/llama.cpp "$src"
    fi
    # Supply-chain pin: LLAMA_CPP_REF=<tag|commit> builds exactly that ref;
    # unset keeps rolling master (new model archs land there weekly), but the
    # commit actually built is recorded in bin/llama-server.commit either way,
    # so a known-good build can be pinned after the fact.
    if [ -n "${LLAMA_CPP_REF:-}" ]; then
      say "llama: pinning to $LLAMA_CPP_REF"
      git -C "$src" fetch --depth 1 origin "$LLAMA_CPP_REF" \
        && git -C "$src" checkout -q FETCH_HEAD \
        || die "could not fetch LLAMA_CPP_REF=$LLAMA_CPP_REF"
    else
      git -C "$src" checkout -q master 2>/dev/null || true
      git -C "$src" pull --ff-only || true
    fi
    say "llama: building llama-server ($cuda_flag) — this takes a while"
    # shellcheck disable=SC2086
    cmake -S "$src" -B "$src/build" $cuda_flag $platform_flags \
          -DLLAMA_CURL=OFF \
          -DBUILD_SHARED_LIBS=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF \
          -DLLAMA_BUILD_SERVER=ON ${VINUR_LLAMA_CMAKE_EXTRA:-} >/dev/null \
      || die "cmake configure failed — see above"
    cmake --build "$src/build" --target llama-server -j"$(vk_ncpu)" \
      || die "llama-server build failed — see above"
    mkdir -p bin
    cp "$src/build/bin/llama-server" bin/
    git -C "$src" rev-parse HEAD > bin/llama-server.commit 2>/dev/null || true
    say "installed bin/llama-server @ $(cut -c1-12 bin/llama-server.commit 2>/dev/null || echo '?') (found automatically by serving.py + run-reranker.sh)"
  fi
fi

# ── 3b. vLLM serving venv (its own uv project — see serving/pyproject.toml) ─
if [ "$WITH_SERVING" -eq 1 ]; then
  say "syncing serving/.venv (vLLM — GPU box only; multi-GB torch/CUDA download)"
  (cd serving && UV_PROJECT_ENVIRONMENT="$PWD/.venv" vk_uv sync --inexact) \
    || die "serving sync failed — see above"
  # Heads-up that has bitten in the wild: vLLM's JIT kernel paths (FlashInfer —
  # the ONLY NVFP4/FP8-MoE implementation on consumer Blackwell) need the CUDA
  # TOOLKIT (nvcc), which the driver alone doesn't provide.
  if command -v nvidia-smi >/dev/null 2>&1 && ! command -v nvcc >/dev/null 2>&1 \
     && [ -z "${CUDA_HOME:-}" ] && [ ! -e /usr/local/cuda ]; then
    warn "NVIDIA GPU present but no CUDA toolkit (nvcc) found.  NVFP4/FP8-MoE"
    warn "models (and other JIT kernel paths) will fail to start with"
    warn "'Could not find nvcc'.  See serving/README.md -> Troubleshooting for"
    warn "the toolkit-only install (your driver is left untouched)."
  fi
  say "declare the models in config.toml's [[serving.llms]] and start with ./vinur.sh"
fi

# ── 4. system binaries we cannot pip-install (offer, never block) ───────────
if [ "$WITH_PDF" -eq 1 ]; then
  vk_require_tools "tesseract:tesseract|tesseract-ocr" ocrmypdf || {
    warn "OCR fallback unavailable — scanned PDFs with no text layer will be skipped."
    warn "(PDFs with a real text layer ingest fine without it; re-run once installed.)"
  }
fi

# ── 5. seed config.toml once (never clobber an existing one) ────────────────
if [ ! -f config.toml ] && [ -f config.example.toml ]; then
  cp config.example.toml config.toml
  say "seeded config.toml from config.example.toml — edit 'sources', 'embed_url', etc."
else
  say "config.toml left as-is"
fi

# ── 6. post-install verification (offline; no embed endpoint needed) ────────
if [ "$RUN_TEST" -eq 1 ]; then
  say "verifying the install (offline smoke test)"
  FIX="$(mktemp -d)"
  trap 'rm -rf "$FIX"' EXIT
  bash tests/make_fixtures.sh "$FIX" >/dev/null
  if "$PY" tests/smoke.py "$FIX"; then
    say "smoke test passed"
  else
    die "smoke test failed — see output above"
  fi
fi

# ── done ────────────────────────────────────────────────────────────────────
cat <<EOF

$(say "install complete")
  Next:
    1. edit ./config.toml      (sources, embed_url, zim_path, port, auth_token)
    2. ./ingest.sh             (crawl documents; add --wikipedia for a ZIM)
    3. ./run.sh                (start the query service)

  run.sh / ingest.sh auto-use $VENV_DIR when present.
EOF
