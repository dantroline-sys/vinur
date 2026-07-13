# Knowledge-host filesystem confinement — source this from every script here.
#
# THE GUARANTEE: everything the knowledge host writes stays inside this folder:
#   var/         indexes + databases (index.db, kb.db, library.db, lance/) — the
#                config defaults point here; absolute paths in config.toml are
#                honoured if you deliberately choose somewhere else
#   var/cache/   third-party caches (XDG, HF)
#   var/tmp/     temp files (TMPDIR — OCR scratch, mktemp, pip build isolation)
#   models/      the reranker GGUF fetched by run-reranker.sh
#   .venv/       the virtualenv built by install.sh
#   config.toml  seeded once by install.sh, then yours
#
# Reads are unrestricted — `sources`, `zim_path`, `library_sources` can point
# anywhere. Process-scoped: affects these scripts only, not your shell.

KH_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export XDG_CACHE_HOME="$KH_ROOT/var/cache"
export HF_HOME="$KH_ROOT/var/cache/huggingface"
export TMPDIR="$KH_ROOT/var/tmp"
# uv (the python env manager — see vk_uv below): wheel cache + any CPython
# interpreters it downloads both stay in-tree (the second line matters — the
# default would be ~/.local/share/uv).
export UV_CACHE_DIR="$KH_ROOT/var/cache/uv"
export UV_PYTHON_INSTALL_DIR="$KH_ROOT/var/uv/python"
mkdir -p "$KH_ROOT/var/cache" "$KH_ROOT/var/tmp"

# ── vk_require_tools: check for system tools, offer to install the missing ──
# Same helper as assistant/env.sh — kept in both so each component stands alone.
# Usage:   vk_require_tools "tesseract:tesseract|tesseract-ocr" ocrmypdf || ...
# Spec:    tool[:package] — package may be "dnfname|aptname" where they differ.
vk_require_tools() {
    local mgr="" pick=1 spec tool pkg missing=() pkgs=()
    if command -v dnf >/dev/null 2>&1;      then mgr="dnf install -y"
    elif command -v apt-get >/dev/null 2>&1; then mgr="apt-get install -y"; pick=2
    elif command -v pacman >/dev/null 2>&1;  then mgr="pacman -S --needed --noconfirm"
    elif command -v zypper >/dev/null 2>&1;  then mgr="zypper install -y"
    fi
    for spec in "$@"; do
        tool="${spec%%:*}"
        command -v "$tool" >/dev/null 2>&1 && continue
        pkg="${spec#*:}"; [ "$pkg" = "$spec" ] && pkg="$tool"
        if [ "$pick" -eq 2 ]; then pkg="${pkg##*|}"; else pkg="${pkg%%|*}"; fi
        missing+=("$tool"); pkgs+=("$pkg")
    done
    [ "${#missing[@]}" -eq 0 ] && return 0
    echo "Missing system tools: ${missing[*]}"
    if [ -z "$mgr" ]; then
        echo "No known package manager found — install them yourself, then re-run."
        return 1
    fi
    if [ -t 0 ] || [ "${VINKONA_ASSUME_TTY:-}" = 1 ]; then
        printf "Install now with 'sudo %s %s'? [Y/n]: " "$mgr" "${pkgs[*]}"
        local answer; read -r answer
        case "$answer" in
            n*|N*) echo "Skipped — install them and re-run."; return 1 ;;
        esac
        # shellcheck disable=SC2086
        sudo $mgr "${pkgs[@]}" || { echo "Package install failed — install manually, then re-run."; return 1; }
        hash -r
        local t
        for t in "${missing[@]}"; do
            command -v "$t" >/dev/null 2>&1 || { echo "Still missing after install: $t"; return 1; }
        done
        echo "System tools installed."
        return 0
    fi
    echo "Non-interactive shell — run:  sudo $mgr ${pkgs[*]}   then re-run this script."
    return 1
}

# ── vk_uv: run uv, bootstrapping it in-tree on first use ────────────────────
# uv (https://docs.astral.sh/uv/) builds .venv from pyproject.toml + uv.lock —
# same pinned set on every platform, and it downloads a matching CPython itself
# (into var/uv/python) if the system one doesn't satisfy requires-python.
# A system-wide uv is used when present; otherwise one standalone binary is
# fetched into ./bin (UV_UNMANAGED_INSTALL = no PATH/rc edits, no self-update
# state — in-tree like everything else). The venv it makes is a plain venv.
vk_uv() {
    local uv
    uv="$(command -v uv 2>/dev/null || true)"
    [ -n "$uv" ] || uv="$KH_ROOT/bin/uv"
    if [ ! -x "$uv" ]; then
        vk_require_tools curl || return 1
        echo "==> fetching uv (one-time, into bin/)" >&2
        mkdir -p "$KH_ROOT/bin"
        curl -LsSf https://astral.sh/uv/install.sh \
                | env UV_UNMANAGED_INSTALL="$KH_ROOT/bin" sh >&2 \
            || { echo "could not bootstrap uv — install it yourself (https://docs.astral.sh/uv/) and re-run" >&2; return 1; }
        uv="$KH_ROOT/bin/uv"
    fi
    "$uv" "$@"
}
