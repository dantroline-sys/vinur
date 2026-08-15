"""Cooperative GPU leases — Vinkona yields the GPUs to the live assistant.

Vinkona publishes two lease files in her repo's ``logs/control/``:

  * ``lm_fast.busy`` — a live chat is using the fast LM (4090).  Pause the distil
    (first-pass extraction) stage; keep verify running.
  * ``lm_big.busy``  — Vinkona is doing big-LM work (research/briefing/deliberation,
    3090).  Pause the verify/reconcile stage; keep distil running.

A file's contents are a unix expiry timestamp (float).  ``held`` ⇔ a file exists
AND ``float(contents) > time.time()``.  A missing, unparseable, or expired file is
NOT held — we never block on it, and a crashed Vinkona auto-releases within ~15s as the
freshness stamp lapses.  We only READ these files; they are Vinkona's to write/delete.

A tier may have SEVERAL holders on Vinkona's side (the bridge's per-stream hold, the
research worker's phase hold), each with its own ``<name>.<holder>.busy`` file so one
release can never drop another's live lease.  The tier is held while ANY of its files
(the legacy ``<name>.busy`` included) is unexpired.
"""
from __future__ import annotations

import os
import time

FAST = "lm_fast"     # the 4090 (fast extractor)
BIG = "lm_big"       # the 3090 (big verifier / reconciler)

# Default: the paired Vinkona checkout's control dir, with the two repos cloned
# side by side (<parent>/vinur + <parent>/vinkona).  A machine with no assistant
# simply never sees a lease file — fail open, never held.
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DEFAULT = os.path.join(os.path.dirname(_REPO), "vinkona", "assistant", "logs", "control")


def control_dir(cfg=None) -> str:
    """Resolve the lease directory: $VINKONA_CONTROL_DIR > cfg['control_dir'] > default."""
    d = os.environ.get("VINKONA_CONTROL_DIR") or (cfg or {}).get("control_dir") or _DEFAULT
    return os.path.expanduser(d)


def _live(path: str, now: float) -> bool:
    try:
        with open(path) as fh:
            return float(fh.read().strip()) > now
    except (OSError, ValueError):
        return False


def is_held(name: str, cfg=None) -> bool:
    """True iff the named lease is currently held by ANY holder.  Any error => not
    held (fail open)."""
    d = control_dir(cfg)
    now = time.time()
    if _live(os.path.join(d, name + ".busy"), now):
        return True
    try:
        files = os.listdir(d)
    except OSError:
        return False
    return any(f.startswith(name + ".") and f.endswith(".busy") and f != name + ".busy"
               and _live(os.path.join(d, f), now) for f in files)
