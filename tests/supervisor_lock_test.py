"""Supervisor single-instance claim — the July #6 start-race fix, vinur side.

Two racing `start`s both passed the advisory read_state()+alive() check and
booted two whole stacks onto the same GPUs; _claim_lock's kernel flock admits
exactly one.  Pure stdlib + a temp dir; nothing is spawned.
"""
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from knowledgehost import supervisor as sup  # noqa: E402

PASS = FAIL = 0


def check(name, cond):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  ok  {name}")
    else:
        FAIL += 1
        print(f"FAIL  {name}")


def main():
    with tempfile.TemporaryDirectory() as d:
        old = sup.LOCK
        sup.LOCK = Path(d) / "run" / "supervisor.lock"   # parent must be created too
        try:
            f1 = sup._claim_lock()
            check("first claim wins", f1 is not None)
            check("lock file created (parents included)", sup.LOCK.exists())
            f2 = sup._claim_lock()
            check("second claim is refused while the first lives", f2 is None)
            f1.close()                                   # holder dies → kernel drops the lock
            f3 = sup._claim_lock()
            check("claim succeeds again after the holder is gone", f3 is not None)
            if f3:
                f3.close()
        finally:
            sup.LOCK = old
    print(f"\n{PASS} passed, {FAIL} failed")
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
