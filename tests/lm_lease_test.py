"""lm_lease reader battery — the knowledge-host side of Vinkona's busy leases.

The host only READS lease files; Vinkona writes them.  Since the per-holder
scheme (one ``<name>.<holder>.busy`` file per holder, July #5) the reader must
treat a tier as held while ANY of its files — legacy or per-holder — is
unexpired, and keep failing open on garbage.  Pure stdlib + a temp dir.
"""
import os
import sys
import tempfile
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from knowledgehost import lm_lease  # noqa: E402

PASS = FAIL = 0


def check(name, cond):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  ok  {name}")
    else:
        FAIL += 1
        print(f"FAIL  {name}")


def _stamp(d, fname, ttl):
    with open(os.path.join(d, fname), "w") as fh:
        fh.write(repr(time.time() + ttl))


def main():
    with tempfile.TemporaryDirectory() as d:
        os.environ["VINKONA_CONTROL_DIR"] = d
        try:
            check("empty dir: not held", not lm_lease.is_held(lm_lease.BIG))

            _stamp(d, "lm_big.busy", 5)
            check("legacy file holds", lm_lease.is_held(lm_lease.BIG))
            os.unlink(os.path.join(d, "lm_big.busy"))

            _stamp(d, "lm_big.worker-123.busy", 5)
            check("a holder file holds", lm_lease.is_held(lm_lease.BIG))
            check("holder file for big never holds fast", not lm_lease.is_held(lm_lease.FAST))

            _stamp(d, "lm_big.worker-123.busy", -5)
            check("an expired holder file does not hold", not lm_lease.is_held(lm_lease.BIG))

            _stamp(d, "lm_big.bridge-9-1.busy", 5)
            check("one live holder among expired siblings holds",
                  lm_lease.is_held(lm_lease.BIG))

            with open(os.path.join(d, "lm_fast.cascade-1.busy"), "w") as fh:
                fh.write("not-a-number")
            check("a corrupt holder file fails open (not held)",
                  not lm_lease.is_held(lm_lease.FAST))

            _stamp(d, "lm_fast.busy", 5)
            check("legacy + corrupt holder: legacy still holds",
                  lm_lease.is_held(lm_lease.FAST))
        finally:
            os.environ.pop("VINKONA_CONTROL_DIR", None)

    # missing control dir entirely — the no-assistant box: fail open
    os.environ["VINKONA_CONTROL_DIR"] = os.path.join(tempfile.gettempdir(),
                                                     "vinur-no-such-dir-xyz")
    try:
        check("absent control dir: not held", not lm_lease.is_held(lm_lease.BIG))
    finally:
        os.environ.pop("VINKONA_CONTROL_DIR", None)

    print(f"\n{PASS} passed, {FAIL} failed")
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
