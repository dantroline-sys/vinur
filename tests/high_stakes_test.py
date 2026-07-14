"""high_stakes_extra: the config extension point that lets a domain overlay extend
the rigor heuristic without forking grounding.py.  Pure python, no services."""
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from knowledgehost import grounding, config  # noqa: E402

FAILED = []


def check(label, cond):
    print(("  ok  " if cond else "FAIL  ") + label)
    if not cond:
        FAILED.append(label)


def main():
    # baseline: built-in pattern fires, neutral question doesn't
    check("built-in consequential wording is high rigor",
          grounding.default_rigor("what is a safe dose here?") == "high")
    check("neutral question is low rigor",
          grounding.default_rigor("how do I proof sourdough?") == "low")

    # extension: overlay vocabulary registered at runtime
    n = grounding.extend_high_stakes([r"\bload-?bearing\b", r"\bgas (line|leak)\b"])
    check("patterns register", n >= 2)
    check("extended vocabulary now trips high rigor",
          grounding.default_rigor("can I remove a load-bearing wall myself?") == "high")
    check("other queries stay low", grounding.default_rigor("what colour is teal?") == "low")

    # idempotent + fail-soft
    n2 = grounding.extend_high_stakes([r"\bload-?bearing\b", "(broken[regex", "", None])
    check("re-registering and junk patterns are harmless", n2 == n)

    # the config route: high_stakes_extra in config.toml lands in the heuristic
    with tempfile.NamedTemporaryFile("w", suffix=".toml", delete=False) as f:
        f.write('high_stakes_extra = ["\\\\bscaffolding collapse\\\\b"]\n')
        path = f.name
    try:
        cfg = config.load_config(path)
        check("config key survives the DEFAULTS merge",
              cfg["high_stakes_extra"] == ["\\bscaffolding collapse\\b"])
        check("config-registered pattern trips high rigor",
              grounding.default_rigor("what causes a scaffolding collapse?") == "high")
    finally:
        os.unlink(path)

    if FAILED:
        print(f"\n{len(FAILED)} FAILED")
        raise SystemExit(1)
    print("\nALL OK")


if __name__ == "__main__":
    main()
