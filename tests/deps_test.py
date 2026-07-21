#!/usr/bin/env python
"""The dependency ratchet: knowledgehost's core imports NOTHING third-party at
module scope.  That is the repo's strongest supply-chain property — the host
serves, ingests text, and answers on a bare interpreter; every heavy package
(numpy, lancedb, pyarrow, spacy, usearch, pymupdf, …) is a lazy import behind
a capability it merely improves, and vLLM's whole pin-forest lives in its own
venv under serving/.  This test keeps that true by construction: a new
top-level third-party import anywhere in knowledgehost/ fails here, and adding
one is a decision taken by editing this file in the same commit.

Stdlib only; parses with ast, executes nothing.
"""
import ast
import sys
from pathlib import Path

PKG = Path(__file__).resolve().parent.parent / "knowledgehost"

# Deliberate exceptions to "no hard third-party imports".  Empty, and the point
# is that it stays that way.
HARD_ALLOWED: set = set()

OK = 0


def ok(label):
    global OK
    OK += 1
    print(f"  ok {OK:2d}  {label}")


std = set(sys.stdlib_module_names)
local = {p.stem for p in PKG.glob("*.py")}
hard, soft = {}, {}
for f in sorted(PKG.glob("*.py")):
    tree = ast.parse(f.read_text())
    for node in ast.walk(tree):
        mods = []
        if isinstance(node, ast.Import):
            mods = [a.name.split(".")[0] for a in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            mods = [node.module.split(".")[0]]
        for m in mods:
            if m in std or m in local:
                continue
            (hard if node.col_offset == 0 else soft).setdefault(m, set()).add(f.name)

grown = set(hard) - HARD_ALLOWED
assert not grown, (
    f"NEW hard third-party import(s): "
    f"{ {m: sorted(hard[m]) for m in sorted(grown)} } — the core must import on a "
    "bare interpreter.  Make it lazy (in-function or try/except with a graceful "
    "degrade), or add it to HARD_ALLOWED here as a deliberate decision.")
ok("knowledgehost core has ZERO hard third-party imports (bare interpreter suffices)")

stale = HARD_ALLOWED - set(hard)
assert not stale, f"stale allowlist entries: {sorted(stale)} — ratchet down"
ok("allowlist matches reality")

# The lazy surface is allowed to grow, but it should be KNOWN — a new name here
# is fine, it just has to be added consciously.
LAZY_KNOWN = {"numpy", "lancedb", "pyarrow", "pylance", "spacy", "usearch",
              "fitz", "pymupdf", "trafilatura", "ebooklib", "libzim",
              "bs4", "sqlcipher3", "PIL"}
unknown = set(soft) - LAZY_KNOWN - set(hard)
assert not unknown, (
    f"lazy import(s) not in LAZY_KNOWN: {sorted(unknown)} — probably fine, but "
    "add them here so the optional surface stays inventoried.")
ok(f"lazy/optional surface inventoried: {sorted(set(soft))}")

print(f"deps_test: {OK} checks OK")
