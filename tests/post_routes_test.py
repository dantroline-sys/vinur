#!/usr/bin/env python
"""_do_POST gates every request through one allow-list tuple
(`if path not in (...): 404`) BEFORE dispatching to the per-path handlers.  Add a
handler but forget to list its path and the route is dead — it 404s "not found"
before the handler is ever reached (exactly how /queue/clear shipped broken).

This parses server.py and asserts every `path == "..."` handler inside _do_POST is
present in that allow-list.  Pure-AST, no server process needed.

    python tests/post_routes_test.py
"""

import ast
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / "knowledgehost" / "server.py"


def _find_func(tree, name):
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    return None


def main():
    tree = ast.parse(SRC.read_text())
    fn = _find_func(tree, "_do_POST")
    if fn is None:
        print("FAIL: _do_POST not found in server.py")
        return 1

    allowed = None          # the `path not in (...)` tuple
    handled = set()         # every `path == "..."` comparator
    for node in ast.walk(fn):
        if not isinstance(node, ast.Compare) or not (
                isinstance(node.left, ast.Name) and node.left.id == "path"):
            continue
        op = node.ops[0]
        if isinstance(op, ast.NotIn) and isinstance(node.comparators[0], (ast.Tuple, ast.List)):
            allowed = {e.value for e in node.comparators[0].elts
                       if isinstance(e, ast.Constant) and isinstance(e.value, str)}
        elif isinstance(op, ast.Eq) and isinstance(node.comparators[0], ast.Constant):
            handled.add(node.comparators[0].value)

    if allowed is None:
        print("FAIL: could not find the `path not in (...)` allow-list in _do_POST")
        return 1

    missing = sorted(handled - allowed)
    if missing:
        print("FAIL: these _do_POST handlers are unreachable — not in the allow-list "
              "(they 404 'not found'):")
        for p in missing:
            print(f"  {p}")
        return 1
    print(f"post_routes_test: OK — all {len(handled)} POST handlers are registered "
          f"in the allow-list ({len(allowed)} paths)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
