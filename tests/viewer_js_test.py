#!/usr/bin/env python
"""The viewer ships a ~2500-line inline <script> embedded in a Python triple-quoted
string (knowledgehost.viewer.INDEX_HTML).  That embedding has a sharp edge: a `\\n`
written inside a single/double-quoted JS string is eaten by Python and emitted as a
REAL newline, which is a JavaScript SyntaxError (only backtick template literals
tolerate a literal newline).  py_compile can't see it — the Python parses fine — so
a broken page still serves HTTP 200 and the browser silently runs NO script.

This checks the RUNTIME string (what the browser actually receives), not the source
form, by handing the extracted script to `node --check`.  Skips (does not fail) when
node isn't installed.

    python tests/viewer_js_test.py
"""

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from knowledgehost.viewer import INDEX_HTML   # noqa: E402


def main():
    node = shutil.which("node")
    if not node:
        print("viewer_js_test: SKIPPED (node not installed)")
        return 0
    if "<script>" not in INDEX_HTML or "</script>" not in INDEX_HTML:
        print("FAIL: no <script> block in INDEX_HTML")
        return 1
    js = INDEX_HTML.split("<script>", 1)[1].rsplit("</script>", 1)[0]
    with tempfile.NamedTemporaryFile("w", suffix=".js", delete=False) as f:
        f.write(js)
        path = f.name
    r = subprocess.run([node, "--check", path], capture_output=True, text=True)
    Path(path).unlink(missing_ok=True)
    if r.returncode == 0:
        print(f"viewer_js_test: OK — runtime inline script parses ({len(js)} bytes)")
        return 0
    print("FAIL: the SERVED inline script is not valid JS (what the browser runs):")
    print(r.stderr.rstrip() or r.stdout.rstrip())
    return 1


if __name__ == "__main__":
    sys.exit(main())
