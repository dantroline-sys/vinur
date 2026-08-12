"""The file-browser endpoint (server._fs_list / _fs_roots) that backs the collect
wizard's document + target pickers.

Checks: dirs-first ordering, dotfiles hidden, file sizes, parent link, ingestable
`exts` passthrough, a FILE path resolving to its containing directory, a bogus path
resolving up to the nearest readable ancestor, and roots including Home + a
configured source dir.

Run:  python tests/fs_browse_test.py     (stdlib only)
"""
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from knowledgehost.server import _fs_list, _fs_roots   # noqa: E402

PASS = FAIL = 0


def check(name, cond):
    global PASS, FAIL
    if cond:
        PASS += 1; print(f"  ok  {name}")
    else:
        FAIL += 1; print(f"FAIL  {name}")


def main():
    with tempfile.TemporaryDirectory() as td:
        root = Path(td) / "Documents"
        (root / "Science").mkdir(parents=True)
        (root / "paper.pdf").write_text("x" * 42)
        (root / "notes.txt").write_text("y" * 5)
        (root / ".secret").write_text("z")

        r = _fs_list({"extensions": [".pdf", ".txt"]}, str(root))
        names = [e["name"] for e in r["entries"]]
        check("ok + lists the dir", r["ok"] and r["path"] == os.path.realpath(str(root)))
        check("dotfiles hidden", ".secret" not in names)
        check("dirs sort before files", names == ["Science", "notes.txt", "paper.pdf"])
        pdf = next(e for e in r["entries"] if e["name"] == "paper.pdf")
        sci = next(e for e in r["entries"] if e["name"] == "Science")
        check("file carries size, dir does not", pdf["size"] == 42 and sci["dir"] and sci["size"] is None)
        check("parent points one level up", r["parent"] == os.path.realpath(str(root.parent)))
        check("ingestable exts passed through (lowercased)", r["exts"] == [".pdf", ".txt"])
        check("small dir is not truncated", r["truncated"] is False)

        # a FILE path resolves to its containing directory (the picker opens there)
        rf = _fs_list({}, str(root / "paper.pdf"))
        check("a file path opens its folder", rf["ok"] and rf["path"] == os.path.realpath(str(root)))

        # a bogus path walks up to the nearest readable ancestor (never errors out)
        rb = _fs_list({}, str(root / "nope" / "gone" / "x"))
        check("bogus path resolves up to an existing dir",
              rb["ok"] and os.path.isdir(rb["path"]))

        # empty path → a curated root (Home first)
        re = _fs_list({}, "")
        check("empty path lands on a readable dir", re["ok"] and os.path.isdir(re["path"]))

        # roots: Home always, plus a configured source dir that exists
        roots = _fs_roots({"sources": [str(root)], "library_sources": [],
                           "quarantine_dir": ""})
        labels = {r0["label"] for r0 in roots}
        paths = {r0["path"] for r0 in roots}
        check("roots include Home", "Home" in labels)
        check("roots include the configured source dir",
              os.path.realpath(str(root)) in paths)
        check("roots are de-duplicated + all real dirs",
              len(paths) == len(roots) and all(os.path.isdir(p) for p in paths))

    print()
    if FAIL:
        print(f"{FAIL} FAILURE(S)")
        return 1
    print(f"fs_browse_test: ALL OK ({PASS} checks)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
