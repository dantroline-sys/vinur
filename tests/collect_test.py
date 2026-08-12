"""Acceptance test for `collect` — add one document to a shareable .kdb collection
under a named bundle (pack.add_to_collection / collection_preview).

Covers: clean-room create of a fresh file (master untouched, manifest names the
bundle, sources tagged into it); additive MERGE of a second doc into the same file
(counts grow, `added` reports only the new rows); idempotent re-add (content-hash
ids ⇒ no new rows); one-bundle-per-file refusal; dry-run previews + writes nothing;
plain-.kdb-only guard.

Like pack_test, the distill pipeline is patched with a fake that populates the
scratch kb, so the split/merge/manifest machinery all run for real — no LM needed.

Run:  python tests/collect_test.py     (stdlib only)
"""
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from knowledgehost import bundles as B          # noqa: E402
from knowledgehost import pack as P             # noqa: E402
from knowledgehost.kb import KB                 # noqa: E402


def check(label, cond):
    if cond:
        print(f"  ok  {label}")
    else:
        print(f"  FAIL  {label}")
        check.failed += 1
check.failed = 0


def fake_pipeline(doc, node, label, *, license="CC-BY-4.0", holder="A. Author"):
    """A _pipeline stand-in that drops one source + one node + one card (all citing
    `doc`) into the scratch kb, exactly like a real clean-room build would."""
    def run(scfg, src, say, **_kw):        # absorbs label=/report= that add_to_collection passes
        kb = KB(scfg)
        kb.db.execute(
            "INSERT INTO source_registry(doc_id,title,source_type,trust_weight,"
            "status,bundle,license,license_holder) VALUES(?,?,?,?,?,?,?,?)",
            (doc, "T:" + label, "book", 0.9, "active", "base", license, holder))
        sup = json.dumps([{"doc_id": doc, "trust": 0.9}])
        kb.db.execute(
            "INSERT INTO nodes(id,label,kind,summary,aliases,support,status,embedding)"
            " VALUES(?,?, 'entity',?, '[]',?, 'active',NULL)",
            (node, label, "about " + label, sup))
        kb.add_card(node, title="Use " + label, goal="g", steps=["a", "b"], doc_id=doc)
        kb.db.commit()
        kb.close()
        return {"fake": True}
    return run


def main():
    pipe0 = P._pipeline
    with tempfile.TemporaryDirectory() as td:
        src1 = Path(td) / "a.txt"; src1.write_text("alpha")
        src2 = Path(td) / "b.txt"; src2.write_text("beta")

        # a populated master — a clean-room build must never touch it
        master = os.path.join(td, "master", "kb.db")
        m = KB({"kb_path": master})
        m.db.execute("INSERT INTO source_registry(doc_id,title,source_type,trust_weight,"
                     "status,bundle) VALUES('doc:mine','Mine','book',0.9,'active','base')")
        m.db.commit(); m.close()
        master_bytes = Path(master).read_bytes()

        cfg = {"pack_build_dir": os.path.join(td, "build"), "kb_path": master,
               "embed_model": "nomic-embed-text-v1.5.f16.gguf"}
        target = os.path.join(td, "share", "geo.kdb")

        try:
            # ── preview a fresh (non-existent) target ────────────────────────────
            pv = P.collection_preview(cfg, target, "geo")
            check("preview: fresh target is compatible + not existing",
                  pv["ok"] and pv["compatible"] and not pv["exists"])

            # ── create: first document makes the file ────────────────────────────
            P._pipeline = fake_pipeline("doc:rocks", "n_rock", "rock")
            prog = []
            r1 = P.add_to_collection(cfg, str(src1), target, "geo",
                                     report=lambda ph, **k: prog.append((ph, k)))
            check("create: file made, created=True", r1["created"] and Path(target).exists())
            # (the fake _pipeline skips ingest/distill/link phases; add_to_collection
            #  still drives export + done, which is the wiring under test)
            phases = [p for p, _ in prog]
            check("progress: reporter fired export → done, steps=5 throughout",
                  phases == ["export", "done"] and all(k.get("steps") == 5 for _, k in prog))
            check("progress: 'done' carries created + added for the bar",
                  dict(prog)["done"].get("created") is True and dict(prog)["done"].get("added"))
            check("clean room: master kb byte-identical after the build",
                  Path(master).read_bytes() == master_bytes)
            i1 = B.inspect_bundle_file(target)
            check("create: manifest names bundle 'geo'",
                  (i1["manifest"] or {}).get("name") == "geo")
            check("create: 1 source / 1 node / 1 card",
                  i1["counts"].get("source_registry") == 1 and i1["counts"].get("nodes") == 1
                  and i1["counts"].get("procedure_cards") == 1)
            check("create: the source is tagged into bundle 'geo'",
                  all(s["bundle"] == "geo" for s in i1["sources"]))

            # ── add a DIFFERENT document to the SAME file/bundle → merge grows ───
            P._pipeline = fake_pipeline("doc:water", "n_water", "water")
            r2 = P.add_to_collection(cfg, str(src2), target, "geo")
            i2 = B.inspect_bundle_file(target)
            check("add: created=False (merged into existing)", not r2["created"])
            check("add: file now holds 2 sources / 2 nodes",
                  i2["counts"]["source_registry"] == 2 and i2["counts"]["nodes"] == 2)
            check("add: `added` reports exactly the 1 new node",
                  (r2["added"] or {}).get("nodes") == 1)

            # ── idempotent: re-adding the SAME doc changes nothing ───────────────
            P._pipeline = fake_pipeline("doc:water", "n_water", "water")
            r3 = P.add_to_collection(cfg, str(src2), target, "geo")
            i3 = B.inspect_bundle_file(target)
            check("idempotent: counts unchanged on re-add (content-hash ids)",
                  i3["counts"]["nodes"] == 2 and i3["counts"]["source_registry"] == 2)
            check("idempotent: `added` nodes == 0", (r3["added"] or {}).get("nodes", 0) == 0)

            # ── one bundle per file ──────────────────────────────────────────────
            P._pipeline = fake_pipeline("doc:x", "n_x", "xenon")
            try:
                P.add_to_collection(cfg, str(src1), target, "chemistry")
                check("one-bundle-per-file: a different bundle is refused", False)
            except ValueError as e:
                check("one-bundle-per-file: refused, names the rule",
                      "one bundle per file" in str(e))
            check("one-bundle-per-file: refusal wrote nothing (still 2 nodes)",
                  B.inspect_bundle_file(target)["counts"]["nodes"] == 2)

            # ── dry-run previews, writes nothing ─────────────────────────────────
            before = Path(target).read_bytes()
            dr = P.add_to_collection(cfg, str(src1), target, "geo", dry_run=True)
            check("dry-run: nothing written", Path(target).read_bytes() == before)
            check("dry-run: flagged, not-created, reports current counts",
                  dr.get("dry_run") and not dr["created"] and dr["current"].get("nodes") == 2)

            # ── plain-.kdb-only guard ────────────────────────────────────────────
            try:
                P.add_to_collection(cfg, str(src1), target + ".gz", "geo")
                check("plain-.kdb-only: compress/encrypt target refused", False)
            except ValueError as e:
                check("plain-.kdb-only: compress/encrypt target refused",
                      "plain .kdb" in str(e))

            # ── real CLI dispatch (a dry-run — no LM): the parser-builds gate can't
            #    catch a crash at DISPATCH time (e.g. touching an unopened `store`) ─
            repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            cli = subprocess.run(
                [sys.executable, "-m", "knowledgehost", "collect", str(src1),
                 "--to", os.path.join(td, "cli", "z.kdb"), "--bundle", "geo", "--dry-run"],
                cwd=repo, capture_output=True, text=True)
            check("CLI: `collect --dry-run` dispatches cleanly (exit 0, no traceback)",
                  cli.returncode == 0 and "Traceback" not in cli.stderr
                  and "dry-run" in (cli.stdout + cli.stderr))
        finally:
            P._pipeline = pipe0

    print()
    if check.failed:
        print(f"{check.failed} FAILURE(S)")
        return 1
    print("collect_test: ALL OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
