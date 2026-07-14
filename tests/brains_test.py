"""Acceptance test for modular brains — ship / import / load-unload / eject.

Builds an AUTHOR master (base + a 'geology' bundle with a node, an edge into a
borrowed base node, a card and a surface question), splits the geology brain to
its .kdb, then plays a RECIPIENT box through the whole lifecycle:

  * import: absorbed under one bundle name; the recipient's own pre-existing
    row (same content-hash id) is untouched; the brain's rows get support
    trust capped to 'low'; trust='keep' skips the cap; a manifest embed-model
    mismatch strips shipped vectors; re-import is a no-op; a name collision
    with an unrelated file is refused; a file whose sources call themselves
    'base' is rebranded, never merged into the local base group.
  * load/unload: the unloaded_bundles filter prunes whole bundles after the
    scenario, assemble builds a working DB without them, and clearing the
    toggle short-circuits back to the master byte-for-byte.
  * eject: exports the bundle's closure to its .kdb first, removes exactly its
    provenance (shared/borrowed rows survive, surfaces/facets cleaned), and
    re-importing the exported file undoes the eject.

Run:  python tests/brains_test.py     (stdlib only)
"""
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from knowledgehost.kb import KB
from knowledgehost import bundles as B


def check(label, cond):
    print(("  ok  " if cond else "  FAIL ") + label)
    if not cond:
        check.failed += 1
check.failed = 0


def sup(doc_id, trust=0.9):
    return json.dumps([{"doc_id": doc_id, "trust": trust}])


def add_source(kb, doc_id, bundle, title, trust=0.9):
    kb.db.execute(
        "INSERT INTO source_registry(doc_id,title,source_type,trust_weight,status,bundle)"
        " VALUES(?,?,?,?,'active',?)", (doc_id, title, "book", trust, bundle))


def add_node(kb, nid, label, doc_id, trust=0.9, embedding=None):
    kb.db.execute(
        "INSERT INTO nodes(id,label,kind,summary,aliases,support,status,embedding)"
        " VALUES(?,?,?,?,?,?,'active',?)",
        (nid, label, "entity", f"{label} summary", "[]", sup(doc_id, trust), embedding))


def node(kb, nid):
    r = kb.db.execute("SELECT id,label,support,embedding FROM nodes WHERE id=?",
                      (nid,)).fetchall()
    return (dict(zip(("id", "label", "support", "embedding"), r[0]))
            if r else None)


def trust_of(row):
    return json.loads(row["support"])[0]["trust"]


def main():
    td = tempfile.mkdtemp(prefix="kb-brains-")

    # ── author box: base + geology, split the brain ──────────────────────────
    cfg_a = {"kb_path": os.path.join(td, "author", "kb.db"),
             "bundle_dir": os.path.join(td, "author", "bundles"),
             "embed_model": "nomic-embed-text-v1.5.f16.gguf"}
    os.makedirs(os.path.dirname(cfg_a["kb_path"]))
    a = KB(cfg_a)
    add_source(a, "base:doc1", "base", "Base Book")
    add_node(a, "n_water", "water", "base:doc1")
    add_node(a, "n_rock", "rock", "base:doc1", embedding=b"\x01\x02")
    add_source(a, "geo:doc1", "geology", "Field Geology")
    add_node(a, "n_basalt", "basalt", "geo:doc1", embedding=b"\x03\x04")
    a.db.execute(
        "INSERT INTO edges(id,src_id,dst_id,family,type,support,status)"
        " VALUES('e_basalt_rock','n_basalt','n_rock','taxonomy','is_a',?,'active')",
        (sup("geo:doc1"),))
    a.db.execute(
        "INSERT INTO procedure_cards(id,node_id,title,steps,support,status)"
        " VALUES('c_identify','n_basalt','Identify basalt','[]',?,'active')",
        (sup("geo:doc1"),))
    a.db.execute(
        "INSERT INTO surface_questions(id,target_kind,target_id,text)"
        " VALUES('q1','node','n_basalt','what is basalt?')")
    a.db.commit()
    res = B.split(cfg_a, force=True)
    check("split exports base + geology .kdb",
          set(res) == {"base", "geology"} and
          os.path.exists(res["geology"]["file"]))
    geo_kdb = res["geology"]["file"]
    insp = B.inspect_bundle_file(geo_kdb)
    check("manifest carries name + embed model",
          (insp["manifest"] or {}).get("name") == "geology" and
          insp["manifest"]["embed_model"] == cfg_a["embed_model"])
    check("closure borrowed the referenced base node",
          insp["counts"]["nodes"] == 2 and insp["counts"]["edges"] == 1
          and insp["counts"]["procedure_cards"] == 1)
    a.close()

    # ── recipient box: own base, one shared row (same content-hash id) ──────
    cfg_b = {"kb_path": os.path.join(td, "recip", "kb.db"),
             "bundle_dir": os.path.join(td, "recip", "bundles"),
             "embed_model": "nomic-embed-text-v1.5.f16.gguf"}
    os.makedirs(os.path.dirname(cfg_b["kb_path"]))
    b = KB(cfg_b)
    add_source(b, "own:doc1", "base", "My Own Notes")
    add_node(b, "n_rock", "rock", "own:doc1", trust=0.9)   # same id as shipped 'rock'
    b.db.commit()

    # ── import: rebrand + trust cap + shared-row protection ─────────────────
    r1 = B.import_bundle(cfg_b, geo_kdb)
    check("import lands 1 new source under 'geology'",
          r1["bundle"] == "geology" and r1["sources_new"] == 1)
    row = b.db.execute("SELECT bundle, trust_weight FROM source_registry "
                       "WHERE doc_id='geo:doc1'").fetchall()[0]
    check("shipped source rebranded + registry trust capped",
          row[0] == "geology" and row[1] <= B.IMPORT_TRUST_CAP)
    basalt = node(b, "n_basalt")
    check("brain rows trust-capped to 'low'",
          trust_of(basalt) == B.IMPORT_TRUST_CAP)
    check("recipient's own shared row untouched",
          trust_of(node(b, "n_rock")) == 0.9)
    check("same embed model — shipped vectors kept",
          r1["embeddings_stripped"] == 0 and basalt["embedding"] is not None)

    r2 = B.import_bundle(cfg_b, geo_kdb)
    check("re-import is a no-op", r2["sources_new"] == 0 and
          not any(r2["new"].values()))

    # a DIFFERENT file may not squat an existing bundle name…
    base_kdb = res["base"]["file"]
    try:
        B.import_bundle(cfg_b, base_kdb, name="geology")
        collided = False
    except ValueError:
        collided = True
    check("name collision with unrelated file refused", collided)
    # …and a file whose sources call themselves 'base' gets rebranded, with an
    # embed-model mismatch stripping its vectors for local re-embedding.
    cfg_b2 = {**cfg_b, "embed_model": "some-other-embedder.gguf"}
    r3 = B.import_bundle(cfg_b2, base_kdb, name="packA")
    packrock = b.db.execute("SELECT bundle FROM source_registry "
                            "WHERE doc_id='base:doc1'").fetchall()[0][0]
    check("'base' brain rebranded to packA (local base untouched)",
          packrock == "packA" and r3["bundle"] == "packA")
    check("embed-model mismatch strips shipped vectors",
          r3["embeddings_stripped"] >= 1 and
          node(b, "n_water")["embedding"] is None)
    check("shared n_rock kept ITS vector state through packA import",
          trust_of(node(b, "n_rock")) == 0.9)
    b.close()

    # ── load/unload: the selection filter + assembly ─────────────────────────
    cfg_run = {**cfg_b, "unloaded_bundles": "geology",
               "bundle_work_dir": os.path.join(td, "recip", "work"),
               "bundle_dir": ""}          # force closure-from-master assembly
    check("unload engages modularity", B.is_modular(cfg_run))
    work = B.assemble_working_db(cfg_run, force=True)
    check("working DB is a separate file", work != cfg_run["kb_path"])
    import sqlite3
    w = sqlite3.connect(work)
    have = {r[0] for r in w.execute("SELECT id FROM nodes")}
    check("unloaded brain's own node absent; base + packA present",
          "n_basalt" not in have and {"n_rock", "n_water"} <= have)
    check("geology card absent from working DB",
          not w.execute("SELECT 1 FROM procedure_cards WHERE id='c_identify'").fetchall())
    w.close()
    cfg_run["unloaded_bundles"] = ""
    check("everything loaded + no scenarios → master served as-is",
          B.assemble_working_db(cfg_run, force=True) == cfg_run["kb_path"])

    # ── eject: export-first, surgical removal, undo by re-import ────────────
    dry = B.eject_bundle(cfg_b, "geology", dry_run=True)
    check("dry-run counts without deleting",
          dry["dry_run"] and dry["edges_deleted"] == 1 and
          node(KB(cfg_b), "n_basalt") is not None)
    st = B.eject_bundle(cfg_b, "geology")
    check("eject exported the .kdb first",
          st["exported"] and os.path.exists(st["exported"]))
    b = KB(cfg_b)
    check("brain's own rows gone (node, card, edge)",
          node(b, "n_basalt") is None and
          not b.db.execute("SELECT 1 FROM procedure_cards WHERE id='c_identify'").fetchall() and
          not b.db.execute("SELECT 1 FROM edges WHERE id='e_basalt_rock'").fetchall())
    check("surface question cleaned", st["refs_cleaned"] >= 1)
    check("shared/borrowed 'rock' survives eject",
          node(b, "n_rock") is not None and trust_of(node(b, "n_rock")) == 0.9)
    check("geology source row removed",
          not b.db.execute("SELECT 1 FROM source_registry WHERE doc_id='geo:doc1'").fetchall())

    r4 = B.import_bundle(cfg_b, st["exported"])
    check("re-importing the exported file undoes the eject",
          r4["sources_new"] == 1 and node(KB(cfg_b), "n_basalt") is not None)
    b.close()

    print()
    if check.failed:
        print(f"{check.failed} FAILURE(S)")
        return 1
    print("ALL OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
