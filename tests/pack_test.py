"""Acceptance test for VINUR-PACK-01 — the clean-room knowledge-pack producer.

Covers §5 (license gate matrix, override fills EMPTY only + attested,
--allow-unlicensed → shareable:false), §3/§4 (build → packs/<slug>/ artifact +
sidecar, manifest v2 fields, same-version refusal, --force), §6 (gzip
round-trip through import's magic sniff), §7 (encrypted round-trip — skipped
with a note when no SQLCipher driver), and §8 (compat decision table: schema/
card-family refusals name the remedy, vocab-hash + shareable warnings, sidecar
disagreement refuses, --verify catches post-export tampering).

The distill pipeline itself is exercised by the standalone suite; here it is
patched with a fake that populates the scratch kb, so the gate/manifest/
packaging/import machinery all run for real.

Run:  python tests/pack_test.py     (stdlib only)
"""
import json
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from knowledgehost import bundles as B
from knowledgehost import dbcrypt
from knowledgehost import pack as P
from knowledgehost.kb import KB, SCHEMA_VERSION


def check(label, cond):
    if cond:
        print(f"  ok  {label}")
    else:
        print(f"  FAIL  {label}")
        check.failed += 1
check.failed = 0


def fake_pipeline(license="CC-BY-4.0", holder="A. Author"):
    """A _pipeline stand-in that populates the scratch kb like a real build."""
    def run(scfg, src, say):
        kb = KB(scfg)
        kb.db.execute(
            "INSERT INTO source_registry(doc_id,title,source_type,trust_weight,"
            "status,bundle,license,license_holder)"
            " VALUES('doc:rocks','Rocks: A Primer','book',0.9,'active','base',?,?)",
            (license, holder))
        sup = json.dumps([{"doc_id": "doc:rocks", "trust": 0.9}])
        kb.db.execute(
            "INSERT INTO nodes(id,label,kind,summary,aliases,support,status,embedding)"
            " VALUES('n_rock','rock','entity','a solid aggregate of minerals','[]',"
            "?, 'active',NULL)", (sup,))
        kb.add_card("n_rock", title="Identify a rock", goal="know your rocks",
                    steps=["look", "scratch test"], doc_id="doc:rocks")
        kb.db.commit()
        kb.close()
        return {"fake": True}
    return run


def edit_manifest(kdb, fn):
    """Read-modify-write a pack file's manifest (tamper helper)."""
    con = B._connect(kdb)
    man = B.read_manifest(con)
    fn(man)
    con.execute("DELETE FROM bundle_manifest")
    con.execute("INSERT INTO bundle_manifest(json) VALUES(?)", (json.dumps(man),))
    con.commit()
    con.close()


def main():
    # ── §5 license classes ───────────────────────────────────────────────────
    cases = {"CC0-1.0": "free", "Public Domain": "free", "CC-BY-4.0": "attribution",
             "MIT": "attribution", "Apache-2.0": "attribution",
             "CC-BY-SA-4.0": "share-alike", "GPL-3.0": "share-alike",
             "CC-BY-NC-4.0": "nc", "CC-BY-ND-4.0": "refused",
             "CC-BY-NC-ND-4.0": "refused", "": "refused", "unknown": "refused",
             "proprietary": "refused"}
    bad = {k: (P.license_class(k), v) for k, v in cases.items()
           if P.license_class(k) != v}
    check("license_class matrix (PD/BY/SA/NC/ND/unknown)", not bad or print(bad))

    rows = [{"doc_id": "d1", "title": "T1", "license": "CC-BY-4.0",
             "license_holder": "H"},
            {"doc_id": "d2", "title": "T2", "license": ""}]
    g = P.gate(rows)
    check("gate refuses on an unlicensed source; table names the remedies",
          not g["ok"] and g["refused"][0]["doc_id"] == "d2"
          and "--allow-unlicensed" in P.gate_table(g))
    g2 = P.gate(rows, override="CC0-1.0")
    check("--license fills EMPTY only (attested) — detected licenses untouched",
          g2["ok"] and g2["filled"] == ["d2"]
          and [p for p in g2["per_source"] if p["doc_id"] == "d2"][0]["attested"]
          and [p for p in g2["per_source"] if p["doc_id"] == "d1"][0]["license"] == "CC-BY-4.0")
    check("effective = most restrictive present (BY beats CC0)",
          g2["effective"] == "CC-BY-4.0" and g2["shareable"])
    g3 = P.gate(rows, allow_unlicensed=True)
    check("--allow-unlicensed builds but stamps shareable:false / UNSHAREABLE",
          g3["ok"] and not g3["shareable"] and g3["effective"] == "UNSHAREABLE")

    with tempfile.TemporaryDirectory() as td:
        srcdir = Path(td) / "book"
        srcdir.mkdir()
        (srcdir / "rocks.txt").write_text("rocks are neat")
        # a populated master on the producer box — §10.1 says a build never
        # touches it (byte-compare after the build proves the clean room)
        master_cfg = {"kb_path": os.path.join(td, "master", "kb.db")}
        m = KB(master_cfg)
        m.db.execute("INSERT INTO source_registry(doc_id,title,source_type,"
                     "trust_weight,status,bundle) VALUES('doc:mine','Mine',"
                     "'book',0.9,'active','base')")
        m.db.commit()
        m.close()
        master_bytes = Path(master_cfg["kb_path"]).read_bytes()
        cfg = {"pack_dir": os.path.join(td, "packs"),
               "pack_build_dir": os.path.join(td, "build"),
               "kb_path": master_cfg["kb_path"],
               "embed_model": "nomic-embed-text-v1.5.f16.gguf",
               "distill_model": "big", "distill_max_tokens": 8192}
        pipe0 = P._pipeline
        P._pipeline = fake_pipeline()
        try:
            # ── §3 build + §4 manifest ───────────────────────────────────────
            res = P.build_pack(cfg, str(srcdir), title="Rocks", author="Dan",
                               version="1.0.0", describe="rock knowledge")
            check("clean room: the master kb is byte-identical after the build",
                  Path(master_cfg["kb_path"]).read_bytes() == master_bytes)
            art = Path(res["artifact"])
            check("artifact lands at packs/<slug>/<slug>-<version>.kdb",
                  art.name == "book-1.0.0.kdb" and art.parent.name == "book"
                  and art.exists())
            side = json.loads(Path(res["sidecar"]).read_text())
            man = res["manifest"]
            check("sidecar is the manifest, byte-comparable", side == man)
            check("manifest v2: format/kind/pack/compat/pipeline blocks",
                  man["format"] == 2 and man["kind"] == "pack"
                  and man["pack"]["producer"]["name"] == "Dan"
                  and man["compat"]["schema_version"] == SCHEMA_VERSION
                  and "procedures" in man["compat"]["card_families"]
                  and man["compat"]["requires"] == []
                  and man["licensing"]["effective"] == "CC-BY-4.0"
                  and man["licensing"]["attribution"])
            con = B._connect(str(art))
            got_hash = P.content_hash(con)
            con.close()
            check("content_hash survives VACUUM (recompute matches manifest)",
                  got_hash == man["content"]["content_hash"])
            check("scratch removed on success",
                  not (Path(td) / "build" / "book").exists())
            try:
                P.build_pack(cfg, str(srcdir), version="1.0.0")
                check("same-version rebuild refused without --force", False)
            except ValueError as e:
                check("same-version rebuild refused without --force",
                      "--force" in str(e))
            P.build_pack(cfg, str(srcdir), version="1.0.0", force=True)
            check("--force replaces the artifact", art.exists())

            # ── §5 in the build path ─────────────────────────────────────────
            P._pipeline = fake_pipeline(license="", holder="")
            try:
                P.build_pack(cfg, str(srcdir), name="nolic", version="1.0.0")
                check("unlicensed source refuses the build", False)
            except ValueError as e:
                check("unlicensed source refuses the build",
                      "license gate" in str(e) and "--allow-unlicensed" in str(e))
            res_p = P.build_pack(cfg, str(srcdir), name="private", version="1.0.0",
                                 allow_unlicensed=True)
            check("--allow-unlicensed pack stamped shareable:false",
                  res_p["manifest"]["licensing"]["shareable"] is False)
            res_a = P.build_pack(cfg, str(srcdir), name="attested", version="1.0.0",
                                 license_override="CC0-1.0")
            att = res_a["manifest"]["licensing"]["per_source"][0]
            check("--license override lands attested in the manifest",
                  att["license"] == "CC0-1.0" and att["attested"] is True)
            P._pipeline = fake_pipeline()

            # ── §6 gzip + §8 import round-trip (virgin consumer bootstrap) ───
            res_gz = P.build_pack(cfg, str(srcdir), name="gzpack", version="1.0.0",
                                  compress=True)
            gz = Path(res_gz["artifact"])
            check("compressed artifact is .kdb.gz + sidecar beside it",
                  gz.name.endswith(".kdb.gz") and Path(res_gz["sidecar"]).exists())
            cons = {"kb_path": os.path.join(td, "cons", "kb.db"),
                    "bundle_dir": os.path.join(td, "cons", "bundles"),
                    "embed_model": cfg["embed_model"]}
            r = B.import_bundle(cons, str(gz))
            check("gz pack imports (magic sniff + virgin bootstrap)",
                  r["sources_new"] == 1)
            k = KB(cons)
            n = k.db.execute("SELECT COUNT(*) FROM nodes").fetchone()[0]
            c = k.db.execute("SELECT COUNT(*) FROM procedure_cards").fetchone()[0]
            k.close()
            check("imported pack is queryable knowledge (node + card landed)",
                  n == 1 and c == 1)
            msgs = []
            B.import_bundle(cons, str(gz), log_fn=msgs.append)
            check("re-import is a no-op with the seed-not-tissue note upstream",
                  any("0 new source" in m for m in msgs))

            # ── §8 compat decision table ─────────────────────────────────────
            plain = Path(res["artifact"])
            for tamper, expect, hard in (
                    (lambda m: m["compat"].__setitem__("schema_version", 999),
                     "update vinur", True),
                    (lambda m: m["compat"]["card_families"].__setitem__("weird", 9),
                     "doesn't know the card family", True)):
                t = Path(td) / "tampered.kdb"
                t.write_bytes(plain.read_bytes())
                edit_manifest(str(t), tamper)
                try:
                    B.import_bundle({"kb_path": os.path.join(td, "c2", "kb.db"),
                                     "bundle_dir": os.path.join(td, "c2", "b"),
                                     "embed_model": cfg["embed_model"]}, str(t),
                                    name="tampered")
                    check(f"hard compat refusal: {expect}", False)
                except ValueError as e:
                    check(f"hard compat refusal: {expect}", expect in str(e))
                t.unlink()
            t = Path(td) / "warny.kdb"
            t.write_bytes(plain.read_bytes())
            edit_manifest(str(t), lambda m: (
                m["compat"].__setitem__("vocab_hash", "sha256:dead"),
                m["licensing"].__setitem__("shareable", False)))
            warns = []
            B.import_bundle({"kb_path": os.path.join(td, "c3", "kb.db"),
                             "bundle_dir": os.path.join(td, "c3", "b"),
                             "embed_model": cfg["embed_model"]}, str(t),
                            name="warny", log_fn=warns.append)
            check("soft compat: vocab drift + shareable:false WARN and proceed",
                  any("vocabularies differ" in m for m in warns)
                  and any("NOT SHAREABLE" in m for m in warns))
            t.unlink()

            # ── §8 sidecar + --verify tamper checks ──────────────────────────
            sidep = Path(res["sidecar"])
            side_orig = sidep.read_text()
            tampered = json.loads(side_orig)
            tampered["content"]["content_hash"] = "sha256:beef"
            sidep.write_text(json.dumps(tampered))
            try:
                B.import_bundle({"kb_path": os.path.join(td, "c4", "kb.db"),
                                 "bundle_dir": os.path.join(td, "c4", "b"),
                                 "embed_model": cfg["embed_model"]}, str(plain))
                check("sidecar disagreement refuses (no --verify needed)", False)
            except ValueError as e:
                check("sidecar disagreement refuses (no --verify needed)",
                      "sidecar" in str(e))
            sidep.write_text(side_orig)
            con = B._connect(str(plain))
            con.execute("UPDATE nodes SET label='igneous rock' WHERE id='n_rock'")
            con.commit()
            con.close()
            try:
                B.import_bundle({"kb_path": os.path.join(td, "c5", "kb.db"),
                                 "bundle_dir": os.path.join(td, "c5", "b"),
                                 "embed_model": cfg["embed_model"]}, str(plain),
                                verify=True)
                check("--verify catches content altered after export", False)
            except ValueError as e:
                check("--verify catches content altered after export",
                      "altered after export" in str(e))

            # ── §7 encryption (optional driver) ──────────────────────────────
            if dbcrypt.available():
                os.environ["PACK_PASSPHRASE"] = "hunter2"
                res_e = P.build_pack(cfg, str(srcdir), name="secret",
                                     version="1.0.0", compress=True, encrypt=True)
                enc = Path(res_e["artifact"])
                check("encrypted artifact is .kdb.gz.enc", enc.name.endswith(".kdb.gz.enc"))
                try:
                    B.import_bundle({"kb_path": os.path.join(td, "c6", "kb.db"),
                                     "bundle_dir": os.path.join(td, "c6", "b"),
                                     "embed_model": cfg["embed_model"]}, str(enc))
                    check("encrypted pack without passphrase refuses w/ remedy", False)
                except ValueError as e:
                    check("encrypted pack without passphrase refuses w/ remedy",
                          "passphrase" in str(e))
                r = B.import_bundle({"kb_path": os.path.join(td, "c6", "kb.db"),
                                     "bundle_dir": os.path.join(td, "c6", "b"),
                                     "embed_model": cfg["embed_model"]}, str(enc),
                                    passphrase="hunter2")
                check("encrypted round-trip imports", r["sources_new"] == 1)
            else:
                print("  note  §7 encryption checks skipped — no SQLCipher driver "
                      "(pip install sqlcipher3-binary)")
        finally:
            P._pipeline = pipe0

    print()
    if check.failed:
        print(f"{check.failed} FAILURE(S)")
        return 1
    print("ALL OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
