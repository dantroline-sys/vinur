"""VINUR-OPS-01 gates (§6) on the golden fixtures (§7): the op-annotation surface for
external-oracle consumers.  Pure python, no services.  Region values ("rust:", …) are
ALLOWED here — fixtures are exempt from the neutrality gate they themselves enforce."""
import hashlib
import json
import os
import re
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from knowledgehost import bundles, config, query as query_mod, tools as tools_mod
from knowledgehost.kb import KB, _pack

FAILED = []
ENGINE_DIR = Path(__file__).resolve().parent.parent / "knowledgehost"

OP = "rust:op:std::vec::Vec::push#inherent"
HAZ = "rust:diag:card:E0502"
UNKNOWN = "rust:op:acme::Thing::frob#inherent"


def check(label, cond):
    print(("  ok  " if cond else "FAIL  ") + label)
    if not cond:
        FAILED.append(label)


class StubEmbedder:
    def embed_one(self, text, kind=None):
        return [1.0] + [0.0] * 7


def _cfg(tmp, **over):
    cfg = dict(config.DEFAULTS)
    cfg["kb_path"] = str(Path(tmp) / "kb.db")
    cfg["bundle_dir"] = str(Path(tmp) / "bundles")
    cfg["ops_regions"] = ["rust=rust-coding"]
    cfg["ask_exclude_facets"] = ["domain:rust-coding"]
    cfg.update(over)
    return cfg


def _fixture_kb(tmp):
    """One region op-node with a hazard card + one general node, both embedded and
    both supported by registry docs (the region doc in bundle rust-base)."""
    kb = KB(_cfg(tmp))
    now = time.time()
    vec = _pack([1.0] + [0.0] * 7)
    kb.db.execute("INSERT INTO source_registry(doc_id,title,source_type,trust_weight,"
                  "regime,status,bundle) VALUES(?,?,?,?,?,?,?)",
                  ("rustdoc1", "error index", "reference", 1.0, "empirical", "active",
                   "rust-base"))
    sup = json.dumps([{"doc_id": "rustdoc1"}])
    kb.db.execute("INSERT INTO nodes(id,label,kind,summary,aliases,support,status,"
                  "embedding) VALUES(?,?,?,?,?,?,?,?)",
                  (OP, "Vec::push", "fn", "", "[]", sup, "active", vec))
    kb.db.execute("INSERT INTO procedure_cards(id,node_id,title,card_type,criteria,"
                  "support,status,card_hash,created_at,updated_at) VALUES(?,?,?,?,?,?,?,?,?,?)",
                  (HAZ, OP, "cannot borrow as mutable more than once",
                   "hazard", json.dumps({"code": "E0502", "severity": "error"}),
                   sup, "active", "h1", now, now))
    gen = kb._new_node("General borrowing advice", "concept", "", [1.0] + [0.0] * 7, [])
    kb.db.commit()
    kb.reload()
    return kb, gen


def main():
    tmp = tempfile.mkdtemp(prefix="ops01-")
    kb, gen_id = _fixture_kb(tmp)

    # ── gate 1: key-set identity (containment + non-suppression in one) ─────────
    req = [OP, UNKNOWN, OP, gen_id]                    # known + unknown + duplicate + out-of-region
    res = kb.annotate_ops(req)
    check("gate1: response keys ≡ requested id set",
          set(res["annotations"].keys()) == {OP, UNKNOWN, gen_id})
    check("gate1: unknown id present, bare", res["annotations"][UNKNOWN] == {"annotated": False})
    check("gate1: OUT-OF-REGION id never joins even though the node exists",
          res["annotations"][gen_id] == {"annotated": False})
    check("fixture1: hazard caveat attached with severity",
          res["annotations"][OP]["annotated"] is True
          and res["annotations"][OP]["caveats"] == [
              {"card_id": HAZ, "severity": "error",
               "title": "cannot borrow as mutable more than once"}])
    check("fixture1: reserved fields relayed inert",
          res["annotations"][OP]["rank"] is None
          and res["annotations"][OP]["anti_pattern_of"] == [])
    check("coverage reported, not gated", res["requested"] == 3 and res["joined"] == 1)

    # ── gate 5 (coding→general): a general card attached to the op never leaves ─
    cid, _ = kb.add_card(OP, title="general how-to", goal="g", steps=["s"], doc_id="rustdoc1")
    kb.db.commit()
    res_g = kb.annotate_ops([OP])
    check("gate5a: general-region card id does not appear in caveats",
          all(c["card_id"] == HAZ for c in res_g["annotations"][OP]["caveats"]))

    # ── gate 7: context_features are inert on the key set ───────────────────────
    with_f = kb.annotate_ops(req, {"receiver_type": "Vec<T>", "ownership": "ref_mut"})
    check("gate7: identical key set with and without context_features",
          set(with_f["annotations"].keys()) == set(res.keys() if False else res["annotations"].keys()))

    # ── gate 2: determinism ─────────────────────────────────────────────────────
    a = json.dumps(kb.annotate_ops(req), sort_keys=True)
    b = json.dumps(kb.annotate_ops(req), sort_keys=True)
    check("gate2: identical (request, graph_version) → identical bytes", a == b)
    check("gate2: graph_version is a content digest",
          kb.annotate_ops(req)["graph_version"].startswith("sha256:"))

    # ── gate 3: purity — the surface writes nothing ─────────────────────────────
    before = kb._raw.total_changes
    kb.annotate_ops(req, {"feature": "value"})
    check("gate3: zero db writes during ops_annotate", kb._raw.total_changes == before)

    # ── facet derivation (§4.3): domain facet comes from the id region ──────────
    kb.facetize()
    check("region rows derive domain:<tag> from the id prefix",
          kb.get_facets("node", OP).get("domain") == ["rust-coding"])
    check("general rows keep their own domain facets",
          kb.get_facets("node", gen_id).get("domain", []) != ["rust-coding"])

    # ── gate 5 (general→coding): conversational exclusion + explicit opt-in ─────
    excl = ["domain:rust-coding"]
    ans = query_mod.answer(kb, StubEmbedder(), "how do I push to a vec",
                           exclude_facets=excl)
    ids = {it.get("id") for it in ans.get("items", [])}
    check("gate5b: excluded region never enters the kb_ask pool",
          not any(str(i).startswith("rust:") for i in ids))
    ans2 = query_mod.answer(kb, StubEmbedder(), "how do I push to a vec",
                            facets={"domain": ["rust-coding"]}, exclude_facets=[])
    ids2 = {it.get("id") for it in ans2.get("items", [])}
    check("gate5b: explicit opt-in readmits the region", any(
        str(i).startswith("rust:") for i in ids2))

    # tool-layer opt-in arithmetic (naming the axis lifts only that exclusion)
    t = tools_mod.Tools(None, StubEmbedder(), _cfg(tmp), kb=kb)
    check("naming the axis lifts the exclusion",
          t._ask_exclusions({"domain": ["rust-coding"]}) == [])
    check("unrelated facets leave the exclusion standing",
          t._ask_exclusions({"time_frame": ["current"]}) == ["domain:rust-coding"])

    # ── §3.4 catalogue gating + request validation ───────────────────────────────
    names = [x["name"] for x in t.catalogue()["tools"]]
    check("ops_annotate advertised when a region is configured", "ops_annotate" in names)
    kb_bare = KB(_cfg(tempfile.mkdtemp(prefix="ops01b-"), ops_regions=[]))
    t_bare = tools_mod.Tools(None, StubEmbedder(), _cfg(tmp, ops_regions=[]), kb=kb_bare)
    check("fixture7: no regions → tool absent from the catalogue",
          "ops_annotate" not in [x["name"] for x in t_bare.catalogue()["tools"]])
    check("empty ops rejected", t.call("ops_annotate", {"ops": []})["ok"] is False)
    check("oversize batch rejected",
          t.call("ops_annotate", {"ops": ["x"] * 501})["ok"] is False)
    ok = t.call("ops_annotate", {"ops": [OP]})
    check("tool round-trip serves the annotation",
          ok["ok"] and json.loads(ok["result"])["annotations"][OP]["annotated"] is True)

    # ── gate 4: reserved fields survive a bundle round-trip ─────────────────────
    kb.db.execute("UPDATE nodes SET observed_count=7, validated_count=3, "
                  "last_observed=1234.5, conditional_rank=0.5, "
                  "anti_pattern_of=? WHERE id=?", (json.dumps([HAZ]), OP))
    kb.db.commit()
    out = bundles.split(_cfg(tmp), only={"rust-base"})
    bfile = out["rust-base"]["file"]
    tmp2 = tempfile.mkdtemp(prefix="ops01c-")
    cfg2 = _cfg(tmp2)
    KB(cfg2).close()                                    # create the fresh master schema
    bundles.import_bundle(cfg2, bfile, name="rust-base", trust="keep")
    kb2 = KB(cfg2)
    r = kb2.db.execute("SELECT observed_count, validated_count, last_observed, "
                       "conditional_rank, anti_pattern_of FROM nodes WHERE id=?",
                       (OP,)).fetchone()
    check("gate4: reserved learned fields survive export→import byte-identically",
          r is not None and r["observed_count"] == 7 and r["validated_count"] == 3
          and r["last_observed"] == 1234.5 and r["conditional_rank"] == 0.5
          and json.loads(r["anti_pattern_of"]) == [HAZ])
    res2 = kb2.annotate_ops([OP], None)
    check("gate4: annotate relays the populated learned fields verbatim",
          res2["annotations"][OP]["rank"] == 0.5
          and res2["annotations"][OP]["anti_pattern_of"] == [HAZ])
    check("learned-field change moves graph_version",
          res2["graph_version"] != res["graph_version"])

    # ── gate 6: neutrality — the engine holds no region values ──────────────────
    hits = []
    # word-start "rust" followed by region punctuation or quoted bare — NOT the
    # "rust" inside "trust"/"frustrate"
    pat = re.compile(r"(?<![A-Za-z])rust(?:[-:]|['\"])", re.IGNORECASE)
    for py in sorted(ENGINE_DIR.glob("*.py")):
        for i, line in enumerate(py.read_text().splitlines(), 1):
            if pat.search(line):
                hits.append(f"{py.name}:{i}")
    check("gate6: engine sources are region-value-free", hits == [])
    if hits:
        print("   leaked:", ", ".join(hits))

    if FAILED:
        print(f"\n{len(FAILED)} FAILED")
        raise SystemExit(1)
    print("\nALL OK")


if __name__ == "__main__":
    main()
