"""VINUR-LEX-01 §11 acceptance tests — byte-exact conformance.

Compiles the §11.1 test lexicon against the canon registry, then verifies vectors
V1–V8 byte-for-byte against the §8.3 canonical serialization (lexicon_version matched
by ^[0-9a-f]{64}$ and substituted, per §11).  Also: the §4 tokenizer consequences, the
§6.3 compiler validation (C1–C4, EMPTY_NORM, warns; fail ⇒ report only), the §9 error
contract, and determinism across repeat matches and a fresh load.

Run:  python tests/lex_acceptance.py     (from the repo root; stdlib only)
"""
import json
import os
import re
import sqlite3
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from knowledgehost.conflict import ensure_schema
from knowledgehost.lex import LexError, Matcher, compile_lexicon, norm_token, tokenize

US = "\x1f"
LV = "LVLVLVLVLVLVLVLVLVLVLVLVLVLVLVLVLVLVLVLVLVLVLVLVLVLVLVLVLVLVLVLV"  # placeholder

ALIAS_DDL = """CREATE TABLE alias (
  alias_id      INTEGER PRIMARY KEY,
  node_id       TEXT    NOT NULL,
  surface       TEXT    NOT NULL,
  norm_seq      TEXT    NOT NULL,
  n_tokens      INTEGER NOT NULL CHECK (n_tokens BETWEEN 1 AND 8),
  alias_type    TEXT    NOT NULL CHECK (alias_type IN
                  ('preferred','synonym','index_term','variant','abbrev','informal','inflection')),
  weight        REAL    NOT NULL CHECK (weight > 0.0 AND weight <= 1.0),
  case_mode     TEXT    NOT NULL DEFAULT 'fold' CHECK (case_mode IN ('fold','exact','caps')),
  fuzzy_allowed INTEGER NOT NULL DEFAULT 1 CHECK (fuzzy_allowed IN (0, 1)),
  origin        TEXT    NOT NULL,
  derived_from  INTEGER REFERENCES alias(alias_id),
  status        TEXT    NOT NULL DEFAULT 'active' CHECK (status IN ('active','retired'))
)"""

LEXICON = [  # (alias_id, node_id, surface, alias_type, weight, case_mode, fuzzy_allowed)
    (1, "pub:proc.galvanising", "galvanising", "preferred", 1.00, "fold", 1),
    (2, "pub:proc.galvanising", "galvanizing", "variant", 0.85, "fold", 1),   # -s/-z spelling variant
    (3, "pub:defect.brittle_fracture", "brittle fracture", "preferred", 1.00, "fold", 1),
    (4, "pub:defect.fracture", "fracture", "preferred", 1.00, "fold", 1),
    (5, "pub:mat.polyethylene", "PE", "abbrev", 0.80, "caps", 0),
    (6, "pub:phys.potential_energy", "PE", "abbrev", 0.80, "caps", 0),
    (7, "pub:alloy.inconel", "inconel", "preferred", 1.00, "fold", 0),  # look-alike w/ 'incoloy': no fuzzy
    (8, "pub:test.brinell_hardness", "Brinell hardness", "preferred", 1.00, "fold", 1),
    (9, "pub:defect.cold_work_marks", "cold work marks", "informal", 0.70, "fold", 1),
]

V1 = ("{\"matcher_version\":\"1.0.0\",\"lexicon_version\":\"LVLVLVLVLVLVLVLVLVLVLVLVLVLVLVLVLVLVLVLVLVLVLVLVLVLVLVLVLVLVLVLV\",\"norm_version\":\"NORM-1\",\"tok_version\":\"TOK-1\",\"text\":\"Cold-work marks after galvanising.\",\"tokens\":[{\"i\":0,\"surface\":\"Cold\",\"norm\":\"cold\",\"char_start\":0,\"char_end\":4,\"corrected_from\":null,\"edit_distance\":0},{\"i\":1,\"surface\":\"work\",\"norm\":\"work\",\"char_start\":5,\"char_end\":9,\"corrected_from\":null,\"edit_distance\":0},{\"i\":2,\"surface\":\"marks\",\"norm\":\"marks\",\"char_start\":10,\"char_end\":15,\"corrected_from\":null,\"edit_distance\":0},{\"i\":3,\"surface\":\"after\",\"norm\":\"after\",\"char_start\":16,\"char_end\":21,\"corrected_from\":null,\"edit_distance\":0},{\"i\":4,\"surface\":\"galvanising\",\"norm\":\"galvanising\",\"char_start\":22,\"char_end\":33,\"corrected_from\":null,\"edit_distance\":0}],\"spans\":[{\"span_id\":0,\"tok_start\":0,\"tok_end\":3,\"char_start\":0,\"char_end\":15,\"surface_original\":\"Cold-work marks\",\"matched_norm\":\"cold work marks\",\"fuzzy\":false,\"candidates\":[{\"alias_id\":9,\"node_id\":\"pub:defect.cold_work_marks\",\"alias_type\":\"informal\",\"weight\":0.7000,\"score\":0.7000}]},{\"span_id\":1,\"tok_start\":4,\"tok_end\":5,\"char_start\":22,\"char_end\":33,\"surface_original\":\"galvanising\",\"matched_norm\":\"galvanising\",\"fuzzy\":false,\"candidates\":[{\"alias_id\":1,\"node_id\":\"pub:proc.galvanising\",\"alias_type\":\"preferred\",\"weight\":1.0000,\"score\":1.0000}]}],\"unmatched_token_indices\":[3],\"flags\":[]}")

V2 = ("{\"matcher_version\":\"1.0.0\",\"lexicon_version\":\"LVLVLVLVLVLVLVLVLVLVLVLVLVLVLVLVLVLVLVLVLVLVLVLVLVLVLVLVLVLVLVLV\",\"norm_version\":\"NORM-1\",\"tok_version\":\"TOK-1\",\"text\":\"Query PE.\",\"tokens\":[{\"i\":0,\"surface\":\"Query\",\"norm\":\"query\",\"char_start\":0,\"char_end\":5,\"corrected_from\":null,\"edit_distance\":0},{\"i\":1,\"surface\":\"PE\",\"norm\":\"pe\",\"char_start\":6,\"char_end\":8,\"corrected_from\":null,\"edit_distance\":0}],\"spans\":[{\"span_id\":0,\"tok_start\":1,\"tok_end\":2,\"char_start\":6,\"char_end\":8,\"surface_original\":\"PE\",\"matched_norm\":\"pe\",\"fuzzy\":false,\"candidates\":[{\"alias_id\":5,\"node_id\":\"pub:mat.polyethylene\",\"alias_type\":\"abbrev\",\"weight\":0.8000,\"score\":0.8000},{\"alias_id\":6,\"node_id\":\"pub:phys.potential_energy\",\"alias_type\":\"abbrev\",\"weight\":0.8000,\"score\":0.8000}]}],\"unmatched_token_indices\":[0],\"flags\":[]}")

V3 = ("{\"matcher_version\":\"1.0.0\",\"lexicon_version\":\"LVLVLVLVLVLVLVLVLVLVLVLVLVLVLVLVLVLVLVLVLVLVLVLVLVLVLVLVLVLVLVLV\",\"norm_version\":\"NORM-1\",\"tok_version\":\"TOK-1\",\"text\":\"made of pe\",\"tokens\":[{\"i\":0,\"surface\":\"made\",\"norm\":\"made\",\"char_start\":0,\"char_end\":4,\"corrected_from\":null,\"edit_distance\":0},{\"i\":1,\"surface\":\"of\",\"norm\":\"of\",\"char_start\":5,\"char_end\":7,\"corrected_from\":null,\"edit_distance\":0},{\"i\":2,\"surface\":\"pe\",\"norm\":\"pe\",\"char_start\":8,\"char_end\":10,\"corrected_from\":null,\"edit_distance\":0}],\"spans\":[],\"unmatched_token_indices\":[0,1,2],\"flags\":[]}")

V4 = ("{\"matcher_version\":\"1.0.0\",\"lexicon_version\":\"LVLVLVLVLVLVLVLVLVLVLVLVLVLVLVLVLVLVLVLVLVLVLVLVLVLVLVLVLVLVLVLV\",\"norm_version\":\"NORM-1\",\"tok_version\":\"TOK-1\",\"text\":\"hot galvenising\",\"tokens\":[{\"i\":0,\"surface\":\"hot\",\"norm\":\"hot\",\"char_start\":0,\"char_end\":3,\"corrected_from\":null,\"edit_distance\":0},{\"i\":1,\"surface\":\"galvenising\",\"norm\":\"galvanising\",\"char_start\":4,\"char_end\":15,\"corrected_from\":\"galvenising\",\"edit_distance\":1}],\"spans\":[{\"span_id\":0,\"tok_start\":1,\"tok_end\":2,\"char_start\":4,\"char_end\":15,\"surface_original\":\"galvenising\",\"matched_norm\":\"galvanising\",\"fuzzy\":true,\"candidates\":[{\"alias_id\":1,\"node_id\":\"pub:proc.galvanising\",\"alias_type\":\"preferred\",\"weight\":1.0000,\"score\":0.8000}]}],\"unmatched_token_indices\":[0],\"flags\":[]}")

V5 = ("{\"matcher_version\":\"1.0.0\",\"lexicon_version\":\"LVLVLVLVLVLVLVLVLVLVLVLVLVLVLVLVLVLVLVLVLVLVLVLVLVLVLVLVLVLVLVLV\",\"norm_version\":\"NORM-1\",\"tok_version\":\"TOK-1\",\"text\":\"cast inconl\",\"tokens\":[{\"i\":0,\"surface\":\"cast\",\"norm\":\"cast\",\"char_start\":0,\"char_end\":4,\"corrected_from\":null,\"edit_distance\":0},{\"i\":1,\"surface\":\"inconl\",\"norm\":\"inconl\",\"char_start\":5,\"char_end\":11,\"corrected_from\":null,\"edit_distance\":0}],\"spans\":[],\"unmatched_token_indices\":[0,1],\"flags\":[{\"type\":\"fuzzy_suppressed\",\"stage\":\"token\",\"token_index\":1,\"nearest\":\"inconel\",\"distance\":1}]}")

V6 = ("{\"matcher_version\":\"1.0.0\",\"lexicon_version\":\"LVLVLVLVLVLVLVLVLVLVLVLVLVLVLVLVLVLVLVLVLVLVLVLVLVLVLVLVLVLVLVLV\",\"norm_version\":\"NORM-1\",\"tok_version\":\"TOK-1\",\"text\":\"brittle fracture observed\",\"tokens\":[{\"i\":0,\"surface\":\"brittle\",\"norm\":\"brittle\",\"char_start\":0,\"char_end\":7,\"corrected_from\":null,\"edit_distance\":0},{\"i\":1,\"surface\":\"fracture\",\"norm\":\"fracture\",\"char_start\":8,\"char_end\":16,\"corrected_from\":null,\"edit_distance\":0},{\"i\":2,\"surface\":\"observed\",\"norm\":\"observed\",\"char_start\":17,\"char_end\":25,\"corrected_from\":null,\"edit_distance\":0}],\"spans\":[{\"span_id\":0,\"tok_start\":0,\"tok_end\":2,\"char_start\":0,\"char_end\":16,\"surface_original\":\"brittle fracture\",\"matched_norm\":\"brittle fracture\",\"fuzzy\":false,\"candidates\":[{\"alias_id\":3,\"node_id\":\"pub:defect.brittle_fracture\",\"alias_type\":\"preferred\",\"weight\":1.0000,\"score\":1.0000}]}],\"unmatched_token_indices\":[2],\"flags\":[]}")

V7 = ("{\"matcher_version\":\"1.0.0\",\"lexicon_version\":\"LVLVLVLVLVLVLVLVLVLVLVLVLVLVLVLVLVLVLVLVLVLVLVLVLVLVLVLVLVLVLVLV\",\"norm_version\":\"NORM-1\",\"tok_version\":\"TOK-1\",\"text\":\"Brinell's hardness\",\"tokens\":[{\"i\":0,\"surface\":\"Brinell's\",\"norm\":\"brinell\",\"char_start\":0,\"char_end\":9,\"corrected_from\":null,\"edit_distance\":0},{\"i\":1,\"surface\":\"hardness\",\"norm\":\"hardness\",\"char_start\":10,\"char_end\":18,\"corrected_from\":null,\"edit_distance\":0}],\"spans\":[{\"span_id\":0,\"tok_start\":0,\"tok_end\":2,\"char_start\":0,\"char_end\":18,\"surface_original\":\"Brinell's hardness\",\"matched_norm\":\"brinell hardness\",\"fuzzy\":false,\"candidates\":[{\"alias_id\":8,\"node_id\":\"pub:test.brinell_hardness\",\"alias_type\":\"preferred\",\"weight\":1.0000,\"score\":1.0000}]}],\"unmatched_token_indices\":[],\"flags\":[]}")

V8 = ("{\"matcher_version\":\"1.0.0\",\"lexicon_version\":\"LVLVLVLVLVLVLVLVLVLVLVLVLVLVLVLVLVLVLVLVLVLVLVLVLVLVLVLVLVLVLVLV\",\"norm_version\":\"NORM-1\",\"tok_version\":\"TOK-1\",\"text\":\"\",\"tokens\":[],\"spans\":[],\"unmatched_token_indices\":[],\"flags\":[]}")


def alias_row(aid, node, surface, atype, weight, cmode, fuzzy):
    toks = tokenize(surface)
    norms = [norm_token(t["surface"]) for t in toks]
    return (aid, node, surface, US.join(norms), len(toks), atype, weight, cmode,
            fuzzy, "pub", None, "active")


def make_db(path, rows, nodes):
    conn = sqlite3.connect(path)
    ensure_schema(conn)
    conn.executemany("INSERT INTO conflict_node(node_id,label,kind) VALUES(?,?,'concept')",
                     [(n, n) for n in nodes])
    conn.execute(ALIAS_DDL)
    conn.executemany("INSERT INTO alias VALUES(?,?,?,?,?,?,?,?,?,?,?,?)", rows)
    conn.commit()
    conn.close()


def run_vector(m, name, text, expected):
    got = m.match_json(text).decode("utf-8")
    want = expected.replace(LV, m.lexicon_version)
    if got != want:
        print(f"FAIL {name}")
        for i, (a, b) in enumerate(zip(got, want)):
            if a != b:
                print(f"  first divergence at byte {i}: got {got[i:i+70]!r} want {want[i:i+70]!r}")
                break
        else:
            print(f"  length differs: got {len(got)} want {len(want)}")
        sys.exit(1)
    print(f"PASS {name}  ({len(got)} bytes, byte-exact)")


def main():
    # §4 normative tokenizer consequences
    assert [t["surface"] for t in tokenize("Cold-work")] == ["Cold", "work"]
    assert [t["surface"] for t in tokenize("0.5")] == ["0.5"]
    assert [t["surface"] for t in tokenize("1,000")] == ["1,000"]
    assert [t["surface"] for t in tokenize(".5")] == ["5"]
    assert [t["surface"] for t in tokenize("H2O/CO2")] == ["H2O", "CO2"]
    assert [t["surface"] for t in tokenize("Brinell's")] == ["Brinell's"]
    assert norm_token("Brinell's") == "brinell" and norm_token("'s") == ""
    print("PASS §4/§3 tokenizer + normalizer consequences")

    with tempfile.TemporaryDirectory() as td:
        db = os.path.join(td, "lex.db")
        art = os.path.join(td, "artifacts")
        nodes = sorted({r[1] for r in LEXICON})
        make_db(db, [alias_row(*r) for r in LEXICON], nodes)

        rep = compile_lexicon(db, art)
        assert rep["ok"] and not rep["findings"], rep["findings"]
        assert re.fullmatch(r"[0-9a-f]{64}", rep["lexicon_version"])
        print(f"PASS compile (9 aliases, vocab {rep['counts']['vocab_size']}, no findings)")

        m = Matcher.load(art)
        assert m.lexicon_version == rep["lexicon_version"]

        run_vector(m, "V1 multi-token + hyphen", "Cold-work marks after galvanising.", V1)
        run_vector(m, "V2 ambiguous abbrev    ", "Query PE.", V2)
        run_vector(m, "V3 caps case fails     ", "made of pe", V3)
        run_vector(m, "V4 fuzzy corrected     ", "hot galvenising", V4)
        run_vector(m, "V5 fuzzy suppressed    ", "cast inconl", V5)
        run_vector(m, "V6 leftmost-longest    ", "brittle fracture observed", V6)
        run_vector(m, "V7 possessive          ", "Brinell's hardness", V7)
        run_vector(m, "V8 empty input         ", "", V8)

        # §8.2 determinism: repeat + fresh load byte-identical
        a = m.match_json("Cold-work marks after galvanising.")
        b = Matcher.load(art).match_json("Cold-work marks after galvanising.")
        assert a == b
        print("PASS determinism (repeat match + fresh load byte-identical)")

        # §9 matcher error contract
        for bad, code in [(123, "E_INVALID_INPUT"), ("x\ud800y", "E_INVALID_INPUT"),
                          ("a" * 4097, "E_INPUT_TOO_LONG")]:
            try:
                m.match(bad)
                print(f"FAIL: expected {code}"); sys.exit(1)
            except LexError as e:
                assert e.code == code, f"got {e.code}, want {code}"
        try:
            Matcher.load(os.path.join(td, "nowhere"))
            print("FAIL: expected E_ARTIFACT_MISSING"); sys.exit(1)
        except LexError as e:
            assert e.code == "E_ARTIFACT_MISSING"
        meta_path = os.path.join(art, "lexicon.meta.json")
        meta = json.loads(open(meta_path, encoding="utf-8").read())
        meta["norm_version"] = "NORM-0"
        open(meta_path, "w", encoding="utf-8").write(json.dumps(meta))
        try:
            Matcher.load(art)
            print("FAIL: expected E_ARTIFACT_MISMATCH"); sys.exit(1)
        except LexError as e:
            assert e.code == "E_ARTIFACT_MISMATCH"
        print("PASS §9 error contract (invalid input / too long / artifact missing+mismatch)")

        # §6.3 compiler validation: collect ALL findings; fail ⇒ report only, no artifacts
        db2 = os.path.join(td, "bad.db")
        art2 = os.path.join(td, "artifacts2")
        bad_nodes = ["pub:x1"] + [f"pub:a{i}" for i in range(1, 7)]
        bad_rows = [
            alias_row(201, "pub:x1", "pe", "abbrev", 0.8, "fold", 1),          # C1
            alias_row(202, "pub:x1", "the", "synonym", 0.9, "fold", 1),        # C2
            alias_row(203, "pub:ghost", "ghostterm", "preferred", 1.0, "fold", 1),  # C3
            (204, "pub:x1", "flange", "wrongseq", 1, "preferred", 1.0, "fold", 1,
             "pub", None, "active"),                                           # C4
            alias_row(205, "pub:x1", "'s", "variant", 0.5, "fold", 1),         # EMPTY_NORM (+C1)
        ] + [alias_row(210 + i, f"pub:a{i}", "amb", "synonym", 0.9, "fold", 1)
             for i in range(1, 6)]                                             # AMBIGUOUS_NORM
        bad_rows.append((216, "pub:a6", "amb", "amb", 1, "synonym", 0.9, "fold", 1,
                         "user", None, "active"))                              # ORIGIN_SHADOW
        make_db(db2, bad_rows, bad_nodes)
        rep2 = compile_lexicon(db2, art2)
        codes = {(f["level"], f["code"]) for f in rep2["findings"]}
        for want in [("ERROR", "C1"), ("ERROR", "C2"), ("ERROR", "C3"), ("ERROR", "C4"),
                     ("ERROR", "EMPTY_NORM"), ("WARN", "AMBIGUOUS_NORM"), ("WARN", "ORIGIN_SHADOW")]:
            assert want in codes, f"missing finding {want}: {sorted(codes)}"
        assert not rep2["ok"]
        assert os.path.exists(os.path.join(art2, "compile_report.json"))
        assert not os.path.exists(os.path.join(art2, "lexicon.json")), \
            "artifacts must not be written when ERRORs exist"
        print("PASS §6.3 compiler validation (C1–C4 + EMPTY_NORM errors, warns, report-only on fail)")

        print("\nALL ACCEPTANCE TESTS PASS — VINUR-LEX-01 §11 conformant")


if __name__ == "__main__":
    main()
