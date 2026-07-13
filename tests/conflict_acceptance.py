"""VINUR-CONF-01 §11 acceptance tests — byte-exact conformance.

Builds the §11.1 ruleset (workshop + travel in one graph), runs OPS / TRAV-1 / TRAV-2 /
TRAV-3, and compares the canonical output byte-for-byte against §11.3.  Per §11, the
harness rebuilds the ruleset, so ``ruleset_version`` is matched by ``^[0-9a-f]{64}$`` and
substituted into the expected strings; everything else must be byte-identical.

Run:  python tests/conflict_acceptance.py     (from the repo root; stdlib only)
"""
import json
import os
import re
import sqlite3
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from knowledgehost.conflict import Checker, ConfError, canonical_json, ensure_schema

CAVEAT = ("no_known_conflicts means only that no ratified conflict rule fired for the "
          "checked actions against the provided state under closed-world predicate "
          "assumptions; it is NOT a safety determination. Unrepresented interactions, "
          "absent state, and unresolved quantities are not excluded.")

RS = "0" * 64  # spec §11.3 records the concrete value; the harness substitutes the rebuilt one

EXPECTED_OPS = '{"checker_version":"1.0.0","ruleset_version":"' + RS + '","card_id":"card:ops.residue_removal","clearance":"conflicts_found","findings":[{"action":"act:apply_solvent_cleaner","edge_id":"E1","relation_type":"incompatible","disposition":"fire","reason":"triggered","severity":"severe","recommended_disposition":"warn_strong","mechanism":{"mechanism_id":"mech:polymer_attack","label":"Solvent attack on polycarbonate","explanation":"Aggressive solvents craze and embrittle polycarbonate on contact; risk of the housing cracking under subsequent load.","conditionality_class":"acute_competition"},"rationale":"Aggressive solvents attack polycarbonate housings on contact."},{"action":"act:apply_solvent_cleaner","edge_id":"E4","relation_type":"incompatible","disposition":"flag_for_human","reason":"indeterminate","severity":"caution","recommended_disposition":"human_review","mechanism":null,"rationale":"High-concentration solvent caution on stubborn residue; concentration-conditional."},{"action":"act:apply_solvent_cleaner","edge_id":"E5","relation_type":"antagonizes","disposition":"flag_for_human","reason":"unratified_rule","severity":"caution","recommended_disposition":"human_review","mechanism":null,"rationale":"Possible softening of fresh paint by solvent vapours (proposed, unreviewed)."}],"checked":{"actions":["act:apply_solvent_cleaner","act:apply_protective_coating"],"edges_consulted":["E1","E4","E5","E8"],"overrides_applied":[{"action":"act:apply_solvent_cleaner","edge_id":"E8","override_id":"O3","justification":"Antique-care caution handled by dedicated preparation guidance; generic advisory suppressed."}]},"coverage":{"caveat":"no_known_conflicts means only that no ratified conflict rule fired for the checked actions against the provided state under closed-world predicate assumptions; it is NOT a safety determination. Unrepresented interactions, absent state, and unresolved quantities are not excluded.","not_evaluated":[{"edge_id":"E4","reason":"indeterminate_condition"},{"edge_id":"E5","reason":"unratified_rule"}]}}'

EXPECTED_TRAV1 = '{"checker_version":"1.0.0","ruleset_version":"' + RS + '","card_id":"card:trav.mel_lhr","clearance":"conflicts_found","findings":[{"action":"act:board_intl_flight","edge_id":"E2","relation_type":"requires","disposition":"fire","reason":"triggered","severity":"prohibitive","recommended_disposition":"block","mechanism":{"mechanism_id":"mech:carrier_boarding_rule","label":"Carrier boarding validity rule","explanation":"Carriers deny boarding when document validity at travel date is below the destination\'s required margin.","conditionality_class":"threshold"},"rationale":"Schengen entry requires at least 3 months document validity beyond travel date."}],"checked":{"actions":["act:board_intl_flight"],"edges_consulted":["E2"],"overrides_applied":[]},"coverage":{"caveat":"' + CAVEAT + '","not_evaluated":[]}}'

EXPECTED_TRAV2 = '{"checker_version":"1.0.0","ruleset_version":"' + RS + '","card_id":"card:trav.mel_lhr","clearance":"no_known_conflicts","findings":[],"checked":{"actions":["act:board_intl_flight"],"edges_consulted":["E2"],"overrides_applied":[]},"coverage":{"caveat":"' + CAVEAT + '","not_evaluated":[]}}'

EXPECTED_TRAV3 = '{"checker_version":"1.0.0","ruleset_version":"' + RS + '","card_id":"card:trav.mel_lhr","clearance":"review_required","findings":[{"action":"act:board_intl_flight","edge_id":"E2","relation_type":"requires","disposition":"flag_for_human","reason":"indeterminate","severity":"prohibitive","recommended_disposition":"human_review","mechanism":{"mechanism_id":"mech:carrier_boarding_rule","label":"Carrier boarding validity rule","explanation":"Carriers deny boarding when document validity at travel date is below the destination\'s required margin.","conditionality_class":"threshold"},"rationale":"Schengen entry requires at least 3 months document validity beyond travel date."}],"checked":{"actions":["act:board_intl_flight"],"edges_consulted":["E2"],"overrides_applied":[]},"coverage":{"caveat":"' + CAVEAT + '","not_evaluated":[{"edge_id":"E2","reason":"indeterminate_condition"}]}}'


def j(expr) -> str:
    return json.dumps(expr, separators=(",", ":"))


def build_ruleset(path: str) -> None:
    conn = sqlite3.connect(path)
    ensure_schema(conn)
    nodes = [
        ("act:apply_solvent_cleaner", "apply solvent cleaner", "action"),
        ("act:apply_aggressive_solvent", "apply an aggressive solvent", "class"),
        ("act:apply_protective_coating", "apply a protective coating", "action"),
        ("act:apply_surface_finish", "apply a surface finish", "class"),
        ("act:board_intl_flight", "board an international flight", "action"),
        ("state:stubborn_residue", "stubborn surface residue", "predicate"),
        ("state:polycarbonate_housing", "polycarbonate housing", "predicate"),
        ("state:recently_painted", "recently painted surface", "predicate"),
        ("state:antique", "antique piece", "predicate"),
        ("dest:schengen", "Schengen-area destination", "predicate"),
    ]
    conn.executemany("INSERT INTO conflict_node(node_id,label,kind) VALUES(?,?,?)", nodes)
    conn.executemany("INSERT INTO conflict_is_a(child,parent,status) VALUES(?,?,'ratified')", [
        ("act:apply_solvent_cleaner", "act:apply_aggressive_solvent"),
        ("act:apply_protective_coating", "act:apply_surface_finish"),
    ])
    conn.executemany(
        "INSERT INTO mechanism(mechanism_id,label,explanation,conditionality_class) VALUES(?,?,?,?)", [
            ("mech:polymer_attack", "Solvent attack on polycarbonate",
             "Aggressive solvents craze and embrittle polycarbonate on contact; risk of the "
             "housing cracking under subsequent load.",
             "acute_competition"),
            ("mech:carrier_boarding_rule", "Carrier boarding validity rule",
             "Carriers deny boarding when document validity at travel date is below the "
             "destination's required margin.", "threshold"),
        ])
    edges = [
        ("E1", "act:apply_aggressive_solvent", "incompatible", "severe",
         j({"op": "presence", "pred": "state:polycarbonate_housing"}),
         "mech:polymer_attack", "ratified", "pub",
         "Aggressive solvents attack polycarbonate housings on contact.", None),
        ("E4", "act:apply_solvent_cleaner", "incompatible", "caution",
         j({"op": "all_of", "args": [
             {"op": "presence", "pred": "state:stubborn_residue"},
             {"op": "compare", "field": "solvent_concentration_pct", "cmp": ">",
              "operand": {"lit": 50}}]}),
         None, "ratified", "pub",
         "High-concentration solvent caution on stubborn residue; concentration-conditional.", None),
        ("E5", "act:apply_aggressive_solvent", "antagonizes", "caution",
         j({"op": "presence", "pred": "state:recently_painted"}),
         None, "proposed", "pub",
         "Possible softening of fresh paint by solvent vapours (proposed, unreviewed).", None),
        ("E8", "act:apply_aggressive_solvent", "incompatible", "advisory",
         j({"op": "presence", "pred": "state:antique"}),
         None, "ratified", "pub",
         "Advisory caution for aggressive solvents on antique pieces.", None),
        ("E2", "act:board_intl_flight", "requires", "prohibitive",
         j({"op": "all_of", "args": [
             {"op": "presence", "pred": "dest:schengen"},
             {"op": "compare", "field": "passport_validity_months_at_travel", "cmp": "<",
              "operand": {"lit": 3}}]}),
         "mech:carrier_boarding_rule", "ratified", "pub",
         "Schengen entry requires at least 3 months document validity beyond travel date.", None),
    ]
    conn.executemany(
        "INSERT INTO conflict_edge(edge_id,subject,relation_type,severity,fire_when,"
        "mechanism_id,status,authority,rationale,source_ref) VALUES(?,?,?,?,?,?,?,?,?,?)", edges)
    conn.execute(
        "INSERT INTO conflict_override(override_id,on_node,targets_edge_id,status,justification,"
        "source_ref) VALUES(?,?,?,?,?,?)",
        ("O3", "act:apply_solvent_cleaner", "E8", "ratified",
         "Antique-care caution handled by dedicated preparation guidance; generic advisory suppressed.",
         None))
    conn.commit()
    conn.close()


def run_case(checker, name, card, state, expected):
    out = canonical_json(checker.check(card, state))
    want = expected.replace(RS, checker.ruleset_version)
    if out != want:
        print(f"FAIL {name}")
        print("  got : " + out)
        print("  want: " + want)
        # first divergence, for fast diagnosis
        for i, (a, b) in enumerate(zip(out, want)):
            if a != b:
                print(f"  first divergence at byte {i}: got {out[i:i+60]!r} want {want[i:i+60]!r}")
                break
        sys.exit(1)
    print(f"PASS {name}  ({len(out)} bytes, byte-exact)")


def main():
    with tempfile.TemporaryDirectory() as td:
        db = os.path.join(td, "conf.db")
        build_ruleset(db)
        checker = Checker.load(db)

        assert re.fullmatch(r"[0-9a-f]{64}", checker.ruleset_version), "ruleset_version format"
        print(f"ruleset_version = {checker.ruleset_version}  (harness-rebuilt; format-matched per §11)")

        ops_card = {"card_id": "card:ops.residue_removal",
                    "actions": ["act:apply_solvent_cleaner", "act:apply_protective_coating"]}
        ops_state = {"predicates": ["state:stubborn_residue",
                                    "state:polycarbonate_housing",
                                    "state:recently_painted", "state:antique"], "fields": {}}
        trav_card = {"card_id": "card:trav.mel_lhr", "actions": ["act:board_intl_flight"]}

        run_case(checker, "OPS   ", ops_card, ops_state, EXPECTED_OPS)
        run_case(checker, "TRAV-1", trav_card,
                 {"predicates": ["dest:schengen"],
                  "fields": {"passport_validity_months_at_travel": 2.47}}, EXPECTED_TRAV1)
        run_case(checker, "TRAV-2", trav_card,
                 {"predicates": ["dest:schengen"],
                  "fields": {"passport_validity_months_at_travel": 9.0}}, EXPECTED_TRAV2)
        run_case(checker, "TRAV-3", trav_card,
                 {"predicates": ["dest:schengen"], "fields": {}}, EXPECTED_TRAV3)

        # ── determinism (§9): repeat run and a fresh load are byte-identical ──
        a = canonical_json(checker.check(ops_card, ops_state))
        b = canonical_json(Checker.load(db).check(ops_card, ops_state))
        assert a == b, "determinism across runs/loads"
        print("PASS determinism (repeat check + fresh load byte-identical)")

        # ── §9 (1.1/1.2 errata): the scope-STRUCTURAL tables are hashed too ──
        # is_a / uses / member_of / acts_via all change what fires, so any row
        # change must move ruleset_version — including a PROPOSED is_a row that
        # changes no behaviour (the audit trail must move even when firing
        # doesn't).  Reverting must restore the version exactly.
        base = checker.ruleset_version
        for table, ins, undo in [
            ("conflict_is_a (proposed)",
             "INSERT INTO conflict_is_a(child,parent,status) VALUES("
             "'act:apply_solvent_cleaner','act:apply_surface_finish','proposed')",
             "DELETE FROM conflict_is_a WHERE parent='act:apply_surface_finish' "
             "AND child='act:apply_solvent_cleaner'"),
            ("uses",
             "INSERT INTO uses(action,resource) VALUES("
             "'act:apply_solvent_cleaner','act:apply_aggressive_solvent')",
             "DELETE FROM uses"),
            ("member_of",
             "INSERT INTO member_of(child,grouper,grouper_type) VALUES("
             "'act:apply_solvent_cleaner','act:apply_aggressive_solvent','class')",
             "DELETE FROM member_of"),
            ("acts_via",
             "INSERT INTO acts_via(resource,mechanism,role) VALUES("
             "'act:apply_solvent_cleaner','act:apply_aggressive_solvent','mechanism')",
             "DELETE FROM acts_via"),
        ]:
            conn = sqlite3.connect(db); conn.execute(ins); conn.commit(); conn.close()
            moved = Checker.load(db).ruleset_version
            assert moved != base, f"{table}: row change did not move ruleset_version (the 1.0 audit gap)"
            conn = sqlite3.connect(db); conn.execute(undo); conn.commit(); conn.close()
            restored = Checker.load(db).ruleset_version
            assert restored == base, f"{table}: revert did not restore ruleset_version"
        print("PASS §9/1.1 scope-structural rows move ruleset_version (revert restores it)")

        # ── §10 error contract ──
        for bad_card, bad_state, code in [
            ({"card_id": "c", "actions": ["act:nonexistent"]},
             {"predicates": [], "fields": {}}, "E_UNKNOWN_NODE"),
            ({"card_id": "c", "actions": ["act:board_intl_flight"]},
             {"predicates": "oops", "fields": {}}, "E_MALFORMED_STATE"),
            ({"card_id": "c", "actions": ["act:board_intl_flight"]},
             {"predicates": [], "fields": {"x": True}}, "E_MALFORMED_STATE"),
            ({"card_id": "c", "actions": "act:board_intl_flight"},
             {"predicates": [], "fields": {}}, "E_MALFORMED_STATE"),
        ]:
            try:
                checker.check(bad_card, bad_state)
                print(f"FAIL error contract: expected {code}"); sys.exit(1)
            except ConfError as e:
                assert e.code == code, f"expected {code}, got {e.code}"
        print("PASS §10 error contract (E_UNKNOWN_NODE / E_MALFORMED_STATE)")

        # depth>4 expression must be rejected at LOAD (fail loud, never silently skipped)
        db2 = os.path.join(td, "bad.db")
        conn = sqlite3.connect(db2); ensure_schema(conn)
        conn.execute("INSERT INTO conflict_node(node_id) VALUES('act:x')")
        deep = {"op": "not", "arg": {"op": "not", "arg": {"op": "not", "arg": {
                "op": "not", "arg": {"op": "presence", "pred": "p"}}}}}
        conn.execute("INSERT INTO conflict_edge VALUES('B1','act:x','requires','caution',?,"
                     "NULL,'ratified','pub','r',NULL)", (json.dumps(deep),))
        conn.commit(); conn.close()
        try:
            Checker.load(db2)
            print("FAIL: depth-5 expression accepted"); sys.exit(1)
        except ConfError as e:
            assert e.code == "E_BAD_EXPRESSION"
        print("PASS E_BAD_EXPRESSION at load (depth bound enforced)")

        # a PROPOSED override must not suppress (spec §7 K3)
        db3 = os.path.join(td, "prop.db")
        conn = sqlite3.connect(db3); ensure_schema(conn)
        conn.executemany("INSERT INTO conflict_node(node_id) VALUES(?)", [("act:y",), ("state:p",)])
        conn.execute("INSERT INTO conflict_edge VALUES('X1','act:y','incompatible','severe',?,"
                     "NULL,'ratified','pub','r',NULL)",
                     (j({"op": "presence", "pred": "state:p"}),))
        conn.execute("INSERT INTO conflict_override VALUES('OV1','act:y','X1','proposed','j',NULL)")
        conn.commit(); conn.close()
        out = Checker.load(db3).check({"card_id": "c", "actions": ["act:y"]},
                                      {"predicates": ["state:p"], "fields": {}})
        assert out["clearance"] == "conflicts_found" and not out["checked"]["overrides_applied"], \
            "proposed override must not suppress"
        print("PASS proposed override does not suppress")

        print("\nALL ACCEPTANCE TESTS PASS — VINUR-CONF-01 §11 conformant")


if __name__ == "__main__":
    main()
