"""VINUR-CONF-01 — conflict & dependency relations with mechanism routing.

The inverse polarity of retrieval: a **hit is bad news**.  ``Checker.check`` looks for a
ratified reason to veto or caution each action a card names, evaluated against a supplied
state.  Runs after ranking, before presentation, as a default-deny veto layer.  Pure graph
evaluation — no ML, no network, no LM at runtime (spec §2).

Representational commitments (spec §1):
  * a typed relation family (incompatible / requires / mutually_exclusive / antagonizes),
    NOT one generic "conflict" scalar — dependency danger lives in an *absent* precondition;
  * mechanism routing — the "why" is a shared node cited by many edges, never duplicated
    into cards.

Safety posture, mechanical form: an unratified edge can *raise* a review but can never
*clear* an action; a field that cannot be evaluated escalates (`INDETERMINATE`), never
clears; `no_known_conflicts` is not a safety determination and always carries the caveat
verbatim (§8.3).  The checker cannot emit "safe"/"cleared"/"approved" — deliberately.

Node identity: subjects, predicates and mechanisms use the **canon registry**
(``conflict_node``) — namespaced, human-stable IDs (``act:…``, ``state:…``, ``mech:…``),
NOT the distilled graph's content-hash IDs, so ratified edges never churn when a distilled
label is rephrased.  The subject-side ancestor-walk runs only on **ratified ``conflict_is_a``
rows local to this layer** (curated; never imported/statistical is_a).

Interpretation notes (where the spec leaves latitude):
  * K0 "every referenced node MUST exist" — enforced for card actions at check time and for
    edge subjects / override on_nodes / is_a endpoints at load time.  State *predicates* are
    an open vocabulary (state derivation is out of scope, §2) and are matched as opaque IDs.
  * An override suppresses a non-discarded edge BEFORE K4 classification (K3), so a
    suppressed INDETERMINATE edge yields an overrides_applied entry and no not_evaluated row.
  * ``not_evaluated`` entries are deduplicated on (edge_id, reason).
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path

CHECKER_VERSION = "1.0.0"
# 1.1 erratum (§9): the hash also covers the scope-STRUCTURAL tables
# (conflict_is_a, uses, member_of, acts_via) — they all affect what fires,
# so changing them must change ruleset_version.  At 1.0 only
# edges/overrides/mechanisms were hashed; the is_a walk already made that a
# gap, and the §4 scope linkage widened it.
# 1.2 erratum: domain-neutral vocabulary — relation 'incompatible' (was
# 'contraindicated'), table 'uses' (was 'administers'), column 'resource'
# (was 'substance'), grouper inheritance generalized to class_grouper_types.
_ALGO = "VINUR-CONF-01/1.2"

RELATION_TYPES = ("incompatible", "requires", "mutually_exclusive", "antagonizes")
SEVERITIES = ("advisory", "caution", "severe", "prohibitive")
_SEV_RANK = {s: i + 1 for i, s in enumerate(SEVERITIES)}
_FIRE_DISPOSITION = {"prohibitive": "block", "severe": "warn_strong",
                     "caution": "warn", "advisory": "note"}
_CMPS = ("<", "<=", ">", ">=", "==")
_MAX_DEPTH = 4

CAVEAT = ("no_known_conflicts means only that no ratified conflict rule fired for the "
          "checked actions against the provided state under closed-world predicate "
          "assumptions; it is NOT a safety determination. Unrepresented interactions, "
          "absent state, and unresolved quantities are not excluded.")


class ConfError(Exception):
    """§10 error contract: E_UNKNOWN_NODE | E_BAD_EXPRESSION | E_MALFORMED_STATE."""

    def __init__(self, code: str, message: str):
        self.code = code
        self.message = message
        super().__init__(f"{code}: {message}")


# ── schema (§4 DDL verbatim + the canon registry and curated is_a) ─────────────────────
SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS mechanism (
  mechanism_id        TEXT PRIMARY KEY,
  label               TEXT NOT NULL,
  explanation         TEXT NOT NULL,
  conditionality_class TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS conflict_edge (
  edge_id       TEXT PRIMARY KEY,
  subject       TEXT NOT NULL,
  relation_type TEXT NOT NULL CHECK (relation_type IN
                  ('incompatible','requires','mutually_exclusive','antagonizes')),
  severity      TEXT NOT NULL CHECK (severity IN ('advisory','caution','severe','prohibitive')),
  fire_when     TEXT NOT NULL,
  mechanism_id  TEXT REFERENCES mechanism(mechanism_id),
  status        TEXT NOT NULL CHECK (status IN ('ratified','proposed','deprecated')),
  authority     TEXT NOT NULL,
  rationale     TEXT NOT NULL,
  source_ref    TEXT
);
CREATE INDEX IF NOT EXISTS idx_edge_subject ON conflict_edge (subject)
  WHERE status <> 'deprecated';
CREATE TABLE IF NOT EXISTS conflict_override (
  override_id      TEXT PRIMARY KEY,
  on_node          TEXT NOT NULL,
  targets_edge_id  TEXT NOT NULL REFERENCES conflict_edge(edge_id),
  status           TEXT NOT NULL CHECK (status IN ('ratified','proposed')),
  justification    TEXT NOT NULL,
  source_ref       TEXT
);
CREATE TABLE IF NOT EXISTS conflict_node (
  node_id TEXT PRIMARY KEY,          -- canon registry: 'act:…' / 'state:…' / 'dest:…'
  label   TEXT NOT NULL DEFAULT '',
  kind    TEXT NOT NULL DEFAULT ''   -- action | class | predicate | … (advisory only)
);
CREATE TABLE IF NOT EXISTS conflict_is_a (
  child  TEXT NOT NULL,
  parent TEXT NOT NULL,
  status TEXT NOT NULL DEFAULT 'ratified'
         CHECK (status IN ('ratified','proposed','deprecated')),
  PRIMARY KEY (child, parent)
);
-- §4 scope linkage: an action reaches conflicts asserted on the resource(s) it uses
-- (and their class groupers).  'uses' is any action→thing linkage — applying a
-- treatment to a surface, running a process on a material, feeding an input to a tool.
CREATE TABLE IF NOT EXISTS uses (
  action   TEXT NOT NULL,
  resource TEXT NOT NULL,
  PRIMARY KEY (action, resource)
);
CREATE TABLE IF NOT EXISTS member_of (
  child        TEXT NOT NULL,
  grouper      TEXT NOT NULL,
  grouper_type TEXT NOT NULL,        -- open vocabulary; 'class' = formal classification
  PRIMARY KEY (child, grouper)       --   (inheritance-recommended, see class_grouper_types)
);
CREATE TABLE IF NOT EXISTS acts_via (
  resource  TEXT NOT NULL,
  mechanism TEXT NOT NULL,
  role      TEXT NOT NULL DEFAULT 'mechanism'  CHECK (role IN ('mechanism','effect')),
  PRIMARY KEY (resource, mechanism, role)
);
"""


def ensure_schema(conn: sqlite3.Connection) -> None:
    """Create the conflict-layer tables (idempotent).  Lives in kb.db at MASTER level and
    is scenario-exempt: bundle/scenario assembly and any exposure mask must carry these
    tables verbatim — the veto layer is always-on regardless of which knowledge is loaded."""
    conn.executescript(SCHEMA_SQL)
    conn.commit()


def migrate_vocab(conn: sqlite3.Connection) -> dict:
    """One-shot migration of a pre-1.2 database to the 1.2 domain-neutral vocabulary
    (see spec Errata 1.2).  Idempotent: a 1.2 database is untouched.  Returns counts."""
    def _has_table(name):
        return conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
                            (name,)).fetchone() is not None

    def _cols(name):
        return [r[1] for r in conn.execute(f"PRAGMA table_info({name})")]

    done = {"uses_rows": 0, "acts_via_rows": 0, "edges_mapped": 0}
    ensure_schema(conn)
    if _has_table("administers"):                       # administers(action,substance) -> uses
        done["uses_rows"] = conn.execute(
            "INSERT OR IGNORE INTO uses(action,resource) "
            "SELECT action,substance FROM administers").rowcount
        conn.execute("DROP TABLE administers")
    if "substance" in _cols("acts_via"):                # old columns + role values
        conn.execute("ALTER TABLE acts_via RENAME TO acts_via_pre12")
        ensure_schema(conn)
        done["acts_via_rows"] = conn.execute(
            "INSERT OR IGNORE INTO acts_via(resource,mechanism,role) "
            "SELECT substance,mechanism,CASE role WHEN 'moa' THEN 'mechanism' "
            "WHEN 'pe' THEN 'effect' ELSE role END FROM acts_via_pre12").rowcount
        conn.execute("DROP TABLE acts_via_pre12")
    ddl = conn.execute("SELECT sql FROM sqlite_master WHERE type='table' "
                       "AND name='conflict_edge'").fetchone()[0]
    if "'contraindicated'" in ddl:   # old CHECK is baked in: rebuild the table (even if
                                     # empty — the old CHECK would reject 'incompatible')
        conn.execute("ALTER TABLE conflict_edge RENAME TO conflict_edge_pre12")
        conn.execute("DROP INDEX IF EXISTS idx_edge_subject")
        ensure_schema(conn)
        done["edges_mapped"] = conn.execute(
            "INSERT INTO conflict_edge SELECT edge_id,subject,"
            "CASE relation_type WHEN 'contraindicated' THEN 'incompatible' "
            "ELSE relation_type END,severity,fire_when,mechanism_id,status,"
            "authority,rationale,source_ref FROM conflict_edge_pre12").rowcount
        conn.execute("DROP TABLE conflict_edge_pre12")
    conn.commit()
    return done


# ── fire_when expression grammar (§5.2, bounded) ───────────────────────────────────────
def validate_expression(expr, depth: int = 1) -> None:
    """Reject anything outside the §5.2 shapes or deeper than 4 — the bound that blocks
    creep toward a general rule language.  Raises ConfError(E_BAD_EXPRESSION)."""
    if depth > _MAX_DEPTH:
        raise ConfError("E_BAD_EXPRESSION", f"nesting depth exceeds {_MAX_DEPTH}")
    if not isinstance(expr, dict):
        raise ConfError("E_BAD_EXPRESSION", "expression node must be an object")
    op = expr.get("op")
    if op in ("presence", "absence"):
        if set(expr) != {"op", "pred"} or not isinstance(expr["pred"], str) or not expr["pred"]:
            raise ConfError("E_BAD_EXPRESSION", f"bad {op} node")
    elif op == "compare":
        if set(expr) != {"op", "field", "cmp", "operand"}:
            raise ConfError("E_BAD_EXPRESSION", "bad compare node keys")
        if not isinstance(expr["field"], str) or not expr["field"]:
            raise ConfError("E_BAD_EXPRESSION", "compare needs a field name")
        if expr["cmp"] not in _CMPS:
            raise ConfError("E_BAD_EXPRESSION", f"cmp must be one of {_CMPS}")
        operand = expr["operand"]
        if not (isinstance(operand, dict) and len(operand) == 1
                and (("lit" in operand and isinstance(operand["lit"], (int, float))
                      and not isinstance(operand["lit"], bool))
                     or ("field" in operand and isinstance(operand["field"], str)
                         and operand["field"]))):
            raise ConfError("E_BAD_EXPRESSION", "operand must be {'lit':number} or {'field':name}")
    elif op == "not":
        if set(expr) != {"op", "arg"}:
            raise ConfError("E_BAD_EXPRESSION", "bad not node")
        validate_expression(expr["arg"], depth + 1)
    elif op in ("all_of", "any_of"):
        if set(expr) != {"op", "args"} or not isinstance(expr["args"], list) or not expr["args"]:
            raise ConfError("E_BAD_EXPRESSION", f"bad {op} node")
        for a in expr["args"]:
            validate_expression(a, depth + 1)
    else:
        raise ConfError("E_BAD_EXPRESSION", f"unknown op: {op!r}")


def eval_expression(expr: dict, active: frozenset, fields: dict):
    """Three-valued evaluation (§5.4): True / False / None (=INDETERMINATE).
    Predicates are closed-world (never INDETERMINATE); fields are open-world (a missing
    field is unknown, never zero) — so an unevaluable condition escalates, never clears."""
    op = expr["op"]
    if op == "presence":
        return expr["pred"] in active
    if op == "absence":
        return expr["pred"] not in active
    if op == "compare":
        if expr["field"] not in fields:
            return None
        left = float(fields[expr["field"]])
        operand = expr["operand"]
        if "lit" in operand:
            right = float(operand["lit"])
        else:
            if operand["field"] not in fields:
                return None
            right = float(fields[operand["field"]])
        return {"<": left < right, "<=": left <= right, ">": left > right,
                ">=": left >= right, "==": left == right}[expr["cmp"]]
    if op == "not":
        v = eval_expression(expr["arg"], active, fields)
        return None if v is None else (not v)
    if op == "all_of":
        vals = [eval_expression(a, active, fields) for a in expr["args"]]
        if any(v is False for v in vals):
            return False
        if any(v is None for v in vals):
            return None
        return True
    if op == "any_of":
        vals = [eval_expression(a, active, fields) for a in expr["args"]]
        if any(v is True for v in vals):
            return True
        if any(v is None for v in vals):
            return None
        return False
    raise ConfError("E_BAD_EXPRESSION", f"unknown op: {op!r}")   # pragma: no cover


# ── canonical serialization (§9) ───────────────────────────────────────────────────────
def canonical_json(obj) -> str:
    """UTF-8, compact separators, insertion-order keys, non-ASCII raw."""
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))


def _row_dump(row: dict) -> str:
    return json.dumps(row, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


class Checker:
    """§7: loads edges, mechanisms, overrides, curated is_a and the canon registry into
    memory; ``check`` is then pure, deterministic and reentrant (inputs never mutated)."""

    def __init__(self):
        self.nodes: frozenset = frozenset()
        self.is_a: dict = {}                  # child -> sorted [parents] (ratified only)
        self.mechanisms: dict = {}            # id -> {mechanism_id,label,explanation,conditionality_class}
        self.edges: list = []                 # non-deprecated, parsed + validated
        self.overrides: dict = {}             # targets_edge_id -> sorted [override rows] (ratified only)
        self.ruleset_version: str = ""
        # §4 scope-linkage: empty ⇒ behaviour is byte-identical to is_a-only.
        self.uses: dict = {}                  # action -> sorted [resource]
        self.groupers: dict = {}              # child -> sorted [(grouper, grouper_type)]
        self.acts_via: dict = {}              # resource -> sorted [mechanism]
        # class-grouper inheritance: which grouper_type values a veto propagates
        # through (RECOMMENDED for formal classifications; () disables)
        self.class_grouper_types: tuple = ("class",)
        self.scope_mechanism = False          # mechanism-class inheritance: OPTIONAL
                                              # (sharing a mechanism is weak grounds; alarm-fatigue risk)

    @classmethod
    def load(cls, db, *, class_grouper_types: tuple = ("class",),
             scope_mechanism: bool = False) -> "Checker":
        conn = sqlite3.connect(str(db))
        conn.row_factory = sqlite3.Row
        try:
            self = cls()
            self.class_grouper_types = tuple(class_grouper_types)
            self.scope_mechanism = scope_mechanism
            self.nodes = frozenset(
                r["node_id"] for r in conn.execute("SELECT node_id FROM conflict_node"))

            # Full rows are kept for the §9/1.1 ruleset hash below; the graphs are
            # built from the same fetch so hash and behaviour can never diverge.
            isa_rows = [dict(r) for r in conn.execute(
                "SELECT child,parent,status FROM conflict_is_a")]
            for r in isa_rows:
                for end in (r["child"], r["parent"]):
                    if end not in self.nodes:
                        raise ConfError("E_UNKNOWN_NODE", f"is_a endpoint not in registry: {end}")
                if r["status"] == "ratified":     # only curated, ratified is_a extends a veto
                    self.is_a.setdefault(r["child"], []).append(r["parent"])
            for child in self.is_a:
                self.is_a[child].sort()

            # §4 scope-linkage (all endpoints must resolve — a mis-linked veto is invisible harm)
            use_rows = [dict(r) for r in conn.execute(
                "SELECT action,resource FROM uses")]
            for r in use_rows:
                for end in (r["action"], r["resource"]):
                    if end not in self.nodes:
                        raise ConfError("E_UNKNOWN_NODE", f"uses endpoint not in registry: {end}")
                self.uses.setdefault(r["action"], []).append(r["resource"])
            mem_rows = [dict(r) for r in conn.execute(
                "SELECT child,grouper,grouper_type FROM member_of")]
            for r in mem_rows:
                for end in (r["child"], r["grouper"]):
                    if end not in self.nodes:
                        raise ConfError("E_UNKNOWN_NODE", f"member_of endpoint not in registry: {end}")
                self.groupers.setdefault(r["child"], []).append((r["grouper"], r["grouper_type"]))
            via_rows = [dict(r) for r in conn.execute(
                "SELECT resource,mechanism,role FROM acts_via")]
            for r in via_rows:
                for end in (r["resource"], r["mechanism"]):
                    if end not in self.nodes:
                        raise ConfError("E_UNKNOWN_NODE", f"acts_via endpoint not in registry: {end}")
                self.acts_via.setdefault(r["resource"], []).append(r["mechanism"])
            for d in (self.uses, self.acts_via):
                for k in d:
                    d[k].sort()
            for k in self.groupers:
                self.groupers[k].sort()

            mech_rows = [dict(r) for r in conn.execute(
                "SELECT mechanism_id,label,explanation,conditionality_class FROM mechanism")]
            self.mechanisms = {m["mechanism_id"]: m for m in mech_rows}

            edge_rows = [dict(r) for r in conn.execute(
                "SELECT edge_id,subject,relation_type,severity,fire_when,mechanism_id,"
                "status,authority,rationale,source_ref FROM conflict_edge")]
            for e in edge_rows:
                if e["status"] == "deprecated":
                    continue                      # inert for firing; still hashed below
                if e["subject"] not in self.nodes:
                    raise ConfError("E_UNKNOWN_NODE",
                                    f"edge {e['edge_id']} subject not in registry: {e['subject']}")
                if e["mechanism_id"] is not None and e["mechanism_id"] not in self.mechanisms:
                    raise ConfError("E_UNKNOWN_NODE",
                                    f"edge {e['edge_id']} cites unknown mechanism: {e['mechanism_id']}")
                try:
                    expr = json.loads(e["fire_when"])
                except ValueError:
                    raise ConfError("E_BAD_EXPRESSION",
                                    f"edge {e['edge_id']}: fire_when is not valid JSON")
                try:
                    validate_expression(expr)
                except ConfError as err:          # fail LOUD at load — a bad ratified rule
                    raise ConfError(err.code, f"edge {e['edge_id']}: {err.message}")
                self.edges.append({**e, "_expr": expr})
            self.edges.sort(key=lambda e: e["edge_id"])

            ov_rows = [dict(r) for r in conn.execute(
                "SELECT override_id,on_node,targets_edge_id,status,justification,source_ref "
                "FROM conflict_override")]
            for o in ov_rows:
                if o["on_node"] not in self.nodes:
                    raise ConfError("E_UNKNOWN_NODE",
                                    f"override {o['override_id']} on_node not in registry: {o['on_node']}")
                if o["status"] == "ratified":     # a proposed override MUST NOT suppress (§7 K3)
                    self.overrides.setdefault(o["targets_edge_id"], []).append(o)
            for k in self.overrides:
                self.overrides[k].sort(key=lambda o: o["override_id"])

            # §9 (1.1/1.2): sha256 over the canonical dump of ALL rows of every
            # table that affects what fires — edges, overrides, mechanisms, and
            # the scope-structural tables (is_a, uses, member_of, acts_via).
            # Deprecated/proposed rows included: any change ⇒ new version.  List
            # order is fixed by the spec; each list is sorted; algo string last.
            h = hashlib.sha256()
            for chunk in (sorted(_row_dump(e) for e in edge_rows),
                          sorted(_row_dump(o) for o in ov_rows),
                          sorted(_row_dump(m) for m in mech_rows),
                          sorted(_row_dump(r) for r in isa_rows),
                          sorted(_row_dump(r) for r in use_rows),
                          sorted(_row_dump(r) for r in mem_rows),
                          sorted(_row_dump(r) for r in via_rows)):
                for line in chunk:
                    h.update(line.encode("utf-8"))
                    h.update(b"\n")
            h.update(_ALGO.encode("utf-8"))
            self.ruleset_version = h.hexdigest()
            return self
        finally:
            conn.close()

    def _ancestors(self, node: str) -> list:
        """Bounded, cycle-safe subject-side walk over ratified is_a (§3)."""
        out, seen, queue = [], {node}, list(self.is_a.get(node, []))
        while queue:
            cur = queue.pop(0)
            if cur in seen:
                continue
            seen.add(cur)
            out.append(cur)
            queue.extend(self.is_a.get(cur, []))
        return out

    def _scope(self, action: str) -> set:
        """Subject scope for an action (§4): the action, its is_a ancestors, the
        resources it uses + THEIR is_a ancestors (REQUIRED), the resources' class
        groupers (RECOMMENDED, grouper_type ∈ class_grouper_types), and their mechanism
        classes (OPTIONAL, default off — sharing a mechanism is a weak basis for
        inheriting an incompatibility).  With no uses/member_of/acts_via rows this is
        exactly {action}∪is_a-ancestors."""
        scope = {action} | set(self._ancestors(action))
        for res in self.uses.get(action, ()):
            scope.add(res)
            scope.update(self._ancestors(res))
            if self.class_grouper_types:
                for grouper, gtype in self.groupers.get(res, ()):
                    if gtype in self.class_grouper_types:
                        scope.add(grouper)
                        scope.update(self._ancestors(grouper))
            if self.scope_mechanism:
                scope.update(self.acts_via.get(res, ()))
        return scope

    # ── the checker (§7, K0–K5) ────────────────────────────────────────────────────────
    def check(self, card: dict, state: dict) -> dict:
        # K0 — resolve + validate.  Malformed inputs raise ConfError; a well-formed pair
        # can never raise (§10) — unrepresentable conditions surface as flag_for_human.
        if not (isinstance(card, dict) and isinstance(card.get("card_id"), str)
                and card["card_id"] and isinstance(card.get("actions"), list)
                and all(isinstance(a, str) and a for a in card["actions"])):
            raise ConfError("E_MALFORMED_STATE",
                            "card must be {card_id: str, actions: [node_id, ...]}")
        if not (isinstance(state, dict) and isinstance(state.get("predicates"), list)
                and all(isinstance(p, str) for p in state["predicates"])
                and isinstance(state.get("fields"), dict)
                and all(isinstance(k, str) and isinstance(v, (int, float))
                        and not isinstance(v, bool) for k, v in state["fields"].items())):
            raise ConfError("E_MALFORMED_STATE",
                            "state must be {predicates: [str], fields: {str: number}}")
        actions = list(card["actions"])
        for a in actions:
            if a not in self.nodes:
                raise ConfError("E_UNKNOWN_NODE", f"unknown action node: {a}")

        active = frozenset(state["predicates"]) | frozenset(actions)
        fields = dict(state["fields"])

        findings: list = []
        consulted: set = set()
        overrides_applied: list = []
        not_evaluated: set = set()

        for idx, action in enumerate(actions):
            scope = self._scope(action)
            # K1 — candidates: every non-deprecated edge attached to the action, an is_a
            # ancestor, or (§4) a resource it uses / that resource's groupers
            # (interactions live at the class or resource; cards name the leaf action).
            for e in self.edges:
                if e["subject"] not in scope:
                    continue
                consulted.add(e["edge_id"])
                # K2 — three-valued evaluation; FALSE discards.
                result = eval_expression(e["_expr"], active, fields)
                if result is False:
                    continue
                # K3 — a ratified override on a more-specific node suppresses (logged,
                # never silent).  Default-deny: no override ⇒ the inherited edge stands.
                ovs = [o for o in self.overrides.get(e["edge_id"], ())
                       if o["on_node"] in scope]
                if ovs:
                    overrides_applied.append({
                        "action": action, "edge_id": e["edge_id"],
                        "override_id": ovs[0]["override_id"],
                        "justification": ovs[0]["justification"]})
                    continue
                # K4 — classify & emit.
                mech = None
                if e["mechanism_id"] is not None:
                    m = self.mechanisms[e["mechanism_id"]]
                    mech = {"mechanism_id": m["mechanism_id"], "label": m["label"],
                            "explanation": m["explanation"],
                            "conditionality_class": m["conditionality_class"]}
                if result is None:
                    disposition, reason, rec = "flag_for_human", "indeterminate", "human_review"
                    not_evaluated.add((e["edge_id"], "indeterminate_condition"))
                elif e["status"] == "proposed":
                    disposition, reason, rec = "flag_for_human", "unratified_rule", "human_review"
                    not_evaluated.add((e["edge_id"], "unratified_rule"))
                else:
                    disposition, reason = "fire", "triggered"
                    rec = _FIRE_DISPOSITION[e["severity"]]
                findings.append({
                    "_action_idx": idx,
                    "action": action, "edge_id": e["edge_id"],
                    "relation_type": e["relation_type"], "disposition": disposition,
                    "reason": reason, "severity": e["severity"],
                    "recommended_disposition": rec, "mechanism": mech,
                    "rationale": e["rationale"]})

        # §8.2 — deterministic ordering throughout.
        findings.sort(key=lambda f: (f["_action_idx"], 0 if f["disposition"] == "fire" else 1,
                                     -_SEV_RANK[f["severity"]], f["edge_id"]))
        for f in findings:
            del f["_action_idx"]

        # K5 — clearance + assemble (§8.1 key order).
        if any(f["disposition"] == "fire" for f in findings):
            clearance = "conflicts_found"
        elif findings:
            clearance = "review_required"
        else:
            clearance = "no_known_conflicts"

        return {
            "checker_version": CHECKER_VERSION,
            "ruleset_version": self.ruleset_version,
            "card_id": card["card_id"],
            "clearance": clearance,
            "findings": findings,
            "checked": {
                "actions": actions,
                "edges_consulted": sorted(consulted),
                "overrides_applied": sorted(overrides_applied,
                                            key=lambda o: (o["action"], o["edge_id"])),
            },
            "coverage": {
                "caveat": CAVEAT,
                "not_evaluated": [{"edge_id": e, "reason": r}
                                  for e, r in sorted(not_evaluated)],
            },
        }
