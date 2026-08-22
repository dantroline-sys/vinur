#!/usr/bin/env python
"""yield_audit.py — documents that were 'distilled' but yielded (almost)
nothing, and the reset that puts them back on the queue."""
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from knowledgehost import yield_audit as Y          # noqa: E402
from knowledgehost.kb import KB                     # noqa: E402
from knowledgehost.store import SqliteStore         # noqa: E402
from knowledgehost import dedupe as dd              # noqa: E402

OK = 0


def ok(name):
    global OK
    OK += 1
    print(f"  ok {OK:2d}  {name}")


d = Path(tempfile.mkdtemp())
store = SqliteStore({"db_path": str(d / "store.db"), "embed_dim": 8})
kb = KB({"kb_path": str(d / "kb.db"), "embed_dim": 8})


def chunks(doc, n):
    return [{"id": f"{doc}#{i}", "path_or_url": doc, "title": doc.split("/")[-1],
             "section": "", "text": f"text of {doc} chunk {i} " * 5, "tokens": 30}
            for i in range(n)]


def item(table, iid, doc):
    sup = json.dumps([{"doc_id": doc, "chunk_id": f"{doc}#0", "evidence": "…"}])
    if table == "nodes":
        kb.db.execute("INSERT INTO nodes(id,label,kind,summary,support,status) "
                      "VALUES(?,?,?,?,?,'active')", (iid, iid, "concept", "", sup))
    elif table == "edges":
        kb.db.execute("INSERT INTO edges(id,src_id,dst_id,family,type,support,status) "
                      "VALUES(?,?,?,?,?,?,'active')", (iid, "a", "b", "f", "t", sup))
    else:
        kb.db.execute("INSERT INTO procedure_cards(id,title,support,status) "
                      "VALUES(?,?,?,'active')", (iid, iid, sup))


# healthy ×3: 5 chunks landed, 2 items per chunk (the corpus's normal rate)
for h in ("healthy", "healthy2", "healthy3"):
    store.add_chunks(chunks(f"/docs/{h}.pdf", 5))
    for i in range(5):
        kb.mark_distilled(f"/docs/{h}.pdf#{i}")
    for i in range(8):
        item("nodes", f"{h}-n{i}", f"/docs/{h}.pdf")
    for i in range(2):
        item("procedure_cards", f"{h}-c{i}", f"/docs/{h}.pdf")
# zero-yield: 5 chunks landed + 1 furniture chunk (skipped, never stamped), nothing cites it
store.add_chunks(chunks("/docs/empty.pdf", 6))
for i in range(5):
    kb.mark_distilled(f"/docs/empty.pdf#{i}")
    kb.claim_text(dd.text_hash(f"text of /docs/empty.pdf chunk {i} " * 5), f"/docs/empty.pdf#{i}")
kb.mark_recarded("/docs/empty.pdf#0", 3)
kb.mark_zone_skipped("/docs/empty.pdf#5", "references")     # the furniture chunk
# starved: 10 chunks landed, one lonely node (5% of expected)
store.add_chunks(chunks("/docs/starved.pdf", 10))
for i in range(10):
    kb.mark_distilled(f"/docs/starved.pdf#{i}")
item("edges", "s-e0", "/docs/starved.pdf")
# tiny: 2 chunks, nothing — too small to judge
store.add_chunks(chunks("/docs/tiny.md", 2))
for i in range(2):
    kb.mark_distilled(f"/docs/tiny.md#{i}")
# all dupes: stamped, but every stamp is a dupe mark — never saw an extractor
store.add_chunks(chunks("/docs/copy.pdf", 4))
for i in range(4):
    kb.record_dupe(f"/docs/copy.pdf#{i}", f"/docs/healthy.pdf#{i}", "", kind="exact")
    kb.mark_distilled(f"/docs/copy.pdf#{i}")
# queued: never distilled at all
store.add_chunks(chunks("/docs/queued.pdf", 3))
kb.db.commit()

y = Y.doc_yield(kb.db)
assert y["/docs/healthy.pdf"] == {"nodes": 8, "edges": 0, "cards": 2}, y
assert y["/docs/starved.pdf"] == {"nodes": 0, "edges": 1, "cards": 0}, y
assert "/docs/empty.pdf" not in y
ok("doc_yield counts the items citing each document, per table")

res = Y.audit(store, kb)
assert res["ok"] and res["median_per_chunk"] == 2.0, res
flagged = {f["doc_id"]: f for f in res["flagged"]}
assert set(flagged) == {"/docs/empty.pdf", "/docs/starved.pdf"}, set(flagged)
ok("audit flags the zero-yield and the starved document — and only those")

e = flagged["/docs/empty.pdf"]
assert e["landed"] == 5 and e["items"] == 0 and e["ratio"] == 0.0 and "NOTHING" in e["reason"], e
assert e["zoned"] == 1 and e["chunks"] == 6
ok("…zero-yield: landed excludes the furniture chunk, the reason says nothing landed")

s = flagged["/docs/starved.pdf"]
assert s["landed"] == 10 and s["items"] == 1 and s["expected"] == 20.0 and s["ratio"] == 0.05, s
assert "20 expected" in s["reason"]
ok("…starved: expected = landed × corpus median, ratio 5%")

assert res["flagged"][0]["doc_id"] == "/docs/empty.pdf"
ok("worst first")

assert "/docs/tiny.md" not in flagged and "/docs/copy.pdf" not in flagged and "/docs/queued.pdf" not in flagged
ok("too-small, all-duplicate and still-queued documents are not judged")

assert Y.audit(store, kb, ratio=0.01)["flagged_total"] == 1
assert Y.audit(store, kb, min_chunks=1)["flagged_total"] == 3       # tiny now judged
ok("ratio and min_chunks are honoured")

# reset: back on the queue, claims released, furniture kept, items kept
r = Y.reset_docs(store, kb, ["/docs/empty.pdf", "/nope/missing.pdf"])
assert r["ok"] and r["chunks"] == 5 and r["per_doc"]["/docs/empty.pdf"] == 5 \
    and r["per_doc"]["/nope/missing.pdf"] == 0, r
assert not kb.is_distilled("/docs/empty.pdf#0") and not kb.is_distilled("/docs/empty.pdf#4")
assert kb.db.execute("SELECT COUNT(*) FROM recarded_chunks WHERE chunk_id LIKE '/docs/empty%'").fetchone()[0] == 0
assert kb.db.execute("SELECT COUNT(*) FROM zone_skips WHERE chunk_id='/docs/empty.pdf#5'").fetchone()[0] == 1
assert kb.claim_text(dd.text_hash("text of /docs/empty.pdf chunk 1 " * 5), "newcomer") == "newcomer"
ok("reset un-stamps distil+recard, releases text claims, keeps the furniture skip")

q = store.distill_queue(kb.path)
pend = {x["doc"]: x["pending"] for x in q["docs"]}
assert pend.get("/docs/empty.pdf") == 5 and pend.get("/docs/queued.pdf") == 3, pend
ok("the reset document is back in the distil queue (furniture excluded)")

assert kb.is_distilled("/docs/healthy.pdf#0") and Y.doc_yield(kb.db)["/docs/healthy.pdf"]["nodes"] == 8
ok("untouched documents keep their stamps and items")

# ── the two routes over live HTTP (auth on) ─────────────────────────────────
import json as _json
import threading
import urllib.error
import urllib.request
from types import SimpleNamespace
from knowledgehost.config import load_config      # noqa: E402
from knowledgehost.server import KnowledgeHostServer   # noqa: E402

# re-stamp empty.pdf so the audit flags it again, then exercise the routes
for i in range(5):
    kb.mark_distilled(f"/docs/empty.pdf#{i}")
scfg = load_config()
scfg.update({"host": "127.0.0.1", "port": 0, "auth_token": "s3cret",
             "control_dir": str(d / "ctrl"), "kb_path": kb.path,
             "db_path": store.cfg["db_path"]})
httpd = KnowledgeHostServer(scfg, store, SimpleNamespace(), kb=kb)
port = httpd.server_address[1]
threading.Thread(target=httpd.serve_forever, daemon=True).start()


def call(path, body=None, tok="s3cret"):
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}{path}",
        data=_json.dumps(body).encode() if body is not None else None,
        headers={"Content-Type": "application/json",
                 **({"Authorization": f"Bearer {tok}"} if tok else {})},
        method="POST" if body is not None else "GET")
    try:
        with urllib.request.urlopen(req, timeout=5) as r:
            return r.status, _json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, _json.loads(e.read())


try:
    code, res = call("/yield_audit")
    assert code == 200 and res["ok"] and {f["doc_id"] for f in res["flagged"]} == \
        {"/docs/empty.pdf", "/docs/starved.pdf"}, (code, res)
    ok("GET /yield_audit reports the flagged documents")
    code, res = call("/yield_audit?ratio=0.01")
    assert code == 200 and res["flagged_total"] == 1, res
    ok("…with the ratio as a query parameter")

    code, res = call("/redistil", {"docs": ["/docs/empty.pdf"]}, tok="")
    assert code == 401, (code, res)
    ok("POST /redistil needs the auth token")
    code, res = call("/redistil", {"docs": []})
    assert code == 400 and "nothing" in res["error"], (code, res)
    ok("…an empty request is refused")
    code, res = call("/redistil", {"docs": ["/docs/empty.pdf"]})
    assert code == 200 and res["ok"] and res["chunks"] == 5 and "job" not in res, (code, res)
    assert not kb.is_distilled("/docs/empty.pdf#2")
    ok("…one document re-queued, no pass started unless asked")
    code, res = call("/redistil", {"all_flagged": True})
    # empty.pdf is already back on the queue (so no longer judged) — only starved remains
    assert code == 200 and res["ok"] and res["docs"] == 1 and res["chunks"] == 10, (code, res)
    assert not kb.is_distilled("/docs/starved.pdf#0") and kb.is_distilled("/docs/healthy.pdf#0")
    ok("…all_flagged re-queues every still-flagged document and nothing else")
    code, res = call("/yield_audit")
    assert code == 200 and res["flagged_total"] == 0, res
    ok("…after which the audit is clean (re-queued docs are pending, not judged)")
finally:
    httpd.shutdown()

kb.close(); store.close()
print(f"yield_audit_test: {OK} checks OK")
