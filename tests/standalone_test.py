"""Standalone-box surface: [serving] config, supervisor service list, serving
argv builders, the LAN bind guard, and the /drop research hand-off lane
(validated write + live HTTP round-trip against a real server).

Run:  python tests/standalone_test.py     (stdlib only)
"""
import json
import os
import sys
import tempfile
import threading
import urllib.error
import urllib.request
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from knowledgehost import research, serving
from knowledgehost.config import load_config
from knowledgehost.server import KnowledgeHostServer, check_bind_auth
from knowledgehost.supervisor import services_for

PASS = 0


def ok(label):
    global PASS
    PASS += 1
    print(f"  ok {PASS:2d}  {label}")


DROP = """---
provenance: vinkona
kb_query: "test question"
---

# Question
test question

## Answer
tested.
"""


def main():
    # ── config: defaults + partial merge ─────────────────────────────────
    cfg = load_config()
    assert cfg["serving"]["llms"] == [] and cfg["serving"]["embed"]["port"] == 11437
    assert cfg["serving"]["embed"] is not None
    ok("serving defaults present")

    with tempfile.TemporaryDirectory() as td:
        toml = Path(td) / "c.toml"
        toml.write_text('[[serving.llms]]\nname = "primary"\nengine = "vllm"\n'
                        'model = "org/model-awq"\nport = 11438\n'
                        'args = ["--max-model-len", "16384"]\n'
                        '[serving.embed]\nenabled = true\n'
                        '[serving.reranker]\nenabled = true\n')
        c2 = load_config(str(toml))
        assert c2["serving"]["embed"]["enabled"] is True
        assert c2["serving"]["embed"]["port"] == 11437, "partial table must keep defaults"
        assert load_config()["serving"]["embed"]["enabled"] is False, "DEFAULTS must not be mutated"
        ok("partial [serving] table merges over defaults")

        # ── supervisor service list ──────────────────────────────────────
        svcs = services_for(c2)
        names = [s["name"] for s in svcs]
        assert names == ["llm-primary", "embed", "reranker", "kb"], names
        rr = next(s for s in svcs if s["name"] == "reranker")
        assert rr["env"]["PORT"] == "11439"          # parsed from rerank_url
        assert svcs[-1]["name"] == "kb"
        ok("services_for: LMs first, kb last, reranker port from rerank_url")

        # ── serving argv builders ────────────────────────────────────────
        try:
            serving.llm_argv(c2["serving"]["llms"][0])
            raise AssertionError("vllm without serving/.venv must raise")
        except FileNotFoundError as e:
            assert "--serving" in str(e)
        ok("vllm engine points at install.sh --serving when venv missing")

        gguf = Path(td) / "m.gguf"
        gguf.write_bytes(b"GGUF")
        os.environ["LLAMA_SERVER"] = "/usr/bin/true"
        argv = serving.llm_argv({"name": "x", "engine": "llama", "model": str(gguf),
                                 "port": 11440, "args": ["-ngl", "0"]})
        assert argv[:3] == ["/usr/bin/true", "-m", str(gguf)]
        assert argv[-2:] == ["-ngl", "0"] and "11440" in argv    # override wins (last flag)
        ok("llama engine argv (args appended after defaults)")

        try:
            serving.llm_argv({"name": "x", "engine": "tgi", "model": "m", "port": 1})
            raise AssertionError("unknown engine must raise")
        except ValueError:
            ok("unknown engine rejected")

        # first-class vLLM tuning keys → CLI flags (NVFP4 + fp8 KV case)
        vdir = Path(td) / "serving" / ".venv" / "bin"
        vdir.mkdir(parents=True)
        (vdir / "vllm").write_text("")
        full = {"name": "p", "engine": "vllm", "model": "org/M-NVFP4", "port": 11438,
                "quantization": "modelopt", "kv_cache_dtype": "fp8",
                "max_model_len": 16384, "gpu_memory_utilization": 0.9,
                "max_num_seqs": 32, "tensor_parallel": 2, "enforce_eager": True,
                "trust_remote_code": False, "served_model_name": "primary",
                "args": ["--kv-cache-dtype", "auto"]}
        argv = serving.llm_argv(full, root=Path(td))
        s = " ".join(argv)
        assert "--quantization modelopt" in s and "--kv-cache-dtype fp8" in s
        assert "--max-model-len 16384" in s and "--gpu-memory-utilization 0.9" in s
        assert "--max-num-seqs 32" in s and "--tensor-parallel-size 2" in s
        assert "--served-model-name primary" in s
        assert "--enforce-eager" in s and "--trust-remote-code" not in s
        assert argv[-2:] == ["--kv-cache-dtype", "auto"], "args must come LAST (override)"
        ok("vLLM keys map to flags; false flags omitted; args override last")

        lean = serving.llm_argv({"name": "p", "engine": "vllm",
                                 "model": "org/M", "port": 1}, root=Path(td))
        assert not any(a.startswith("--kv-cache") or a.startswith("--max-model")
                       for a in lean), "unset keys must not emit flags"
        ok("unset keys leave vLLM defaults untouched")

        argv = serving.llm_argv({"name": "g", "engine": "llama", "model": str(gguf),
                                 "port": 11440, "ctx_size": 8192, "n_gpu_layers": 0})
        s = " ".join(argv)
        assert "-c 8192" in s and "-ngl 0" in s
        ok("llama first-class keys: ctx_size (-c) + n_gpu_layers (-ngl)")

        # container engine: official image under podman/docker
        centry = {"name": "primary", "engine": "container", "model": "org/M-NVFP4",
                  "port": 11438, "runtime": "podman",
                  "image": "docker.io/vllm/vllm-openai:v1",
                  "kv_cache_dtype": "fp8", "env": {"HF_TOKEN": "hf_x"},
                  "args": ["--enable-prefix-caching"]}
        argv = serving.llm_argv(centry, root=Path(td))
        s = " ".join(argv)
        assert argv[:4] == ["podman", "run", "--rm", "--name"]
        assert "vinur-llm-primary" in argv and "--replace" in argv
        assert "--device nvidia.com/gpu=all" in s and "--ipc=host" in s
        assert "-p 127.0.0.1:11438:8000" in s
        assert f"-v {td}/var/cache/huggingface:/root/.cache/huggingface:z" in s
        assert "-e HF_TOKEN=hf_x" in s
        img = argv.index("docker.io/vllm/vllm-openai:v1")
        assert argv[img + 1] == "org/M-NVFP4", "model positional after image"
        assert "--kv-cache-dtype fp8" in s and argv[-1] == "--enable-prefix-caching"
        ok("container engine (podman): CDI GPU, :z cache mount, -e env, keys map")

        argv = serving.llm_argv({**centry, "runtime": "docker"}, root=Path(td))
        s = " ".join(argv)
        assert "--gpus all" in s and "--replace" not in s
        ok("container engine (docker): --gpus all, no podman-only flags")

        assert serving.weights_status("container", "org/M")["status"] == "missing"
        ccfg = {"serving": {"llms": [{"name": "c", "engine": "container",
                                      "model": "org/M-NVFP4", "port": 1}]}}
        assert serving.toolkit_warning(ccfg, toolkit_present=False) is None
        ok("container entries: HF-cache weights check, exempt from toolkit warning")

        # llama-server resolution: $LLAMA_SERVER > bin/ > PATH > sibling vinkona
        saved_ls = os.environ.pop("LLAMA_SERVER", None)
        try:
            vroot = Path(td) / "vinur-root"
            assert serving.find_llama_server(vroot) is None
            sib = Path(td) / "vinkona" / "assistant" / "bin" / "llama-server"
            sib.parent.mkdir(parents=True)
            sib.write_text("#!/bin/sh\n")
            sib.chmod(0o755)
            vroot2 = Path(td) / "vinur"          # sibling of the vinkona dir
            vroot2.mkdir()
            assert serving.find_llama_server(vroot2) == str(sib)
            own = vroot2 / "bin" / "llama-server"
            own.parent.mkdir()
            own.write_text("#!/bin/sh\n")
            own.chmod(0o755)
            assert serving.find_llama_server(vroot2) == str(own), "in-tree beats sibling"
            os.environ["LLAMA_SERVER"] = "/x/custom"
            assert serving.find_llama_server(vroot2) == "/x/custom", "env wins"
        finally:
            if saved_ls is None:
                os.environ.pop("LLAMA_SERVER", None)
            else:
                os.environ["LLAMA_SERVER"] = saved_ls
        ok("find_llama_server: env > in-tree bin/ > sibling vinkona; None when absent")

        # CUDA_HOME probe (the vLLM 'Could not find nvcc' crash).  PATH is
        # emptied so a real nvcc on the test box can't shadow the fixtures.
        saved_path = os.environ.get("PATH", "")
        os.environ["PATH"] = ""
        try:
            assert serving.cuda_home_probe({"CUDA_HOME": "/x"}) is None, "set env wins"
            assert serving.cuda_home_probe({}, prefixes=(str(Path(td) / "nope"),)) is None
            croot = Path(td) / "cudaland"
            for name in ("cuda-12.4", "cuda-13.0"):
                d = croot / name / "bin"
                d.mkdir(parents=True)
                (d / "nvcc").write_text("")
            got = serving.cuda_home_probe({}, prefixes=(str(croot),))
            assert got == str(croot / "cuda-13.0"), f"newest toolkit wins: {got}"
        finally:
            os.environ["PATH"] = saved_path
        ok("cuda_home_probe: honours existing env, finds newest cuda-* install")

        ea = serving.embed_argv(c2, "/x/nomic.gguf")
        assert "--embedding" in ea and "11437" in ea and "-ub" in ea
        ok("embed argv (llama-server --embedding, batch-safe)")

        # ── LAN bind guard ───────────────────────────────────────────────
        check_bind_auth({"host": "127.0.0.1", "port": 1, "auth_token": ""})
        check_bind_auth({"host": "0.0.0.0", "port": 1, "auth_token": "s3"})
        try:
            check_bind_auth({"host": "0.0.0.0", "port": 1, "auth_token": ""})
            raise AssertionError("LAN bind without token must refuse")
        except SystemExit:
            pass
        os.environ["VINUR_ALLOW_UNAUTHED_LAN"] = "1"
        check_bind_auth({"host": "0.0.0.0", "port": 1, "auth_token": ""})
        del os.environ["VINUR_ALLOW_UNAUTHED_LAN"]
        ok("bind guard: loopback free, LAN needs token (env override honoured)")

        # ── write_drop validation ────────────────────────────────────────
        dcfg = {"research_solved_dir": str(Path(td) / "solved")}
        name = "0123456789abcdef.md"
        assert research.write_drop(dcfg, name, DROP) == {"ok": True, "changed": True}
        assert (Path(td) / "solved" / name).read_text() == DROP
        assert research.write_drop(dcfg, name, DROP) == {"ok": True, "changed": False}
        ok("write_drop: atomic write + byte-identical no-op")
        for bad_name in ("../evil.md", "x.md", "0123456789ABCDEF.md", "a" * 16):
            try:
                research.write_drop(dcfg, bad_name, DROP)
                raise AssertionError(f"bad name accepted: {bad_name}")
            except ValueError:
                pass
        try:
            research.write_drop(dcfg, name, "# not a drop\n")
            raise AssertionError("non-vinkona content accepted")
        except ValueError:
            pass
        try:
            research.write_drop({"research_solved_dir": ""}, name, DROP)
            raise AssertionError("unconfigured dir accepted")
        except ValueError:
            pass
        ok("write_drop: traversal/name/provenance/unconfigured all rejected")

        # ── /drop over live HTTP (auth on) ───────────────────────────────
        scfg = load_config()
        libtoml = Path(td) / "live.toml"
        libtoml.write_text('# comment kept\nport = 8771\n[serving]\nswap_timeout_s = 60\n')
        scfg.update({"host": "127.0.0.1", "port": 0, "auth_token": "s3cret",
                     "research_solved_dir": str(Path(td) / "solved2"),
                     "control_dir": str(Path(td) / "ctrl"),
                     "_config_path": str(libtoml)})
        tools = SimpleNamespace()
        httpd = KnowledgeHostServer(scfg, SimpleNamespace(), tools, kb=None)
        port = httpd.server_address[1]
        t = threading.Thread(target=httpd.serve_forever, daemon=True)
        t.start()
        try:
            def post(tok):
                req = urllib.request.Request(
                    f"http://127.0.0.1:{port}/drop",
                    data=json.dumps({"name": name, "content": DROP}).encode(),
                    headers={"Content-Type": "application/json",
                             **({"Authorization": f"Bearer {tok}"} if tok else {})},
                    method="POST")
                try:
                    with urllib.request.urlopen(req, timeout=5) as r:
                        return r.status, json.loads(r.read())
                except urllib.error.HTTPError as e:
                    return e.code, json.loads(e.read())

            code, res = post("s3cret")
            assert code == 200 and res == {"ok": True, "changed": True}, (code, res)
            assert (Path(td) / "solved2" / name).read_text() == DROP
            code, res = post("s3cret")
            assert res == {"ok": True, "changed": False}
            code, _ = post("")
            assert code == 401, code
            code, _ = post("wrong")
            assert code == 401, code
            ok("/drop live: 200+write, idempotent, 401 without/with wrong token")

            # GET /drop: the exporter handshake (accepts + inventory), authed
            def get_drop(tok):
                req = urllib.request.Request(
                    f"http://127.0.0.1:{port}/drop",
                    headers={"Authorization": f"Bearer {tok}"} if tok else {})
                try:
                    with urllib.request.urlopen(req, timeout=5) as r:
                        return r.status, json.loads(r.read())
                except urllib.error.HTTPError as e:
                    return e.code, {}

            code, _ = get_drop("")
            assert code == 401, code
            code, hs = get_drop("s3cret")
            assert code == 200 and hs["ok"] and hs["accepts"], hs
            assert name in hs["drops"] and len(hs["drops"][name]) == 16, hs
            noaccept = research.drop_inventory({"research_solved_dir": ""})
            assert noaccept["ok"] and noaccept["accepts"] is False
            ok("GET /drop handshake: 401 unauthed; inventory served; accepts=false w/o dir")

            # ── /library/root: set the trusted root from the panel ───────
            libroot = Path(td) / "TheLibrary"
            (libroot / "papers").mkdir(parents=True)

            def post_root(tok, root):
                req = urllib.request.Request(
                    f"http://127.0.0.1:{port}/library/root",
                    data=json.dumps({"root": root}).encode(),
                    headers={"Content-Type": "application/json",
                             **({"Authorization": f"Bearer {tok}"} if tok else {})},
                    method="POST")
                try:
                    with urllib.request.urlopen(req, timeout=5) as r:
                        return r.status, json.loads(r.read())
                except urllib.error.HTTPError as e:
                    return e.code, json.loads(e.read())

            code, _r = post_root("", str(libroot))
            assert code == 401, code
            code, _r = post_root("s3cret", "relative/path")
            assert code == 400 and "absolute" in _r["error"], _r
            code, _r = post_root("s3cret", str(Path(td) / "no-such-dir"))
            assert code == 400 and "not a directory" in _r["error"], _r
            code, r = post_root("s3cret", str(libroot))
            assert code == 200 and r["ok"] and r["root"] == str(libroot), r
            assert [s["name"] for s in r["subdirs"]] == ["papers"], r
            assert scfg["library_root"] == str(libroot), "live cfg must update"
            text = libtoml.read_text()
            root_at = text.index("library_root")
            assert root_at < text.index("[serving]"), "must land ABOVE the table"
            assert "# comment kept" in text and "swap_timeout_s = 60" in text
            code, r = post_root("s3cret", str(libroot))     # idempotent re-set
            assert code == 200 and text.count("library_root") == 1
            ok("/library/root: authed, validated, live-applied, written above [serving]")

            # the return leg: open kb gaps ride the handshake, verbatim,
            # most-asked first, closed/blank ones filtered out
            httpd.kb = SimpleNamespace(list_gaps=lambda n=100: [
                {"query_text": "How do  plasmids replicate?", "intent": "ask",
                 "effect_label": "", "count": 7, "status": "open"},
                {"query_text": "answered already", "intent": "", "effect_label": "",
                 "count": 3, "status": "acquired"},
                {"query_text": "  ", "intent": "", "effect_label": "",
                 "count": 2, "status": "open"}])
            code, hs = get_drop("s3cret")
            assert hs["gaps"] == [{"query": "How do  plasmids replicate?",
                                   "count": 7, "intent": "ask"}], hs.get("gaps")
            httpd.kb = None
            code, hs = get_drop("s3cret")
            assert "gaps" not in hs, "no kb loaded -> no gaps key, drops still served"
            ok("handshake return leg: open gaps only, verbatim; absent without a kb")

            # ── POST /gaps/close: the Curation tab's manual dismiss ────────
            closed = []

            def post_gap(tok, body):
                req = urllib.request.Request(
                    f"http://127.0.0.1:{port}/gaps/close",
                    data=json.dumps(body).encode(),
                    headers={"Content-Type": "application/json",
                             **({"Authorization": f"Bearer {tok}"} if tok else {})},
                    method="POST")
                try:
                    with urllib.request.urlopen(req, timeout=5) as r:
                        return r.status, json.loads(r.read())
                except urllib.error.HTTPError as e:
                    return e.code, json.loads(e.read())

            code, _ = post_gap(None, {"query": "x"})
            assert code == 401
            code, _ = post_gap("s3cret", {"query": "x"})       # kb is None right now
            assert code == 400
            httpd.kb = SimpleNamespace(
                close_gap=lambda q, status="acquired": closed.append((q, status)) or 1)
            code, r = post_gap("s3cret", {"query": "x", "status": "bogus"})
            assert code == 400 and "dismissed|acquired" in r["error"]
            code, r = post_gap("s3cret", {"query": "How do  plasmids replicate?"})
            assert code == 200 and r["closed"] == 1
            assert closed == [("How do  plasmids replicate?", "dismissed")], closed
            httpd.kb = None
            ok("/gaps/close: authed, status-validated, defaults to dismissed")

            # ── Sources progress: the store-level join + /browse enrichment ──
            import sqlite3
            from knowledgehost.store import SqliteStore
            spcfg = json.loads(json.dumps(
                {k: v for k, v in scfg.items() if isinstance(v, (str, int, float, bool, list, dict))}))
            spcfg["db_path"] = str(Path(td) / "prog.sqlite3")
            pstore = SqliteStore(spcfg)
            docfile = Path(td) / "book.pdf"
            docfile.write_bytes(b"x")
            for i in range(4):
                pstore.db.execute(
                    "INSERT INTO chunks(id,source_type,title,section,path_or_url,"
                    "text,tokens,version,ingested_at) VALUES (?,?,?,?,?,?,?,?,0)",
                    (f"c{i}", "pdf", "Book", "", str(docfile), "t", 3, 1))
            pstore.db.execute(
                "INSERT INTO chunks(id,source_type,title,section,path_or_url,"
                "text,tokens,version,ingested_at) VALUES ('w0','wikipedia','W','',"
                "'zim://Foo','t',3,1,0)")
            pstore.db.commit()
            for i in range(2):                     # ingested but NEVER distilled
                pstore.db.execute(
                    "INSERT INTO chunks(id,source_type,title,section,path_or_url,"
                    "text,tokens,version,ingested_at) VALUES (?,?,?,?,?,?,?,?,9)",
                    (f"q{i}", "pdf", "Fresh", "", "new-book.pdf", "t", 3, 1))
            pstore.db.commit()
            kbfile = Path(td) / "prog-kb.db"
            kcon = sqlite3.connect(kbfile)
            kcon.execute("CREATE TABLE distilled_chunks(chunk_id TEXT PRIMARY KEY)")
            kcon.executemany("INSERT INTO distilled_chunks VALUES (?)",
                             [("c0",), ("c1",), ("c3",)])
            kcon.execute("CREATE TABLE source_registry(doc_id TEXT PRIMARY KEY)")
            kcon.executemany("INSERT INTO source_registry VALUES (?)",
                             [(str(docfile),), ("zim://Foo",)])
            kcon.commit()
            kcon.close()
            prog = pstore.source_progress(str(kbfile), [str(docfile), "zim://Foo", "ghost"])
            assert prog[str(docfile)] == {"chunks": 4, "distilled": 3}, prog
            assert prog["zim://Foo"] == {"chunks": 1, "distilled": 0}
            assert "ghost" not in prog
            assert pstore.source_progress(str(Path(td) / "no-such-kb.db"),
                                          [str(docfile)]) == {}, "bad kb -> {}"
            srows = [{"doc_id": str(docfile), "title": "Book", "source_type": "pdf",
                      "trust_weight": 1.0, "regime": "empirical", "status": "active",
                      "bundle": "base", "license": "", "license_holder": "",
                      "license_url": ""},
                     {"doc_id": "zim://Foo", "title": "W", "source_type": "wikipedia",
                      "trust_weight": 0.6, "regime": "empirical", "status": "active",
                      "bundle": "base", "license": "", "license_holder": "",
                      "license_url": ""}]
            httpd.kb = SimpleNamespace(list_sources=lambda n=200: srows)
            store0, httpd.store = httpd.store, pstore
            scfg["_master_kb_path"] = str(kbfile)
            try:
                with urllib.request.urlopen(
                        f"http://127.0.0.1:{port}/browse?kind=sources", timeout=5) as r:
                    rows = json.loads(r.read())["rows"]
            finally:
                httpd.kb, httpd.store = None, store0
                scfg.pop("_master_kb_path", None)
            by = {r0["doc_id"]: r0 for r0 in rows}
            assert by[str(docfile)]["pct"] == 75 and by[str(docfile)]["chunks"] == 4
            assert by[str(docfile)]["file_time"], "a real file gets its mtime"
            assert by["zim://Foo"]["pct"] == 0 and by["zim://Foo"]["file_time"] == ""
            ok("sources progress: per-doc distilled % + file date over /browse")

            # the QUEUE: ingested-but-never-distilled docs surface + are counted
            pq = pstore.pending_sources(str(kbfile), 50)
            assert pq["total_docs"] == 3 and pq["pending_docs"] == 1, pq
            assert [r0["doc_id"] for r0 in pq["rows"]] == ["new-book.pdf"]
            assert pq["rows"][0]["chunks"] == 2 and pq["rows"][0]["title"] == "Fresh"
            assert pstore.pending_sources(str(Path(td) / "no-such.db"), 5) == {}
            httpd.kb = SimpleNamespace(list_sources=lambda n=200: srows)
            store0, httpd.store = httpd.store, pstore
            scfg["_master_kb_path"] = str(kbfile)
            try:
                with urllib.request.urlopen(
                        f"http://127.0.0.1:{port}/browse?kind=sources", timeout=5) as r:
                    res = json.loads(r.read())
            finally:
                httpd.kb, httpd.store = None, store0
                scfg.pop("_master_kb_path", None)
            assert res["totals"] == {"docs": 3, "queued": 1}, res["totals"]
            assert len(res["pending"]) == 1
            p0 = res["pending"][0]
            assert p0["doc_id"] == "new-book.pdf" and p0["status"] == "queued"
            assert p0["pct"] == 0 and p0["distilled"] == 0 and p0["file_time"] == ""
            ok("queued sources: counted in totals + listed with status=queued")

            # the vinkona exporter's client speaks the same lane
            sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent
                                   / "vinkona" / "assistant"))
            try:
                from research_export import _hash16, _post_drop, negotiate_drop
                assert _post_drop(f"http://127.0.0.1:{port}", "s3cret", name, DROP) is False
                assert _post_drop(f"http://127.0.0.1:{port}", "s3cret",
                                  "fedcba9876543210.md", DROP) is True
                try:
                    _post_drop(f"http://127.0.0.1:{port}", "s3cret", "bad name.md", DROP)
                    raise AssertionError("rejected drop must raise client-side")
                except Exception as e:
                    assert not isinstance(e, AssertionError)
                ok("vinkona _post_drop round-trip (idempotent + error surfaced)")

                status, hs = negotiate_drop(f"http://127.0.0.1:{port}", "s3cret")
                assert status == "ok" and hs["accepts"], (status, hs)
                # THE cross-repo contract: the host's inventory hash must equal
                # vinkona's local fingerprint, or skip-if-held would never skip.
                assert hs["drops"][name] == _hash16(DROP), (hs["drops"][name], _hash16(DROP))
                status, _ = negotiate_drop(f"http://127.0.0.1:{port}", "wrong")
                assert status == "denied", status
                status, _ = negotiate_drop("http://127.0.0.1:1", "s3cret")
                assert status == "down", status
                ok("vinkona negotiate_drop vs the real host: ok/denied/down + hash contract")
            except ImportError:
                print("  --    (vinkona checkout not adjacent — client round-trip skipped)")
        finally:
            httpd.shutdown()

    # ── DistillLM: 404 model-name self-heal (llama-server ignored the request
    #    "model" field; vLLM validates it — llama-era names must reconcile) ──
    import http.server
    from knowledgehost.distill import DistillLM

    class _LM(http.server.BaseHTTPRequestHandler):
        served = ["served-name"]

        def _json(self, code, obj):
            body = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            self._json(200, {"data": [{"id": i} for i in self.served]})

        def do_POST(self):
            n = int(self.headers.get("Content-Length") or 0)
            req = json.loads(self.rfile.read(n))
            if req.get("model") not in self.served:
                self._json(404, {"message": "model does not exist"})
            else:
                self._json(200, {"choices": [{"message": {"content": "{}"}}]})

        def log_message(self, *a):
            pass

    lm_srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _LM)
    threading.Thread(target=lm_srv.serve_forever, daemon=True).start()
    lm_url = f"http://127.0.0.1:{lm_srv.server_address[1]}"
    lm_cfg = {"distill_url": lm_url, "distill_model": "stale-gguf-name",
              "distill_timeout_s": 5}

    lm = DistillLM(lm_cfg)
    assert lm.warmup() is True, "single-model server: 404 should self-heal"
    assert lm.model == "served-name", lm.model
    assert lm.warmup() is True
    ok("DistillLM: stale name adopts the single served id (warmup survives)")

    _LM.served = ["model-a", "model-b"]
    lm2 = DistillLM(lm_cfg)
    try:
        lm2._post({"model": lm2.model, "messages": []})
        raise AssertionError("ambiguous server must not silently adopt")
    except urllib.error.HTTPError as e:
        assert "model-a" in str(e.reason) and "model-b" in str(e.reason), e.reason
    assert lm2.model == "stale-gguf-name"
    ok("DistillLM: several served ids -> 404 surfaces them, no silent pick")

    # Through the REAL flow: warmup() folds HTTPError into False, so the
    # mismatch must reach the operator via the WARNING log instead.
    import logging as _logging

    class _Grab(_logging.Handler):
        records = []

        def emit(self, record):
            type(self).records.append(record.getMessage())

    grab = _Grab()
    _logging.getLogger("distill").addHandler(grab)
    try:
        lm3 = DistillLM(lm_cfg)
        assert lm3.warmup() is False, "ambiguous server: warmup stays False"
        warned = [m for m in _Grab.records if "model-a" in m and "model-b" in m]
        assert warned and "stale-gguf-name" in warned[0], _Grab.records
    finally:
        _logging.getLogger("distill").removeHandler(grab)
    ok("DistillLM: warmup() path logs the served names (not a silent 'down')")

    lm_srv.shutdown()

    # ── config loading: library keys + the loud TOML-placement warnings ──────
    import contextlib
    import io

    with tempfile.TemporaryDirectory() as td:
        good = Path(td) / "good.toml"
        good.write_text('library_root = "lib"\n'
                        'library_sources = ["lib/papers", "/abs/books"]\n'
                        '[[serving.llms]]\nname = "p"\nengine = "vllm"\n'
                        'model = "m"\nport = 1\n')
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            c = load_config(str(good))
        assert c["library_root"].endswith("/lib") and Path(c["library_root"]).is_absolute()
        assert c["library_sources"][0].endswith("/lib/papers")
        assert c["library_sources"][1] == "/abs/books"
        assert err.getvalue() == "", err.getvalue()
        ok("library_root/library_sources load + resolve when placed top-level")

        trap = Path(td) / "trap.toml"
        trap.write_text('[serving]\nswap_timeout_s = 60\n'
                        'library_root = "lib"\n'          # swallowed by [serving]
                        'librari_sources = ["x"]\n')      # typo, also inside table
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            c = load_config(str(trap))
        assert not c["library_root"], "swallowed key must not leak to top level"
        msg = err.getvalue()
        assert "library_root" in msg and "INSIDE [serving]" in msg and "ABOVE" in msg, msg
        ok("a top-level key below a [table] header warns loudly instead of vanishing")

        typo = Path(td) / "typo.toml"
        typo.write_text('librari_root = "lib"\n')
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            load_config(str(typo))
        msg = err.getvalue()
        assert "unknown key 'librari_root'" in msg and "library_root" in msg, msg
        ok("an unknown/typo'd key warns with the close-match suggestion")

    # ── distill stage counters: why "0 cards" happened, not just that it did ──
    from knowledgehost import distill as D

    D._stage_reset()
    D._stage_add(proc_offered=3, crit_offered=2)
    D._stage_add(proc_kept=1)
    s = D.stage_stats()
    assert (s["proc_offered"], s["crit_offered"], s["proc_kept"], s["crit_kept"]) \
        == (3, 2, 1, 0), s
    assert "offered 3 proc / 2 crit" in D._stage_line()
    ok("stage counters: offered/kept accumulate; progress line renders them")

    import logging as _lg

    class _DGrab(_lg.Handler):
        msgs = []

        def emit(self, r):
            type(self).msgs.append(r.getMessage())

    dgrab = _DGrab()
    D.log.addHandler(dgrab)
    lvl0 = D.log.level
    D.log.setLevel(_lg.INFO)
    seq0 = D._distill_sequential
    try:
        # offered-but-dropped -> the validation warning names the counts
        D._distill_sequential = lambda *a, **k: (D._stage_add(proc_offered=4),
                                                 {"chunks": 5, "cards": 0})[1]
        res = D.distill_corpus(None, None, [object()], None, {"verify": False})
        assert res["proc_offered"] == 4 and res["cards"] == 0
        assert any("validation dropped" in m for m in _DGrab.msgs), _DGrab.msgs
        # offered-nothing -> the corpus/empty-array-exit explanation
        _DGrab.msgs.clear()
        D._distill_sequential = lambda *a, **k: {"chunks": 5, "cards": 0}
        res = D.distill_corpus(None, None, [object()], None, {"verify": False})
        assert res["proc_offered"] == 0
        assert any("offered no procedures/criteria" in m for m in _DGrab.msgs)
        # cards flowing -> no diagnosis noise
        _DGrab.msgs.clear()
        D._distill_sequential = lambda *a, **k: {"chunks": 5, "cards": 3}
        D.distill_corpus(None, None, [object()], None, {"verify": False})
        assert not _DGrab.msgs, _DGrab.msgs
    finally:
        D._distill_sequential = seq0
        D.log.removeHandler(dgrab)
        D.log.setLevel(lvl0)
    ok("distill_corpus: 0-card runs log WHICH drought it was; healthy runs stay quiet")

    # ── parallel fan-out: one vLLM endpoint becomes N in-flight request slots ──
    vcfg = {"verify": False, "ingest_log_every": 0, "serving": {"llms": [
        {"name": "primary", "engine": "container", "port": 11438, "exclusive": True},
        {"name": "tiny", "engine": "llama", "port": 11441},
        {"name": "capped", "engine": "vllm", "port": 11450, "max_num_seqs": 3}]}}
    lm_v = SimpleNamespace(url="http://127.0.0.1:11438")
    lm_l = SimpleNamespace(url="http://127.0.0.1:11441")
    lm_c = SimpleNamespace(url="http://127.0.0.1:11450")
    lm_r = SimpleNamespace(url="http://10.0.0.7:8000")   # remote: engine unknowable
    assert D._endpoint_fanout(vcfg, lm_v) == 8
    assert D._endpoint_fanout(vcfg, lm_l) == 1
    assert D._endpoint_fanout(vcfg, lm_c) == 3, "entry's max_num_seqs caps auto"
    assert D._endpoint_fanout(vcfg, lm_r) == 1, "foreign endpoint stays sequential"
    assert D._endpoint_fanout({**vcfg, "distill_parallel": 4}, lm_r) == 4, "knob wins"
    assert D._endpoint_fanout({**vcfg, "distill_parallel": 1}, lm_v) == 1, "knob forces serial"
    fanned = D._fan_out(vcfg, [lm_v, lm_l])
    assert len(fanned) == 9 and fanned[0] is lm_v
    assert all(f.url == lm_v.url for f in fanned[:8])
    assert len({id(f) for f in fanned[:8]}) == 8, "clones are distinct pool entries"
    ok("_endpoint_fanout/_fan_out: vLLM->8 (max_num_seqs caps), llama/remote->1, knob overrides")

    # dispatch: a single batching endpoint now takes the PARALLEL path
    got = {}
    par0, seq0 = D._distill_parallel, D._distill_sequential

    def fake_par(store, kb, lms, embedder, cfg, **k):
        got["par"] = list(lms)
        return {"chunks": 0, "cards": 0}

    def fake_seq(store, kb, lm, embedder, cfg, **k):
        got["seq"] = lm
        return {"chunks": 0, "cards": 0}

    try:
        D._distill_parallel, D._distill_sequential = fake_par, fake_seq
        D.distill_corpus(None, None, [lm_v], None, vcfg)
        assert len(got.pop("par")) == 8 and "seq" not in got
        D.distill_corpus(None, None, [lm_l], None, vcfg)
        assert got.pop("seq") is lm_l and "par" not in got
    finally:
        D._distill_parallel, D._distill_sequential = par0, seq0
    ok("distill_corpus: one vLLM endpoint -> parallel x8; llama endpoint -> sequential")

    print(f"standalone_test: {PASS} checks OK")


if __name__ == "__main__":
    main()
