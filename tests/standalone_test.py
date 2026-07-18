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
        scfg.update({"host": "127.0.0.1", "port": 0, "auth_token": "s3cret",
                     "research_solved_dir": str(Path(td) / "solved2"),
                     "control_dir": str(Path(td) / "ctrl")})
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

            # the vinkona exporter's client speaks the same lane
            sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent
                                   / "vinkona" / "assistant"))
            try:
                from research_export import _post_drop
                assert _post_drop(f"http://127.0.0.1:{port}", "s3cret", name, DROP) is False
                assert _post_drop(f"http://127.0.0.1:{port}", "s3cret",
                                  "fedcba9876543210.md", DROP) is True
                try:
                    _post_drop(f"http://127.0.0.1:{port}", "s3cret", "bad name.md", DROP)
                    raise AssertionError("rejected drop must raise client-side")
                except Exception as e:
                    assert not isinstance(e, AssertionError)
                ok("vinkona _post_drop round-trip (idempotent + error surfaced)")
            except ImportError:
                print("  --    (vinkona checkout not adjacent — client round-trip skipped)")
        finally:
            httpd.shutdown()

    print(f"standalone_test: {PASS} checks OK")


if __name__ == "__main__":
    main()
