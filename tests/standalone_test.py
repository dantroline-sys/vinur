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

    print(f"standalone_test: {PASS} checks OK")


if __name__ == "__main__":
    main()
