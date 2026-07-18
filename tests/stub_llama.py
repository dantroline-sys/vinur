#!/usr/bin/env python3
"""Stub llama-server for tests/swap_live.sh: binds --host/--port and answers
/health with 503 (loading) for $STUB_DELAY seconds, then 200 — the same
readiness shape the real llama-server and vLLM expose."""
import os
import sys
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

args = sys.argv[1:]
port = int(args[args.index("--port") + 1]) if "--port" in args else 8080
host = args[args.index("--host") + 1] if "--host" in args else "127.0.0.1"
ready_at = time.time() + float(os.environ.get("STUB_DELAY", "2"))


class H(BaseHTTPRequestHandler):
    def do_GET(self):
        code = 200 if time.time() >= ready_at else 503
        self.send_response(code)
        self.send_header("Content-Length", "0")
        self.end_headers()

    def log_message(self, *_):
        pass


HTTPServer((host, port), H).serve_forever()
