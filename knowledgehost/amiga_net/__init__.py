"""amiga_net — the egress broker (AMIGA-OPS-01 §3.3).

The single place in this codebase that opens an outbound network connection.
Everything else talks to loopback peers or calls this package:

    from knowledgehost.amiga_net import broker
    data = broker.request("check model catalogue", url)          # small calls
    broker.download("model weights", url, dest, sha256=...)      # big files

Deny-by-default: a request is allowed only when it matches a rule in the
egress.toml policy file at the repo root — name, host patterns, port, methods,
and a plain-language purpose.  A rule may be a LEASE (ttl_seconds/max_uses):
it grants nothing until an operation opens it, and it closes itself — so an
idle Vinur has zero standing egress.  Every decision is appended to the audit
log (var/log/egress.jsonl): timestamp, component, purpose, destination, rule,
verdict, bytes.  Never bodies.

    python3 -m knowledgehost.amiga_net.status     # the user's window

Downloads resume (HTTP Range) and verify sha256 when the caller knows it.
The transfer engine is aria2c when installed (segmented, -c -x4), wget -c as
second choice, and a pure-stdlib single stream otherwise — so macOS and
Windows work with nothing installed.  Engines are subprocesses the broker
spawns and accounts for; they are not independent egress.

This package is the in-process forerunner of the native snitch daemon: the
policy file is plain data and the API is transport-shaped (B-13), so swapping
the implementation out later changes no caller.
"""
from . import broker  # noqa: F401  (the public surface)
