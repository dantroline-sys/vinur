"""Always-on performance telemetry (VINUR-UI-01, Stage 6).

A cheap sampler thread banks time-series history into its own SQLite file
(``var/metrics.db`` — never kb.db: telemetry is not knowledge), so the Stats
tab and A/B comparisons have data from the moment the server first ran, not
from the moment someone opened a chart.  Per tick it collects:

  * **nvidia-smi** — per-GPU utilisation %, VRAM, power, temperature
    (one short subprocess; no binary → no GPU series, logged once).
  * **vLLM /metrics** — requests running, requests WAITING (the queue),
    KV-cache %, token counters — from every local [[serving.llms]] entry
    with a batching engine that answers.  Metric names drift between vLLM
    releases, so parsing is by-name with aliases and skips the unknown.
  * **KB counts** — the cached ``kb.counts()`` (nodes/edges/cards/…), so
    throughput rates can be computed over ANY window server-side; the
    chunk count (a table scan) is sampled on a slower cadence.
  * **Ops transitions** — the single-slot runner's job start/end (with the
    parsed OPS_RESULT stats) as labelled events.  The Prioritizer drives
    the same runner, so panel jobs and autopilot steps both annotate the
    charts for free.

Everything degrades gracefully; a sampling failure skips that source for
the tick and never kills the thread.
"""
from __future__ import annotations

import json
import logging
import sqlite3
import subprocess
import threading
import time
import urllib.request
from pathlib import Path

log = logging.getLogger("knowledgehost.metrics")

ROOT = Path(__file__).resolve().parent.parent

# chunk-count (and other table-scan) sampling: every Nth tick
SLOW_EVERY = 12
# retention prune: roughly hourly at the default 5s interval
PRUNE_EVERY = 720


def db_path(cfg: dict) -> str:
    return str(Path(cfg.get("metrics_db") or (ROOT / "var" / "metrics.db")).expanduser())


_SCHEMA = """
CREATE TABLE IF NOT EXISTS samples(
  ts     REAL NOT NULL,
  series TEXT NOT NULL,
  value  REAL NOT NULL);
CREATE INDEX IF NOT EXISTS samples_series_ts ON samples(series, ts);
CREATE INDEX IF NOT EXISTS samples_ts ON samples(ts);
CREATE TABLE IF NOT EXISTS events(
  ts    REAL NOT NULL,
  kind  TEXT NOT NULL,
  label TEXT NOT NULL,
  data  TEXT);
CREATE INDEX IF NOT EXISTS events_ts ON events(ts);
"""


class MetricsStore:
    """Telemetry SQLite store.  Short-lived connection per operation — the
    sampler thread writes, HTTP handler threads read, and nobody shares a
    connection across threads."""

    def __init__(self, path: str):
        self.path = str(path)
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        with self._con() as con:
            con.executescript(_SCHEMA)

    def _con(self) -> sqlite3.Connection:
        con = sqlite3.connect(self.path, timeout=10)
        con.execute("PRAGMA journal_mode=WAL")
        return con

    def add(self, ts: float, kv: dict) -> None:
        if not kv:
            return
        with self._con() as con:
            con.executemany("INSERT INTO samples(ts, series, value) VALUES (?,?,?)",
                            [(ts, k, float(v)) for k, v in kv.items()])

    def event(self, kind: str, label: str, data: dict | None = None,
              ts: float | None = None) -> None:
        with self._con() as con:
            con.execute("INSERT INTO events(ts, kind, label, data) VALUES (?,?,?,?)",
                        (ts if ts is not None else time.time(), kind, label,
                         json.dumps(data) if data else None))

    def prune(self, keep_days: float) -> None:
        cutoff = time.time() - float(keep_days) * 86400
        with self._con() as con:
            con.execute("DELETE FROM samples WHERE ts < ?", (cutoff,))
            con.execute("DELETE FROM events WHERE ts < ?", (cutoff,))

    def history(self, mins: float, step: float = 0, now: float | None = None) -> dict:
        """Bucket-averaged series + raw events over the last `mins` minutes.
        Auto step targets ≤ ~600 points per series so a 7-day chart is one
        light payload, not four million rows."""
        now = now if now is not None else time.time()
        span = max(60.0, float(mins) * 60)
        t0 = now - span
        bucket = max(float(step) or span / 600, 1.0)
        series: dict = {}
        with self._con() as con:
            rows = con.execute(
                "SELECT series, CAST((ts - ?) / ? AS INTEGER) AS b, AVG(value) "
                "FROM samples WHERE ts >= ? GROUP BY series, b ORDER BY b",
                (t0, bucket, t0)).fetchall()
            evs = con.execute(
                "SELECT ts, kind, label, data FROM events WHERE ts >= ? ORDER BY ts",
                (t0,)).fetchall()
        for name, b, avg in rows:
            series.setdefault(name, []).append(
                [round(t0 + (b + 0.5) * bucket, 1), round(avg, 3)])
        events = []
        for ts, kind, label, data in evs:
            try:
                d = json.loads(data) if data else None
            except ValueError:
                d = None
            events.append({"ts": ts, "kind": kind, "label": label, "data": d})
        return {"ok": True, "now": now, "bucket": bucket,
                "series": series, "events": events}


# ── GPU: nvidia-smi ──────────────────────────────────────────────────────────

_GPU_QUERY = "index,utilization.gpu,memory.used,memory.total,power.draw,temperature.gpu"
_GPU_FIELDS = ("util", "vram_mb", "vram_total_mb", "power_w", "temp_c")


def sample_gpu(timeout: float = 3.0) -> dict:
    """Per-GPU series, or {} when nvidia-smi is missing/failing.  Values the
    driver reports as "[N/A]" are skipped, not zeroed."""
    try:
        out = subprocess.run(
            ["nvidia-smi", f"--query-gpu={_GPU_QUERY}", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=timeout)
    except (OSError, subprocess.SubprocessError):
        return {}
    if out.returncode != 0:
        return {}
    kv: dict = {}
    for line in out.stdout.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 1 + len(_GPU_FIELDS):
            continue
        idx = parts[0]
        for name, raw in zip(_GPU_FIELDS, parts[1:]):
            try:
                kv[f"gpu{idx}.{name}"] = float(raw)
            except ValueError:
                pass
    return kv


# ── vLLM: the /metrics Prometheus endpoint ───────────────────────────────────

# (accepted metric names…) -> our short series name.  gpu_cache_usage_perc is
# 0..1 in vLLM — scaled to % below.  Token *_total are cumulative counters;
# the UI turns them into tokens/s by differencing.
_PROM_MAP = (
    (("vllm:num_requests_running",), "running", 1.0),
    (("vllm:num_requests_waiting",), "waiting", 1.0),
    (("vllm:gpu_cache_usage_perc", "vllm:kv_cache_usage_perc"), "kv_pct", 100.0),
    (("vllm:prompt_tokens_total",), "prompt_toks", 1.0),
    (("vllm:generation_tokens_total",), "gen_toks", 1.0),
)


def parse_prom(text: str) -> dict:
    """The handful of series we chart, summed across label sets.  Anything
    unparseable is skipped — vLLM's exposition is large and version-drifty."""
    out: dict = {}
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        name = line.split("{", 1)[0].split(" ", 1)[0]
        for accepted, series, scale in _PROM_MAP:
            if name in accepted:
                try:
                    val = float(line.rsplit(" ", 1)[1])
                except (ValueError, IndexError):
                    break
                out[series] = out.get(series, 0.0) + val * scale
                break
    return out


def vllm_targets(cfg: dict) -> list:
    """(entry_name, metrics_url) for every LOCAL batching-engine entry.  A
    0.0.0.0/blank host is probed via loopback; foreign hosts are skipped
    (a remote box runs its own collector)."""
    local = {"127.0.0.1", "localhost", "::1", "0.0.0.0", ""}
    out = []
    for e in (cfg.get("serving") or {}).get("llms") or []:
        if str(e.get("engine")) not in ("vllm", "container"):
            continue
        host = str(e.get("host") or "127.0.0.1").lower()
        port = int(e.get("port") or 0)
        if not port or host not in local:
            continue
        name = str(e.get("name") or port).replace(".", "_")
        out.append((name, f"http://127.0.0.1:{port}/metrics"))
    return out


def sample_vllm(cfg: dict, timeout: float = 2.0) -> dict:
    """vllm.<entry>.<series> for every target that answers.  A dead endpoint
    (swapped out, restarting) is silently skipped — that's normal life in
    exclusive swap mode, not an error."""
    kv: dict = {}
    for name, url in vllm_targets(cfg):
        try:
            with urllib.request.urlopen(url, timeout=timeout) as r:
                text = r.read().decode("utf-8", "replace")
        except Exception:
            continue
        for series, val in parse_prom(text).items():
            kv[f"vllm.{name}.{series}"] = val
    return kv


# ── the sampler thread ───────────────────────────────────────────────────────

class Sampler(threading.Thread):
    """One tick every `stats_interval_s`: collect, write one batch, watch the
    ops runner for job transitions.  Every failure is per-source and
    non-fatal — a broken nvidia-smi must never cost KB history."""

    def __init__(self, cfg: dict, store: MetricsStore, *,
                 counts_fn=None, slow_fn=None, runner=None):
        super().__init__(daemon=True, name="metrics-sampler")
        self.cfg = cfg
        self.store = store
        self.counts_fn = counts_fn        # cheap (cached) -> sampled every tick
        self.slow_fn = slow_fn            # table scans    -> every SLOW_EVERY ticks
        self.runner = runner
        self.interval = float(cfg.get("stats_interval_s", 5) or 0)
        self.keep_days = float(cfg.get("stats_keep_days", 14) or 14)
        self._stop = threading.Event()
        self._ticks = 0
        self._op_seen = None              # runner job identity = its `started` stamp
        self._op_running = False
        self._gpu_warned = False

    def stop(self) -> None:
        self._stop.set()

    def run(self) -> None:
        while not self._stop.wait(self.interval):
            try:
                self.tick()
            except Exception:             # pragma: no cover - belt and braces
                log.exception("metrics tick failed")

    def tick(self) -> None:
        now = time.time()
        self._ticks += 1
        kv: dict = {}
        gpu = sample_gpu()
        if not gpu and not self._gpu_warned:
            self._gpu_warned = True
            log.info("metrics: nvidia-smi not answering — no GPU series "
                     "(fine on a non-NVIDIA box)")
        kv.update(gpu)
        kv.update(sample_vllm(self.cfg))
        for fn, slow in ((self.counts_fn, False), (self.slow_fn, True)):
            if fn is None or (slow and self._ticks % SLOW_EVERY != 1):
                continue
            try:
                for k, v in (fn() or {}).items():
                    if isinstance(v, (int, float)):
                        kv[k] = v
            except Exception:             # a mid-swap KB handle, etc.
                pass
        self.store.add(now, kv)
        self._watch_ops(now)
        if self._ticks % PRUNE_EVERY == 0:
            try:
                self.store.prune(self.keep_days)
            except Exception:             # pragma: no cover
                pass

    def _watch_ops(self, now: float) -> None:
        """Job start/end events from the single-slot runner.  `started` is the
        job's identity, so a job that began AND finished between ticks still
        gets both events (op_start back-stamped to its real start time)."""
        if self.runner is None:
            return
        try:
            st = self.runner.status()
        except Exception:                 # pragma: no cover
            return
        started = st.get("started")
        if started and started != self._op_seen:
            self._op_seen = started
            self._op_running = True
            self.store.event("op_start", st.get("command") or "?",
                             {"argv": st.get("argv") or []}, ts=started)
        if self._op_running and started == self._op_seen and not st.get("running"):
            self._op_running = False
            data = {"exit_code": st.get("exit_code")}
            try:
                res = self.runner.result() or {}
                data.update({k: v for k, v in res.items() if k != "command"})
            except Exception:             # pragma: no cover
                pass
            self.store.event("op_end", st.get("command") or "?", data, ts=now)
