"""metrics.py — Prometheus-compatible metrics HTTP server (no extra dependencies)."""

import asyncio
import json
import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, HTTPServer
from statistics import mean, median, stdev
from threading import Thread
from typing import Optional

log = logging.getLogger("metrics")


# ── Metric primitives ─────────────────────────────────────────────────────────

class Counter:
    def __init__(self, name: str, help: str):
        self.name = name
        self.help = help
        self._v: dict = defaultdict(int)

    def inc(self, amount: int = 1, **labels):
        self._v[tuple(sorted(labels.items()))] += amount

    def get(self, **labels) -> int:
        return self._v[tuple(sorted(labels.items()))]

    def render(self) -> str:
        lines = [f"# HELP {self.name} {self.help}", f"# TYPE {self.name} counter"]
        for key, val in self._v.items():
            lbl = ",".join(f'{k}="{v}"' for k, v in key)
            lines.append(f"{self.name}{{{lbl}}} {val}" if lbl else f"{self.name} {val}")
        return "\n".join(lines)


class Gauge:
    def __init__(self, name: str, help: str):
        self.name  = name
        self.help  = help
        self._v    = 0.0

    def set(self, v: float):  self._v = v
    def inc(self, v: float = 1): self._v += v
    def dec(self, v: float = 1): self._v -= v

    def render(self) -> str:
        return f"# HELP {self.name} {self.help}\n# TYPE {self.name} gauge\n{self.name} {self._v}"


class Histogram:
    _BUCKETS = (100, 250, 500, 750, 1000, 1500, 2000, 3000, float("inf"))

    def __init__(self, name: str, help: str):
        self.name  = name
        self.help  = help
        self._obs: list[float] = []

    def observe(self, v: float):
        self._obs.append(v)

    def render(self) -> str:
        lines = [f"# HELP {self.name} {self.help}", f"# TYPE {self.name} histogram"]
        if not self._obs:
            lines += [f"{self.name}_count 0", f"{self.name}_sum 0"]
            return "\n".join(lines)
        for b in self._BUCKETS:
            c    = sum(1 for o in self._obs if o <= b)
            label = "+Inf" if b == float("inf") else str(b)
            lines.append(f'{self.name}_bucket{{le="{label}"}} {c}')
        lines += [f"{self.name}_count {len(self._obs)}", f"{self.name}_sum {sum(self._obs):.1f}"]
        return "\n".join(lines)

    def summary(self) -> dict:
        if not self._obs:
            return {}
        s = sorted(self._obs)
        n = len(s)
        return {
            "count":  n,
            "mean":   round(mean(self._obs), 1),
            "median": round(median(self._obs), 1),
            "p95":    round(s[int(n * 0.95)], 1),
            "p99":    round(s[int(n * 0.99)], 1),
            "max":    round(max(self._obs), 1),
            "stdev":  round(stdev(self._obs), 1) if n > 1 else 0.0,
        }


# ── Collector ─────────────────────────────────────────────────────────────────

class MetricsCollector:
    """Central registry. Feed with record_utterance() after each utterance."""

    def __init__(self):
        self.utterances   = Counter("voice_pipeline_utterances_total",   "Total utterances")
        self.errors       = Counter("voice_pipeline_errors_total",       "Errors by stage")
        self.retries      = Counter("voice_pipeline_retries_total",      "Retries by stage")
        self.barge_ins    = Counter("voice_pipeline_barge_ins_total",    "Barge-in interrupts")
        self.wake_detects = Counter("voice_pipeline_wake_detections_total", "Wake word detections")
        self.active       = Gauge("voice_pipeline_active_sessions",      "Active sessions")
        self.asr_latency  = Histogram("voice_pipeline_asr_latency_ms",  "ASR latency ms")
        self.llm_ttft     = Histogram("voice_pipeline_llm_ttft_ms",     "LLM TTFT ms")
        self.tts_ttfb     = Histogram("voice_pipeline_tts_ttfb_ms",     "TTS TTFB ms")
        self.total_lat    = Histogram("voice_pipeline_total_latency_ms", "End-to-end latency ms")
        self._start       = time.time()
        self._all         = [self.utterances, self.errors, self.retries, self.barge_ins,
                             self.wake_detects, self.active,
                             self.asr_latency, self.llm_ttft, self.tts_ttfb, self.total_lat]

    def record_utterance(self, record: dict):
        self.utterances.inc()
        for attr, key in [(self.asr_latency, "asr_latency_ms"),
                          (self.llm_ttft,    "llm_ttft_ms"),
                          (self.tts_ttfb,    "tts_ttfb_ms"),
                          (self.total_lat,   "total_ms")]:
            if record.get(key):
                attr.observe(record[key])

    def inc_error(self, stage: str):   self.errors.inc(stage=stage)
    def inc_retry(self, stage: str):   self.retries.inc(stage=stage)
    def inc_barge_in(self):            self.barge_ins.inc()
    def inc_wake(self):                self.wake_detects.inc()

    def render_prometheus(self) -> str:
        uptime = time.time() - self._start
        parts  = [f"# HELP voice_pipeline_uptime_seconds Uptime\n"
                  f"# TYPE voice_pipeline_uptime_seconds gauge\n"
                  f"voice_pipeline_uptime_seconds {uptime:.1f}"]
        parts += [m.render() for m in self._all]
        return "\n".join(parts) + "\n"

    def summary_json(self) -> str:
        return json.dumps({
            "uptime_s":         round(time.time() - self._start, 1),
            "utterances":       self.utterances.get(),
            "barge_ins":        self.barge_ins.get(),
            "wake_detections":  self.wake_detects.get(),
            "errors":           {s: self.errors.get(stage=s) for s in ("asr", "llm", "tts")},
            "asr_latency_ms":   self.asr_latency.summary(),
            "llm_ttft_ms":      self.llm_ttft.summary(),
            "tts_ttfb_ms":      self.tts_ttfb.summary(),
            "total_latency_ms": self.total_lat.summary(),
        }, indent=2)

    async def serve(self, port: int = 9090):
        collector = self

        class _Handler(BaseHTTPRequestHandler):
            def log_message(self, *_): pass
            def do_GET(self):
                if self.path == "/metrics":
                    body = collector.render_prometheus().encode()
                    ct   = "text/plain; version=0.0.4; charset=utf-8"
                elif self.path in ("/summary", "/"):
                    body = collector.summary_json().encode()
                    ct   = "application/json"
                elif self.path == "/health":
                    body = b'{"status":"ok"}'
                    ct   = "application/json"
                else:
                    self.send_response(404); self.end_headers(); return
                self.send_response(200)
                self.send_header("Content-Type", ct)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        server = HTTPServer(("", port), _Handler)
        Thread(target=server.serve_forever, daemon=True).start()
        log.info(f"Metrics: http://localhost:{port}/metrics | /summary | /health")
        try:
            while True:
                await asyncio.sleep(3600)
        except asyncio.CancelledError:
            server.shutdown()
