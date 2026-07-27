"""Vercel Function: POST /api/predict

Runs the same scoring core as the local server. Vercel maps api/<name>.py to /api/<name>
and loads the top-level `handler`. Models are read from backend/models at cold start,
which measured 1.17 s locally — the whole point of dropping pandas and swapping the
xgboost wheel for xgboost-cpu was to keep this bundle under the 500 MB function limit.
"""
from __future__ import annotations

import json
import sys
from http.server import BaseHTTPRequestHandler
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))

from core import MAX_BATCH, boot, predict  # noqa: E402

boot()  # once per cold start, not per request


class handler(BaseHTTPRequestHandler):
    def _send(self, code: int, payload: dict) -> None:
        body = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        try:
            n = int(self.headers.get("Content-Length", 0))
            req = json.loads(self.rfile.read(n) or b"{}")
            raw = req.get("smiles", "")
            items = [s for s in (raw if isinstance(raw, list) else raw.splitlines()) if s.strip()]
            if len(items) > MAX_BATCH:
                self._send(413, {"error": f"limit of {MAX_BATCH} compounds per run — "
                                          f"{len(items)} were submitted"})
                return
            self._send(200, {"results": predict(items), "max_batch": MAX_BATCH})
        except Exception as exc:
            self._send(500, {"error": str(exc)})
