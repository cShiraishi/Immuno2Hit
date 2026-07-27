"""Vercel Function: GET /api/health

Deliberately imports nothing heavy — the frontend probes this to decide whether a
prediction backend exists, and it must answer even while the predict function is cold.
"""
from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler


class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        body = json.dumps({"ok": True, "service": "immuno2hit"}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)
