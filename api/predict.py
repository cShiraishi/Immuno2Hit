"""Vercel Function: POST /api/predict

Takes fingerprints, not SMILES. The browser computes them with RDKit.js, which is
bit-for-bit identical to RDKit Python for the three fingerprints these models use, so
this bundle needs no RDKit — 148 MB of the 500 MB function limit saved.

Body: {"compounds": [{"smiles": "...", "fps": {"Morgan": "0101...", "MACCS": "...",
                                               "RDKit": "..."}}, ...]}
"""
from __future__ import annotations

import json
import sys
from http.server import BaseHTTPRequestHandler
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))

from core import MAX_BATCH, boot, predict_fps  # noqa: E402

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
            compounds = req.get("compounds") or []
            if len(compounds) > MAX_BATCH:
                self._send(413, {"error": f"limit of {MAX_BATCH} compounds per run — "
                                          f"{len(compounds)} were submitted"})
                return
            self._send(200, {"results": predict_fps(compounds), "max_batch": MAX_BATCH})
        except Exception as exc:
            self._send(500, {"error": str(exc)})
