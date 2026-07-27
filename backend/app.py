"""Immuno2Hit — local server for the TREM2 / CD28 / IDO1 models.

Serves the frontend and the same /api/predict the Vercel function exposes; the scoring
itself lives in core.py so both entry points share one implementation.

    python app.py            # http://127.0.0.1:8765
    python app.py --port 9000
"""
from __future__ import annotations

import json
import os
import sys
import threading
import warnings
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

warnings.filterwarnings("ignore")

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

# the frontend sits at the repository root, where Vercel publishes it
STATIC = HERE / "static" if (HERE / "static").is_dir() else HERE.parent / "static"

from core import ALLOWED_ORIGIN, MAX_BATCH, MODELS, TARGETS, boot, predict  # noqa: E402

class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):  # keep the console readable
        pass

    def _send(self, code: int, body: bytes, ctype: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        # the frontend may be served from a different origin (e.g. Vercel) than this API
        self.send_header("Access-Control-Allow-Origin", ALLOWED_ORIGIN)
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", ALLOWED_ORIGIN)
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Access-Control-Allow-Methods", "POST, GET, OPTIONS")
        self.end_headers()

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            self._send(200, (STATIC / "index.html").read_bytes(), "text/html; charset=utf-8")
        elif self.path == "/api/health":
            # the frontend probes this to decide whether scoring is available
            self._send(200, json.dumps({"ok": True, "service": "immuno2hit"}).encode(),
                       "application/json")
        elif self.path == "/api/targets":
            meta = {t: {"models": [m["model_key"] for m in MODELS[t]],
                        "rule": c["rule_label"], "note": c["note"], "original": c["original"]}
                    for t, c in TARGETS.items()}
            self._send(200, json.dumps(meta).encode(), "application/json")
        else:
            self._send(404, b"not found", "text/plain")

    def do_POST(self):
        if self.path != "/api/predict":
            self._send(404, b"not found", "text/plain")
            return
        n = int(self.headers.get("Content-Length", 0))
        try:
            req = json.loads(self.rfile.read(n) or b"{}")
            raw = req.get("smiles", "")
            items = raw if isinstance(raw, list) else raw.splitlines()
            items = [s for s in items if s.strip()]
            if len(items) > MAX_BATCH:
                # refuse rather than silently scoring the first 100 and reporting success
                self._send(413, json.dumps({
                    "error": f"limit of {MAX_BATCH} compounds per run — {len(items)} were submitted"
                }).encode(), "application/json")
                return
            body = json.dumps({"results": predict(items), "max_batch": MAX_BATCH}).encode()
            self._send(200, body, "application/json")
        except Exception as exc:  # surface the reason instead of a blank page
            self._send(500, json.dumps({"error": str(exc)}).encode(), "application/json")


def main() -> None:
    port = int(os.environ.get("PORT", 8765))
    if "--port" in sys.argv:
        port = int(sys.argv[sys.argv.index("--port") + 1])
    print("Immuno2Hit — loading models (the CD28 k-NN is 110 MB, takes a few seconds)...")
    boot()
    url = f"http://127.0.0.1:{port}"
    print(f"\nReady: {url}\nCtrl+C to stop.\n")
    threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    try:
        ThreadingHTTPServer((os.environ.get("HOST", "127.0.0.1"), port), Handler).serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
