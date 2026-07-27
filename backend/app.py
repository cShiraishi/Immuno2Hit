"""Immuno2Hit — local prediction platform for the TREM2 / CD28 / IDO1 models.

Paste SMILES, get the per-model probability, applicability-domain status and the
consensus verdict each paper defines. Stdlib HTTP server only — no Streamlit, no
external web dependency.

    python app.py            # http://127.0.0.1:8765
    python app.py --port 9000
"""
from __future__ import annotations

import json
import os
import pickle
import sys
import threading
import warnings
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

# the frontend lives at the repository root so that Vercel, which only deploys the
# static site, never sees this directory's requirements.txt and tries to build Python
STATIC = HERE / "static" if (HERE / "static").is_dir() else HERE.parent / "static"

from fingerprints import featurize_one  # noqa: E402

# Deliberately NOT importing from predict.py: that module pulls in pandas for its
# DataFrame output, and pandas is 39.8 MB installed on Linux. The API only needs
# pickle and numpy, and the 40 MB matters against a serverless bundle limit.


def load(path) -> dict:
    """Load a packaged model .pkl."""
    with open(path, "rb") as fh:
        return pickle.load(fh)


def _ad_predict(ad: dict, X):
    """Mean Tanimoto distance to the k nearest training neighbours, and the AD verdict."""
    distances, _ = ad["nn"].kneighbors(X, n_neighbors=ad["k_neighbors"])
    mean_dists = distances.mean(axis=1)
    return mean_dists <= ad["threshold_AD"], mean_dists

# Each target: the models, the consensus rule, and how much to trust it.
TARGETS = {
    "TREM2": {
        "files": ["RDKit_SVM", "RDKit_LogisticRegression", "Morgan_SVM"],
        "rule": "majority",
        "rule_label": "majority vote — 2 of 3 members above their own Youden threshold",
        "note": "Original models from the paper.",
        "base_rate": "18.1% of a 66,560-compound Enamine library clears this gate; the paper applies five further filters to reach 70 hits.",
        "original": True,
    },
    "CD28": {
        "files": ["Morgan_XGBoost", "Morgan_KNN"],
        "rule": "mean",
        "rule_label": "mean probability ≥ 0.60, inside the applicability domain",
        "note": "Original models from the paper, trained on the REINVENT4-augmented set.",
        "base_rate": "3.7% of 8,960 compounds clear this gate; 14 survived the full funnel.",
        "original": True,
    },
    "IDO1": {
        "files": ["RDKit_XGBoost", "Morgan_RandomForest", "MACCS_SVM"],
        "rule": "mean",
        "rule_label": "mean probability ≥ 0.60, inside the applicability domain",
        "note": "Baseline consensus, RETRAINED — the original models were never saved. "
                "Regimes A and C exist in the paper to show that augmentation inflates the "
                "metric, so they are deliberately left out of a prediction platform.",
        "base_rate": "0.2% of 4,800 compounds clear this gate; 49 formed the consensus core.",
        "original": False,
    },
}

MEAN_GATE = 0.60
MAX_BATCH = 100  # compounds per run
ALLOWED_ORIGIN = os.environ.get("ALLOWED_ORIGIN", "*")
MODELS: dict[str, list[dict]] = {}


def boot() -> None:
    for target, cfg in TARGETS.items():
        MODELS[target] = [load(HERE / "models" / target / f"{name}.pkl") for name in cfg["files"]]
        print(f"  {target}: {len(MODELS[target])} models")


def depict(smiles: str) -> str | None:
    try:
        from rdkit import Chem
        from rdkit.Chem.Draw import rdMolDraw2D
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        drawer = rdMolDraw2D.MolDraw2DSVG(260, 190)
        opts = drawer.drawOptions()
        opts.clearBackground = False
        rdMolDraw2D.PrepareAndDrawMolecule(drawer, mol)
        drawer.FinishDrawing()
        svg = drawer.GetDrawingText()
        # the <?xml ...?> prolog is not valid inside innerHTML and disturbs the layout
        return svg[svg.index("<svg"):] if "<svg" in svg else svg
    except Exception:
        return None


def score_target(target: str, smiles: str) -> dict:
    cfg = TARGETS[target]
    members, probs, votes, inside_all = [], [], [], True

    for payload in MODELS[target]:
        X = featurize_one(smiles, payload["fp"], payload["n_bits"], payload["radius"])
        if X is None:
            return {"valid": False}
        X = X.reshape(1, -1)
        prob = float(payload["model"].predict_proba(X)[0, 1])
        probs.append(prob)

        cut = payload.get("decision_threshold")
        vote = None
        if cut is not None:
            vote = prob >= cut
            votes.append(vote)

        ad = payload.get("ad")
        inside = dist = None
        if ad:
            ok, d = _ad_predict(ad, X)
            inside, dist = bool(ok[0]), float(d[0])
            inside_all = inside_all and inside

        members.append({
            "key": payload["model_key"],
            "prob": prob,
            "threshold": cut,
            "vote": vote,
            "inside_ad": inside,
            "ad_distance": dist,
        })

    mean = float(np.mean(probs))
    if cfg["rule"] == "majority":
        active = sum(bool(v) for v in votes) > len(votes) / 2
        detail = f"{sum(bool(v) for v in votes)}/{len(votes)} members positive"
    else:
        active = mean >= MEAN_GATE and inside_all
        detail = f"mean {mean:.3f} (cut-off {MEAN_GATE:.2f})"
        if not inside_all:
            detail += " · outside the applicability domain"

    return {
        "valid": True,
        "members": members,
        "prob_mean": mean,
        "active": bool(active),
        "detail": detail,
        "inside_ad": inside_all,
        "rule_label": cfg["rule_label"],
        "note": cfg["note"],
        "base_rate": cfg["base_rate"],
        "original": cfg["original"],
    }


def predict(smiles_list: list[str]) -> list[dict]:
    out = []
    for smi in smiles_list:
        smi = smi.strip()
        if not smi:
            continue
        svg = depict(smi)
        row = {"smiles": smi, "valid": svg is not None, "svg": svg, "targets": {}}
        if row["valid"]:
            for target in TARGETS:
                row["targets"][target] = score_target(target, smi)
        out.append(row)
    return out


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
