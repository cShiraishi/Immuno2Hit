"""Scoring core for Immuno2Hit — shared by the local server and the Vercel function.

Holds the model registry, the consensus rules and the per-compound scoring. No HTTP and
no pandas, so a serverless bundle carries only numpy, scikit-learn, xgboost and RDKit.
"""
from __future__ import annotations

import json
import os
import pickle
import sys
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))


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


