"""Load the packaged .pkl models and score SMILES with them.

    from predict import load, screen
    models = [load("models/TREM2/RDKit_SVM.pkl"), load("models/TREM2/RDKit_LogisticRegression.pkl"),
              load("models/TREM2/Morgan_SVM.pkl")]
    df = screen(models, ["CCOc1ccccc1", "c1ccccc1"], vote="majority")

Command line:
    python predict.py models/IDO1/*.pkl --smiles library.csv --out scored.csv
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fingerprints import featurize  # noqa: E402
from qsar_ad import ApplicabilityDomain  # noqa: F401,E402  (kept importable for downstream use)


def load(path) -> dict:
    with open(path, "rb") as fh:
        return pickle.load(fh)


def predict_one(payload: dict, smiles) -> pd.DataFrame:
    """Score SMILES with a single packaged model.

    Returns one row per input SMILES: probability, class, and applicability-domain
    columns when the model carries an AD. Invalid SMILES come back as NaN.
    """
    smiles = list(smiles)
    X, kept = featurize(smiles, payload["fp"], payload["n_bits"], payload["radius"])
    key = payload["model_key"]
    out = pd.DataFrame({"SMILES": smiles})
    out[f"{key}_Prob"] = np.nan
    if len(kept) == 0:
        return out

    proba = payload["model"].predict_proba(X)[:, 1]
    out.loc[kept, f"{key}_Prob"] = proba

    cut = payload.get("decision_threshold")
    if cut is not None:
        out[f"{key}_Class"] = (out[f"{key}_Prob"] >= cut).astype("Int64")

    ad = payload.get("ad")
    if ad:
        inside, dist = _ad_predict(ad, X)
        out[f"{key}_AD_Distance"] = np.nan
        out[f"{key}_AD"] = pd.Series([pd.NA] * len(smiles), dtype="object")
        out.loc[kept, f"{key}_AD_Distance"] = dist
        out.loc[kept, f"{key}_AD"] = np.where(inside, "Inside", "Outside")
    return out


def _ad_predict(ad: dict, X):
    distances, _ = ad["nn"].kneighbors(X, n_neighbors=ad["k_neighbors"])
    mean_dists = distances.mean(axis=1)
    return mean_dists <= ad["threshold_AD"], mean_dists


def screen(payloads, smiles, vote: str = "majority") -> pd.DataFrame:
    """Score SMILES with an ensemble and add the consensus columns.

    vote="majority" flags a compound active when more than half the members call it
    active at their own threshold (TREM2's >=2/3 rule). vote="mean" only reports the
    mean probability, which is how the CD28 and IDO1 papers gate (P >= 0.6).
    """
    smiles = list(smiles)
    frames = [predict_one(p, smiles) for p in payloads]
    merged = frames[0][["SMILES"]].copy()
    for f in frames:
        merged = merged.join(f.drop(columns=["SMILES"]))

    prob_cols = [c for c in merged.columns if c.endswith("_Prob")]
    merged["Prob_mean"] = merged[prob_cols].mean(axis=1)

    class_cols = [c for c in merged.columns if c.endswith("_Class")]
    if vote == "majority" and class_cols:
        votes = merged[class_cols].sum(axis=1)
        merged["Consensus_Active"] = (votes > len(class_cols) / 2).astype("Int64")
    elif vote == "mean":
        merged["Consensus_Active"] = (merged["Prob_mean"] >= 0.6).astype("Int64")
    # a SMILES RDKit could not parse has no verdict, not a negative one
    merged.loc[merged["Prob_mean"].isna(), "Consensus_Active"] = pd.NA
    return merged


def _read_smiles(arg: str) -> list[str]:
    p = Path(arg)
    if not p.exists():
        return [arg]
    if p.suffix.lower() in (".csv", ".tsv"):
        df = pd.read_csv(p, sep="\t" if p.suffix.lower() == ".tsv" else ",")
        col = next((c for c in df.columns if "smiles" in c.lower()), df.columns[0])
        return df[col].astype(str).tolist()
    return [line.strip() for line in p.read_text().splitlines() if line.strip()]


if __name__ == "__main__":
    args = sys.argv[1:]
    if "--smiles" not in args:
        print(__doc__)
        sys.exit(1)
    i = args.index("--smiles")
    model_paths, smi_arg = args[:i], args[i + 1]
    out_path = None
    if "--out" in args:
        out_path = args[args.index("--out") + 1]

    payloads = [load(p) for p in model_paths]
    result = screen(payloads, _read_smiles(smi_arg))
    if out_path:
        result.to_csv(out_path, index=False)
        print(f"{len(result)} rows -> {out_path}")
    else:
        print(result.to_string(index=False))
