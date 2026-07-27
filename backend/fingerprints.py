"""Fingerprint generation, identical to QSAR_curadoria/ScreenSAR `ModeladorQSAR.gerar_dados`."""
from __future__ import annotations

import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, MACCSkeys

RDLogger.DisableLog("rdApp.*")


def featurize_one(smiles: str, fp: str, n_bits: int = 1024, radius: int = 2):
    """Fingerprint a single SMILES. Returns a 1-D array, or None if RDKit rejects it."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    if fp == "MACCS":
        bits = MACCSkeys.GenMACCSKeys(mol)
    elif fp == "RDKit":
        bits = Chem.RDKFingerprint(mol, maxPath=7, fpSize=n_bits, nBitsPerHash=2)
    elif fp == "Morgan":
        bits = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    else:
        raise ValueError(f"unknown fingerprint {fp!r} (use MACCS, Morgan or RDKit)")
    return np.array(bits)


def featurize(smiles_list, fp: str, n_bits: int = 1024, radius: int = 2):
    """Fingerprint many SMILES. Returns (X, kept_indices) — invalid SMILES are dropped."""
    X, kept = [], []
    for i, smi in enumerate(smiles_list):
        arr = featurize_one(smi, fp, n_bits, radius)
        if arr is not None:
            X.append(arr)
            kept.append(i)
    return np.array(X), kept
