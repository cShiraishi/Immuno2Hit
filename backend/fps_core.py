"""Scoring from pre-computed fingerprints — the path used by the web API.

The browser computes the fingerprints with RDKit.js and posts the bit strings here, so
this module needs neither RDKit nor SMILES parsing. That is what takes the serverless
bundle from 399 MB to 251 MB: RDKit alone is 148 MB installed on Linux.

Safe because RDKit.js is bit-for-bit identical to RDKit Python for the three fingerprints
these models use — verified over Morgan-1024, MACCS-167 and RDKit-1024 with zero
differing bits. If that ever stops holding, predictions change silently, so the parity
check belongs in any future test suite.
"""
from __future__ import annotations

import numpy as np


def bits_to_array(bitstring: str, expected: int) -> np.ndarray | None:
    """Turn a '0101...' string from RDKit.js into the model's input vector."""
    if not isinstance(bitstring, str) or len(bitstring) != expected:
        return None
    try:
        return np.frombuffer(bitstring.encode(), dtype=np.uint8).astype(np.float64) - 48.0
    except Exception:
        return None


def expected_length(payload: dict) -> int:
    """MACCS keys are 167 bits; Morgan and RDKit use the configured bit count."""
    return 167 if payload["fp"] == "MACCS" else int(payload["n_bits"])
