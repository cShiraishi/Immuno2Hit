"""Applicability domain used by the packaged models.

Self-contained copy of ``src.core.applicability_domain.ApplicabilityDomain`` from
QSAR_curadoria/ScreenSAR, so the pickles in ``models/`` load without the original
repository on sys.path. Same maths: mean Jaccard (Tanimoto) distance to the k
nearest training neighbours, cutoff at <D> + z * std(D).
"""
from __future__ import annotations

import numpy as np
from sklearn.neighbors import NearestNeighbors


class ApplicabilityDomain:
    def __init__(self, k_neighbors: int = 5, z_threshold: float = 3.0, metric: str = "jaccard"):
        self.k = k_neighbors
        self.z = z_threshold
        self.metric = metric
        self.model_nn = None
        self.threshold_AD = None
        self.mean_dist_train = None
        self.std_dist_train = None
        self.training_distances = None

    def fit(self, X):
        X = np.asarray(X)
        self.model_nn = NearestNeighbors(n_neighbors=self.k + 1, metric=self.metric, n_jobs=-1)
        self.model_nn.fit(X)
        distances, _ = self.model_nn.kneighbors(X)
        mean_distances = np.mean(distances[:, 1:], axis=1)
        self.training_distances = mean_distances
        self.mean_dist_train = float(np.mean(mean_distances))
        self.std_dist_train = float(np.std(mean_distances))
        self.threshold_AD = self.mean_dist_train + self.z * self.std_dist_train
        return self

    def predict(self, X_new):
        """Return (is_inside, mean_distance) for new fingerprints."""
        X_new = np.asarray(X_new)
        distances, _ = self.model_nn.kneighbors(X_new, n_neighbors=self.k)
        mean_dists = np.mean(distances, axis=1)
        return mean_dists <= self.threshold_AD, mean_dists
