"""
Compute the Local Entropy Score (LES)
"""
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import numpy as np
from sklearn.neighbors import NearestNeighbors


def compute_local_entropy_score(X, y, n_neighbors=10):
    """
    Compute Local Entropy Score (LES) for each sample.
    """
    X = np.asarray(X)
    y = np.asarray(y)
    N = len(y)

    classes = np.unique(y)
    K = len(classes)

    class_to_idx = {c: i for i, c in enumerate(classes)}
    y_idx = np.array([class_to_idx[yi] for yi in y])

    k = min(n_neighbors, N - 1)
    if k < 1:
        return np.zeros(N), np.zeros((N, K))

    nn = NearestNeighbors(n_neighbors=k + 1, algorithm='auto', metric='euclidean')
    nn.fit(X)

    _, indices = nn.kneighbors(X)
    neighbor_indices = indices[:, 1:]
    neighbor_labels = y_idx[neighbor_indices]

    neighbor_class_dist = np.zeros((N, K))
    for j in range(K):
        neighbor_class_dist[:, j] = np.mean(neighbor_labels == j, axis=1)

    H = _compute_entropy_vectorized(neighbor_class_dist)

    return H, neighbor_class_dist


def _compute_entropy_vectorized(p):
    """
    Compute Shannon entropy for each row of probability matrix
    """
    with np.errstate(divide='ignore', invalid='ignore'):
        log_p = np.log(p)
        log_p = np.where(p > 0, log_p, 0)

    H = np.abs(-np.sum(p * log_p, axis=1))

    return H


def compute_normalized_les(X, y, n_neighbors=10):
    """
    Compute normalized LES in [0, 1].
    """
    H, neighbor_class_dist = compute_local_entropy_score(X, y, n_neighbors)

    K = neighbor_class_dist.shape[1]
    logK = np.log(K) if K > 1 else 1.0

    if logK > 0:
        H_normalized = H / logK
    else:
        H_normalized = np.zeros_like(H)

    return H_normalized, H, neighbor_class_dist
