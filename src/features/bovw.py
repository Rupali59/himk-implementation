"""
Bag of Visual Words (BoVW) with histogram matching score and K-medoid codebook.

Algorithm 5.1: Histogram Matching Score between two histograms h_i, h_j:
  score = (1 / num_bins) * sum_b min(h_i(b), h_j(b))

Algorithm 5.2: K-medoid using this score (max score = nearest); medoid = representative of cluster.
"""

from typing import Optional

import numpy as np


def histogram_matching_score(hi: np.ndarray, hj: np.ndarray) -> float:
    """
    Histogram matching score for two histogram vectors (Algorithm 5.1 in thesis).
    Higher score = more similar. Range [0, 1] if histograms are normalized.
    """
    hi = np.asarray(hi, dtype=np.float64).ravel()
    hj = np.asarray(hj, dtype=np.float64).ravel()
    if hi.shape != hj.shape:
        raise ValueError("Histogram shapes must match")
    n = hi.size
    if n == 0:
        return 0.0
    score = np.minimum(hi, hj).sum()
    return float(score / n)


def pairwise_histogram_scores(X: np.ndarray) -> np.ndarray:
    """Compute pairwise histogram matching scores for rows of X. Returns (n, n) matrix."""
    X = np.asarray(X, dtype=np.float64)
    n = X.shape[0]
    out = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1):
            s = histogram_matching_score(X[i], X[j])
            out[i, j] = out[j, i] = s
    return out


def kmedoid_histogram(
    X: np.ndarray,
    k: int,
    max_iter: int = 100,
    random_state: Optional[int] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    K-medoid clustering using histogram matching score (Algorithm 5.2).
    X: (n_samples, n_features) e.g. frame HoG vectors.
    Returns: (medoid_indices, labels) where medoid_indices is shape (k,) and labels (n_samples,).
    """
    X = np.asarray(X, dtype=np.float64)
    n = X.shape[0]
    if n < k:
        raise ValueError("Need at least k samples")
    rng = np.random.default_rng(random_state)
    medoid_idx = rng.choice(n, size=k, replace=False)
    labels = np.zeros(n, dtype=int)

    for _ in range(max_iter):
        # Assign each point to cluster with max histogram score to medoid
        for i in range(n):
            best = -1.0
            best_j = 0
            for j in range(k):
                s = histogram_matching_score(X[i], X[medoid_idx[j]])
                if s > best:
                    best = s
                    best_j = j
            labels[i] = best_j

        # Update medoid: for each cluster, choose sample that maximizes sum of scores to others in cluster
        new_medoid_idx = np.zeros(k, dtype=int)
        for j in range(k):
            mask = labels == j
            indices = np.where(mask)[0]
            if len(indices) == 0:
                new_medoid_idx[j] = medoid_idx[j]
                continue
            best_sum = -1.0
            best_idx = indices[0]
            for cand in indices:
                total = sum(
                    histogram_matching_score(X[cand], X[other])
                    for other in indices
                )
                if total > best_sum:
                    best_sum = total
                    best_idx = cand
            new_medoid_idx[j] = best_idx

        if np.array_equal(new_medoid_idx, medoid_idx):
            break
        medoid_idx = new_medoid_idx

    return medoid_idx, labels


def build_codebook(
    frame_features: list[np.ndarray],
    k: int,
    max_samples: Optional[int] = None,
    random_state: Optional[int] = None,
) -> np.ndarray:
    """
    Build BoVW codebook from a list of per-video frame feature arrays.
    Each element of frame_features is (T_v, D). We pool all vectors (optionally subsample),
    then run K-medoid. Returns codebook matrix (k, D).
    """
    all_vectors = np.vstack([f for f in frame_features if len(f) > 0])
    if all_vectors.size == 0:
        raise ValueError("No frame features to build codebook")
    if max_samples is not None and len(all_vectors) > max_samples:
        rng = np.random.default_rng(random_state)
        idx = rng.choice(len(all_vectors), size=max_samples, replace=False)
        all_vectors = all_vectors[idx]
    medoid_idx, _ = kmedoid_histogram(all_vectors, k, random_state=random_state)
    return all_vectors[medoid_idx]


def frame_sequence_to_bow(
    frame_sequence: np.ndarray,
    codebook: np.ndarray,
) -> np.ndarray:
    """
    Convert a single video's frame sequence (T, D) to a bag-of-words histogram (k,)
    by assigning each frame to nearest codebook entry (max histogram matching score).
    """
    T, D = frame_sequence.shape
    k = codebook.shape[0]
    hist = np.zeros(k)
    for t in range(T):
        best = -1.0
        best_j = 0
        for j in range(k):
            s = histogram_matching_score(frame_sequence[t], codebook[j])
            if s > best:
                best = s
                best_j = j
        hist[best_j] += 1
    if hist.sum() > 0:
        hist = hist / hist.sum()
    return hist
