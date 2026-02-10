"""
HMM-based Intermediate Matching Kernel (HIMK) - complete sequence matching.

K_HIMK(X_m, X_n) = sum_{i=1}^{N} sum_{q=1}^{Q_i} k(x*_miq, x*_niq)
where x*_iq = argmax_t R_iq(x_t | X, lambda), R_iq = nu_it * gamma_iq(x_t).
Uses a single CDHMM (trained on all training data) to define state/component indices.
"""

from typing import Callable, List, Optional

import numpy as np

from .hmm_utils import compute_R_iq, train_single_cdhmm, virtual_vectors_from_R


def base_kernel_linear(x: np.ndarray, y: np.ndarray) -> float:
    """Linear base kernel k(x,y) = x'y."""
    return float(np.dot(x.ravel(), y.ravel()))


def base_kernel_rbf(x: np.ndarray, y: np.ndarray, gamma: float = 0.01) -> float:
    """RBF base kernel k(x,y) = exp(-gamma * ||x-y||^2). Clipped for numerical stability."""
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    d = np.dot(x - y, x - y)
    if not np.isfinite(d) or d < 0:
        return 0.0
    return float(np.clip(np.exp(-gamma * d), 0.0, 1.0))


def compute_virtual_vectors(
    model,
    X: np.ndarray,
) -> List[np.ndarray]:
    """
    For sequence X and CDHMM model, compute virtual vectors x*_iq for all (i,q).
    Returns list of length n_states * n_mix, each element shape (D,).
    """
    R = compute_R_iq(model, X)
    return virtual_vectors_from_R(X, R)


def himk_single(
    model,
    X_m: np.ndarray,
    X_n: np.ndarray,
    base_kernel: Callable[[np.ndarray, np.ndarray], float],
) -> float:
    """
    HIMK between two sequences X_m and X_n using the given CDHMM and base kernel.
    """
    vv_m = compute_virtual_vectors(model, X_m)
    vv_n = compute_virtual_vectors(model, X_n)
    if len(vv_m) != len(vv_n):
        raise ValueError("Virtual vector count mismatch (different HMM?): {} vs {}".format(len(vv_m), len(vv_n)))
    return sum(base_kernel(a, b) for a, b in zip(vv_m, vv_n))


def himk_gram(
    model,
    X_list: List[np.ndarray],
    base_kernel: Callable[[np.ndarray, np.ndarray], float],
    sym: bool = True,
) -> np.ndarray:
    """
    Compute HIMK Gram matrix K[i,j] = HIMK(X_list[i], X_list[j]).
    If sym=True, only compute lower triangle and mirror.
    """
    n = len(X_list)
    K = np.zeros((n, n))
    for i in range(n):
        vv_i = compute_virtual_vectors(model, X_list[i])
        for j in range(i + 1 if sym else n):
            vv_j = compute_virtual_vectors(model, X_list[j])
            k_val = sum(base_kernel(a, b) for a, b in zip(vv_i, vv_j))
            K[i, j] = np.clip(k_val, 0.0, 1e10) if np.isfinite(k_val) else 0.0
            if sym and i != j:
                K[j, i] = K[i, j]
    return K


def himk_train_test(
    model,
    X_train: List[np.ndarray],
    X_test: List[np.ndarray],
    base_kernel: Callable[[np.ndarray, np.ndarray], float],
) -> np.ndarray:
    """
    Compute kernel matrix between test and train: K_test_train[i, j] = HIMK(X_test[i], X_train[j]).
    Returns (n_test, n_train).
    """
    n_test = len(X_test)
    n_train = len(X_train)
    K = np.zeros((n_test, n_train))
    for i in range(n_test):
        vv_i = compute_virtual_vectors(model, X_test[i])
        for j in range(n_train):
            vv_j = compute_virtual_vectors(model, X_train[j])
            k_val = sum(base_kernel(a, b) for a, b in zip(vv_i, vv_j))
            K[i, j] = np.clip(k_val, 0.0, 1e10) if np.isfinite(k_val) else 0.0
    return K


def build_himk_model_and_gram(
    X_train: List[np.ndarray],
    y_train: np.ndarray,
    n_states: int = 15,
    n_mix: int = 3,
    base_kernel: Optional[Callable] = None,
    base_kernel_gamma: float = 0.01,
    random_state: Optional[int] = None,
):
    """
    Train one CDHMM on all X_train, then compute train-train HIMK Gram matrix.
    Returns (model, K_gram).
    base_kernel: if None, use RBF with base_kernel_gamma.
    """
    if base_kernel is None:
        gamma = base_kernel_gamma
        base_kernel = lambda x, y: base_kernel_rbf(x, y, gamma=gamma)
    model = train_single_cdhmm(
        X_train,
        n_states=n_states,
        n_mix=n_mix,
        n_iter=50,
        random_state=random_state,
    )
    K = himk_gram(model, X_train, base_kernel)
    return model, K
