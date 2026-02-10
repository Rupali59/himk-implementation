"""
CDHMM utilities for HIMK: train one GMM-HMM per class or one global HMM,
and compute state posteriors (nu_it) and component responsibilities (gamma_iq) for R_iq.
"""

from typing import List, Optional

import numpy as np

try:
    from hmmlearn import hmm
except ImportError:
    hmm = None


def get_gmm_responsibility(
    x: np.ndarray,
    weights: np.ndarray,
    means: np.ndarray,
    covars: np.ndarray,
    covariance_type: str = "diag",
) -> np.ndarray:
    """
    P(component q | x) for a single GMM. Returns shape (n_components,).
    """
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 1:
        x = x.reshape(1, -1)
    n_components = len(weights)
    log_prob = np.zeros(n_components)
    for q in range(n_components):
        mu = means[q]
        if covariance_type == "diag":
            var = covars[q]
            diff = x - mu
            log_prob[q] = -0.5 * (np.log(2 * np.pi * var).sum() + ((diff ** 2) / var).sum())
        else:
            from scipy.stats import multivariate_normal
            log_prob[q] = multivariate_normal.logpdf(x, mu, covars[q])
    log_prob += np.log(weights + 1e-300)
    log_prob -= np.max(log_prob)
    prob = np.exp(log_prob)
    prob = prob / (prob.sum() + 1e-300)
    return prob.ravel()


def forward_backward_gamma(
    log_A: np.ndarray,
    log_pi: np.ndarray,
    log_B: np.ndarray,
) -> np.ndarray:
    """
    State posteriors gamma_t(i) = P(s_t = i | X, lambda) via forward-backward.
    log_A: (n_states, n_states), log_B: (T, n_states), log_pi: (n_states,).
    Returns (T, n_states).
    """
    T, N = log_B.shape
    # Forward
    log_alpha = np.zeros((T, N))
    log_alpha[0] = log_pi + log_B[0]
    for t in range(1, T):
        log_alpha[t] = np.logaddexp.reduce(
            log_alpha[t - 1].reshape(-1, 1) + log_A.T, axis=0
        ) + log_B[t]
    # Backward
    log_beta = np.zeros((T, N))
    log_beta[T - 1] = 0.0
    for t in range(T - 2, -1, -1):
        log_beta[t] = np.logaddexp.reduce(
            log_A + log_B[t + 1] + log_beta[t + 1], axis=1
        )
    # Gamma
    log_gamma = log_alpha + log_beta
    log_gamma -= np.logaddexp.reduce(log_gamma, axis=1, keepdims=True)
    return np.exp(log_gamma)


def train_cdhmms_per_class(
    sequences_per_class: List[List[np.ndarray]],
    n_states_per_class: List[int],
    n_mix: int = 3,
    covariance_type: str = "diag",
    n_iter: int = 50,
    random_state: Optional[int] = None,
):
    """
    Train one CDHMM (GMM emissions) per class.
    sequences_per_class[c] = list of (T_c, D) arrays for class c.
    n_states_per_class[c] = number of states for class c (thesis Table 6.1).
    Returns list of hmmlearn GMMHMM models (or our wrapper if hmmlearn not available).
    """
    if hmm is None:
        raise ImportError("hmmlearn is required for CDHMM. Install with: pip install hmmlearn")
    models = []
    for c, seqs in enumerate(sequences_per_class):
        X = np.vstack(seqs)
        lengths = [s.shape[0] for s in seqs]
        n_states = n_states_per_class[c]
        model = hmm.GMMHMM(
            n_components=n_states,
            n_mix=n_mix,
            covariance_type=covariance_type,
            n_iter=n_iter,
            random_state=random_state,
            min_covar=1e-3,
        )
        model.fit(X, lengths=lengths)
        models.append(model)
    return models


def train_single_cdhmm(
    all_sequences: List[np.ndarray],
    n_states: int,
    n_mix: int = 3,
    covariance_type: str = "diag",
    n_iter: int = 50,
    random_state: Optional[int] = None,
):
    """
    Train one CDHMM on all sequences (for a common kernel topology).
    all_sequences: list of (T, D) arrays.
    """
    if hmm is None:
        raise ImportError("hmmlearn is required. pip install hmmlearn")
    X = np.vstack(all_sequences)
    lengths = [s.shape[0] for s in all_sequences]
    model = hmm.GMMHMM(
        n_components=n_states,
        n_mix=n_mix,
        covariance_type=covariance_type,
        n_iter=n_iter,
        random_state=random_state,
        min_covar=1e-3,
    )
    model.fit(X, lengths=lengths)
    return model


def _log_gmm_density(
    X: np.ndarray,
    weights: np.ndarray,
    means: np.ndarray,
    covars: np.ndarray,
    covariance_type: str,
) -> np.ndarray:
    """Log density of GMM at each row of X. Returns (X.shape[0],)."""
    from scipy.special import logsumexp
    X = np.asarray(X, dtype=np.float64)
    n_comp = len(weights)
    log_prob = np.zeros((X.shape[0], n_comp))
    for q in range(n_comp):
        mu = means[q]
        if covariance_type == "diag":
            var = np.asarray(covars[q]).ravel()
            diff = X - mu
            log_prob[:, q] = -0.5 * (np.log(2 * np.pi * var + 1e-300).sum() + ((diff ** 2) / (var + 1e-300)).sum(axis=1))
        else:
            from scipy.stats import multivariate_normal
            for i in range(X.shape[0]):
                log_prob[i, q] = multivariate_normal.logpdf(X[i], mu, covars[q])
        log_prob[:, q] += np.log(weights[q] + 1e-300)
    return logsumexp(log_prob, axis=1)


def compute_log_B_gmmhmm(model, X: np.ndarray) -> np.ndarray:
    """Log emission log P(x_t | state i) for GMMHMM. Returns (T, n_states)."""
    T, _ = X.shape
    n_states = model.n_components
    log_B = np.zeros((T, n_states))
    for i in range(n_states):
        log_B[:, i] = _log_gmm_density(
            X, model.weights_[i], model.means_[i], model.covars_[i],
            model.covariance_type,
        )
    return log_B


def compute_R_iq(
    model,
    X: np.ndarray,
) -> np.ndarray:
    """
    Compute R_iq(x_t | X, lambda) = nu_it * gamma_iq(x_t) for all t, i, q.
    Returns array of shape (T, n_states, n_mix).
    """
    X = np.asarray(X, dtype=np.float64)
    T = X.shape[0]
    n_states = model.n_components
    n_mix = getattr(model, "n_mix", 1)

    # Log emission per state (GMM density)
    log_B = compute_log_B_gmmhmm(model, X)
    log_A = np.log(model.transmat_ + 1e-300)
    log_pi = np.log(model.startprob_ + 1e-300)
    gamma = forward_backward_gamma(log_A, log_pi, log_B)  # (T, n_states)

    if n_mix == 1:
        return gamma[:, :, np.newaxis]

    # GMM responsibility per state
    R = np.zeros((T, n_states, n_mix))
    for t in range(T):
        xt = X[t : t + 1]
        for i in range(n_states):
            weights = model.weights_[i]
            means = model.means_[i]
            covars = model.covars_[i]
            resp = get_gmm_responsibility(
                xt, weights, means, covars, model.covariance_type
            )
            R[t, i, :] = gamma[t, i] * resp
    return R


def virtual_vectors_from_R(
    X: np.ndarray,
    R: np.ndarray,
) -> List[np.ndarray]:
    """
    For sequence X and R_iq(t), compute x*_iq = argmax_t R_iq(x_t) for each (i,q).
    Returns list of arrays: for each (i,q) the vector x_t that maximizes R_iq(t).
    Flattened as list of length n_states * n_mix, each element shape (D,).
    """
    T, n_states, n_mix = R.shape
    D = X.shape[1]
    virtuals = []
    for i in range(n_states):
        for q in range(n_mix):
            t_star = np.argmax(R[:, i, q])
            virtuals.append(X[t_star].ravel().astype(np.float64))
    return virtuals
