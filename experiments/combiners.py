"""Member-probability combination strategies for AraSeg ensembling.

Each combiner takes a list of per-word probability arrays (one per member, all
the same length for a given document) and returns one combined array. Kept
separate from ensemble.py so the math is unit-testable with no model loading.
Pure numpy — runs on the offline KISSKI venv (no sklearn).
"""
import numpy as np

_EPS = 1e-6


def _stack(probs):
    arr = np.stack(probs, axis=0).astype(np.float64)   # (M, N)
    return np.clip(arr, _EPS, 1.0 - _EPS)


def combine_prob(probs, weights=None):
    """Plain (or weighted) probability mean — the original behaviour."""
    return np.average(_stack(probs), axis=0, weights=weights).astype(np.float32)


def combine_logit(probs, weights=None):
    """Average in log-odds space, map back with sigmoid. Removes the pile-up at
    exactly 0.5 that flattens the threshold curve when averaging overconfident
    members in probability space."""
    arr = _stack(probs)
    logits = np.log(arr / (1.0 - arr))
    mean_logit = np.average(logits, axis=0, weights=weights)
    return (1.0 / (1.0 + np.exp(-mean_logit))).astype(np.float32)


def combine_rank(probs, weights=None):
    """Average per-member rank-normalised scores within the document. Robust to
    miscalibration: only each member's internal ordering matters."""
    arr = _stack(probs)
    n = arr.shape[1]
    ranks = np.empty_like(arr)
    for i in range(arr.shape[0]):
        ranks[i] = np.argsort(np.argsort(arr[i])) / max(n - 1, 1)
    return np.average(ranks, axis=0, weights=weights).astype(np.float32)


COMBINERS = {"prob": combine_prob, "logit": combine_logit, "rank": combine_rank}


def _logodds(member_probs):
    """(N, M) probabilities -> (N, M) clipped log-odds features."""
    p = np.clip(np.asarray(member_probs, dtype=np.float64), _EPS, 1.0 - _EPS)
    return np.log(p / (1.0 - p))


def fit_stacker(member_probs, gold, l2=1.0, iters=800, lr=0.3):
    """Tiny L2-regularised logistic-regression meta-learner over member probs.

    member_probs: (N_words, M) member probabilities.  gold: (N_words,) 0/1.
    Returns (weights (M,), bias). Dependency-free full-batch gradient descent so
    it runs on the offline cluster venv. Features are member log-odds, which keeps
    the meta-model close to a calibrated weighted average and reduces overfit on
    the small (~222-doc) dev set.
    """
    X = _logodds(member_probs)
    y = np.asarray(gold, dtype=np.float64)
    # standardise features for stable, scale-free L2
    mu, sd = X.mean(0), X.std(0) + _EPS
    Xs = (X - mu) / sd
    n, m = Xs.shape
    w = np.zeros(m)
    b = 0.0
    for _ in range(iters):
        z = Xs @ w + b
        p = 1.0 / (1.0 + np.exp(-z))
        err = p - y
        gw = Xs.T @ err / n + l2 * w / n
        gb = err.mean()
        w -= lr * gw
        b -= lr * gb
    # fold standardisation back so apply_stacker takes raw probs
    w_raw = w / sd
    b_raw = b - float(w @ (mu / sd))
    return w_raw, b_raw


def apply_stacker(member_probs, w, b):
    """Map (N_words, M) member probs to a combined probability via the fitted
    logistic meta-model (operates on log-odds features, matching fit_stacker)."""
    z = _logodds(member_probs) @ np.asarray(w) + b
    return (1.0 / (1.0 + np.exp(-z))).astype(np.float32)
