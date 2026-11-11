"""Evaluation helpers: validation-chosen F1 threshold and paired bootstrap confidence intervals."""
from __future__ import annotations

import numpy as np


def best_f1_threshold(y_true, scores) -> float:
    """Threshold maximising F1 on (validation) data; used frozen on the test period."""
    from sklearn.metrics import precision_recall_curve

    p, r, t = precision_recall_curve(y_true, scores)
    f = 2 * p * r / (p + r + 1e-12)
    return float(t[np.nanargmax(f[:-1])])


def paired_bootstrap_ci(a, b, reps: int = 2000, alpha: float = 0.05, seed: int = 1):
    """CI of ``sum(a) - sum(b)`` by resampling paired units (here: test time steps).

    ``a`` and ``b`` hold one value per unit (e.g. illicit found per step for two queues).
    Returns ``(lower, upper)``.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.shape != b.shape:
        raise ValueError("a and b must be paired (same shape)")
    rng = np.random.default_rng(seed)
    n = len(a)
    idx = rng.integers(0, n, size=(reps, n))
    d = a[idx].sum(1) - b[idx].sum(1)
    return float(np.percentile(d, 100 * alpha / 2)), float(np.percentile(d, 100 * (1 - alpha / 2)))
