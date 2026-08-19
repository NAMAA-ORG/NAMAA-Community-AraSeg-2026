"""Pure-numpy document scoring — the F1 math, with no torch import.

Split out of metrics.py so the CPU-only tools (bootstrap_paired.py, fit_decoder.py)
can score predictions on a laptop from cached probabilities, without a torch install.
metrics.py re-exports these, so nothing else changes.
"""
from typing import Dict, List

import numpy as np


def _doc_f1(pred: np.ndarray, gold: np.ndarray) -> Dict[str, float]:
    tp = int(np.sum((pred == 1) & (gold == 1)))
    fp = int(np.sum((pred == 1) & (gold == 0)))
    fn = int(np.sum((pred == 0) & (gold == 1)))
    p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return {"precision": p, "recall": r, "f1": f}


def _macro(scores: List[Dict]) -> Dict[str, float]:
    return {k: float(np.mean([s[k] for s in scores])) for k in ("precision", "recall", "f1")}
