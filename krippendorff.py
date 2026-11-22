"""Krippendorff's alpha for binary ratings.

Nominal alpha on a complete items-by-raters matrix of 0/1 labels. This is
the human-agreement figure reported for the pass/fail annotations.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def krippendorff_binary(M: NDArray[np.generic]) -> float:
    """Krippendorff's alpha for a binary rating matrix.

    ``M`` has shape ``(items, raters)`` and values 0/1. Every item must have
    the same number of ratings; missing values are not supported.
    """
    n, k = M.shape
    # Disagreeing unordered pairs / total unordered pairs. No extra halving:
    # De below is also a proportion of unordered pairs, so the two must match.
    Do = (M.sum(1) * (k - M.sum(1))).sum() / (n * k * (k - 1) / 2)
    p = M.mean()
    De = 2 * p * (1 - p) * (n * k) / (n * k - 1)
    return float(1 - Do / De) if De > 0 else float("nan")
