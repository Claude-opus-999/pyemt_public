"""Input and output validation for LCP operations."""

from __future__ import annotations

import numpy as np


def validate_frequency_vector(freq: np.ndarray) -> None:
    """Check that *freq* is a valid frequency vector."""
    arr = np.asarray(freq, dtype=float)
    if arr.ndim != 1:
        raise ValueError("freq must be a 1-D array")
    if len(arr) == 0:
        raise ValueError("freq must not be empty")
    if np.any(arr <= 0):
        raise ValueError("freq values must all be > 0")


def validate_zy_matrices(
    freq: np.ndarray,
    Z: np.ndarray,
    Y: np.ndarray,
) -> None:
    """Check that Z and Y matrices have expected shape and no NaN/Inf."""
    Z = np.asarray(Z)
    Y = np.asarray(Y)
    K = len(freq)

    if Z.ndim != 3 or Z.shape[0] != K:
        raise ValueError(f"Z must have shape ({K}, n, n), got {Z.shape}")
    if Y.ndim != 3 or Y.shape[0] != K:
        raise ValueError(f"Y must have shape ({K}, n, n), got {Y.shape}")
    if Z.shape[1] != Z.shape[2]:
        raise ValueError(f"Z[:,i,j] is not square: {Z.shape}")
    if Y.shape[1] != Y.shape[2]:
        raise ValueError(f"Y[:,i,j] is not square: {Y.shape}")
    if Z.shape[1] != Y.shape[1]:
        raise ValueError(
            f"Z and Y conductor counts differ: {Z.shape[1]} vs {Y.shape[1]}"
        )
    if np.any(np.isnan(Z)) or np.any(np.isinf(Z)):
        raise ValueError("Z contains NaN or Inf")
    if np.any(np.isnan(Y)) or np.any(np.isinf(Y)):
        raise ValueError("Y contains NaN or Inf")
