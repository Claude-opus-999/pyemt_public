"""Waveform comparison — interpolation-aware comparison for external tool benchmarks."""

from typing import Optional, Tuple

import numpy as np


def compare_waveform(
    t_sim: np.ndarray,
    y_sim: np.ndarray,
    t_ref: np.ndarray,
    y_ref: np.ndarray,
    *,
    interp: bool = True,
    window: Optional[Tuple[float, float]] = None,
) -> Tuple[float, float, float]:
    """Compare two waveforms, optionally interpolating to a common time grid.

    Returns
    -------
    max_err : float
        Maximum absolute error.
    rms_err : float
        Root-mean-square error.
    rel_peak_err : float
        Relative peak error.
    """
    y_s = np.asarray(y_sim, dtype=float).ravel()
    y_r = np.asarray(y_ref, dtype=float).ravel()
    t_s = np.asarray(t_sim, dtype=float).ravel()
    t_r = np.asarray(t_ref, dtype=float).ravel()

    if window is not None:
        w0, w1 = window
        mask_s = (t_s >= w0) & (t_s <= w1)
        mask_r = (t_r >= w0) & (t_r <= w1)
        t_s, y_s = t_s[mask_s], y_s[mask_s]
        t_r, y_r = t_r[mask_r], y_r[mask_r]

    if interp and not np.array_equal(t_s, t_r):
        y_s_interp = np.interp(t_r, t_s, y_s)
        max_err = float(np.max(np.abs(y_s_interp - y_r)))
        rms_err = float(np.sqrt(np.mean((y_s_interp - y_r) ** 2)))
        rel_peak = abs(np.max(np.abs(y_s_interp)) - np.max(np.abs(y_r))) / max(np.max(np.abs(y_r)), 1e-30)
    else:
        min_len = min(len(y_s), len(y_r))
        max_err = float(np.max(np.abs(y_s[:min_len] - y_r[:min_len])))
        rms_err = float(np.sqrt(np.mean((y_s[:min_len] - y_r[:min_len]) ** 2)))
        rel_peak = abs(np.max(np.abs(y_s)) - np.max(np.abs(y_r))) / max(np.max(np.abs(y_r)), 1e-30)

    return max_err, rms_err, float(rel_peak)
