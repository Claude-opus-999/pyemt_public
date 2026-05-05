"""Validation metrics for comparing simulation output to references."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np


@dataclass
class ValidationResult:
    """Unified container for a single validation case."""

    name: str
    category: str
    passed: bool
    metrics: Dict[str, float] = field(default_factory=dict)
    tolerances: Dict[str, float] = field(default_factory=dict)
    waveforms: Dict[str, np.ndarray] = field(default_factory=dict)
    references: Dict[str, str] = field(default_factory=dict)
    notes: str = ""

    def summary(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        lines = [f"[{status}] {self.category}/{self.name}"]
        for k, v in self.metrics.items():
            tol = self.tolerances.get(k)
            tol_str = f" (tol: {tol})" if tol is not None else ""
            lines.append(f"  {k}: {v:.6g}{tol_str}")
        if self.notes:
            lines.append(f"  note: {self.notes}")
        return "\n".join(lines)


def max_abs_error(sim, ref):
    """Maximum absolute error between sim and reference waveforms."""
    return float(np.max(np.abs(np.asarray(sim, dtype=float) - np.asarray(ref, dtype=float))))


def rms_error(sim, ref):
    """Root-mean-square error between sim and reference waveforms."""
    e = np.asarray(sim, dtype=float) - np.asarray(ref, dtype=float)
    return float(np.sqrt(np.mean(e * e)))


def relative_peak_error(sim, ref, eps=1e-30):
    """Relative error of peak absolute values."""
    sim_peak = float(np.max(np.abs(sim)))
    ref_peak = float(np.max(np.abs(ref)))
    return float(abs(sim_peak - ref_peak) / max(ref_peak, eps))


def peak_value(y):
    """Peak absolute value of a waveform."""
    return float(np.max(np.abs(y)))


def time_of_peak(t, y):
    """Time at which |y| reaches its maximum."""
    return float(t[int(np.argmax(np.abs(y)))])


def final_value_error(sim, ref, n_tail=10):
    """Relative error between average of last *n_tail* points."""
    s = np.mean(np.asarray(sim, dtype=float)[-n_tail:])
    r = np.mean(np.asarray(ref, dtype=float)[-n_tail:])
    denom = max(abs(r), 1e-30)
    return float(abs(s - r) / denom)


def check_metrics(metrics: Dict[str, float], tolerances: Dict[str, float]) -> bool:
    """Return True if all metrics are within their tolerances."""
    for name, tol in tolerances.items():
        if name in metrics and abs(metrics[name]) > tol:
            return False
    return True
