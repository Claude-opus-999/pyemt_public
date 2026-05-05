"""Plot validation waveforms for visual inspection."""

from typing import List, Optional
import os

import numpy as np

_PLT = None


def _ensure_matplotlib():
    global _PLT
    if _PLT is None:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            _PLT = plt
        except ImportError:
            raise ImportError("matplotlib is required for plot_report")


def plot_comparison(
    t: np.ndarray,
    y_sim: np.ndarray,
    y_ref: np.ndarray,
    *,
    title: str = "Validation",
    sim_label: str = "EMTP",
    ref_label: str = "Reference",
    xlabel: str = "Time (s)",
    ylabel: str = "",
    save_path: Optional[str] = None,
) -> None:
    """Plot simulated vs reference waveforms."""
    _ensure_matplotlib()
    fig, ax = _PLT.subplots(figsize=(9, 5))
    ax.plot(t, np.asarray(y_sim).ravel(), label=sim_label, linewidth=1.2)
    ax.plot(t, np.asarray(y_ref).ravel(), label=ref_label, linestyle="--", linewidth=1.2)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, dpi=150)
    _PLT.close(fig)


def plot_waveforms(
    t: np.ndarray,
    waveforms: dict,
    *,
    title: str = "",
    xlabel: str = "Time (s)",
    ylabel: str = "",
    save_path: Optional[str] = None,
) -> None:
    """Plot multiple named waveforms on one axes."""
    _ensure_matplotlib()
    fig, ax = _PLT.subplots(figsize=(9, 5))
    for label, y in waveforms.items():
        ax.plot(t, np.asarray(y).ravel(), label=label, linewidth=1.2)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, dpi=150)
    _PLT.close(fig)
