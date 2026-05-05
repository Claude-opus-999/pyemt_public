"""Export validation results: NPZ waveforms, JSON metrics, Markdown summary."""

import json
import os
from pathlib import Path
from typing import Optional

import numpy as np


def export_waveforms(path: str, **waveforms: np.ndarray) -> None:
    """Save named waveform arrays to a compressed .npz file."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    np.savez_compressed(path, **{k: np.asarray(v) for k, v in waveforms.items()})


def export_metrics(path: str, metrics: dict) -> None:
    """Save metrics dict as JSON."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, default=_json_default)


def export_summary(path: str, result) -> None:
    """Save a human-readable Markdown summary."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    status = "PASS" if result.passed else "FAIL"
    lines = [
        f"# Validation: {result.name}",
        f"",
        f"**Status:** {status}  ",
        f"**Category:** {result.category}  ",
        f"",
        f"## Metrics",
        f"",
        f"| Metric | Value | Tolerance |",
        f"|--------|-------|-----------|",
    ]
    for k, v in result.metrics.items():
        tol = result.tolerances.get(k, "—")
        lines.append(f"| {k} | {v:.6g} | {tol} |")
    if result.references:
        lines.append("")
        lines.append("## References")
        for k, v in result.references.items():
            lines.append(f"- **{k}**: {v}")
    if result.notes:
        lines.append("")
        lines.append(f"## Notes")
        lines.append(result.notes)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def _json_default(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
