"""ResultBundle — structured simulation output container."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict


@dataclass
class ResultBundle:
    """Aggregated results from a single simulation run.

    Attributes
    ----------
    case_name:
        Name from the case config.
    success:
        ``True`` if the simulation completed without error.
    metrics:
        Scalar metrics (e.g. peak voltages, flashover count).
    waveforms:
        Time-series arrays keyed by signal name.
    metadata:
        Run metadata (elapsed time, dt, etc.).
    result_dir:
        Output directory (if one was requested).
    error:
        Exception message when ``success`` is ``False``.
    """

    case_name: str
    success: bool
    metrics: Dict[str, Any]
    waveforms: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)
    result_dir: Path | None = None
    error: str | None = None
