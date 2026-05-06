"""LCP type specifications."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Optional

import numpy as np


class LCPLineType(str, Enum):
    """Supported line types for LCP-based fitULM generation."""
    OHL_DERI_SEMLYEN = "ohl_deri_semlyen"
    PIPE_TYPE_CABLE = "pipe_type_cable"
    MULTI_ARMORED_CABLE = "multi_armored_cable"


@dataclass
class LCPFitULMSpec:
    """Complete specification for LCP-based fitULM generation.

    Parameters
    ----------
    line_type:
        The line category (overhead, pipe-type cable, armored cable).
    name:
        Unique name for this line, used for cache filenames.
    length:
        Line length in metres.
    freq:
        Frequency vector in Hz (1-D array, all values > 0).
    geometry_config:
        Line-type-specific geometry configuration.
    output_path:
        Destination for the generated fitULM file.
        When None, auto-generated from *cache_dir* and a content hash (PR-6).
    soil_config:
        Soil parameters.  Defaults to rho=100 Ω·m, εr=10, μr=1.
    vf_config:
        :class:`~LCP.vector_fitting_v411_independent.IterativePoleFindingConfig`
        or None for defaults.
    precision:
        Decimal precision for fitULM file export.
    use_freq_dependent:
        Frequency-dependent transformation mode: ``"auto"`` | ``"yes"`` | ``"no"``.
    enforce_passivity:
        When True, enforce passivity during fitting.
    cache_dir:
        Directory for auto-generated fitULM cache files (PR-6).
    verbose:
        When True, print progress messages.
    """

    line_type: LCPLineType
    name: str
    length: float
    freq: np.ndarray
    geometry_config: Any
    output_path: Optional[Path] = None
    soil_config: Optional[Any] = None
    vf_config: Optional[Any] = None
    precision: int = 16
    use_freq_dependent: str = "auto"
    enforce_passivity: bool = True
    cache_dir: Path = field(default_factory=lambda: Path(".lcp_cache"))
    verbose: bool = False
