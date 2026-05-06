"""pylcp — Python Line Constants Program for EMTP.

Automatic generation of fitULM files from line geometry and soil parameters.
"""

__all__ = [
    "LCPLineType",
    "LCPFitULMSpec",
    "LCPError",
    "LCPInputError",
    "LCPGenerationError",
    "LCPFittingError",
    "FitULMExportError",
    "LCPFitULMGenerator",
    "validate_frequency_vector",
    "validate_zy_matrices",
    "compute_ohl_zy",
    "compute_pipe_type_cable_zy",
    "compute_multi_armored_cable_zy",
]

from .specs import LCPLineType, LCPFitULMSpec
from .exceptions import (
    LCPError, LCPInputError, LCPGenerationError, LCPFittingError,
    FitULMExportError,
)
from .lcp_fitulm_generator import LCPFitULMGenerator
from .generation.ohl_deri_semlyen import compute_ohl_zy
from .generation.pipe_type_cable import compute_pipe_type_cable_zy
from .generation.multi_armored_cable import compute_multi_armored_cable_zy
