"""EMTP nonlinear models — PSCAD-style segmented MOA and LPM flashover.

Usage::

    from emtp.nonlinear import (
        SegmentedMOAResistor,
        InsulatorFlashoverLPM,
        LPMConfig,
        LPMInsulatorType,
        SegmentedSolverHelper,
    )
"""

try:
    from nonlinear_models_pscad import (
        InsulatorFlashoverLPM,
        LPMConfig,
        LPMInsulatorType,
        SegmentedSolverHelper,
        SegmentedMOAResistor,
    )
except ImportError:
    InsulatorFlashoverLPM = None
    LPMConfig = None
    LPMInsulatorType = None
    SegmentedSolverHelper = None
    SegmentedMOAResistor = None

__all__ = [
    "InsulatorFlashoverLPM",
    "LPMConfig",
    "LPMInsulatorType",
    "SegmentedSolverHelper",
    "SegmentedMOAResistor",
]
