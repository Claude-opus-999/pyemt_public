"""Legacy compatibility entry point.

Preferred import::

    from emtp import EMTPSolver

This module is kept for backward compatibility.  Existing scripts
that use ``from emtp_solver_v3 import EMTPSolver`` continue to work.
"""

from emtp.solver import EMTPSolver  # noqa: E402, F401

# ---------------------------------------------------------------------------
# Re-export symbols that external code imports directly from this module
# (NodeBook / NodeIndexer used by tests/test_nodes.py,
#  SparseLinearSolver / ValidationReport / ValidationIssue used by
#  tests/test_solver_regression.py, etc.)
# ---------------------------------------------------------------------------
from emtp.nodes import NodeBook, NodeIndexer                  # noqa: E402, F401
from emtp.types import (                                       # noqa: E402, F401
    VoltageSource, ValidationIssue, ValidationReport,
    RHSPlan, ElementType, Branch, CurrentSource, LineData,
)
from emtp.sparse_solver import (                               # noqa: E402, F401
    _SPARSE_SOLVER_NAME, _sparse_factorize, SparseLinearSolver,
)
from emtp.stamping import COOStamper, StampingEngine           # noqa: E402, F401
from emtp.devices import (                                      # noqa: E402, F401
    Device,
    ResistorDevice,
    InductorDevice,
    CapacitorDevice,
    SwitchDevice,
    SeriesRLDevice,
    NonlinearResistorDevice,
    LPMFlashoverDevice,
    _update_series_rl_history_static,
)
from emtp.runtime import DynamicDeviceRuntime                  # noqa: E402, F401

# ---------------------------------------------------------------------------
# Optional module re-exports (preserve original try/except fallback structure)
# ---------------------------------------------------------------------------

try:
    from nonlinear_models_pscad import (
        InsulatorFlashoverLPM,
        LPMConfig,
        LPMInsulatorType,
        SegmentedSolverHelper,
        SegmentedMOAResistor,
    )
    NONLINEAR_AVAILABLE = True
except ImportError:
    NONLINEAR_AVAILABLE = False

    class _UnavailableNonlinear:
        def __init__(self, *args, **kwargs):
            raise ImportError(
                "nonlinear_models_pscad.py is required for MOA/LPM nonlinear components"
            )

    class SegmentedSolverHelper:
        def register(self, *args, **kwargs):
            raise ImportError(
                "nonlinear_models_pscad.py is required for segmented nonlinear components"
            )
        def reset_all(self):
            return None
        def check_all_segments(self, voltages):
            return False, {}

    InsulatorFlashoverLPM = _UnavailableNonlinear
    LPMConfig = _UnavailableNonlinear
    LPMInsulatorType = _UnavailableNonlinear
    SegmentedMOAResistor = _UnavailableNonlinear

try:
    from transmission_line_emtp_v2 import (
        BergeronLine,
        TransmissionLineInterface,
    )
    TRANSMISSION_LINE_AVAILABLE = True
except ImportError:
    TRANSMISSION_LINE_AVAILABLE = False

    class TransmissionLineInterface:
        """Placeholder used when transmission_line_emtp_v2.py is not installed."""
        pass

    class BergeronLine(TransmissionLineInterface):
        def __init__(self, *args, **kwargs):
            raise ImportError(
                "transmission_line_emtp_v2.py is required for BergeronLine components"
            )

try:
    from ulm_transmission_line_PARA import (
        FitULMData,
        FitULMReader,
        ULMLine,
        ULMModel,
        ULMBatchPack,
    )
    ULM_AVAILABLE = True
except ImportError:
    ULM_AVAILABLE = False
    ULMLine = None
    ULMModel = None
    FitULMReader = None
    FitULMData = None
    ULMBatchPack = None

try:
    from umec_transformer import (
        UMECTransformer,
        UMECTransformerData,
        WindingType,
        create_umec_transformer_3ph_bank,
    )
    UMEC_AVAILABLE = True
except ImportError:
    UMEC_AVAILABLE = False
    UMECTransformer = None
    UMECTransformerData = None
    WindingType = None
    create_umec_transformer_3ph_bank = None

try:
    from atp_lightning_current_generator_simplified import (
        BaseLightningCurrentSource,
        TWOEXPFCurrentSource,
        HEIDLERFCurrentSource,
        create_lightning_current_source,
        create_standard_twoexpf_current_source,
    )
except ImportError:
    BaseLightningCurrentSource = ()
    TWOEXPFCurrentSource = None
    HEIDLERFCurrentSource = None
    create_lightning_current_source = None
    create_standard_twoexpf_current_source = None

__all__ = ["EMTPSolver"]
