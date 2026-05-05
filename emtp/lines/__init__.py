"""EMTP transmission-line models.

Bergeron (lossless constant-parameter)::

    from emtp.lines.bergeron import BergeronLine, BergeronLineDevice, TransmissionLineFactory

ULM (frequency-dependent universal line model)::

    from emtp.lines.ulm import ULMLine, ULMModel, FitULMData, FitULMReader
"""

from .bergeron import BergeronLineDevice
from .ulm import ULMLineDevice

try:
    from transmission_line_emtp_v2 import (
        BergeronLine,
        TransmissionLineInterface,
        TransmissionLineFactory,
        DelayBuffer,
    )
except ImportError:
    BergeronLine = None
    TransmissionLineInterface = None
    TransmissionLineFactory = None
    DelayBuffer = None

try:
    from ulm_transmission_line_PARA import (
        FitULMData,
        FitULMReader,
        ULMLine,
        ULMModel,
        ULMBatchPack,
    )
except ImportError:
    FitULMData = None
    FitULMReader = None
    ULMLine = None
    ULMModel = None
    ULMBatchPack = None

__all__ = [
    "BergeronLine",
    "BergeronLineDevice",
    "ULMLineDevice",
    "TransmissionLineInterface",
    "TransmissionLineFactory",
    "DelayBuffer",
    "FitULMData",
    "FitULMReader",
    "ULMLine",
    "ULMModel",
    "ULMBatchPack",
]
