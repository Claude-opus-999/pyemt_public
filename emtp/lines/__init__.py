"""EMTP transmission-line models.

Bergeron (lossless constant-parameter)::

    from emtp.lines.bergeron import BergeronLine, TransmissionLineFactory

ULM (frequency-dependent universal line model)::

    from emtp.lines.ulm import ULMLine, ULMModel, FitULMData, FitULMReader
"""

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
    "TransmissionLineInterface",
    "TransmissionLineFactory",
    "DelayBuffer",
    "FitULMData",
    "FitULMReader",
    "ULMLine",
    "ULMModel",
    "ULMBatchPack",
]
