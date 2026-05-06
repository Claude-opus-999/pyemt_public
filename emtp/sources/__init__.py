"""EMTP source models — lightning current generators (ATP-compatible).

Usage::

    from emtp.sources.lightning import (
        create_lightning_current_source,
        create_standard_twoexpf_current_source,
        TWOEXPFCurrentSource,
        HEIDLERFCurrentSource,
    )
"""

try:
    from atp_lightning_current_generator_simplified import (
        BaseLightningCurrentSource,
        TWOEXPFCurrentSource,
        HEIDLERFCurrentSource,
        LightningWaveform,
        STANDARD_DOUBLE_EXPONENTIAL_PARAMS,
        create_lightning_current_source,
        create_standard_twoexpf_current_source,
    )
except ImportError:
    BaseLightningCurrentSource = None
    TWOEXPFCurrentSource = None
    HEIDLERFCurrentSource = None
    LightningWaveform = None
    STANDARD_DOUBLE_EXPONENTIAL_PARAMS = None
    create_lightning_current_source = None
    create_standard_twoexpf_current_source = None

__all__ = [
    "BaseLightningCurrentSource",
    "TWOEXPFCurrentSource",
    "HEIDLERFCurrentSource",
    "LightningWaveform",
    "STANDARD_DOUBLE_EXPONENTIAL_PARAMS",
    "create_lightning_current_source",
    "create_standard_twoexpf_current_source",
]
