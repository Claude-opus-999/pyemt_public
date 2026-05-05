"""Legacy compatibility module for EMTP base types.

Prefer the new import path::

    from emtp.types import Branch, CurrentSource, ElementType, LineData

This module is kept for backward compatibility with existing scripts
and will continue to work without deprecation warnings.
"""

# Re-export base types from the canonical location
from emtp.types import (
    ElementType,
    Branch,
    CurrentSource,
    LineData,
)

# Lightning waveform compatibility wrappers
try:
    from lightning_waveform import (
        STANDARD_DOUBLE_EXPONENTIAL_PARAMS,
        LightningWaveform,
        create_custom_waveform,
        create_lightning_waveform,
    )
except ImportError:
    from atp_lightning_current_generator_simplified import (
        STANDARD_DOUBLE_EXPONENTIAL_PARAMS,
        LightningWaveform,
        create_lightning_current_source,
        create_standard_twoexpf_current_source,
    )

    def create_lightning_waveform(*args, **kwargs):
        """Compatibility wrapper around create_lightning_current_source()."""
        return create_lightning_current_source(*args, **kwargs)

    def create_custom_waveform(*args, **kwargs):
        """Compatibility wrapper around create_lightning_current_source()."""
        return create_lightning_current_source(*args, **kwargs)


__all__ = [
    'ElementType', 'Branch', 'CurrentSource', 'LineData',
    'LightningWaveform', 'STANDARD_DOUBLE_EXPONENTIAL_PARAMS',
    'create_lightning_waveform', 'create_custom_waveform',
]
