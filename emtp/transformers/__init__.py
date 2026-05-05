"""EMTP transformer models — UMEC multi-port transformer.

Usage::

    from emtp.transformers.umec import (
        UMECTransformer,
        UMECTransformerData,
        WindingType,
        create_umec_transformer_3ph_bank_data,
    )
"""

try:
    from umec_transformer import (
        UMECTransformer,
        UMECTransformerData,
        WindingType,
        create_umec_transformer_3ph_bank,
    )
except ImportError:
    UMECTransformer = None
    UMECTransformerData = None
    WindingType = None
    create_umec_transformer_3ph_bank = None

__all__ = [
    "UMECTransformer",
    "UMECTransformerData",
    "WindingType",
    "create_umec_transformer_3ph_bank",
]
