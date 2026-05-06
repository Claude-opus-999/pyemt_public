"""RHS Engine — right-hand-side vector construction for MNA.

PR4: Thin wrapper around solver._build_MNA_rhs().  Maintains existing
behaviour while creating a dedicated home for RHS logic that will be
fleshed out in later PRs.
"""

from .rhs_engine import RHSEngine

__all__ = ["RHSEngine"]
