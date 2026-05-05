"""Device Protocol — the abstract interface every branch element implements."""

from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class Device(Protocol):
    """Branch device protocol: every element kind implements these methods.

    This replaces the ``if et == ElementType.X`` dispatch scattered across
    the stamp / RHS / update / history / reset paths with a single virtual
    call per device per step.
    """
    name: str

    def stamp_G(self, stamper: 'COOStamper', indexer: 'NodeIndexer') -> None:
        """Stamp conductance contributions into the MNA G-matrix block."""
        ...

    def stamp_rhs(self, rhs: np.ndarray, indexer: 'NodeIndexer', t: float) -> None:
        """Stamp history / source currents into the MNA RHS vector."""
        ...

    def update_branch_quantities(self, V: np.ndarray, indexer: 'NodeIndexer') -> None:
        """Compute branch voltage & current from the MNA solution V."""
        ...

    def update_history(self, dt: float) -> None:
        """Advance internal history sources after a completed time step."""
        ...

    def reset_state(self) -> None:
        """Clear all dynamic state so the solver can be re-run."""
        ...

    @property
    def is_dynamic(self) -> bool:
        """True if this device contributes history terms to the RHS."""
        ...

    @property
    def element_kind(self) -> str:
        """Short string tag for diagnostics / logging."""
        ...
