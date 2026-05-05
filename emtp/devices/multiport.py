"""MultiPortDevice Protocol — unified dispatch interface for multi-port elements.

Unlike the two-terminal :class:`Device`, a multi-port device may have an
arbitrary number of terminal-pair ports and does not map to a single
:class:`Branch`.  It participates in the same MNA assembly, RHS injection,
post-solve update, history-advance and rebuild-check phases as ordinary
devices, but through its own stamp/update methods.

Examples: Bergeron transmission line, ULM frequency-dependent line,
UMEC multi-winding transformer.
"""

from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class MultiPortDevice(Protocol):
    """Abstract interface for a circuit element with one or more ports.

    Each port is a (node_from, node_to) integer pair.  The MNA assembler
    calls ``stamp_G`` and ``stamp_rhs`` in order; after each linear solve
    it calls ``update_after_solve``; and after probe recording it calls
    ``update_history`` to advance internal state.

    ``check_rebuild_required`` is the mechanism for triggering topology /
    matrix rebuilds (e.g. UMEC saturation segment switching).
    """

    name: str

    # -- port topology --------------------------------------------------------

    @property
    def ports(self) -> tuple:
        """Return ``((nf0, nt0), (nf1, nt1), ...)`` for every port."""
        ...

    @property
    def contributes_G(self) -> bool:
        """Return True if this device contributes conductance to MNA G."""
        ...

    @property
    def is_dynamic(self) -> bool:
        """Return True if this device has history-state needing per-step update."""
        ...

    # -- lifecycle ------------------------------------------------------------

    def register_nodes(self, indexer) -> None:
        """Register every non-ground node with *indexer* before freezing."""
        ...

    def stamp_G(self, stamper, indexer) -> None:
        """Add port conductance contributions to the MNA G-matrix block."""
        ...

    def stamp_rhs(self, rhs: np.ndarray, indexer, t: float) -> None:
        """Add history / source currents to the MNA RHS vector."""
        ...

    def update_after_solve(self, V: np.ndarray, indexer, t: float) -> None:
        """Read port voltages / currents from the fresh MNA solution *V*."""
        ...

    def update_history(self, V: np.ndarray, indexer, dt: float) -> None:
        """Advance internal history sources after a completed time step."""
        ...

    def check_rebuild_required(
        self, V: np.ndarray, indexer, t: float,
    ) -> bool:
        """Return True if the device requires an MNA matrix rebuild.

        Called after each linear solve (before history update).
        """
        return False

    def reset_state(self) -> None:
        """Clear all dynamic state so the solver can be re-run."""
        ...
