"""RHSEngine — builds the MNA right-hand-side vector each time step.

PR4: Thin wrapper.  The solver holds ``self.rhs_engine`` and delegates
``_build_MNA_rhs()`` to it.  Internal logic is unchanged; the engine
is introduced now so that subsequent PRs can refactor history injection,
source pre-sampling, and RHSPlan compilation without touching solver.py.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np


class RHSEngine:
    """Right-hand-side assembler for MNA transient simulation.

    Owns:
    - RHS buffer management (reuse across steps)
    - Current source injection evaluation
    - Voltage source constraint row population
    - Device / multiport / line / transformer history injection
    - RHSPlan compilation and fast-path application

    Parameters
    ----------
    solver: EMTPSolver
        The owning solver (used to access registry, indexer, sources, etc.).
        This reference will be removed in a later PR when RHSEngine takes
        explicit dependencies on registry + indexer + source_sampler.
    """

    def __init__(self, solver):
        self._solver = solver

    # -----------------------------------------------------------------
    # Public API — called once per time step
    # -----------------------------------------------------------------

    def build(self) -> "np.ndarray":
        """Return the MNA right-hand-side vector for the current time step."""
        return self._solver._build_MNA_rhs()

    def build_fast(self) -> "np.ndarray":
        """Return the RHS via precompiled RHSPlan (fast path)."""
        return self._solver._build_rhs_fast()

    def pre_sample_sources(self, n_steps: int) -> None:
        """Pre-sample all independent source waveforms into flat arrays."""
        self._solver._pre_sample_sources(n_steps)

    def invalidate_plan(self) -> None:
        """Mark the compiled RHS plan as dirty (topology changed)."""
        self._solver._rhs_plan_dirty = True
