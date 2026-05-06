"""MNAKernel — owns the G-matrix lifecycle and sparse linear solve.

PR5: Thin wrapper.  The solver holds ``self.kernel`` and delegates:
- ``kernel.ensure_matrix()`` — rebuilds G if dirty
- ``kernel.solve(rhs)`` — factorizes and solves via SuperLU

Internal logic is unchanged; the kernel is introduced now so that
subsequent PRs can refactor layout, topology signatures, and
diagnostics without touching solver.py.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np
    import scipy.sparse as sp


class MNAKernel:
    """Sparse MNA matrix lifecycle manager.

    Owns:
    - Matrix dirty detection and rebuild scheduling
    - StampingEngine (COOStamper + G assembly)
    - LU factorization cache (SuperLU via scipy.sparse.linalg.splu)
    - Linear solve

    Parameters
    ----------
    solver: EMTPSolver
        The owning solver.  The kernel reads the solver's registry,
        indexer, and stamping engine.  This reference will be replaced
        with explicit dependencies in a later PR.
    """

    def __init__(self, solver):
        self._solver = solver

    # -----------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------

    def ensure_matrix(self) -> "sp.csc_matrix":
        """Return the current MNA matrix, rebuilding if needed.

        Side-effects:
        - Updates solver._stats['G_rebuilds'] / ['G_cache_hits']
        - Updates solver._cached_MNA
        """
        s = self._solver
        eng = s._stamping
        if eng.G_dirty or eng.cached_MNA is None:
            s._build_MNA_matrix()
            s._cached_MNA = eng.cached_MNA
            s._stats['G_rebuilds'] = s._stats.get('G_rebuilds', 0) + 1
        else:
            s._cached_MNA = eng.cached_MNA
            s._stats['G_cache_hits'] = s._stats.get('G_cache_hits', 0) + 1
        return s._cached_MNA

    def solve(self, MNA: "sp.csc_matrix", rhs: "np.ndarray") -> "np.ndarray":
        """Solve MNA · x = rhs using the current LU factorization."""
        return self._solver._solve_mna(MNA, rhs)

    @property
    def is_dirty(self) -> bool:
        """True when the G matrix needs to be rebuilt."""
        return self._solver._stamping._G_dirty

    @property
    def cached_matrix(self):
        """The most recently assembled MNA matrix (or None)."""
        return self._solver._stamping._cached_MNA

    def mark_dirty(self) -> None:
        """Force a matrix rebuild on the next ensure_matrix() call."""
        self._solver._stamping.mark_dirty()
