"""MNA Kernel — sparse matrix assembly, LU factorization, and linear solve.

PR5: Thin wrapper around solver._stamping + _solve_mna().  The kernel
owns the G-matrix lifecycle (dirty detection, assembly, caching, solve)
so that solver.py delegates to ``self.kernel`` instead of managing
StampingEngine flags directly.

Later PRs will extract layout computation and topology signature logic.
"""

from .mna_kernel import MNAKernel

__all__ = ["MNAKernel"]
