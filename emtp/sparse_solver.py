"""Sparse linear solver with LU-factorisation caching.

Uses scipy.sparse.linalg.splu (SuperLU backend) for CSC sparse LU
decomposition.  Re-factors only when the matrix identity changes.
"""

from typing import Any

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import splu

# ---------------------------------------------------------------------------
# SuperLU is the sole sparse solver backend.
# KLU / UMFPACK optional acceleration has been removed to avoid
# extra library dependencies and environment variance.
# ---------------------------------------------------------------------------
_SPARSE_SOLVER_NAME: str = "SuperLU"


def _sparse_factorize(A: sp.csc_matrix) -> Any:
    """Sparse LU-decompose a CSC matrix.

    Returns an object satisfying ``.solve(rhs) -> ndarray``.
    Uses scipy.sparse.linalg.splu with the SuperLU backend.
    """
    return splu(A, permc_spec="MMD_AT_PLUS_A")


def sparse_solver_name() -> str:
    return _SPARSE_SOLVER_NAME


class SparseLinearSolver:
    """Sparse LU solver with matrix-id-based caching.

    Independent of MNA / device concerns — it only knows about sparse
    matrices and right-hand sides.  Caches the factorisation and
    re-factors only when *matrix_id* changes.
    """

    _LU_SINGULAR_REG: float = 1e-12

    def __init__(self, allow_singular_regularization: bool = False) -> None:
        self._allow_reg = allow_singular_regularization
        self._lu: Any = None
        self._matrix_id: int = -1

    def solve(
        self,
        A: sp.csc_matrix,
        b: np.ndarray,
        matrix_id: int,
        n_compact: int,
    ) -> np.ndarray:
        """Return x = A^{-1}b, factoring A only when *matrix_id* changes."""
        if matrix_id != self._matrix_id:
            self._lu = self._factorize(A, n_compact)
            self._matrix_id = matrix_id
        try:
            return self._lu.solve(b)
        except (RuntimeError, ValueError) as exc:
            raise RuntimeError("MNA sparse solve failed") from exc

    def invalidate(self) -> None:
        self._lu = None
        self._matrix_id = -1

    def _factorize(self, A: sp.csc_matrix, n_compact: int) -> Any:
        try:
            return _sparse_factorize(A)
        except RuntimeError as exc:
            if not self._allow_reg:
                raise RuntimeError(
                    "MNA matrix is singular. Check floating nodes, "
                    "missing ground reference, open circuits, ideal "
                    "voltage-source loops, or disconnected subcircuits."
                ) from exc
            N = A.shape[0]
            reg_diag = np.zeros(N, dtype=np.float64)
            reg_diag[:n_compact] = self._LU_SINGULAR_REG
            reg = sp.diags(reg_diag, format="csc")
            return _sparse_factorize(A + reg)
