"""MNA matrix assembly and sparse-solve orchestration.

COOStamper accumulates triplet contributions for building the sparse
G-matrix.  StampingEngine manages the assembly lifecycle (begin/finish),
device stamping, voltage-source stamping, and delegates linear solves
to SparseLinearSolver.
"""

from typing import Dict, List, Optional

import numpy as np
import scipy.sparse as sp

from .sparse_solver import SparseLinearSolver
from .nodes import NodeIndexer
from .types import VoltageSource


class COOStamper:
    """Minimal COO triplet accumulator for building a sparse MNA matrix.

    Usage::

        stamper = COOStamper(N)
        for dev in devices:
            dev.stamp_G(stamper, indexer)
        A = stamper.tocsc()
    """

    def __init__(self, size: int) -> None:
        self.rows: List[int] = []
        self.cols: List[int] = []
        self.vals: List[float] = []
        self._size = size

    def add(self, r: int, c: int, v: float) -> None:
        self.rows.append(r)
        self.cols.append(c)
        self.vals.append(v)

    def tocsc(self) -> sp.csc_matrix:
        return sp.coo_matrix(
            (self.vals, (self.rows, self.cols)),
            shape=(self._size, self._size),
        ).tocsc()


class StampingEngine:
    """MNA matrix assembly, sparse solve and associated caching.

    The engine owns the COO-stamper lifecycle so the solver can interpose
    line and transformer stamps between :meth:`begin_G` and :meth:`finish_G`.
    Linear solving is delegated to :class:`SparseLinearSolver`.
    """

    def __init__(
        self,
        indexer: NodeIndexer,
        allow_singular_regularization: bool = False,
    ) -> None:
        self._indexer = indexer
        self._G_dirty: bool = True
        self._cached_MNA: Optional[sp.csc_matrix] = None
        self._matrix_id: int = 0
        self._rhs_buf: Optional[np.ndarray] = None
        self._linear_solver = SparseLinearSolver(
            allow_singular_regularization=allow_singular_regularization,
        )

    # -- public helpers --------------------------------------------------------

    @property
    def n(self) -> int:
        return self._indexer.n

    @property
    def G_dirty(self) -> bool:
        return self._G_dirty

    @property
    def matrix_id(self) -> int:
        return self._matrix_id

    @property
    def cached_MNA(self) -> Optional[sp.csc_matrix]:
        return self._cached_MNA

    @property
    def rhs_buf(self) -> Optional[np.ndarray]:
        return self._rhs_buf

    def mark_dirty(self) -> None:
        self._G_dirty = True
        self._cached_MNA = None
        self._linear_solver.invalidate()

    def ensure_rhs_buf(self, size: int) -> np.ndarray:
        """Return a zeroed RHS buffer of at least *size* elements."""
        buf = self._rhs_buf
        if buf is None or buf.shape[0] < size:
            buf = np.zeros(size, dtype=np.float64)
            self._rhs_buf = buf
        else:
            buf[:size] = 0.0
        return buf

    # -- G-matrix assembly (open/close pattern) --------------------------------

    def begin_G(self, n_compact: int, n_vs: int) -> COOStamper:
        """Create a COO stamper for an *(n_compact + n_vs)* MNA matrix."""
        return COOStamper(n_compact + n_vs)

    def stamp_devices_G(
        self, stamper: COOStamper, devices: List,
    ) -> None:
        for dev in devices:
            dev.stamp_G(stamper, self._indexer)

    def stamp_vs_G(
        self, stamper: COOStamper, vs_list: List[VoltageSource],
    ) -> None:
        n = self._indexer.n
        for k, vs in enumerate(vs_list):
            vs_row = n + k
            if vs.node_pos > 0:
                cp = self._indexer.to_compact(vs.node_pos)
                stamper.add(cp, vs_row, 1.0)
                stamper.add(vs_row, cp, 1.0)
            if vs.node_neg > 0:
                cn = self._indexer.to_compact(vs.node_neg)
                stamper.add(cn, vs_row, -1.0)
                stamper.add(vs_row, cn, -1.0)

    def finish_G(self, stamper: COOStamper) -> sp.csc_matrix:
        """Convert COO -> CSC, cache, bump matrix_id, mark clean."""
        self._cached_MNA = stamper.tocsc()
        self._G_dirty = False
        self._linear_solver.invalidate()
        self._matrix_id += 1
        return self._cached_MNA

    # -- sparse solve (delegates to SparseLinearSolver) ------------------------

    def solve(
        self,
        MNA: sp.csc_matrix,
        rhs: np.ndarray,
        vs_list: List[VoltageSource],
    ) -> np.ndarray:
        """Solve MNA·x = rhs, return V = x[:n], write back VS currents."""
        n = self._indexer.n
        x = self._linear_solver.solve(MNA, rhs, self._matrix_id, n)
        V = x[:n]
        for k, vs in enumerate(vs_list):
            vs.current = -x[n + k]
        return V
