"""MNAAssembler — builds the augmented MNA system matrix and RHS vector.

Extracted from ``EMTPSolver._build_MNA_matrix`` and
``EMTPSolver._build_MNA_rhs`` so the solver delegates assembly
orchestration to a standalone component.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import scipy.sparse as sp

from emtp.nodes import NodeIndexer
from emtp.types import VoltageSource
from emtp.stamping import COOStamper, StampingEngine


class MNAAssembler:
    """Build the (n+m)×(n+m) MNA augmented system.

    Parameters
    ----------
    stamping_engine:
        The engine that manages COO-stamping lifecycle and LU caching.
    indexer:
        Compact node-indexer (must be frozen before assembly).
    """

    def __init__(self, stamping_engine: StampingEngine, indexer: NodeIndexer):
        self._eng = stamping_engine
        self._indexer = indexer

    # -- G-matrix ------------------------------------------------------------

    def begin_G(self, m_vs: int) -> COOStamper:
        return self._eng.begin_G(self._indexer.n, m_vs)

    def finish_G(self, stamper: COOStamper) -> sp.csc_matrix:
        return self._eng.finish_G(stamper)

    def stamp_devices_G(
        self, stamper: COOStamper, devices: List,
    ) -> None:
        self._eng.stamp_devices_G(stamper, devices)

    def stamp_multiport_G(
        self, stamper: COOStamper, multiport_devices: List,
    ) -> None:
        for dev in multiport_devices:
            if dev.contributes_G:
                dev.stamp_G(stamper, self._indexer)

    def stamp_vs_G(
        self, stamper: COOStamper, vs_list: List[VoltageSource],
    ) -> None:
        self._eng.stamp_vs_G(stamper, vs_list)

    # -- RHS vector ----------------------------------------------------------

    def new_rhs(self, size: int) -> np.ndarray:
        return self._eng.ensure_rhs_buf(size)

    def stamp_devices_rhs(
        self, rhs: np.ndarray, devices: List, t: float,
    ) -> None:
        for dev in devices:
            dev.stamp_rhs(rhs, self._indexer, t)

    def stamp_multiport_rhs(
        self, rhs: np.ndarray, multiport_devices: List, t: float,
    ) -> None:
        for dev in multiport_devices:
            dev.stamp_rhs(rhs, self._indexer, t)

    def stamp_current_sources_rhs(
        self,
        rhs: np.ndarray,
        current_sources: Dict,
        t: float,
    ) -> None:
        for source in current_sources.values():
            I_s = source.current_at(t)
            cf = self._indexer.to_compact(source.node_from)
            ct = self._indexer.to_compact(source.node_to)
            if cf >= 0:
                rhs[cf] -= I_s
            if ct >= 0:
                rhs[ct] += I_s

    def stamp_vs_rhs(
        self,
        rhs: np.ndarray,
        vs_list: List[VoltageSource],
        t: float,
    ) -> None:
        n = self._indexer.n
        for k, vs in enumerate(vs_list):
            rhs[n + k] = vs.voltage_at(t)

    def solve(
        self,
        MNA: sp.csc_matrix,
        rhs: np.ndarray,
        vs_list: List[VoltageSource],
    ) -> np.ndarray:
        return self._eng.solve(MNA, rhs, vs_list or [])
