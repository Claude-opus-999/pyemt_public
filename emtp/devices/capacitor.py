"""Capacitor device — implicit trapezoidal rule discretisation."""

import numpy as np

from emtp.nodes import NodeIndexer
from emtp.stamping import COOStamper
from emtp.types import Branch, ElementType


class CapacitorDevice:
    """Capacitor discretised with the implicit trapezoidal rule.

    Geq = 2C / Δt    Ihist_{k+1} = -Ihist_k - 2·Geq·v_k
    """

    def __init__(self, name: str, node_from: int, node_to: int,
                 C: float, dt: float, Rp: float = 0.0) -> None:
        self.name = name
        self._nf = node_from
        self._nt = node_to
        self._C = C
        self._G = 2.0 * C / dt
        self._Rp = Rp if Rp else 0.0
        self._Gd = 1.0 / Rp if Rp and Rp > 0 else 0.0
        self._branch = Branch(
            name=name, element_type=ElementType.CAPACITOR,
            node_from=node_from, node_to=node_to,
            value=C, Geq=self._G, Rp=self._Rp, Geq_damping=self._Gd,
        )

    def stamp_G(self, stamper: COOStamper, indexer: NodeIndexer) -> None:
        cf, ct = indexer.to_compact(self._nf), indexer.to_compact(self._nt)
        g_eq = self._G + self._Gd
        if cf >= 0: stamper.add(cf, cf, g_eq)
        if ct >= 0: stamper.add(ct, ct, g_eq)
        if cf >= 0 and ct >= 0:
            stamper.add(cf, ct, -g_eq)
            stamper.add(ct, cf, -g_eq)

    def stamp_rhs(self, rhs: np.ndarray, indexer: NodeIndexer, t: float) -> None:
        cf, ct = indexer.to_compact(self._nf), indexer.to_compact(self._nt)
        i_eq = self._branch.Ihist
        if cf >= 0: rhs[cf] -= i_eq
        if ct >= 0: rhs[ct] += i_eq

    def update_branch_quantities(self, V: np.ndarray, indexer: NodeIndexer) -> None:
        cf = indexer.to_compact(self._nf)
        ct = indexer.to_compact(self._nt)
        v = (V[cf] if cf >= 0 else 0.0) - (V[ct] if ct >= 0 else 0.0)
        br = self._branch
        br.voltage_prev = br.voltage
        br.voltage = v
        br.current_prev = br.current
        br.current = (self._G + self._Gd) * v + br.Ihist

    def update_history(self, dt: float) -> None:
        br = self._branch
        br.Ihist = -br.Ihist - 2.0 * self._G * br.voltage

    def reset_state(self) -> None:
        br = self._branch
        br.current = 0.0
        br.voltage = 0.0
        br.current_prev = 0.0
        br.voltage_prev = 0.0
        br.current_history.clear()
        br.voltage_history.clear()
        br.Ihist = 0.0

    @property
    def is_dynamic(self) -> bool:
        return True

    @property
    def element_kind(self) -> str:
        return "C"
