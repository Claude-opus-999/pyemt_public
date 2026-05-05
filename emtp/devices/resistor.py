"""Resistor device — pure resistance, no history, no dynamics."""

import numpy as np

from emtp.nodes import NodeIndexer
from emtp.stamping import COOStamper
from emtp.types import Branch, ElementType


class ResistorDevice:
    """Pure resistor.  No history, no dynamics."""

    def __init__(self, name: str, node_from: int, node_to: int, R: float) -> None:
        self.name = name
        self._nf = node_from
        self._nt = node_to
        self._R = R
        self._G = 1.0 / R
        self._branch = Branch(
            name=name, element_type=ElementType.RESISTOR,
            node_from=node_from, node_to=node_to, value=R, Geq=self._G,
        )

    def stamp_G(self, stamper: COOStamper, indexer: NodeIndexer) -> None:
        cf, ct = indexer.to_compact(self._nf), indexer.to_compact(self._nt)
        if cf >= 0: stamper.add(cf, cf, self._G)
        if ct >= 0: stamper.add(ct, ct, self._G)
        if cf >= 0 and ct >= 0:
            stamper.add(cf, ct, -self._G)
            stamper.add(ct, cf, -self._G)

    def stamp_rhs(self, rhs: np.ndarray, indexer: NodeIndexer, t: float) -> None:
        pass

    def update_branch_quantities(self, V: np.ndarray, indexer: NodeIndexer) -> None:
        cf = indexer.to_compact(self._nf)
        ct = indexer.to_compact(self._nt)
        v = (V[cf] if cf >= 0 else 0.0) - (V[ct] if ct >= 0 else 0.0)
        br = self._branch
        br.voltage = v
        br.current = v * self._G

    def update_history(self, dt: float) -> None:
        pass

    def reset_state(self) -> None:
        br = self._branch
        br.current = 0.0
        br.voltage = 0.0
        br.current_prev = 0.0
        br.voltage_prev = 0.0
        br.current_history.clear()
        br.voltage_history.clear()

    @property
    def is_dynamic(self) -> bool:
        return False

    @property
    def element_kind(self) -> str:
        return "R"
