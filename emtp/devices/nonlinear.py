"""Nonlinear resistor device — PSCAD-style segmented MOA arrester."""

from typing import Any

import numpy as np

from emtp.nodes import NodeIndexer
from emtp.stamping import COOStamper
from emtp.types import Branch, ElementType


class NonlinearResistorDevice:
    """PSCAD-style segmented nonlinear resistor (MOA arrester).

    Geq and Ihist are managed externally by the SegmentedSolverHelper
    during iterative resolves.  The device is always dynamic because
    its segment can change at any step.
    """

    def __init__(self, name: str, node_from: int, node_to: int,
                 g_init: float, i_init: float,
                 model: Any, Rp: float) -> None:
        self.name = name
        self._nf = node_from
        self._nt = node_to
        self._model = model
        self._branch = Branch(
            name=name, element_type=ElementType.NONLINEAR_RESISTOR,
            node_from=node_from, node_to=node_to,
            value=0.0, Geq=g_init, Ihist=i_init, Rp=Rp,
            nonlinear_model=model,
        )

    def stamp_G(self, stamper: COOStamper, indexer: NodeIndexer) -> None:
        cf, ct = indexer.to_compact(self._nf), indexer.to_compact(self._nt)
        g_eq = self._branch.Geq
        if cf >= 0: stamper.add(cf, cf, g_eq)
        if ct >= 0: stamper.add(ct, ct, g_eq)
        if cf >= 0 and ct >= 0:
            stamper.add(cf, ct, -g_eq)
            stamper.add(ct, cf, -g_eq)

    def stamp_rhs(self, rhs: np.ndarray, indexer: NodeIndexer, t: float) -> None:
        cf, ct = indexer.to_compact(self._nf), indexer.to_compact(self._nt)
        i_eq = getattr(self._branch, 'Ihist', 0.0)
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
        if br.nonlinear_model is not None:
            br.current = br.nonlinear_model.get_current(v)
        else:
            br.current = v * br.Geq + br.Ihist

    def update_history(self, dt: float) -> None:
        pass  # Geq / Ihist are set externally by seg_helper

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
        return "NR"
