"""Series-RL device — two-terminal branch without internal node."""

import numpy as np

from emtp.nodes import NodeIndexer
from emtp.stamping import COOStamper
from emtp.types import Branch, ElementType


def _update_series_rl_history_static(branch: Branch) -> None:
    """Update the history source of a Series-RL two-terminal branch."""
    p = branch.params
    s = branch.state
    R = p['R']
    G_L = p['G_L']
    denom = p['denom']

    i = branch.current
    v_branch = branch.voltage
    v_L = v_branch - R * i

    hist_raw_old = s.get('Ihist_L_raw', 0.0)
    hist_raw_new = hist_raw_old + 2.0 * G_L * v_L

    s['v_L'] = v_L
    s['Ihist_L_raw'] = hist_raw_new
    branch.Ihist = hist_raw_new / denom


class SeriesRLDevice:
    """Two-terminal series-RL branch without internal node.

    Discretised with trapezoidal rule.  The history source is split
    between the internal inductor history and the series resistance.
    """

    def __init__(self, name: str, node_from: int, node_to: int,
                 R: float, L: float, dt: float) -> None:
        self.name = name
        self._nf = node_from
        self._nt = node_to
        G_L = dt / (2.0 * L)
        denom = 1.0 + R * G_L
        G_eq = G_L / denom
        self._branch = Branch(
            name=name, element_type=ElementType.SERIES_RL,
            node_from=node_from, node_to=node_to,
            value=R, Geq=G_eq, Ihist=0.0,
            params={'R': R, 'L': L, 'G_L': G_L, 'denom': denom},
            state={'Ihist_L_raw': 0.0, 'v_L': 0.0},
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
        br.current = br.Geq * v + br.Ihist

    def update_history(self, dt: float) -> None:
        _update_series_rl_history_static(self._branch)

    def reset_state(self) -> None:
        br = self._branch
        br.current = 0.0
        br.voltage = 0.0
        br.current_prev = 0.0
        br.voltage_prev = 0.0
        br.current_history.clear()
        br.voltage_history.clear()
        br.Ihist = 0.0
        br.state['Ihist_L_raw'] = 0.0
        br.state['v_L'] = 0.0

    @property
    def is_dynamic(self) -> bool:
        return True

    @property
    def element_kind(self) -> str:
        return "SRL"
