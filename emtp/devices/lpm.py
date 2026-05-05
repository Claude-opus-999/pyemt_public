"""LPM flashover device — CIGRE leader-progression-model insulator switch."""

import numpy as np

from emtp.nodes import NodeIndexer
from emtp.stamping import COOStamper
from emtp.types import Branch, ElementType


class LPMFlashoverDevice:
    """CIGRE leader-progression-model insulator flashover switch.

    Wraps a switch-type Branch whose open/close state is driven by the
    LPM physics model rather than timed events.  The solver still
    manages the LPM state machine via ``_update_lpm_states``; this
    device only provides the standard electrical interface.
    """

    def __init__(self, name: str, node_from: int, node_to: int,
                 R_open: float, R_arc: float) -> None:
        self.name = name
        self._nf = node_from
        self._nt = node_to
        self._branch = Branch(
            name=name, element_type=ElementType.SWITCH,
            node_from=node_from, node_to=node_to,
            value=R_open, Geq=1.0 / R_open,
            is_closed=False,
            R_closed=R_arc, R_open=R_open,
            t_close=-1.0, t_open=-1.0,
            state={
                'initially_closed': False,
                'close_done': False,
                'open_done': False,
            },
        )

    def stamp_G(self, stamper: COOStamper, indexer: NodeIndexer) -> None:
        cf, ct = indexer.to_compact(self._nf), indexer.to_compact(self._nt)
        g = self._branch.Geq
        if cf >= 0: stamper.add(cf, cf, g)
        if ct >= 0: stamper.add(ct, ct, g)
        if cf >= 0 and ct >= 0:
            stamper.add(cf, ct, -g)
            stamper.add(ct, cf, -g)

    def stamp_rhs(self, rhs: np.ndarray, indexer: NodeIndexer, t: float) -> None:
        pass

    def update_branch_quantities(self, V: np.ndarray, indexer: NodeIndexer) -> None:
        cf = indexer.to_compact(self._nf)
        ct = indexer.to_compact(self._nt)
        v = (V[cf] if cf >= 0 else 0.0) - (V[ct] if ct >= 0 else 0.0)
        br = self._branch
        br.voltage_prev = br.voltage
        br.voltage = v
        br.current_prev = br.current
        br.current = v * br.Geq

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
        br.is_closed = False
        br.value = br.R_open
        br.Geq = 1.0 / br.R_open
        br.state['initially_closed'] = False
        br.state['close_done'] = False
        br.state['open_done'] = False

    @property
    def is_dynamic(self) -> bool:
        return False

    @property
    def element_kind(self) -> str:
        return "LPM"
