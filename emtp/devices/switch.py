"""Switch device — timed open/close events with topology rebuild."""

import numpy as np

from emtp.nodes import NodeIndexer
from emtp.stamping import COOStamper
from emtp.types import Branch, ElementType


class SwitchDevice:
    """Ideal(ish) switch with timed open / close events.

    No history term, but topology changes require MNA rebuild.
    """

    def __init__(self, name: str, node_from: int, node_to: int,
                 t_close: float, t_open: float,
                 R_closed: float, R_open: float,
                 initially_closed: bool) -> None:
        self.name = name
        self._nf = node_from
        self._nt = node_to
        R_init = R_closed if initially_closed else R_open
        self._branch = Branch(
            name=name, element_type=ElementType.SWITCH,
            node_from=node_from, node_to=node_to,
            value=R_init, Geq=1.0 / R_init,
            is_closed=initially_closed,
            R_closed=R_closed, R_open=R_open,
            t_close=t_close, t_open=t_open,
            state={
                'initially_closed': bool(initially_closed),
                'close_done': False, 'open_done': False,
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

    def update_timed_state(self, t: float) -> bool:
        """Apply timed open/close.  Returns True if topology changed."""
        br = self._branch
        changed = False
        if br.t_close >= 0 and not br.state.get('close_done', False) and t >= br.t_close:
            if not br.is_closed:
                br.is_closed = True
                br.value = br.R_closed
                br.Geq = 1.0 / br.R_closed
                changed = True
            br.state['close_done'] = True
        if br.t_open >= 0 and not br.state.get('open_done', False) and t >= br.t_open:
            if br.is_closed:
                br.is_closed = False
                br.value = br.R_open
                br.Geq = 1.0 / br.R_open
                changed = True
            br.state['open_done'] = True
        return changed

    def reset_state(self) -> None:
        br = self._branch
        br.current = 0.0
        br.voltage = 0.0
        br.current_prev = 0.0
        br.voltage_prev = 0.0
        br.current_history.clear()
        br.voltage_history.clear()
        initially_closed = bool(br.state.get('initially_closed', br.is_closed))
        br.state['close_done'] = False
        br.state['open_done'] = False
        br.is_closed = initially_closed
        br.value = br.R_closed if initially_closed else br.R_open
        br.Geq = 1.0 / br.value

    @property
    def is_dynamic(self) -> bool:
        return False

    @property
    def element_kind(self) -> str:
        return "SW"
