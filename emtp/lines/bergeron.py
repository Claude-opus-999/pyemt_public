"""BergeronLineDevice — wraps transmission_line_emtp_v2.BergeronLine as a
:class:`MultiPortDevice` so it participates in unified MNA assembly,
RHS injection, post-solve update and history advance.
"""

from emtp.devices.multiport import MultiPortDevice


class BergeronLineDevice:
    """MultiPortDevice adapter for a lossless constant-parameter Bergeron line.

    The underlying Bergeron model treats each end as a port referenced to
    ground.  For a single-phase line::

        ports = ((node_k, 0), (node_m, 0))

    The equivalent conductance ``G_eq = 1 / Zc`` is stamped on the diagonal
    at both ends, and the history currents ``I_hist_k`` / ``I_hist_m`` are
    injected into the RHS according to the transmission-line convention
    documented in ``DIRECTION_CONVENTIONS.md``.
    """

    def __init__(self, name: str, impl, node_k: int, node_m: int):
        self.name = name
        self.impl = impl          # BergeronLine (transmission_line_emtp_v2)
        self._node_k = node_k
        self._node_m = node_m
        self._vk: float = 0.0
        self._vm: float = 0.0

    # -- MultiPortDevice port topology ----------------------------------------

    @property
    def ports(self):
        return ((self._node_k, 0), (self._node_m, 0))

    @property
    def contributes_G(self) -> bool:
        return True

    @property
    def is_dynamic(self) -> bool:
        return True

    def register_nodes(self, indexer) -> None:
        indexer.register(self._node_k)
        indexer.register(self._node_m)

    # -- MNA stamping ---------------------------------------------------------

    def stamp_G(self, stamper, indexer) -> None:
        G_eq = float(self.impl.G_eq)
        ck = indexer.to_compact(self._node_k)
        cm = indexer.to_compact(self._node_m)
        stamper.add(ck, ck, G_eq)
        stamper.add(cm, cm, G_eq)

    def stamp_rhs(self, rhs, indexer, t: float) -> None:
        ck = indexer.to_compact(self._node_k)
        cm = indexer.to_compact(self._node_m)
        rhs[ck] -= float(self.impl.I_hist_k)
        rhs[cm] -= float(self.impl.I_hist_m)

    # -- post-solve update ----------------------------------------------------

    def update_after_solve(self, V, indexer, t: float) -> None:
        ck = indexer.to_compact(self._node_k)
        cm = indexer.to_compact(self._node_m)
        self._vk = float(V[ck])
        self._vm = float(V[cm])

    def update_history(self, V, indexer, dt: float) -> None:
        self.impl.update_state(self._vk, self._vm)
        self.impl.update_history_sources()

    def check_rebuild_required(self, V, indexer, t: float) -> bool:
        return False

    def reset_state(self) -> None:
        self._vk = 0.0
        self._vm = 0.0
