"""UMECTransformerDevice — wraps umec_transformer.UMECTransformer as a
:class:`MultiPortDevice` so it participates in unified MNA assembly,
RHS injection, post-solve update, history advance and saturation-driven
rebuild checks.
"""

import numpy as np

from emtp.devices.multiport import MultiPortDevice


class UMECTransformerDevice:
    """MultiPortDevice adapter for a UMEC multi-port transformer.

    Each winding terminal-pair is a port; the underlying UMEC model
    provides a port admittance matrix ``G_tf`` and Norton history vector
    ``I_hist_tf`` via :meth:`get_norton_equivalent`.

    Saturation segment changes are detected by
    :meth:`check_rebuild_required`, which forwards to the underlying
    :meth:`check_saturation` and returns ``True`` to trigger an MNA
    matrix rebuild.
    """

    def __init__(self, name: str, impl):
        self.name = name
        self.impl = impl            # UMECTransformer

    # -- port topology --------------------------------------------------------

    @property
    def ports(self):
        return tuple(self.impl.get_port_nodes())

    @property
    def contributes_G(self) -> bool:
        return True

    @property
    def is_dynamic(self) -> bool:
        return True

    def register_nodes(self, indexer) -> None:
        for nf, nt in self.ports:
            if nf > 0:
                indexer.register(nf)
            if nt > 0:
                indexer.register(nt)

    # -- MNA stamping ---------------------------------------------------------

    def stamp_G(self, stamper, indexer) -> None:
        G_tf, _ = self.impl.get_norton_equivalent()
        mp = G_tf.shape[0]

        for i in range(mp):
            nf_i, nt_i = self.ports[i]
            cf_i = indexer.to_compact(nf_i) if nf_i > 0 else -1
            ct_i = indexer.to_compact(nt_i) if nt_i > 0 else -1
            for j in range(mp):
                nf_j, nt_j = self.ports[j]
                cf_j = indexer.to_compact(nf_j) if nf_j > 0 else -1
                ct_j = indexer.to_compact(nt_j) if nt_j > 0 else -1
                g = G_tf[i, j]
                if cf_i >= 0 and cf_j >= 0:
                    stamper.add(cf_i, cf_j, g)
                if ct_i >= 0 and ct_j >= 0:
                    stamper.add(ct_i, ct_j, g)
                if cf_i >= 0 and ct_j >= 0:
                    stamper.add(cf_i, ct_j, -g)
                if ct_i >= 0 and cf_j >= 0:
                    stamper.add(ct_i, cf_j, -g)

    def stamp_rhs(self, rhs, indexer, t: float) -> None:
        _, I_hist = self.impl.get_norton_equivalent()

        for i, (nf_i, nt_i) in enumerate(self.ports):
            cf_i = indexer.to_compact(nf_i) if nf_i > 0 else -1
            ct_i = indexer.to_compact(nt_i) if nt_i > 0 else -1
            if cf_i >= 0:
                rhs[cf_i] -= I_hist[i]
            if ct_i >= 0:
                rhs[ct_i] += I_hist[i]

    # -- post-solve -----------------------------------------------------------

    def update_after_solve(self, V, indexer, t: float) -> None:
        mp = len(self.ports)
        self._V_ports = np.zeros(mp)
        for k, (nf, nt) in enumerate(self.ports):
            vf = V[indexer.to_compact(nf)] if nf > 0 else 0.0
            vt = V[indexer.to_compact(nt)] if nt > 0 else 0.0
            self._V_ports[k] = vf - vt

    def update_history(self, V, indexer, dt: float) -> None:
        # Use stored port voltages from update_after_solve
        self.impl.update_history(self._V_ports)

    def check_rebuild_required(self, V, indexer, t: float) -> bool:
        if not hasattr(self.impl, 'check_saturation'):
            return False
        # Re-derive port voltages from current V
        mp = len(self.ports)
        V_ports = np.zeros(mp)
        for k, (nf, nt) in enumerate(self.ports):
            vf = V[indexer.to_compact(nf)] if nf > 0 else 0.0
            vt = V[indexer.to_compact(nt)] if nt > 0 else 0.0
            V_ports[k] = vf - vt
        need_update, _ = self.impl.check_saturation(V_ports)
        return need_update

    def reset_state(self) -> None:
        self.impl.reset_state()
