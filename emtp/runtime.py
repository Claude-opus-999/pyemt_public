"""Per-step state management for devices, switches and reactive elements.

Encapsulates the operations that happen at each time step around the
core MNA-solve so the solver's main loop stays short and declarative.
"""

from typing import Any, Dict, List, Tuple

import numpy as np

from emtp.nodes import NodeIndexer
from emtp.types import Branch, ElementType
from emtp.devices.base import Device
from emtp.devices.switch import SwitchDevice


class DynamicDeviceRuntime:
    """Per-step state management for devices, switches and reactive elements."""

    def __init__(self, dt: float) -> None:
        self._dt = dt

    # -- pre-solve ----------------------------------------------------------

    def step_pre_solve(
        self, t: float,
        devices: List[Device],
        lpm_names: set,
    ) -> bool:
        """Apply timed switch events.  Return True if topology changed."""
        changed = False
        for dev in devices:
            if not isinstance(dev, SwitchDevice):
                continue
            if dev.name in lpm_names:
                continue
            if dev.update_timed_state(t):
                changed = True
        return changed

    # -- post-solve (branch V/I update, then history after probes) ---------

    def step_post_solve_V_I(
        self, V: np.ndarray,
        devices: List[Device],
        indexer: NodeIndexer,
        step_idx: int,
        n_steps: int,
        record_history: bool,
        branch_v_bufs: Dict[str, np.ndarray],
        branch_i_bufs: Dict[str, np.ndarray],
    ) -> None:
        """Update branch voltage / current from MNA solution.

        Must be called BEFORE probe recording so that probes see the
        correct per-step branch quantities, not the next-step Ihist.
        """
        use_buf = (
            record_history and step_idx >= 0 and step_idx < n_steps
            and bool(branch_v_bufs)
        )
        for dev in devices:
            et = dev._branch.element_type
            if not record_history and et in (ElementType.RESISTOR, ElementType.SWITCH):
                continue

            dev.update_branch_quantities(V, indexer)

            br = dev._branch
            if use_buf:
                branch_v_bufs[dev.name][step_idx] = br.voltage
                branch_i_bufs[dev.name][step_idx] = br.current
            elif record_history:
                br.voltage_history.append(br.voltage)
                br.current_history.append(br.current)

    def step_post_solve_history(
        self, devices: List[Device],
    ) -> None:
        """Advance reactive history sources (L, C, SRL) after probes."""
        for dev in devices:
            dev.update_history(self._dt)

    # -- post-solve resolve triggers (LPM + UMEC + nonlinear) --------------

    def post_solve_resolve_check(
        self, V: np.ndarray, t: float,
        lpm_elements: Dict[str, Any],
        lpm_node_map: Dict[str, Tuple[int, int]],
        transformers: Dict[str, Any],
        seg_node_map: Dict[str, Tuple[int, int]],
        seg_helper: Any,
        branches: Dict[str, Branch],
        indexer: NodeIndexer,
        mark_dirty_callback,
        stats: Dict[str, Any],
    ) -> bool:
        """Check LPM flashover, UMEC saturation and nonlinear segment changes.

        Returns True if a re-solve is needed (circuit topology changed).
        The caller is expected to re-assemble MNA and solve again.
        """
        any_change = False

        # --- LPM flashover check ---
        if lpm_elements:
            for name, lpm in lpm_elements.items():
                nf, nt = lpm_node_map[name]
                v_branch = 0.0
                if nf > 0:
                    v_branch += V[indexer.to_compact(nf)]
                if nt > 0:
                    v_branch -= V[indexer.to_compact(nt)]

                br = branches[name]
                current_A = br.Geq * v_branch
                state_changed = lpm.update(
                    voltage_V=v_branch,
                    dt=self._dt,
                    current_A=current_A,
                    time=t,
                )

                if state_changed:
                    any_change = True
                    br.is_closed = bool(lpm.is_flashed_over)
                    br.value = lpm.R_current
                    br.Geq = lpm.G_current
                    event = "LPM flashover" if lpm.is_flashed_over else "LPM extinction"
                    mark_dirty_callback(f"{event}: {name}")
                    if lpm.is_flashed_over:
                        stats['lpm_flashovers'] = stats.get('lpm_flashovers', 0) + 1
                    else:
                        stats['lpm_extinctions'] = stats.get('lpm_extinctions', 0) + 1

        # --- UMEC saturation check ---
        if transformers:
            for name, xfmr in transformers.items():
                if not hasattr(xfmr, 'check_saturation'):
                    continue
                port_nodes = xfmr.get_port_nodes()
                V_ports = np.zeros(xfmr.m)
                for k, (nf, nt) in enumerate(port_nodes):
                    v_f = V[indexer.to_compact(nf)] if nf > 0 else 0.0
                    v_t = V[indexer.to_compact(nt)] if nt > 0 else 0.0
                    V_ports[k] = v_f - v_t
                need_update, updates = xfmr.check_saturation(V_ports)
                if need_update:
                    any_change = True
                    stats['transformer_saturation_switches'] += len(updates)
                    mark_dirty_callback(f"UMEC saturation: {name}")

        # --- nonlinear segment check ---
        if seg_node_map:
            voltages = {}
            for name, (nf, nt) in seg_node_map.items():
                v_i = V[indexer.to_compact(nf)] if nf > 0 else 0.0
                v_j = V[indexer.to_compact(nt)] if nt > 0 else 0.0
                voltages[name] = v_i - v_j

            need_resolve, updates = seg_helper.check_all_segments(voltages)
            if need_resolve:
                any_change = True
                for seg_name, (g_new, i_new) in updates.items():
                    br = branches[seg_name]
                    br.Geq = g_new
                    br.Ihist = i_new
                    mark_dirty_callback(f"nonlinear segment: {seg_name}")
                    stats['segment_switches'] = stats.get('segment_switches', 0) + 1

        return any_change
