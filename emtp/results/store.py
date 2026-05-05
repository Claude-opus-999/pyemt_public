"""ResultStore — pre-allocated buffer management for simulation outputs.

Encapsulates the time array, node-voltage matrix, voltage-source current
buffers, and lightweight probe storage that the solver allocates and
populates during :meth:`EMTPSolver.run`.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np


class ResultStore:
    """Pre-allocated store for time-step simulation results.

    Parameters
    ----------
    n_nodes:
        Number of compact node-voltage entries (indexer.n).
    n_steps:
        Number of simulation time steps.
    record_node_voltage:
        Whether to allocate the node-voltage history matrix.
    vs_names:
        Voltage-source names (for current-history buffers).
    record_branch_history:
        Whether to allocate per-branch V/I buffers.
    branch_names:
        Branch names for per-branch history buffers.
    voltage_probe_names:
        Ordered list of voltage-probe names.
    branch_current_probe_names:
        Ordered list of branch-current-probe names.
    """

    def __init__(
        self,
        n_nodes: int,
        n_steps: int,
        *,
        record_node_voltage: bool = True,
        vs_names: Optional[List[str]] = None,
        record_branch_history: bool = False,
        branch_names: Optional[List[str]] = None,
        voltage_probe_names: Optional[List[str]] = None,
        branch_current_probe_names: Optional[List[str]] = None,
    ):
        self.n_steps = n_steps
        self.n_nodes = n_nodes
        self._steps_written = 0

        # -- time array -------------------------------------------------------
        self.time = np.zeros(n_steps, dtype=np.float64)

        # -- node voltage matrix ----------------------------------------------
        if record_node_voltage and n_nodes > 0:
            self.voltage: Optional[np.ndarray] = np.zeros(
                (n_nodes, n_steps), dtype=np.float64,
            )
        else:
            self.voltage = None

        # -- voltage-source current buffers -----------------------------------
        self.vs_current: Dict[str, np.ndarray] = {
            name: np.zeros(n_steps, dtype=np.float64)
            for name in (vs_names or [])
        }

        # -- branch history buffers -------------------------------------------
        if record_branch_history:
            names = branch_names or []
            self.branch_v: Dict[str, np.ndarray] = {
                name: np.zeros(n_steps, dtype=np.float64) for name in names
            }
            self.branch_i: Dict[str, np.ndarray] = {
                name: np.zeros(n_steps, dtype=np.float64) for name in names
            }
        else:
            self.branch_v = {}
            self.branch_i = {}

        # -- lightweight probes -----------------------------------------------
        n_vp = len(voltage_probe_names or [])
        n_cp = len(branch_current_probe_names or [])
        self.voltage_probe_data: Optional[np.ndarray] = (
            np.empty((n_steps, n_vp), dtype=np.float64) if n_vp else None
        )
        self.branch_current_probe_data: Optional[np.ndarray] = (
            np.empty((n_steps, n_cp), dtype=np.float64) if n_cp else None
        )
        self._voltage_probe_names = list(voltage_probe_names or [])
        self._branch_current_probe_names = list(branch_current_probe_names or [])

    # -- per-step recording ---------------------------------------------------

    def record_step(
        self,
        step_idx: int,
        t: float,
        V: np.ndarray,
        *,
        voltage_probe_values: Optional[List[float]] = None,
        branch_current_probe_values: Optional[List[float]] = None,
    ) -> None:
        """Record time, node voltages and optional probe values for one step."""
        self.time[step_idx] = t
        if self.voltage is not None:
            self.voltage[:, step_idx] = V

        if voltage_probe_values and self.voltage_probe_data is not None:
            for j, val in enumerate(voltage_probe_values):
                self.voltage_probe_data[step_idx, j] = val

        if branch_current_probe_values and self.branch_current_probe_data is not None:
            for j, val in enumerate(branch_current_probe_values):
                self.branch_current_probe_data[step_idx, j] = val

        self._steps_written = max(self._steps_written, step_idx + 1)

    def record_branch_history(
        self, step_idx: int, name: str, voltage: float, current: float,
    ) -> None:
        """Record one branch's V/I at *step_idx* (only when pre-allocated)."""
        if name in self.branch_v:
            self.branch_v[name][step_idx] = voltage
        if name in self.branch_i:
            self.branch_i[name][step_idx] = current

    def record_vs_current(self, step_idx: int, name: str, current: float) -> None:
        """Record a voltage-source current for *name* at *step_idx*."""
        buf = self.vs_current.get(name)
        if buf is not None and 0 <= step_idx < len(buf):
            buf[step_idx] = current

    # -- finalization ---------------------------------------------------------

    def finalize(self, indexer) -> None:
        """Trim to actual steps and build the external-id voltage-results dict.

        Must be called once after the main loop completes.
        """
        actual = self._steps_written
        self.time = self.time[:actual]

        if self.voltage is not None:
            self.voltage = self.voltage[:, :actual]

        for name in list(self.vs_current):
            self.vs_current[name] = self.vs_current[name][:actual]

        for name in list(self.branch_v):
            self.branch_v[name] = self.branch_v[name][:actual]
        for name in list(self.branch_i):
            self.branch_i[name] = self.branch_i[name][:actual]

        if self.voltage_probe_data is not None:
            self.voltage_probe_data = self.voltage_probe_data[:actual, :]
        if self.branch_current_probe_data is not None:
            self.branch_current_probe_data = self.branch_current_probe_data[:actual, :]

        # voltage_results dict keyed by external node id
        self.voltage_results: Dict[int, np.ndarray] = {}
        if self.voltage is not None:
            for c in range(self.n_nodes):
                ext_id = indexer.to_external(c)
                self.voltage_results[ext_id] = self.voltage[c, :]
