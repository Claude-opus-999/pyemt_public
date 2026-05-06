"""EventRuntime — per-step physics orchestration.

PR6: Thin wrapper around solver._run_one_step().  The runtime owns
the single-step flow (switch events → solve → branch updates →
probes → history advance), delegating to the solver's existing
methods for each sub-step.

Later PRs will extract ResolveKernel and unify the three-phase
device interface (pre_step, post_solve_check, commit_step).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np


class EventRuntime:
    """Per-time-step event-driven simulation loop.

    Owns:
    - Pre-step switch/event detection
    - Core solve dispatch (linear / segmented / resolve)
    - Post-solve branch V/I update
    - Probe recording
    - History advance

    Parameters
    ----------
    solver: EMTPSolver
        The owning solver.  Will be replaced with explicit
        registry/kernel/rhs/probe dependencies in a later PR.
    """

    def __init__(self, solver):
        self._solver = solver

    # -----------------------------------------------------------------
    # Public API — called once per time step
    # -----------------------------------------------------------------

    def step(self, step_idx: int, n_steps: int, perf_counter) -> None:
        """Execute one full time step, from switch checks through
        history advance.  Identical semantics to solver._run_one_step().
        """
        self._solver._run_one_step(step_idx, n_steps, perf_counter)

    # -----------------------------------------------------------------
    # Sub-step accessors (delegated to solver internals for now)
    # -----------------------------------------------------------------

    def pre_step_switches(self) -> bool:
        """Check timed switch events; return True if topology changed."""
        s = self._solver
        return s._runtime.step_pre_solve(
            s.time, s._devices, set(s._lpm_elements),
        )

    def post_solve_update(self, V: "np.ndarray", step_idx: int, n_steps: int) -> None:
        """Update branch voltages and currents from solution."""
        s = self._solver
        s._runtime.step_post_solve_V_I(
            V, s._devices, s._indexer,
            step_idx, n_steps,
            bool(getattr(s, 'record_branch_history', False)),
            s._branch_v_bufs, s._branch_i_bufs,
        )

    def advance_history(self, step_idx: int) -> None:
        """Advance device, line, and transformer histories."""
        s = self._solver
        s._runtime.step_post_solve_advance(
            s.time, s.dt, step_idx,
            s._devices, set(s._lpm_elements),
            getattr(s, '_line_devices', {}),
            s.transformers,
        )
