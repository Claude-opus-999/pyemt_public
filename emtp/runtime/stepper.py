"""TimeStepper — extracted main loop for EMTPSolver.

Moves the per-step iteration out of ``EMTPSolver.run()`` so the solver
delegates time-stepping to a standalone component.
"""

from __future__ import annotations

import time as _perf_time
from typing import Any


class TimeStepper:
    """Orchestrate the time-step loop.

    The stepper owns the loop structure and timing instrumentation;
    per-step physics is delegated to the solver via ``_run_one_step``.
    """

    def run(self, solver: Any, n_steps: int, timing: dict) -> None:
        """Execute the main time-stepping loop over *n_steps* steps.

        Parameters
        ----------
        solver:
            The EMTPSolver instance (duck-typed — must provide
            ``_run_one_step`` and timing-friendly attributes).
        n_steps:
            Number of time steps.
        timing:
            Mutable timing dict updated with per-phase wall-clock deltas.
        """
        _t = _perf_time.perf_counter

        for step_idx in range(n_steps):
            solver._run_one_step(step_idx, n_steps, _t)

        # Post-loop: export ULM batch state back to per-line models.
        ulm_batch = getattr(solver, '_ulm_batch', None)
        if ulm_batch is not None and hasattr(ulm_batch, 'export_model_state_to_lines'):
            ulm_batch.export_model_state_to_lines()
