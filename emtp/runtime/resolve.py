"""ResolveManager — unified nonlinear / LPM / UMEC re-solve loop.

Extracted from ``EMTPSolver._solve_step`` so the solver delegates the
re-solve orchestration to a standalone component.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict

import numpy as np

logger = logging.getLogger(__name__)


class ResolveManager:
    """Orchestrate the iterative re-solve loop for topology-changing events.

    After each linear (or segmented-nonlinear) solve the manager calls
    *check_fn*, which inspects LPM flashover, UMEC saturation, nonlinear
    segment changes, and :meth:`MultiPortDevice.check_rebuild_required`.
    If any trigger fires, the MNA matrix is marked dirty and the solve
    repeats, up to *max_iter* times.
    """

    def __init__(self, max_iter: int = 5):
        self.max_iter = max_iter

    def solve_with_resolve(
        self,
        solve_fn: Callable[[], np.ndarray],
        check_fn: Callable[[np.ndarray], bool],
        stats: Dict[str, Any],
        t: float,
        logger_: logging.Logger | None = None,
    ) -> np.ndarray:
        """Run *solve_fn*, check, and re-solve if needed.

        Parameters
        ----------
        solve_fn:
            Callable that builds MNA, solves, and returns V.
        check_fn:
            Callable that receives V and returns ``True`` when the
            circuit changed and a re-solve is required.
        stats:
            Mutable stats dict (keys ``segment_resolves``, ``max_seg_iter``).
        t:
            Current simulation time (for log messages only).
        logger_:
            Logger for non-convergence warnings (uses module logger by default).
        """
        _log = logger_ or logger

        for resolve_round in range(self.max_iter):
            V = solve_fn()

            if not check_fn(V):
                return V

            stats['segment_resolves'] = stats.get('segment_resolves', 0) + 1
            if resolve_round + 1 > stats.get('max_seg_iter', 0):
                stats['max_seg_iter'] = resolve_round + 1
        else:
            _log.warning(
                "nonlinear / saturation solver did not converge at t=%g "
                "after %d iterations", t, self.max_iter,
            )
        return V
