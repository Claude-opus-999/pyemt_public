"""ResolveManager — unified nonlinear / LPM / UMEC re-solve loop.

Extracted from ``EMTPSolver._solve_step`` so the solver delegates the
re-solve orchestration to a standalone component.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ResolveEvent:
    """Structured description of a topology-changing event during a solve.

    Emitted by LPM flashover, MOA segment switch, UMEC saturation,
    or :meth:`MultiPortDevice.check_rebuild_required`.
    """

    source: str                    # "LPM", "MOA", "UMEC", "multiport"
    device_name: str               # e.g. "arrestor_1", "T1"
    reason: str                    # e.g. "flashover", "segment_switch"
    requires_matrix_rebuild: bool = True
    severity: str = "info"         # "info", "warning"

    def __str__(self) -> str:
        return (
            f"[{self.severity.upper()}] {self.source}:{self.device_name}"
            f" — {self.reason}"
        )


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
        self._last_events: List[ResolveEvent] = []

    # -- boolean check (legacy, kept for backward compat) --------------------

    def solve_with_resolve(
        self,
        solve_fn: Callable[[], np.ndarray],
        check_fn: Callable[[np.ndarray], bool],
        stats: Dict[str, Any],
        t: float,
        logger_: logging.Logger | None = None,
    ) -> np.ndarray:
        """Run *solve_fn*, check (bool), and re-solve if needed."""
        _log = logger_ or logger
        self._last_events.clear()

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

    # -- event-based check ----------------------------------------------------

    def solve_with_resolve_events(
        self,
        solve_fn: Callable[[], np.ndarray],
        event_check_fn: Callable[[np.ndarray], List[ResolveEvent]],
        stats: Dict[str, Any],
        t: float,
        logger_: logging.Logger | None = None,
    ) -> np.ndarray:
        """Run *solve_fn*, check for :class:`ResolveEvent`\\s, and re-solve.

        *event_check_fn* receives V and returns a list of
        :class:`ResolveEvent`.  The solver calls ``mark_topology_changed``
        for every event with ``requires_matrix_rebuild=True``.
        """
        _log = logger_ or logger
        self._last_events = []

        for resolve_round in range(self.max_iter):
            V = solve_fn()
            events = event_check_fn(V)
            self._last_events = events

            if not any(e.requires_matrix_rebuild for e in events):
                return V

            for e in events:
                if e.requires_matrix_rebuild:
                    _log.debug(
                        "Resolve round %d: %s (source=%s, device=%s)",
                        resolve_round + 1, e.reason, e.source, e.device_name,
                    )

            stats['segment_resolves'] = stats.get('segment_resolves', 0) + 1
            if resolve_round + 1 > stats.get('max_seg_iter', 0):
                stats['max_seg_iter'] = resolve_round + 1
        else:
            event_summary = "; ".join(
                f"{e.source}:{e.device_name}" for e in self._last_events
            )
            _log.warning(
                "nonlinear / saturation solver did not converge at t=%g "
                "after %d iterations (events: %s)",
                t, self.max_iter, event_summary or "none",
            )
        return V

    @property
    def last_events(self) -> List[ResolveEvent]:
        """Events from the most recent check (empty if converged)."""
        return list(self._last_events)
