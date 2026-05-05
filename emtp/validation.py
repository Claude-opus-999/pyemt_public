"""Circuit validation helpers — topology checks, parameter checks, memory warnings.

These functions are used by EMTPSolver.validate_circuit().  They live in a
separate module so the validation logic can be tested and maintained
independently of the solver facade.
"""

from typing import Dict, List, Set

from emtp.types import (
    Branch,
    ElementType,
    ValidationIssue,
    ValidationReport,
)


def finalize_validation(
    issues: List[ValidationIssue], strict: bool,
) -> ValidationReport:
    """Build a ValidationReport and optionally raise on errors."""
    report = ValidationReport(issues=issues)
    if strict and report.has_errors:
        error_msgs = "\n".join(
            f"[{i.code}] {i.message}" for i in report.errors()
        )
        raise RuntimeError(
            f"Circuit validation failed with {len(report.errors())} "
            f"error(s):\n{error_msgs}"
        )
    return report


def check_floating_components(
    adjacency: Dict[int, set],
    grounded: set,
) -> List[List[int]]:
    """Find connected components that have no path to ground.

    Returns a list of floating node groups (each group is a sorted list).
    """
    visited: Set[int] = set()
    floating_components: List[List[int]] = []

    for start in sorted(adjacency):
        if start in visited:
            continue
        stack = [start]
        component: Set[int] = set()
        is_grounded = False
        while stack:
            node = stack.pop()
            if node in visited:
                continue
            visited.add(node)
            component.add(node)
            if node in grounded:
                is_grounded = True
            for nxt in adjacency.get(node, ()):
                if nxt not in visited:
                    stack.append(nxt)
        if not is_grounded:
            floating_components.append(sorted(component))

    return floating_components


def estimate_result_memory_bytes(
    n_steps: int,
    compact_n: int,
    n_voltage_probes: int,
    n_branch_current_probes: int,
    n_branches: int,
    n_voltage_sources: int,
    n_transmission_lines: int,
    n_line_phases: int,
    record_all_node_voltages: bool,
    record_line_history: bool,
    record_branch_history: bool,
    record_source_history: bool,
) -> int:
    """Estimate result buffer memory usage in bytes before running.

    Covers the main arrays allocated during ``run()``: time vector, node
    voltage history (if enabled), probes, line/branch history buffers,
    and source current history.  Assumes float64 storage (8 bytes).
    """
    F64 = 8
    total = n_steps * F64  # time_array

    if record_all_node_voltages:
        total += compact_n * n_steps * F64

    if n_voltage_probes:
        total += n_voltage_probes * n_steps * F64

    if n_branch_current_probes:
        total += n_branch_current_probes * n_steps * F64

    if record_line_history:
        total += 4 * n_transmission_lines * n_line_phases * n_steps * F64

    if record_branch_history:
        total += 2 * n_branches * n_steps * F64  # V, I

    if record_source_history:
        total += n_voltage_sources * n_steps * F64

    return total
