"""Result retrieval helpers — unit conversion, probe access, voltage/current queries.

These standalone functions operate on solver data structures so they can
be tested independently of the full solver loop.
"""

from typing import Optional, Dict

import numpy as np

from emtp.types import Branch, ElementType


def scale_probe_values(values: np.ndarray, unit: Optional[str]) -> np.ndarray:
    """Scale probe results according to unit."""
    if unit is None:
        return values.copy()
    scale = {
        "V": 1.0,
        "kV": 1e-3,
        "mV": 1e3,
        "A": 1.0,
        "kA": 1e-3,
        "mA": 1e3,
    }.get(unit)
    if scale is None:
        raise ValueError(f"Unsupported probe unit: {unit}")
    return values * scale


def scale_values(
    values: np.ndarray,
    unit: Optional[str],
    scale_map: Dict[str, float],
    quantity: str,
) -> np.ndarray:
    """Scale result arrays and reject unknown units explicitly."""
    if unit is None:
        return values.copy()
    if unit not in scale_map:
        supported = ", ".join(scale_map)
        raise ValueError(
            f"Unsupported {quantity} unit: {unit!r}. Supported: {supported}"
        )
    scale = scale_map[unit]
    return values * scale if scale != 1.0 else values.copy()


def node_voltage_from_solution(V: np.ndarray, node: int, to_compact) -> float:
    """Read node voltage from MNA solution vector.  node <= 0 treated as GND."""
    if node <= 0:
        return 0.0
    return float(V[to_compact(node)])


def branch_voltage_from_solution(V: np.ndarray, branch: Branch, to_compact) -> float:
    """Return branch voltage from node_from to node_to for the current solution."""
    vf = node_voltage_from_solution(V, branch.node_from, to_compact)
    vt = node_voltage_from_solution(V, branch.node_to, to_compact)
    return vf - vt


def branch_current_from_solution(V: np.ndarray, branch: Branch, to_compact) -> float:
    """Compute branch current directly from the current MNA solution.

    This is used by lightweight probes so they do not depend on Branch.current,
    which may intentionally be skipped for pure R/SW branches when full branch
    history recording is disabled.
    """
    vbr = branch_voltage_from_solution(V, branch, to_compact)
    et = branch.element_type

    if et == ElementType.RESISTOR:
        return float(vbr / branch.value)
    if et == ElementType.SWITCH:
        return float(branch.Geq * vbr)
    if et in (ElementType.INDUCTOR, ElementType.CAPACITOR):
        return float((branch.Geq + branch.Geq_damping) * vbr + branch.Ihist)
    if et == ElementType.SERIES_RL:
        return float(branch.Geq * vbr + branch.Ihist)
    if et == ElementType.NONLINEAR_RESISTOR:
        if branch.nonlinear_model is not None:
            return float(branch.nonlinear_model.get_current(vbr))
        return float(branch.Geq * vbr + branch.Ihist)

    return float(getattr(branch, "current", 0.0))
