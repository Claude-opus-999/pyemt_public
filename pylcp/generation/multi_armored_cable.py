"""Multi-circuit armored cable Z/Y generation.

Calls into :mod:`LCP.cable_model` for the physics.
"""

from __future__ import annotations

from typing import Any, Optional, Tuple

import numpy as np

N_PER_CABLE = 3  # Core, Sheath, Armor


def _compute_multi_armored_admittance(
    freq: np.ndarray,
    cables: list,
    cable_model: Any,
) -> np.ndarray:
    """Assemble block-diagonal Y matrix from per-cable admittance matrices.

    Each cable contributes a (N_PER_CABLE, N_PER_CABLE) block on the diagonal.
    Off-diagonal blocks are zero (no cross-cable capacitive coupling).
    """
    K = len(freq)
    n_cables = len(cables)
    n_total = N_PER_CABLE * n_cables
    Y = np.zeros((K, n_total, n_total), dtype=complex)

    for i, cable in enumerate(cables):
        Yi = cable_model.compute_armored_cable_admittance(freq, cable)

        expected = (K, N_PER_CABLE, N_PER_CABLE)
        if Yi.shape != expected:
            raise ValueError(
                f"Unexpected admittance shape for cable {i}: "
                f"expected {expected}, got {Yi.shape}"
            )

        a = i * N_PER_CABLE
        b = a + N_PER_CABLE
        Y[:, a:b, a:b] = Yi

    return Y


def compute_multi_armored_cable_zy(
    freq: np.ndarray,
    geometry_config: Any,
    *,
    soil_config: Optional[Any] = None,
    verbose: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Compute multi-circuit armored cable series impedance and shunt admittance.

    Parameters
    ----------
    freq:
        Frequency vector in Hz (1-D, all > 0).
    geometry_config:
        A list of :class:`ArmoredCableGeometry` instances from
        :mod:`LCP.cable_model`, one per cable circuit.
    soil_config:
        Soil parameters.  When None, uses rho=100 Ω·m, εr=10, μr=1.

    Returns
    -------
    freq, Z_matrix (K, n, n), Y_matrix (K, n, n), metadata
    """
    from LCP import cable_model

    from ._soil import resolve_soil_params

    cables = geometry_config if isinstance(geometry_config, list) else [geometry_config]
    rho, mu_r_soil, eps_r, gamma_soil = resolve_soil_params(freq, soil_config)

    Z_matrix = cable_model.compute_multi_cable_impedance(freq, cables, gamma_soil)
    Y_matrix = _compute_multi_armored_admittance(freq, cables, cable_model)

    order = []
    for i in range(len(cables)):
        order.append(f"Core{i+1}")
        order.append(f"Sheath{i+1}")
        order.append(f"Armor{i+1}")

    metadata = {
        "line_type": "multi_armored_cable",
        "conductor_order": order,
        "n_conductors": Z_matrix.shape[1],
        "n_circuits": len(cables),
        "n_per_cable": N_PER_CABLE,
    }

    return freq, Z_matrix, Y_matrix, metadata
