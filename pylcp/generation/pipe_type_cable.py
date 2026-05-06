"""Pipe-type cable Z/Y generation.

Calls into :mod:`LCP.cable_model` for the physics (Ametani 1980).
"""

from __future__ import annotations

from typing import Any, Optional, Tuple

import numpy as np


def compute_pipe_type_cable_zy(
    freq: np.ndarray,
    geometry_config: Any,
    *,
    soil_config: Optional[Any] = None,
    verbose: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Compute pipe-type cable series impedance and shunt admittance matrices.

    Parameters
    ----------
    freq:
        Frequency vector in Hz (1-D, all > 0).
    geometry_config:
        A :class:`PipeTypeCableGeometry` instance from
        :mod:`LCP.cable_model`, or any object with compatible attributes.
    soil_config:
        Soil parameters.  When None, uses rho=100 Ω·m, εr=10, μr=1.

    Returns
    -------
    freq, Z_matrix (K, n, n), Y_matrix (K, n, n), metadata
    """
    from LCP.cable_model import (
        compute_pipe_type_cable_impedance as _compute_Z,
        compute_pipe_type_cable_potential as _compute_P,
    )

    cable = geometry_config

    rho = getattr(soil_config, "resistivity", 100.0) if soil_config else 100.0
    mu_r_soil = getattr(soil_config, "permeability", 1.0) if soil_config else 1.0
    eps_r = getattr(soil_config, "permittivity", 10.0) if soil_config else 10.0

    omega = 2.0 * np.pi * freq
    mu_0 = 4.0 * np.pi * 1e-7
    sigma = 1.0 / rho
    gamma_soil = np.sqrt(1j * omega * mu_0 * mu_r_soil * sigma - omega**2 * mu_0 * mu_r_soil * eps_r * 8.854e-12)

    Z_matrix = _compute_Z(freq, cable, gamma_soil)
    P_matrix = _compute_P(freq, cable)

    K = len(freq)
    n = Z_matrix.shape[1]
    Y_matrix = np.zeros((K, n, n), dtype=complex)
    for k in range(K):
        if np.linalg.cond(P_matrix[k]) < 1e12:
            Y_matrix[k] = 1j * omega[k] * np.linalg.inv(P_matrix[k])
        else:
            Y_matrix[k] = 1j * omega[k] * np.linalg.pinv(P_matrix[k])

    metadata = {
        "conductor_order": [
            "Core1", "Sheath1",
            "Core2", "Sheath2",
            "Core3", "Sheath3",
            "Pipe",
        ],
        "n_conductors": n,
        "P_condition_number": float(np.linalg.cond(P_matrix[0])),
    }

    return freq, Z_matrix, Y_matrix, metadata
