"""Multi-circuit armored cable Z/Y generation.

Calls into :mod:`LCP.cable_model` for the physics.
"""

from __future__ import annotations

from typing import Any, Optional, Tuple

import numpy as np


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
    from LCP.cable_model import (
        compute_multi_cable_impedance as _compute_Z,
        compute_armored_cable_admittance as _compute_Y,
    )

    cables = geometry_config if isinstance(geometry_config, list) else [geometry_config]

    rho = getattr(soil_config, "resistivity", 100.0) if soil_config else 100.0
    mu_r_soil = getattr(soil_config, "permeability", 1.0) if soil_config else 1.0
    eps_r = getattr(soil_config, "permittivity", 10.0) if soil_config else 10.0

    omega = 2.0 * np.pi * freq
    mu_0 = 4.0 * np.pi * 1e-7
    sigma = 1.0 / rho
    gamma_soil = np.sqrt(1j * omega * mu_0 * mu_r_soil * sigma - omega**2 * mu_0 * mu_r_soil * eps_r * 8.854e-12)

    Z_matrix = _compute_Z(freq, cables, gamma_soil)
    Y_matrix = _compute_Y(freq, cables[0] if len(cables) == 1 else cables)

    n_conductors = 3 + 3  # Core + Sheath + Armor per cable = 3 per circuit
    metadata = {
        "conductor_order": [
            f"{prefix}{i+1}"
            for i in range(len(cables))
            for prefix in [f"Core", f"Sheath", f"Armor"]
        ],
        "n_conductors": Z_matrix.shape[1],
        "n_circuits": len(cables),
    }

    return freq, Z_matrix, Y_matrix, metadata
