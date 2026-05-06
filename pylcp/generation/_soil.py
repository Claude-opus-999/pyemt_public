"""Shared soil parameter resolution for cable/line Z/Y generation."""

from __future__ import annotations

from typing import Any, Optional, Tuple

import numpy as np

MU_0 = 4.0 * np.pi * 1e-7


def resolve_soil_params(
    freq: np.ndarray,
    soil_config: Optional[Any] = None,
) -> Tuple[float, float, float, np.ndarray]:
    """Extract soil parameters and compute propagation constant.

    Returns (rho, mu_r_soil, eps_r, gamma_soil).
    """
    rho = getattr(soil_config, "resistivity", 100.0) if soil_config else 100.0
    mu_r_soil = getattr(soil_config, "permeability", 1.0) if soil_config else 1.0
    eps_r = getattr(soil_config, "permittivity", 10.0) if soil_config else 10.0

    omega = 2.0 * np.pi * freq
    sigma = 1.0 / rho
    eps_0 = 8.854187817e-12
    gamma_soil = np.sqrt(
        1j * omega * MU_0 * mu_r_soil * sigma
        - omega ** 2 * MU_0 * mu_r_soil * eps_r * eps_0
    )
    return rho, mu_r_soil, eps_r, gamma_soil
