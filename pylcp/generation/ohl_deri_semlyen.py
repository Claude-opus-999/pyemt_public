"""Overhead line Z/Y generation via Deri-Semlyen complex-depth method.

Calls into :mod:`LCP.ulm_atp_zy_deri_semlyen` for the physics.
"""

from __future__ import annotations

from typing import Any, Optional, Tuple

import numpy as np


def compute_ohl_zy(
    freq: np.ndarray,
    geometry_config: Any,
    *,
    soil_config: Optional[Any] = None,
    verbose: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Compute overhead-line series impedance and shunt admittance matrices.

    Parameters
    ----------
    freq:
        Frequency vector in Hz (1-D, all > 0).
    geometry_config:
        A :class:`MultiConductorLine` instance from
        :mod:`LCP.ulm_atp_zy_deri_semlyen`, or any object with compatible
        ``conductors`` and ``names`` attributes.
    soil_config:
        Soil parameters.  When None, uses rho=100 Ω·m, εr=10, μr=1.
    verbose:
        When True, print progress.

    Returns
    -------
    freq : ndarray
        Original frequency vector.
    Z_matrix : ndarray, shape (K, n, n)
        Series impedance matrix per frequency.
    Y_matrix : ndarray, shape (K, n, n)
        Shunt admittance matrix per frequency.
    metadata : dict
        Conductor names, counts, and ground-wire info.
    """
    from LCP.ulm_atp_zy_deri_semlyen import (
        compute_impedance_matrix,
        compute_admittance_matrix,
        get_constant_soil_params,
    )

    line = geometry_config

    rho = getattr(soil_config, "resistivity", 100.0) if soil_config else 100.0
    eps_r = getattr(soil_config, "permittivity", 10.0) if soil_config else 10.0

    soil = get_constant_soil_params(freq, rho, eps_r)

    Z_result = compute_impedance_matrix(freq, line, soil.p_complex, verbose=verbose)
    Z_matrix = Z_result.Z_matrix

    Y_result = compute_admittance_matrix(freq, line, verbose=verbose)
    Y_matrix = Y_result.Y_matrix

    metadata = {
        "conductor_names": getattr(line, "names", []),
        "is_ground_wire": getattr(line, "is_ground_wire", []),
        "n_original": Z_matrix.shape[1],
        "n_after_reduction": Z_matrix.shape[1],
        "ground_wires_eliminated": False,
    }

    return freq, Z_matrix, Y_matrix, metadata
