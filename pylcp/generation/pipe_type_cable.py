"""Pipe-type cable Z/Y generation.

Calls into :mod:`LCP.cable_model` for the physics (Ametani 1980).
"""

from __future__ import annotations

from typing import Any, Optional, Tuple

import numpy as np


def _safe_invert(M: np.ndarray, threshold: float = 1e12) -> np.ndarray:
    """Invert a matrix, falling back to pseudo-inverse when ill-conditioned.

    Uses try/except on LinAlgError instead of pre-computing the condition
    number, avoiding a redundant SVD in the common well-conditioned case.
    """
    try:
        return np.linalg.inv(M)
    except np.linalg.LinAlgError:
        if np.linalg.cond(M) < threshold:
            raise
        return np.linalg.pinv(M)


def _potential_to_admittance(
    freq: np.ndarray,
    P_matrix: np.ndarray,
    n_conductors: int,
) -> np.ndarray:
    """Convert potential coefficient matrix to shunt admittance.

    Handles both 2D P (n, n) — frequency-independent — and
    3D P (K, n, n) — frequency-dependent.
    """
    omega = 2.0 * np.pi * np.asarray(freq, dtype=float)
    K = len(omega)
    Y = np.zeros((K, n_conductors, n_conductors), dtype=complex)

    P = np.asarray(P_matrix, dtype=complex)

    if P.ndim == 2:
        if P.shape != (n_conductors, n_conductors):
            raise ValueError(
                f"P_matrix 2D shape mismatch: expected "
                f"{(n_conductors, n_conductors)}, got {P.shape}"
            )
        P_inv = _safe_invert(P)
        for k, w in enumerate(omega):
            Y[k] = 1j * w * P_inv
        return Y

    if P.ndim == 3:
        if P.shape != (K, n_conductors, n_conductors):
            raise ValueError(
                f"P_matrix 3D shape mismatch: expected "
                f"{(K, n_conductors, n_conductors)}, got {P.shape}"
            )
        for k, w in enumerate(omega):
            Y[k] = 1j * w * _safe_invert(P[k])
        return Y

    raise ValueError(
        f"Unexpected P_matrix ndim={P.ndim}, shape={P.shape}"
    )


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

    from ._soil import resolve_soil_params

    cable = geometry_config
    rho, mu_r_soil, eps_r, gamma_soil = resolve_soil_params(freq, soil_config)

    Z_matrix = _compute_Z(freq, cable, gamma_soil)
    P_matrix = _compute_P(freq, cable)
    n = Z_matrix.shape[1]

    Y_matrix = _potential_to_admittance(freq, P_matrix, n)

    metadata = {
        "conductor_order": [
            "Core1", "Sheath1",
            "Core2", "Sheath2",
            "Core3", "Sheath3",
            "Pipe",
        ],
        "n_conductors": n,
        "P_matrix_ndim": int(P_matrix.ndim),
        "P_matrix_shape": list(P_matrix.shape),
    }

    return freq, Z_matrix, Y_matrix, metadata
