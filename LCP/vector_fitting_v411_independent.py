# vector_fitting_v411_independent.py module.

"""
*** ULM complete fitting (Morched/Gustavsen/Tartibi 1999, paper-consistent) ***

Full pipeline from frequency-domain line matrices (Z, Y) to a ULM realization
suitable for time-domain simulation. Fits tr(Yc) and the H propagation matrix
via Vector Fitting, extracts modal delays with the Gustavsen 2017 MPS angle
estimator, and assembles matrix residues through the global phase-domain
least-squares formulation of the paper eqs. (7)-(8).

* Key paper-consistency points:
    - Modal grouping threshold (paper eq. 9): Omega Delta_tau < 2 pi eps_deg/360
      => threshold = eps_deg / (360 f_max).
    - Group representative delay (paper section 4.1): tau_rep = min_m(tau_m),
      ensuring causality in time-domain simulation.
    - Global phase-domain LS (paper eqs. 7-8): all group poles appear in all
      H_ij elements, enabling the columnwise realization of section 5.2.

* Exposed API:
    - IterativePoleFindingConfig              : fitting configuration
    - ULMParameters, ULMFittingResult,
      HModeFitResult, HReconstructionMetrics  : result data classes
    - compute_ulm_parameters()                : stage 1 preprocessing
    - perform_ulm_fitting()                   : stage 2 fitting
    - ulm_complete_fitting()                  : top-level entry point
    - write_fitULM(), read_fitULM_header(),
      verify_fitULM_file(), export_ulm_to_fitulm() : fitULM file I/O

* References:
    [1] Morched, Gustavsen, Tartibi, "A universal model for accurate
        calculation of electromagnetic transients on overhead lines and
        underground cables," IEEE Trans. Power Delivery, 1999.
    [2] Zanon, Leal, De Conti, "Implementation of the universal line model
        in the alternative transients program," EPSR, 2021.
"""
__version__ = '4.11'

# Scientific computing modules:
import numpy as np
from scipy import linalg
from dataclasses import dataclass
from typing import Tuple, List, Optional, Dict, Any, Union
from datetime import datetime

# Vector Fitting engine:
from .vf_core import (
    VectorFitResult, vector_fitting, evaluate_rational_fit_s,
)

### -------------------------------------------------- Constants ----------------------------------------------------- ###

MU_0 = 4 * np.pi * 1e-7
EPSILON_0 = 8.854187817e-12
C_LIGHT = 299792458.0

### -------------------------------------------- JSON-safe type coercion -------------------------------------------- ###

def to_python_type(val):
    """Convert numpy scalars/arrays to native Python types for JSON."""
    if isinstance(val, (np.integer, np.int64, np.int32)):
        return int(val)
    if isinstance(val, (np.floating, np.float64, np.float32)):
        return float(val)
    if isinstance(val, np.ndarray):
        return val.tolist()
    if isinstance(val, (np.complexfloating, np.complex128, np.complex64)):
        return complex(val)
    return val

### ----------------------------------------------- Data structures ------------------------------------------------- ###

@dataclass
class HReconstructionMetrics:
    """H matrix reconstruction metrics (paper eqs. 12-13 verification).

    Two reconstruction paths are evaluated:
    - method1: similarity transform H = Ti diag(lambda) Ti^-1
    - method2: D-matrix expansion H = sum_j D_j lambda_j  (paper eq. 13)
    """
    method1_rmse: float
    method1_max_error: float
    method2_rmse: float
    method2_max_error: float
    method_difference: float
    inverse_off_diag_max: float
    inverse_off_diag_mean: float
    D_identity_error: float
    D_orthogonality_error: float
    D_idempotent_error: float
    rmse_per_frequency: Optional[np.ndarray] = None
    worst_frequency_idx: Optional[int] = None
    worst_frequency_Hz: Optional[float] = None
    element_rmse_matrix: Optional[np.ndarray] = None

    def to_dict(self):
        return {
            'method1_rmse': to_python_type(self.method1_rmse),
            'method1_max_error': to_python_type(self.method1_max_error),
            'method2_rmse': to_python_type(self.method2_rmse),
            'method2_max_error': to_python_type(self.method2_max_error),
            'method_difference': to_python_type(self.method_difference),
            'inverse_off_diag_max': to_python_type(self.inverse_off_diag_max),
            'inverse_off_diag_mean': to_python_type(self.inverse_off_diag_mean),
            'D_identity_error': to_python_type(self.D_identity_error),
            'D_orthogonality_error': to_python_type(self.D_orthogonality_error),
            'D_idempotent_error': to_python_type(self.D_idempotent_error),
            'worst_frequency_idx': to_python_type(self.worst_frequency_idx),
            'worst_frequency_Hz': to_python_type(self.worst_frequency_Hz),
        }


@dataclass
class IterativePoleFindingConfig:
    """Iterative pole-finding configuration."""
    Ymin: int = 10
    Ymax: int = 50
    epsY: float = 0.002
    Hmin: int = 20
    Hmax: int = 100
    epsH: float = 0.002
    pole_step: int = 2
    max_vf_iterations: int = 20
    eps_deg: float = 10.0

    nr_max_iter: int = 50
    nr_tol: float = 1e-12
    use_full_frequency_TI: bool = False

    delay_n_decades: float = 4.0
    delay_poly_order: int = 3
    use_delay_optimization: bool = True

    use_freq_dependent_D: str = 'auto'
    freq_dependent_threshold: float = 4.0

    compute_H_reconstruction_metrics: bool = True
    verbose_H_metrics: bool = True


@dataclass
class ULMParameters:
    """Pre-processed data prior to VF fitting."""
    freq: np.ndarray
    Yc_matrix: np.ndarray
    Yc_trace: np.ndarray
    H_matrix: np.ndarray
    H_modes: np.ndarray
    gamma_matrix: np.ndarray
    gamma_modes: np.ndarray
    tau: np.ndarray
    tau_mean: float
    T_ref: np.ndarray
    T_ref_inv: np.ndarray
    D_matrices: np.ndarray
    is_freq_dependent: bool
    TI_matrix: Optional[np.ndarray]
    lambda_matrix: Optional[np.ndarray]
    nf: int
    H_mode_vf_results: Optional[List] = None
    Yc_modal_diag: Optional[np.ndarray] = None
    source: str = 'nr'


@dataclass
class HModeFitResult:
    """Per-group H fitting result with independent matrix residues.

    - mode_index stores group[0] (identification only).
    - tau stores the group representative delay (paper section 4.1: min).
    - poles stores the full pole list including conjugate partners.
    - c_matrix_residues has shape (n_poles_full, nf, nf).
    """
    mode_index: int
    tau: float
    poles: np.ndarray
    residues: np.ndarray
    d: complex
    h: complex
    rmse: float
    D_matrix: np.ndarray
    c_matrix_residues: np.ndarray
    is_freq_dependent: bool = False


@dataclass
class ULMFittingResult:
    """Complete ULM fitting result (Yc + H + reconstruction diagnostics)."""
    nf: int
    n_active_modes: int
    active_modes: List[int]
    mode_groups: List[List[int]]
    poles_Yc: np.ndarray
    k_residues: np.ndarray
    k0: np.ndarray
    tau_all: np.ndarray
    D_matrices: np.ndarray
    H_modes_fits: List[HModeFitResult]
    Yc_trace_rmse: float
    H_modes_rmse: np.ndarray
    H_matrix_rmse: float
    is_passive: bool
    is_freq_dependent: bool = False
    Yc_matrix: Optional[np.ndarray] = None
    H_modes: Optional[np.ndarray] = None
    H_matrix: Optional[np.ndarray] = None
    freqs_Hz: Optional[np.ndarray] = None
    H_reconstruction_metrics: Optional[HReconstructionMetrics] = None
    H_reconstructed: Optional[np.ndarray] = None

### ---------------------------------------- D-matrix construction and checks -------------------------------------- ###

def compute_D_matrices_from_Ti(Ti):
    """D_j = T_I[:,j] outer T_I^-1[j,:] (paper eq. 13)."""
    n = Ti.shape[0]
    Ti_inv = np.linalg.inv(Ti)
    D_matrices = np.zeros((n, n, n), dtype=complex)
    for j in range(n):
        D_matrices[j] = np.outer(Ti[:, j], Ti_inv[j, :])
    return D_matrices


def compute_D_matrices_all_frequencies(TI_matrix):
    """Build D_j for every frequency sample."""
    K, n, _ = TI_matrix.shape
    D_matrices = np.zeros((K, n, n, n), dtype=complex)
    for k in range(K):
        D_matrices[k] = compute_D_matrices_from_Ti(TI_matrix[k])
    return D_matrices


def verify_D_matrix_properties(D_matrices):
    """Check D-matrix properties: sum_j D_j = I, D_i D_j = 0 (i!=j), D_j^2 = D_j."""
    n = D_matrices.shape[0]
    D_sum = np.sum(D_matrices, axis=0)
    identity_error = np.max(np.abs(D_sum - np.eye(n)))

    orthogonality_errors = []
    for i in range(n):
        for j in range(n):
            if i != j:
                orthogonality_errors.append(np.max(np.abs(D_matrices[i] @ D_matrices[j])))

    idempotent_errors = []
    for j in range(n):
        idempotent_errors.append(np.max(np.abs(D_matrices[j] @ D_matrices[j] - D_matrices[j])))

    return {
        'identity_error': float(identity_error),
        'orthogonality_max_error': float(np.max(orthogonality_errors)) if orthogonality_errors else 0.0,
        'orthogonality_mean_error': float(np.mean(orthogonality_errors)) if orthogonality_errors else 0.0,
        'idempotent_max_error': float(np.max(idempotent_errors)) if idempotent_errors else 0.0,
        'idempotent_mean_error': float(np.mean(idempotent_errors)) if idempotent_errors else 0.0,
    }

### ------------------------------------------ H matrix reconstruction ---------------------------------------------- ###

def reconstruct_H_method1(Ti, H_mode_diag):
    """Similarity transform H = Ti diag(lambda) Ti^-1."""
    return Ti @ np.diag(H_mode_diag) @ np.linalg.inv(Ti)


def reconstruct_H_method2(Ti, H_mode_diag):
    """D-matrix expansion H = sum_j D_j lambda_j (paper eq. 13)."""
    n = Ti.shape[0]
    D_matrices = compute_D_matrices_from_Ti(Ti)
    H_phase = np.zeros((n, n), dtype=complex)
    for j in range(n):
        H_phase += D_matrices[j] * H_mode_diag[j]
    return H_phase


def inverse_transform_to_modal(Ti, H_phase):
    """Modal domain H = Ti^-1 H_phase Ti (paper eq. 12)."""
    return np.linalg.inv(Ti) @ H_phase @ Ti


def extract_modal_propagation(Ti, H_phase):
    """Extract modal propagation scalars (diagonal of Ti^-1 H Ti)."""
    return np.diag(inverse_transform_to_modal(Ti, H_phase))

### --------------------------------------- Scalar rational evaluator ----------------------------------------------- ###

def evaluate_scalar_rational(s, poles, residues, d):
    """Evaluate a scalar rational with automatic conjugate-pair inclusion."""
    y = np.full(len(s), complex(d), dtype=complex)
    for p, r in zip(poles, residues):
        y += r / (s - p)
        if np.abs(np.imag(p)) > 1e-10:
            y += np.conj(r) / (s - np.conj(p))
    return y

### -------------------------------------------- H reconstruction RMSE ---------------------------------------------- ###

def compute_H_reconstruction_rmse(freq, H_phase, TI_matrix, verbose=False):
    """Verify H reconstruction through both paper formulas (eqs. 12, 13)
    and compute per-frequency and element-wise RMSE.
    """
    K = len(freq)
    n = H_phase.shape[1]

    method1_errors = np.zeros(K)
    method2_errors = np.zeros(K)
    method_diff = np.zeros(K)
    off_diag_max = np.zeros(K)
    off_diag_mean = np.zeros(K)

    identity_errors = []
    orthogonality_errors = []
    idempotent_errors = []
    element_errors = np.zeros((n, n))

    for k in range(K):
        Ti_k = TI_matrix[k]
        Ti_inv_k = np.linalg.inv(Ti_k)
        H_k = H_phase[k]

        H_modal_k = Ti_inv_k @ H_k @ Ti_k
        H_mode_diag_k = np.diag(H_modal_k)

        H_recon_1 = reconstruct_H_method1(Ti_k, H_mode_diag_k)
        H_recon_2 = reconstruct_H_method2(Ti_k, H_mode_diag_k)

        norm_H = np.linalg.norm(H_k)
        method1_errors[k] = np.linalg.norm(H_recon_1 - H_k) / (norm_H + 1e-15)
        method2_errors[k] = np.linalg.norm(H_recon_2 - H_k) / (norm_H + 1e-15)
        method_diff[k] = np.linalg.norm(H_recon_1 - H_recon_2) / (norm_H + 1e-15)

        mask = ~np.eye(n, dtype=bool)
        off_diag_max[k] = np.max(np.abs(H_modal_k[mask]))
        off_diag_mean[k] = np.mean(np.abs(H_modal_k[mask]))

        props = verify_D_matrix_properties(compute_D_matrices_from_Ti(Ti_k))
        identity_errors.append(props['identity_error'])
        orthogonality_errors.append(props['orthogonality_max_error'])
        idempotent_errors.append(props['idempotent_max_error'])

        element_errors += np.abs(H_recon_1 - H_k) ** 2

    element_errors = np.sqrt(element_errors / K)
    worst_idx = int(np.argmax(method1_errors))

    return HReconstructionMetrics(
        method1_rmse=float(np.mean(method1_errors)),
        method1_max_error=float(np.max(method1_errors)),
        method2_rmse=float(np.mean(method2_errors)),
        method2_max_error=float(np.max(method2_errors)),
        method_difference=float(np.mean(method_diff)),
        inverse_off_diag_max=float(np.max(off_diag_max)),
        inverse_off_diag_mean=float(np.mean(off_diag_mean)),
        D_identity_error=float(np.mean(identity_errors)),
        D_orthogonality_error=float(np.mean(orthogonality_errors)),
        D_idempotent_error=float(np.mean(idempotent_errors)),
        rmse_per_frequency=method1_errors,
        worst_frequency_idx=worst_idx,
        worst_frequency_Hz=float(freq[worst_idx]),
        element_rmse_matrix=element_errors,
    )


def compute_fitted_reconstruction_rmse(freq, H_phase_original, H_phase_fitted, verbose=False):
    """Compute RMSE between a fitted H matrix and the original samples.
    Returns the triple (total_rmse, rmse_per_freq, stats_dict).
    """
    K = len(freq)
    n = H_phase_original.shape[1]

    rmse_per_freq = np.zeros(K)
    max_error_per_freq = np.zeros(K)
    element_errors = np.zeros((n, n))

    for k in range(K):
        diff = H_phase_fitted[k] - H_phase_original[k]
        norm_orig = np.linalg.norm(H_phase_original[k])
        rmse_per_freq[k] = np.linalg.norm(diff) / (norm_orig + 1e-15)
        max_error_per_freq[k] = np.max(np.abs(diff)) / (np.max(np.abs(H_phase_original[k])) + 1e-15)
        element_errors += np.abs(diff) ** 2

    element_errors = np.sqrt(element_errors / K)
    total_diff = H_phase_fitted - H_phase_original
    total_rmse = float(np.linalg.norm(total_diff) / (np.linalg.norm(H_phase_original) + 1e-15))
    worst_idx = int(np.argmax(rmse_per_freq))

    stats = {
        'total_rmse': total_rmse,
        'mean_rmse': float(np.mean(rmse_per_freq)),
        'max_rmse': float(np.max(rmse_per_freq)),
        'min_rmse': float(np.min(rmse_per_freq)),
        'std_rmse': float(np.std(rmse_per_freq)),
        'mean_max_error': float(np.mean(max_error_per_freq)),
        'worst_freq_idx': worst_idx,
        'worst_freq_Hz': float(freq[worst_idx]),
        'element_rmse_matrix': element_errors,
    }
    return total_rmse, rmse_per_freq, stats


def verify_H_reconstruction_complete(ulm_params, H_reconstructed=None, verbose=False):
    """Full H reconstruction verification bundle.

    Combines paper-formula verification (requires TI_matrix) and fitted-
    reconstruction RMSE (requires H_reconstructed).
    """
    results = {}
    freq = ulm_params.freq
    H_phase = ulm_params.H_matrix
    TI_matrix = ulm_params.TI_matrix

    if TI_matrix is not None:
        results['paper_formula_metrics'] = compute_H_reconstruction_rmse(freq, H_phase, TI_matrix)
    else:
        results['paper_formula_metrics'] = None

    if H_reconstructed is not None:
        total_rmse, rmse_per_freq, stats = compute_fitted_reconstruction_rmse(
            freq, H_phase, H_reconstructed)
        results['fitted_rmse'] = total_rmse
        results['fitted_rmse_per_freq'] = rmse_per_freq
        results['fitted_stats'] = stats
    else:
        results['fitted_rmse'] = None
        results['fitted_rmse_per_freq'] = None
        results['fitted_stats'] = None

    return results


def verify_H_reconstruction_standalone(freq, H_phase, Ti_matrix, verbose=False):
    """Standalone wrapper around compute_H_reconstruction_rmse()."""
    return compute_H_reconstruction_rmse(freq, H_phase, Ti_matrix, verbose)

### ---------------------------------------- Verification report generator ------------------------------------------ ###

def generate_verification_report(ulm_params, result, output_path=None):
    """Emit a human-readable verification summary. Writes to output_path
    if provided and returns the report string in all cases.
    """
    lines = [
        "=" * 80,
        "  ULM Fitting Verification Report",
        f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"  Version: {__version__}",
        "=" * 80,
        "",
        "1. Basic Information",
        "-" * 40,
        f"   Frequency range: {ulm_params.freq[0]:.2f} Hz ~ {ulm_params.freq[-1]/1e6:.2f} MHz",
        f"   Number of frequencies: {len(ulm_params.freq)}",
        f"   Number of conductors: {ulm_params.nf}",
        f"   Ti source: {ulm_params.source}",
        f"   D-matrix mode: {'Frequency-dependent' if ulm_params.is_freq_dependent else 'Single-frequency'}",
        "",
        "2. Fitting Results",
        "-" * 40,
        f"   tr(Yc) RMSE: {result.Yc_trace_rmse*100:.4f}%",
        f"   H matrix RMSE: {result.H_matrix_rmse*100:.4f}%",
        f"   Active mode groups: {result.n_active_modes}/{result.nf}",
        f"   Passive: {result.is_passive}",
        "",
        "3. H Mode Fitting Details",
        "-" * 40,
    ]
    for i, fit in enumerate(result.H_modes_fits):
        lines.append(f"   Mode {i}: tau={fit.tau*1e6:.4f} us, "
                     f"RMSE={fit.rmse*100:.4f}%, Poles={len(fit.poles)}")
    lines.append("")

    if result.H_reconstruction_metrics is not None:
        m = result.H_reconstruction_metrics
        lines.extend([
            "4. Paper Formula Verification (Eq.12, 13)",
            "-" * 40,
            f"   Method 1 (Similarity Transform):",
            f"      Mean RMSE: {m.method1_rmse*100:.6f}%",
            f"      Max Error: {m.method1_max_error*100:.6f}%",
            f"   Method 2 (D-matrix Expansion):",
            f"      Mean RMSE: {m.method2_rmse*100:.6f}%",
            f"      Max Error: {m.method2_max_error*100:.6f}%",
            f"   Method Difference: {m.method_difference*100:.2e}%",
            "",
            "5. Inverse Transform Verification (Eq.12)",
            "-" * 40,
            f"   Off-diagonal max: {m.inverse_off_diag_max:.2e}",
            f"   Off-diagonal mean: {m.inverse_off_diag_mean:.2e}",
            "",
            "6. D-matrix Properties",
            "-" * 40,
            f"   Sum D_j = I error: {m.D_identity_error:.2e}",
            f"   Orthogonality error: {m.D_orthogonality_error:.2e}",
            f"   Idempotent error: {m.D_idempotent_error:.2e}",
            "",
            f"   Worst frequency: {m.worst_frequency_Hz:.2f} Hz",
            "",
        ])

    lines.extend(["=" * 80, "  Summary", "=" * 80])

    if result.H_reconstruction_metrics is not None:
        m = result.H_reconstruction_metrics
        if m.method1_rmse < 1e-10:
            lines.append("   Paper formulas verified with machine precision")
        elif m.method1_rmse < 1e-6:
            lines.append("   Paper formulas verified with high precision")
        else:
            lines.append("   Paper formulas have noticeable error")
        if m.inverse_off_diag_max < 1e-10:
            lines.append("   Inverse transform perfectly diagonalizes H")
        elif m.inverse_off_diag_max < 1e-6:
            lines.append("   Inverse transform diagonalizes H with high precision")
        else:
            lines.append("   Inverse transform has off-diagonal residuals")
        if m.D_identity_error < 1e-10:
            lines.append("   D-matrices satisfy all projection properties")
        else:
            lines.append("   D-matrices have property verification errors")

    lines.extend(["", "=" * 80])
    report = "\n".join(lines)

    if output_path is not None:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(report)
    return report

### ------------------------------------- Newton-Raphson eigensolver (T_I tracking) --------------------------------- ###

def normalize_to_zero_phase(T, ref_idx=None):
    """Phase-align a vector T so that the reference element is real-positive,
    then rescale so that max(|T|) = 1. Returns (T_normalized, ref_idx).
    """
    if ref_idx is None:
        ref_idx = np.argmax(np.abs(T))
    ref_val = T[ref_idx]
    if np.abs(ref_val) > 1e-15:
        phase_factor = ref_val / np.abs(ref_val)
        T_phase_aligned = T / phase_factor
    else:
        T_phase_aligned = T.copy()
    max_mag = np.max(np.abs(T_phase_aligned))
    if max_mag > 1e-15:
        T_normalized = T_phase_aligned / max_mag
    else:
        T_normalized = T_phase_aligned
    return T_normalized, int(ref_idx)


def solve_eigenproblem_standard(S):
    """Standard eigen-decomposition with magnitude-descending sort and
    zero-phase normalization of each eigenvector.
    """
    eigenvalues, eigenvectors = np.linalg.eig(S)
    sort_idx = np.argsort(-np.abs(eigenvalues))
    eigenvalues = eigenvalues[sort_idx]
    eigenvectors = eigenvectors[:, sort_idx]
    for j in range(eigenvectors.shape[1]):
        eigenvectors[:, j], _ = normalize_to_zero_phase(eigenvectors[:, j])
    return eigenvalues, eigenvectors


def build_nr_residual(S, v, lam):
    """Augmented residual F = [ (S - lam*I) v ; v^T v - 1 ]."""
    n = len(v)
    F = np.zeros(n + 1, dtype=complex)
    F[:n] = (S - lam * np.eye(n)) @ v
    F[n] = np.dot(v, v) - 1.0
    return F


def build_nr_jacobian(S, v, lam):
    """Jacobian associated with build_nr_residual()."""
    n = len(v)
    J = np.zeros((n + 1, n + 1), dtype=complex)
    J[:n, :n] = S - lam * np.eye(n)
    J[:n, n] = -v
    J[n, :n] = 2 * v
    J[n, n] = 0
    return J


def newton_raphson_eigensolver_single_mode(S, lambda_init, T_col_init, max_iter=50, tol=1e-8):
    """Single-mode NR eigen-solver with unit-norm constraint.
    Returns the tuple (lambda, eigenvector, converged, n_iter).
    """
    lam = complex(lambda_init)
    T = T_col_init.copy().astype(complex)
    T, _ = normalize_to_zero_phase(T)
    for iteration in range(max_iter):
        F = build_nr_residual(S, T, lam)
        if np.linalg.norm(F) < tol:
            return lam, T, True, iteration + 1
        J = build_nr_jacobian(S, T, lam)
        try:
            dX = np.linalg.solve(J, -F)
        except np.linalg.LinAlgError:
            dX = np.linalg.lstsq(J, -F, rcond=None)[0]
        T = T + dX[:-1]
        lam = lam + dX[-1]
    return lam, T, False, max_iter


def newton_raphson_eigensolver_all_modes(S, lambda_init, T_init, max_iter=50, tol=1e-8, verbose=False):
    """Apply newton_raphson_eigensolver_single_mode() to every mode."""
    nf = S.shape[0]
    eigenvalues = np.zeros(nf, dtype=complex)
    eigenvectors = np.zeros((nf, nf), dtype=complex)
    all_converged = True
    total_iters = 0
    for j in range(nf):
        lam, T_col, converged, n_iter = newton_raphson_eigensolver_single_mode(
            S, lambda_init[j], T_init[:, j], max_iter, tol)
        eigenvalues[j] = lam
        eigenvectors[:, j] = T_col
        total_iters += n_iter
        if not converged:
            all_converged = False
    return eigenvalues, eigenvectors, all_converged, total_iters


def compute_TI_over_frequency_newton_raphson(freq, Z_matrix, Y_matrix,
                                              max_iter=50, tol=1e-12, verbose=False):
    """Continuously track T_I across frequency by warm-starting NR from the
    previous sample. Returns (TI_matrix, TI_inv_matrix, lambda_matrix).
    """
    K = len(freq)
    nf = Z_matrix.shape[1]
    TI_matrix = np.zeros((K, nf, nf), dtype=complex)
    TI_inv_matrix = np.zeros((K, nf, nf), dtype=complex)
    lambda_matrix = np.zeros((K, nf), dtype=complex)

    S0 = Y_matrix[0] @ Z_matrix[0]
    eigenvalues, eigenvectors = solve_eigenproblem_standard(S0)
    TI_matrix[0] = eigenvectors
    TI_inv_matrix[0] = np.linalg.inv(eigenvectors)
    lambda_matrix[0] = eigenvalues

    for k in range(1, K):
        S_k = Y_matrix[k] @ Z_matrix[k]
        eigenvalues, eigenvectors, _, _ = newton_raphson_eigensolver_all_modes(
            S_k, lambda_matrix[k - 1], TI_matrix[k - 1], max_iter, tol)
        for j in range(nf):
            eigenvectors[:, j], _ = normalize_to_zero_phase(eigenvectors[:, j])
        TI_matrix[k] = eigenvectors
        TI_inv_matrix[k] = np.linalg.inv(eigenvectors)
        lambda_matrix[k] = eigenvalues

    return TI_matrix, TI_inv_matrix, lambda_matrix

### ---------------------------------------------- Yc and H computation --------------------------------------------- ###

def compute_sqrt_matrix(A):
    """Principal matrix square root via eigen-decomposition, with the
    convention Re(sqrt_eig) >= 0.
    """
    eigenvalues, eigenvectors = np.linalg.eig(A)
    sqrt_eigenvalues = np.sqrt(eigenvalues)
    sqrt_eigenvalues = np.where(np.real(sqrt_eigenvalues) < 0, -sqrt_eigenvalues, sqrt_eigenvalues)
    return eigenvectors @ np.diag(sqrt_eigenvalues) @ np.linalg.inv(eigenvectors)


def compute_Yc_and_H(freq, Z_matrix, Y_matrix, length, T_ref, T_ref_inv):
    """Reference-basis Yc/H/modal decomposition.

    Yc = Z^-1 sqrt(ZY). H = exp(-sqrt(YZ) length). Modal quantities are
    obtained by projecting onto the constant basis (T_ref, T_ref_inv).
    """
    K = len(freq)
    nf = Z_matrix.shape[1]
    Yc_matrix = np.zeros((K, nf, nf), dtype=complex)
    H_matrix = np.zeros((K, nf, nf), dtype=complex)
    H_modes = np.zeros((K, nf), dtype=complex)
    gamma_modes = np.zeros((K, nf), dtype=complex)

    for k in range(K):
        sqrt_ZY = compute_sqrt_matrix(Z_matrix[k] @ Y_matrix[k])
        try:
            Z_inv = np.linalg.inv(Z_matrix[k])
        except np.linalg.LinAlgError:
            Z_inv = np.linalg.pinv(Z_matrix[k])
        Yc_matrix[k] = Z_inv @ sqrt_ZY

        sqrt_YZ = compute_sqrt_matrix(Y_matrix[k] @ Z_matrix[k])
        evals, evecs = np.linalg.eig(sqrt_YZ)
        H_matrix[k] = evecs @ np.diag(np.exp(-evals * length)) @ np.linalg.inv(evecs)

        H_modal = T_ref_inv @ H_matrix[k] @ T_ref
        H_modes[k] = np.diag(H_modal)
        gamma_modal = T_ref_inv @ sqrt_YZ @ T_ref
        gamma_modes[k] = np.diag(gamma_modal)

    return Yc_matrix, H_matrix, H_modes, gamma_modes


def compute_Yc_and_H_from_NR(freq, Z_matrix, Y_matrix, length,
                              TI_matrix, lambda_matrix, verbose=False):
    """NR-consistent Yc/H using the frequency-dependent T_I and lambda
    produced by compute_TI_over_frequency_newton_raphson().

    Guarantees that T_I^-1 H T_I is diagonal to machine precision.
    """
    K = len(freq)
    nf = Z_matrix.shape[1]
    Yc_matrix = np.zeros((K, nf, nf), dtype=complex)
    H_matrix = np.zeros((K, nf, nf), dtype=complex)
    H_modes = np.zeros((K, nf), dtype=complex)
    gamma_modes = np.zeros((K, nf), dtype=complex)

    for k in range(K):
        sqrt_ZY = compute_sqrt_matrix(Z_matrix[k] @ Y_matrix[k])
        try:
            Z_inv = np.linalg.inv(Z_matrix[k])
        except np.linalg.LinAlgError:
            Z_inv = np.linalg.pinv(Z_matrix[k])
        Yc_matrix[k] = Z_inv @ sqrt_ZY

        gamma_j = np.sqrt(lambda_matrix[k])
        gamma_j = np.where(np.real(gamma_j) < 0, -gamma_j, gamma_j)
        gamma_modes[k] = gamma_j

        H_modes[k] = np.exp(-gamma_j * length)

        TI_k = TI_matrix[k]
        TI_k_inv = np.linalg.inv(TI_k)
        H_matrix[k] = TI_k @ np.diag(H_modes[k]) @ TI_k_inv

    return Yc_matrix, H_matrix, H_modes, gamma_modes

### ------------------------------------------- Modal grouping (paper eq. 9) ---------------------------------------- ###

def merge_modes_by_tau(tau, fmax, eps_deg=10.0):
    """Group modes by delay similarity using paper eq. (9):
    Omega Delta_tau < 2 pi eps_deg/360 with Omega = 2 pi fmax, hence
    threshold = eps_deg / (360 fmax).
    Returns (active_modes, groups).
    """
    nf = len(tau)
    threshold = eps_deg / (360.0 * fmax)
    idx_sorted = np.argsort(tau)
    tau_sorted = tau[idx_sorted]
    groups = [[idx_sorted[0]]]
    for i in range(1, nf):
        if np.abs(tau_sorted[i] - tau_sorted[i - 1]) < threshold:
            groups[-1].append(idx_sorted[i])
        else:
            groups.append([idx_sorted[i]])
    active_modes = [g[0] for g in groups]
    return active_modes, groups

### ------------------------------------- Frequency-dependent D-matrix assembly ------------------------------------- ###

def compute_D_matrices_frequency_dependent(TI_matrix, TI_inv_matrix, verbose=False):
    """Build D_j for every frequency sample (outer products from T_I rows/cols)."""
    K, nf, _ = TI_matrix.shape
    D_matrices = np.zeros((K, nf, nf, nf), dtype=complex)
    for k in range(K):
        for j in range(nf):
            D_matrices[k, j] = np.outer(TI_matrix[k, :, j], TI_inv_matrix[k, j, :])
    return D_matrices


def compute_D_matrices_single_frequency(T_ref, T_ref_inv):
    """Build D_j from a single reference (T_ref, T_ref_inv) pair."""
    nf = T_ref.shape[1]
    D_matrices = np.zeros((nf, nf, nf), dtype=complex)
    for j in range(nf):
        D_matrices[j] = np.outer(T_ref[:, j], T_ref_inv[j, :])
    return D_matrices


def diagnose_TI_frequency_variation(TI_matrix, freq, idx_ref):
    """Measure the principal-angle dispersion of T_I subspaces across
    frequency. Returns a dict with max/mean angle and a recommendation:
    'freq_dependent_essential' (>15 deg), 'freq_dependent_recommended' (>5 deg),
    or 'single_freq_acceptable'.
    """
    K, nf = TI_matrix.shape[0], TI_matrix.shape[1]
    T_ref = TI_matrix[idx_ref]
    U_ref, _, _ = np.linalg.svd(T_ref)
    max_angles = []
    for k in range(K):
        U_k, _, _ = np.linalg.svd(TI_matrix[k])
        overlap = U_ref.conj().T @ U_k
        singular_vals = np.linalg.svd(overlap, compute_uv=False)
        angles_deg = np.arccos(np.clip(singular_vals, 0, 1)) * 180 / np.pi
        max_angles.append(np.max(angles_deg))
    max_angles = np.array(max_angles)
    max_angle = float(np.max(max_angles))
    mean_angle = float(np.mean(max_angles))
    k_worst = int(np.argmax(max_angles))
    if max_angle > 15:
        recommendation = 'freq_dependent_essential'
    elif max_angle > 5:
        recommendation = 'freq_dependent_recommended'
    else:
        recommendation = 'single_freq_acceptable'
    return {
        'max_angle': max_angle,
        'mean_angle': mean_angle,
        'worst_freq': float(freq[k_worst]),
        'recommendation': recommendation,
    }

### ------------------------------------------------- Delay extraction ---------------------------------------------- ###

def compute_magnitude_derivatives(omega, mag):
    """Log-log derivative d log|H| / d log omega (piecewise slopes)."""
    K = len(omega)
    A = np.zeros(K - 1)
    for j in range(K - 1):
        if mag[j] > 1e-300 and mag[j + 1] > 1e-300:
            log_mag_ratio = np.log(mag[j + 1] / mag[j])
            log_omega_ratio = np.log(omega[j + 1] / omega[j])
            if np.abs(log_omega_ratio) > 1e-15:
                A[j] = log_mag_ratio / log_omega_ratio
    return A


def mps_angle_singularity_removed(omega, mag, k_eval, n_decades=4.0,
                                  use_prediction=True, poly_order=3,
                                  samples_per_decade=20):
    """Minimum-phase-shift angle estimate with the singularity removed
    (Gustavsen 2017), integrated over a windowed log-frequency band.
    """
    omega_k = omega[k_eval]
    omega_lo = max(omega_k / (10 ** n_decades), omega[0])
    omega_hi = omega_k * (10 ** n_decades)
    A = compute_magnitude_derivatives(omega, mag)
    A_k = A[k_eval] if k_eval < len(A) else (A[-1] if len(A) > 0 else 0.0)

    sum_term = 0.0
    for j in range(len(A)):
        omega_j = omega[j]
        omega_j1 = omega[j + 1]
        if omega_j < omega_lo or omega_j1 > omega_hi * 1.01:
            continue
        A_j = A[j]
        omega_mid = (omega_j + omega_j1) / 2
        u = np.log(omega_mid / omega_k)
        if np.abs(u) < 1e-10:
            continue
        abs_u_half = np.abs(u) / 2
        if abs_u_half < 1e-10:
            B_j = -np.log(abs_u_half)
        elif abs_u_half > 20:
            B_j = 0.0
        else:
            coth_val = 1.0 / np.tanh(abs_u_half)
            B_j = np.log(coth_val) if coth_val > 1e-15 else 0.0
        log_omega_ratio = np.log(omega_j1 / omega_j)
        sum_term += (A_j - A_k) * B_j * log_omega_ratio

    return (np.pi / 2) * A_k + (1.0 / np.pi) * sum_term


def compute_modal_velocity(gamma, omega):
    """Phase velocity v = omega / Im(gamma) with a speed-of-light fallback."""
    beta = np.imag(gamma)
    return omega / beta if np.abs(beta) > 1e-15 else C_LIGHT


def golden_section_search(objective_func, a, b, tol=1e-6, max_iter=50):
    """Golden-section minimisation of a unimodal scalar objective.
    Returns (x_min, f_min).
    """
    phi = (1 + np.sqrt(5)) / 2
    resphi = 2 - phi
    x1 = a + resphi * (b - a)
    x2 = b - resphi * (b - a)
    f1 = objective_func(x1)
    f2 = objective_func(x2)
    for _ in range(max_iter):
        if (b - a) < tol:
            break
        if f1 < f2:
            b = x2
            x2 = x1
            f2 = f1
            x1 = a + resphi * (b - a)
            f1 = objective_func(x1)
        else:
            a = x1
            x1 = x2
            f1 = f2
            x2 = b - resphi * (b - a)
            f2 = objective_func(x2)
    return (x1, f1) if f1 < f2 else (x2, f2)


def extract_optimal_delay_single_mode(omega, s, freq, H_mode, gamma_mode, line_length,
                                      eps_target, n_poles_min, n_poles_max,
                                      n_decades=4.0, use_delay_optimization=True, verbose=False):
    """Extract the optimal modal delay through Gustavsen's two-step method:
    infinite-frequency estimate tau_infty = L/v_1 + phi_MPS/omega_1, then
    golden-section refinement on VF RMSE within [tau_infty, L/v_1].
    Returns the tuple (tau, vf_result).
    """
    K = len(omega)
    mag = np.abs(H_mode)
    v_mode = np.array([compute_modal_velocity(gamma_mode[k], omega[k]) for k in range(K)])
    diff_from_target = np.abs(mag - eps_target)
    k_eval = int(np.argmin(diff_from_target))
    if np.min(mag) > eps_target:
        k_eval = K - 1

    omega_1 = omega[k_eval]
    v_1 = v_mode[k_eval]
    phi_mps = mps_angle_singularity_removed(omega, mag, k_eval, n_decades)
    tau_infty = (line_length / v_1) + (phi_mps / omega_1)
    tau_theoretical = line_length / C_LIGHT
    if tau_infty < tau_theoretical * 0.5:
        tau_infty = tau_theoretical

    if not use_delay_optimization:
        H_compensated = H_mode * np.exp(s * tau_infty)
        try:
            vf_result = vector_fitting(freq, H_compensated, n_poles=n_poles_min,
                                       max_iterations=20, verbose=False, asymp=1)
        except Exception:
            vf_result = None
        return tau_infty, vf_result

    tau_a = max(tau_infty, tau_theoretical * 0.9)
    tau_b = line_length / v_1
    if tau_b <= tau_a:
        tau_b = tau_a * 1.1

    def objective(tau):
        H_comp = H_mode * np.exp(s * tau)
        try:
            vf = vector_fitting(freq, H_comp, n_poles=n_poles_min,
                                max_iterations=15, verbose=False, asymp=1)
            return vf.rmse
        except Exception:
            return float('inf')

    tau_star, _ = golden_section_search(objective, tau_a, tau_b)
    H_compensated = H_mode * np.exp(s * tau_star)
    try:
        best_result = vector_fitting(freq, H_compensated, n_poles=n_poles_min,
                                     max_iterations=20, verbose=False, asymp=1)
    except Exception:
        best_result = None
    return tau_star, best_result


def extract_optimal_delay_all_modes(freq, H_modes, gamma_modes, line_length, config, verbose=False):
    """Apply extract_optimal_delay_single_mode() to every modal channel."""
    K, nf = H_modes.shape
    omega = 2 * np.pi * freq
    s = 1j * omega
    tau_opt = np.zeros(nf)
    vf_results = []
    for m in range(nf):
        tau_m, vf_m = extract_optimal_delay_single_mode(
            omega, s, freq, H_modes[:, m], gamma_modes[:, m],
            line_length, config.epsH, config.Hmin, config.Hmax,
            config.delay_n_decades, config.use_delay_optimization)
        tau_opt[m] = tau_m
        vf_results.append(vf_m)
    return tau_opt, vf_results

### ------------------------------------------- ULM parameter preprocessing ----------------------------------------- ###

# compute_ulm_parameters() subroutine.
def compute_ulm_parameters(freq, Z_matrix, Y_matrix, length,
                           velocity_freq=1e6, config=None,
                           use_freq_dependent=None,
                           pscad_TI_matrix=None, verbose=False):
    """Pre-process (Z, Y) into a ULMParameters bundle suitable for VF fitting.

    - Chooses between single-frequency and frequency-dependent D_j based on
      T_I dispersion (diagnose_TI_frequency_variation()) when 'auto'.
    - T_I either comes from PSCAD (pscad_TI_matrix) or is computed via NR
      tracking starting from a standard eigen decomposition at freq[0].
    - Modal delays are extracted with the Gustavsen 2017 estimator.
    """
    if config is None:
        config = IterativePoleFindingConfig()
    if use_freq_dependent is None:
        use_freq_dependent = config.use_freq_dependent_D

    if isinstance(use_freq_dependent, str):
        _ufd = use_freq_dependent.strip().lower()
        if _ufd in ('auto', 'always', 'never'):
            use_freq_dependent = _ufd
        elif _ufd in ('true', 't', 'yes', 'y', '1'):
            use_freq_dependent = True
        elif _ufd in ('false', 'f', 'no', 'n', '0'):
            use_freq_dependent = False
        else:
            raise ValueError(
                "vector_fitting::ERROR::use_freq_dependent must be bool or "
                "one of 'auto'/'always'/'never' (also accepts 'true'/'false').")

    nf = Z_matrix.shape[1]
    K = len(freq)
    idx_ref = int(np.argmin(np.abs(freq - velocity_freq)))
    freq_span = np.log10(freq[-1] / freq[0])

    TI_matrix = None
    lambda_matrix = None
    is_freq_dep = False
    ti_source = 'nr'
    use_pscad_ti = (pscad_TI_matrix is not None)

    if use_freq_dependent == 'auto':
        if freq_span > config.freq_dependent_threshold:
            use_freq_dependent = True
        else:
            if use_pscad_ti:
                TI_matrix = pscad_TI_matrix
                ti_source = 'pscad'
                if TI_matrix.shape != (K, nf, nf):
                    raise ValueError("vector_fitting::ERROR::PSCAD Ti shape mismatch")
                lambda_matrix = np.zeros((K, nf), dtype=complex)
                for k in range(K):
                    gamma_k = np.sqrt(Z_matrix[k] @ Y_matrix[k])
                    Ti_k_inv = np.linalg.inv(TI_matrix[k])
                    gamma_modal = Ti_k_inv @ gamma_k @ TI_matrix[k]
                    lambda_matrix[k] = np.diag(gamma_modal)
            else:
                TI_matrix, _, lambda_matrix = compute_TI_over_frequency_newton_raphson(
                    freq, Z_matrix, Y_matrix, config.nr_max_iter, config.nr_tol)
            diagnosis = diagnose_TI_frequency_variation(TI_matrix, freq, idx_ref)
            use_freq_dependent = diagnosis['recommendation'] == 'freq_dependent_essential'

    if use_freq_dependent == 'always':
        use_freq_dependent = True
    elif use_freq_dependent == 'never':
        use_freq_dependent = False

    if use_freq_dependent:
        if use_pscad_ti and TI_matrix is None:
            TI_matrix = pscad_TI_matrix
            ti_source = 'pscad'
            if TI_matrix.shape != (K, nf, nf):
                raise ValueError("vector_fitting::ERROR::PSCAD Ti shape mismatch")
            TI_inv_matrix = np.zeros((K, nf, nf), dtype=complex)
            lambda_matrix = np.zeros((K, nf), dtype=complex)
            for k in range(K):
                TI_inv_matrix[k] = np.linalg.inv(TI_matrix[k])
                S_modal = TI_inv_matrix[k] @ (Y_matrix[k] @ Z_matrix[k]) @ TI_matrix[k]
                lambda_matrix[k] = np.diag(S_modal)
        elif TI_matrix is None:
            TI_matrix, TI_inv_matrix, lambda_matrix = compute_TI_over_frequency_newton_raphson(
                freq, Z_matrix, Y_matrix, config.nr_max_iter, config.nr_tol)
        else:
            TI_inv_matrix = np.array([np.linalg.inv(TI_matrix[k]) for k in range(K)])

        T_ref = TI_matrix[idx_ref]
        T_ref_inv = np.linalg.inv(T_ref)
        D_matrices = compute_D_matrices_frequency_dependent(TI_matrix, TI_inv_matrix)
        is_freq_dep = True
    else:
        if config.use_full_frequency_TI and TI_matrix is None:
            TI_matrix, TI_inv_matrix, lambda_matrix = compute_TI_over_frequency_newton_raphson(
                freq, Z_matrix, Y_matrix, config.nr_max_iter, config.nr_tol)
            T_ref = TI_matrix[idx_ref]
            T_ref_inv = TI_inv_matrix[idx_ref]
        else:
            S_ref = Y_matrix[idx_ref] @ Z_matrix[idx_ref]
            eigenvalues, T_ref = solve_eigenproblem_standard(S_ref)
            omega_ref = 2 * np.pi * freq[idx_ref]
            gamma_modes_ref = np.sqrt(eigenvalues)
            gamma_modes_ref = np.where(np.real(gamma_modes_ref) < 0,
                                       -gamma_modes_ref, gamma_modes_ref)
            beta_modes = np.imag(gamma_modes_ref)
            tau_sort = beta_modes * length / omega_ref if omega_ref > 0 else np.abs(eigenvalues)
            sort_idx = np.argsort(tau_sort)
            T_ref = T_ref[:, sort_idx]
            for j in range(nf):
                col = T_ref[:, j]
                max_idx = int(np.argmax(np.abs(col)))
                if np.abs(col[max_idx]) > 1e-15:
                    T_ref[:, j] = col / col[max_idx]
            T_ref_inv = np.linalg.inv(T_ref)

        D_matrices = compute_D_matrices_single_frequency(T_ref, T_ref_inv)
        is_freq_dep = False

    if TI_matrix is not None and lambda_matrix is not None:
        Yc_matrix, H_matrix, H_modes, gamma_modes = compute_Yc_and_H_from_NR(
            freq, Z_matrix, Y_matrix, length, TI_matrix, lambda_matrix)
    else:
        Yc_matrix, H_matrix, H_modes, gamma_modes = compute_Yc_and_H(
            freq, Z_matrix, Y_matrix, length, T_ref, T_ref_inv)

    Yc_trace = np.array([np.trace(Yc_matrix[k]) for k in range(K)])

    tau, H_mode_vf_results = extract_optimal_delay_all_modes(
        freq, H_modes, gamma_modes, length, config)

    gamma_matrix = np.zeros((K, nf, nf), dtype=complex)
    Yc_modal_diag = np.zeros((K, nf), dtype=complex)
    if TI_matrix is not None:
        for k in range(K):
            TI_k_inv = np.linalg.inv(TI_matrix[k])
            gamma_matrix[k] = TI_matrix[k] @ np.diag(gamma_modes[k]) @ TI_k_inv
            Yc_modal = TI_k_inv @ Yc_matrix[k] @ TI_matrix[k]
            Yc_modal_diag[k] = np.diag(Yc_modal)
    else:
        for k in range(K):
            gamma_matrix[k] = T_ref @ np.diag(gamma_modes[k]) @ T_ref_inv
            Yc_modal = T_ref_inv @ Yc_matrix[k] @ T_ref
            Yc_modal_diag[k] = np.diag(Yc_modal)

    return ULMParameters(
        freq=freq, Yc_matrix=Yc_matrix, Yc_trace=Yc_trace,
        H_matrix=H_matrix, H_modes=H_modes,
        gamma_matrix=gamma_matrix, gamma_modes=gamma_modes,
        tau=tau, tau_mean=float(np.mean(tau)),
        T_ref=T_ref, T_ref_inv=T_ref_inv,
        D_matrices=D_matrices, is_freq_dependent=is_freq_dep,
        TI_matrix=TI_matrix, lambda_matrix=lambda_matrix,
        nf=nf, H_mode_vf_results=H_mode_vf_results,
        Yc_modal_diag=Yc_modal_diag, source=ti_source,
    )

### ---------------------------------- Matrix residues (legacy compatibility) --------------------------------------- ###

def compute_matrix_residues(poles, residues, D_matrices, is_freq_dependent):
    """Scalar residues x D_j product, kept for legacy storage/compatibility."""
    n_poles = len(poles)
    if is_freq_dependent:
        K, nf, _ = D_matrices.shape
        c_matrix_residues = np.zeros((n_poles, K, nf, nf), dtype=complex)
        for i in range(n_poles):
            for k in range(K):
                c_matrix_residues[i, k] = D_matrices[k] * residues[i]
    else:
        nf, _ = D_matrices.shape
        c_matrix_residues = np.zeros((n_poles, nf, nf), dtype=complex)
        for i in range(n_poles):
            c_matrix_residues[i] = D_matrices * residues[i]
    return c_matrix_residues

### ------------------------------------- vectfit3-style residue solver --------------------------------------------- ###

def _identify_pole_types(poles, tol=1e-12):
    """Classify poles as real (0), first of conjugate pair (1) or second (2)
    in lockstep with vectfit3.identifyPoles.
    """
    n = len(poles)
    cindex = np.zeros(n, dtype=np.int32)
    for m in range(n):
        if abs(np.imag(poles[m])) > tol:
            if m == 0:
                cindex[m] = 1
            else:
                if cindex[m - 1] == 0 or cindex[m - 1] == 2:
                    cindex[m] = 1
                    if m + 1 < n:
                        cindex[m + 1] = 2
                else:
                    cindex[m] = 2
    return cindex


def _build_vectfit_basis(s, poles, cindex):
    """Real-valued conjugate-paired basis matrix D_k(s) used by vectfit3."""
    N = len(s)
    n = len(poles)
    Dk = np.zeros((N, n), dtype=np.complex128)
    for m in range(n):
        if cindex[m] == 0:
            Dk[:, m] = 1.0 / (s - poles[m])
        elif cindex[m] == 1:
            p = poles[m]
            pc = np.conj(p)
            Dk[:, m]     = 1.0 / (s - p) + 1.0 / (s - pc)
            Dk[:, m + 1] = 1j  / (s - p) - 1j  / (s - pc)
    return Dk


def _solve_residues_vectfit_style(s, F, poles, asymp=1, weights=None):
    """Multi-RHS residue least squares in the vectfit3 conjugate-paired form.
    Returns (C, D_const, E_lin) with C of shape (Nc, n_poles).
    """
    F = np.atleast_2d(F)
    Nc, N = F.shape
    n = len(poles)

    if weights is None:
        w_common = np.ones(N, dtype=np.float64)
        common_weighting = True
    else:
        w_arr = np.asarray(weights)
        if w_arr.ndim == 1:
            w_common = w_arr.astype(np.float64)
            common_weighting = True
        else:
            common_weighting = False

    cindex = _identify_pole_types(poles)
    Dk = _build_vectfit_basis(s, poles, cindex)

    C = np.zeros((Nc, n), dtype=np.complex128)
    D_const = np.zeros(Nc, dtype=np.float64)
    E_lin = np.zeros(Nc, dtype=np.float64)

    def _make_system(w):
        Dk_w = Dk * w[:, None]
        if asymp == 1:
            A = np.zeros((2 * N, n), dtype=np.float64)
        elif asymp == 2:
            A = np.zeros((2 * N, n + 1), dtype=np.float64)
            A[0:N, n] = w
        else:
            A = np.zeros((2 * N, n + 2), dtype=np.float64)
            A[0:N, n] = w
            A[N:2 * N, n + 1] = np.imag(w * s)
        A[0:N, 0:n] = Dk_w.real
        A[N:2 * N, 0:n] = Dk_w.imag
        return A

    if common_weighting:
        A = _make_system(w_common)
        BB = np.zeros((2 * N, Nc), dtype=np.float64)
        BBc = w_common[:, None] * F.T
        BB[0:N, :] = BBc.real
        BB[N:2 * N, :] = BBc.imag

        Escale = np.linalg.norm(A, axis=0)
        Escale[Escale == 0.0] = 1.0
        A = A / Escale
        x, *_ = linalg.lstsq(A, BB, check_finite=False, lapack_driver="gelsy")
        x = (x / Escale[:, None]).T

        C[:, 0:n] = x[:, 0:n]
        if asymp >= 2:
            D_const[:] = np.real(x[:, n])
        if asymp == 3:
            E_lin[:] = np.real(x[:, n + 1])
    else:
        for k in range(Nc):
            wk = np.asarray(weights[k, :], dtype=np.float64)
            A = _make_system(wk)
            BB = np.zeros(2 * N, dtype=np.float64)
            BBc = wk * F[k, :]
            BB[0:N] = BBc.real
            BB[N:2 * N] = BBc.imag

            Escale = np.linalg.norm(A, axis=0)
            Escale[Escale == 0.0] = 1.0
            A = A / Escale
            x, *_ = linalg.lstsq(A, BB, check_finite=False, lapack_driver="gelsy")
            x = x / Escale

            C[k, 0:n] = x[0:n]
            if asymp >= 2:
                D_const[k] = float(np.real(x[n]))
            if asymp == 3:
                E_lin[k] = float(np.real(x[n + 1]))

    # Restore complex residues for conjugate pairs
    for m in range(n):
        if cindex[m] == 1:
            for kk in range(Nc):
                r1 = float(np.real(C[kk, m]))
                r2 = float(np.real(C[kk, m + 1]))
                C[kk, m]     = r1 + 1j * r2
                C[kk, m + 1] = r1 - 1j * r2

    return C, D_const, E_lin


def fit_phase_domain_residues(s, D_matrix, H_mode_delayfree, poles, nf,
                              is_freq_dependent=False, verbose=False):
    """Per-group phase-domain refit of D_merged * H_mode_delayfree (paper
    eq. 15) against matrix residues C[n, i, j]. Legacy fallback for
    fit_H_modes_pscad_style() when no global H matrix is provided.
    """
    K = len(s)
    n_poles = len(poles)
    if n_poles == 0:
        return np.zeros((0, nf, nf), dtype=complex), float('inf')

    target = np.zeros((K, nf, nf), dtype=complex)
    if is_freq_dependent:
        for k in range(K):
            target[k] = D_matrix[k] * H_mode_delayfree[k]
    else:
        for k in range(K):
            target[k] = D_matrix * H_mode_delayfree[k]

    F_multi = target.transpose(1, 2, 0).reshape(nf * nf, K)
    C, _, _ = _solve_residues_vectfit_style(s, F_multi, poles, asymp=1, weights=None)
    c_matrix_residues = C.T.reshape(n_poles, nf, nf)

    inv_sp = 1.0 / (s[:, None] - poles[None, :])
    y_fit_all = np.einsum('kn,nij->kij', inv_sp, c_matrix_residues)
    diff = y_fit_all - target
    rmse = float(np.sqrt(np.sum(np.abs(diff) ** 2) /
                         (np.sum(np.abs(target) ** 2) + 1e-30)))
    return c_matrix_residues, rmse


def merge_D_matrices_for_group(group, D_matrices_all, is_freq_dependent):
    """Sum D_j over all modal indices belonging to a group."""
    if is_freq_dependent:
        K, nf = D_matrices_all.shape[0], D_matrices_all.shape[1]
        D_merged = np.zeros((K, nf, nf), dtype=complex)
        for k in range(K):
            for j in group:
                D_merged[k] += D_matrices_all[k, j]
        return D_merged
    return np.sum(D_matrices_all[group], axis=0)

### ---------------------------------------- Group average target construction -------------------------------------- ###

def compute_group_average_delayfree(group, H_modes, tau, s):
    """Average delay-compensated mode signals inside a group, and pick the
    group representative delay tau_rep = min(tau_m) (paper section 4.1).
    Returns the tuple (H_avg_delayfree, tau_rep, m_ref).
    """
    if len(group) == 1:
        m = group[0]
        return H_modes[:, m] * np.exp(s * tau[m]), float(tau[m]), int(m)

    H_delayfree_stack = np.stack(
        [H_modes[:, m] * np.exp(s * tau[m]) for m in group], axis=0)
    H_avg_delayfree = np.mean(H_delayfree_stack, axis=0)
    tau_rep = float(np.min([tau[m] for m in group]))
    return H_avg_delayfree, tau_rep, int(group[0])


def unique_conjugate_poles(poles):
    """Drop conjugate partners, keeping only upper-half-plane poles and real poles."""
    unique = []
    tol = 1e-9
    for p in poles:
        if np.abs(np.imag(p)) < tol:
            unique.append(np.real(p))
        elif np.imag(p) > 0:
            unique.append(p)
    return np.array(unique, dtype=complex)


def iterative_pole_fitting(freq, data, n_poles_min, n_poles_max, target_error,
                           pole_step=2, max_vf_iterations=20, verbose=False,
                           label="", asymp=None):
    """Sweep pole count from n_poles_min to n_poles_max, returning the first
    fit with rmse < target_error, or the best fit found otherwise.
    """
    n_poles = n_poles_min
    best_result = None
    best_error = float('inf')
    vf_kwargs = {'max_iterations': max_vf_iterations, 'verbose': False}
    if asymp is not None:
        vf_kwargs['asymp'] = asymp
    while n_poles <= n_poles_max:
        try:
            result = vector_fitting(freq, data, n_poles=n_poles, **vf_kwargs)
            current_error = result.rmse
        except Exception:
            n_poles += pole_step
            continue
        if current_error < best_error:
            best_result = result
            best_error = current_error
        if current_error < target_error:
            return result
        n_poles += pole_step
    return best_result

### -------------------------------------- Global phase-domain LS (paper eq. 7-8) ----------------------------------- ###

# fit_H_global_phase_domain() subroutine.
def fit_H_global_phase_domain(s, H_matrix, poles_per_group, tau_groups, nf, verbose=False):
    """Global least-squares fit of H_ij(jw) = sum_k [sum_m c_mk^ij/(jw-p_mk)]
    exp(-jw tau_k) over all groups simultaneously (paper eqs. 7-8).

    All group poles appear in all H elements, enabling columnwise realization
    as described in paper section 5.2. Real conjugate-paired basis follows
    the vectfit3 convention.

    Returns (c_per_group, rmse, H_fit).
    """
    K = len(s)
    n_groups = len(poles_per_group)

    group_cindex = []
    total_cols = 0
    for poles_k in poles_per_group:
        ci = _identify_pole_types(poles_k) if len(poles_k) > 0 else np.array([], dtype=np.int32)
        group_cindex.append(ci)
        total_cols += len(poles_k)

    if total_cols == 0:
        return ([np.zeros((0, nf, nf), dtype=complex) for _ in range(n_groups)],
                float('inf'), np.zeros_like(H_matrix))

    A = np.zeros((2 * K, total_cols), dtype=np.float64)
    col_offset = 0
    for k, (poles_k, ci_k) in enumerate(zip(poles_per_group, group_cindex)):
        n_k = len(poles_k)
        if n_k == 0:
            continue
        delay_k = np.exp(-s * tau_groups[k])
        Dk = _build_vectfit_basis(s, poles_k, ci_k)
        Dk_delayed = Dk * delay_k[:, None]
        A[0:K, col_offset:col_offset + n_k] = Dk_delayed.real
        A[K:2 * K, col_offset:col_offset + n_k] = Dk_delayed.imag
        col_offset += n_k

    B = np.zeros((2 * K, nf * nf), dtype=np.float64)
    for i in range(nf):
        for j in range(nf):
            idx = i * nf + j
            B[0:K, idx] = H_matrix[:, i, j].real
            B[K:2 * K, idx] = H_matrix[:, i, j].imag

    Escale = np.linalg.norm(A, axis=0)
    Escale[Escale == 0.0] = 1.0
    A_scaled = A / Escale
    x, *_ = linalg.lstsq(A_scaled, B, check_finite=False, lapack_driver="gelsy")
    x = x / Escale[:, None]

    c_per_group = []
    col_offset = 0
    for k, (poles_k, ci_k) in enumerate(zip(poles_per_group, group_cindex)):
        n_k = len(poles_k)
        if n_k == 0:
            c_per_group.append(np.zeros((0, nf, nf), dtype=complex))
            continue
        x_k = x[col_offset:col_offset + n_k, :]
        C_k = x_k.astype(np.complex128).copy()
        for m in range(n_k):
            if ci_k[m] == 1:
                for elem in range(nf * nf):
                    r1 = float(np.real(C_k[m, elem]))
                    r2 = float(np.real(C_k[m + 1, elem]))
                    C_k[m, elem] = r1 + 1j * r2
                    C_k[m + 1, elem] = r1 - 1j * r2
        c_per_group.append(C_k.reshape(n_k, nf, nf))
        col_offset += n_k

    H_fit = np.zeros((K, nf, nf), dtype=complex)
    for k, poles_k in enumerate(poles_per_group):
        if len(poles_k) == 0:
            continue
        delay_k = np.exp(-s * tau_groups[k])
        inv_sp = 1.0 / (s[:, None] - poles_k[None, :])
        rational_term = np.einsum('kn,nij->kij', inv_sp, c_per_group[k])
        H_fit += rational_term * delay_k[:, None, None]

    diff = H_fit - H_matrix
    rmse = float(np.linalg.norm(diff) / (np.linalg.norm(H_matrix) + 1e-15))
    return c_per_group, rmse, H_fit

### ------------------------------------------- H modes fitting (two-step) ------------------------------------------ ###

# fit_H_modes_pscad_style() subroutine.
def fit_H_modes_pscad_style(freq, H_modes, tau, mode_groups, D_matrices, config,
                            is_freq_dependent, pre_fit_results=None, verbose=False,
                            H_matrix=None):
    """Two-step paper-consistent H fitting.

    Step 1 (modal domain): per-group scalar VF on the group-average delay-free
    signal to obtain poles p_mk and representative delays tau_k.

    Step 2 (phase domain, paper eqs. 7-8): global LS fit of the complete H
    matrix using all group poles simultaneously. Falls back to per-group fit
    of D_merged * H_avg_delayfree when H_matrix is None.
    """
    s = 1j * 2 * np.pi * freq
    _, nf = H_modes.shape

    poles_per_group = []
    tau_groups = []
    scalar_rmse_list = []
    m_ref_list = []

    for group_idx, group in enumerate(mode_groups):
        H_avg_delayfree, tau_rep, m_ref = compute_group_average_delayfree(
            group, H_modes, tau, s)
        m_ref_list.append(m_ref)
        tau_groups.append(tau_rep)

        best_fit = iterative_pole_fitting(
            freq, H_avg_delayfree, config.Hmin, config.Hmax,
            config.epsH, config.pole_step, config.max_vf_iterations,
            label=f"H[{group_idx}]", asymp=1)

        if best_fit:
            poles_per_group.append(np.asarray(best_fit.poles, dtype=complex))
            scalar_rmse_list.append(float(best_fit.rmse))
        else:
            poles_per_group.append(np.array([], dtype=complex))
            scalar_rmse_list.append(float('inf'))

    if H_matrix is not None:
        c_per_group, _, _ = fit_H_global_phase_domain(
            s, H_matrix, poles_per_group, tau_groups, nf)
    else:
        c_per_group = []
        for group_idx, group in enumerate(mode_groups):
            if len(poles_per_group[group_idx]) > 0:
                H_avg_delayfree, _, _ = compute_group_average_delayfree(
                    group, H_modes, tau, s)
                D_merged = merge_D_matrices_for_group(group, D_matrices, is_freq_dependent)
                c_res, _ = fit_phase_domain_residues(
                    s, D_merged, H_avg_delayfree, poles_per_group[group_idx], nf,
                    is_freq_dependent=is_freq_dependent)
                c_per_group.append(c_res)
            else:
                c_per_group.append(np.zeros((0, nf, nf), dtype=complex))

    H_fits = []
    for group_idx, group in enumerate(mode_groups):
        D_merged = merge_D_matrices_for_group(group, D_matrices, is_freq_dependent)
        H_fits.append(HModeFitResult(
            mode_index=m_ref_list[group_idx],
            tau=tau_groups[group_idx],
            poles=poles_per_group[group_idx],
            residues=np.array([], dtype=complex),
            d=0.0, h=0.0,
            rmse=scalar_rmse_list[group_idx],
            D_matrix=D_merged,
            c_matrix_residues=c_per_group[group_idx],
            is_freq_dependent=is_freq_dependent,
        ))
    return H_fits

### ------------------------------------------- H matrix reconstruction --------------------------------------------- ###

def evaluate_H_matrix_from_fit(s, H_fits, mode_groups, is_freq_dependent=None):
    """Reconstruct the H matrix from independent matrix residues:
        H_tilde(s) = sum_j [sum_n c_n^j / (s - p_n^j)] exp(-s tau_j)
    """
    K = len(s)
    nf = H_fits[0].D_matrix.shape[-1]
    H = np.zeros((K, nf, nf), dtype=complex)

    for fit in H_fits:
        n_poles = len(fit.poles)
        if n_poles == 0:
            continue
        delay_factor = np.exp(-s * fit.tau)
        for k in range(K):
            rational_term = np.zeros((nf, nf), dtype=complex)
            for n in range(n_poles):
                rational_term += fit.c_matrix_residues[n] / (s[k] - fit.poles[n])
            H[k] += rational_term * delay_factor[k]
    return H

### -------------------------------------------------- Yc fitting --------------------------------------------------- ###

# fit_Yc_matrix_fixed_poles() subroutine.
def fit_Yc_matrix_fixed_poles(s, Yc_matrix, poles, verbose=False):
    """Fit Yc_ij(s) = sum_n k_n^ij / (s - p_n) + k0^ij at fixed poles,
    with symmetry k^ij = k^ji enforced upfront.
    Returns (k_residues, k0, avg_error).
    """
    K, nf, _ = Yc_matrix.shape
    n_poles = len(poles)

    if n_poles == 0:
        k0 = np.zeros((nf, nf), dtype=np.float64)
        for i in range(nf):
            for j in range(i, nf):
                k0[i, j] = float(np.real(np.mean(Yc_matrix[:, i, j])))
                if i != j:
                    k0[j, i] = k0[i, j]
        return np.zeros((0, nf, nf), dtype=complex), k0, 0.0

    pairs = [(i, j) for i in range(nf) for j in range(i, nf)]
    Nc = len(pairs)
    F_multi = np.zeros((Nc, K), dtype=complex)
    for idx, (i, j) in enumerate(pairs):
        F_multi[idx, :] = Yc_matrix[:, i, j]

    C, D_const, _ = _solve_residues_vectfit_style(
        s, F_multi, poles, asymp=2, weights=None)

    k_residues = np.zeros((n_poles, nf, nf), dtype=complex)
    k0 = np.zeros((nf, nf), dtype=np.float64)
    for idx, (i, j) in enumerate(pairs):
        k0[i, j] = float(D_const[idx])
        k_residues[:, i, j] = C[idx, :]
        if i != j:
            k0[j, i] = k0[i, j]
            k_residues[:, j, i] = k_residues[:, i, j]

    inv_sp = 1.0 / (s[:, None] - poles[None, :])
    y_fit_all = np.einsum('kn,nij->kij', inv_sp, k_residues) + k0[None, :, :]

    total_error = 0.0
    for i, j in pairs:
        y = Yc_matrix[:, i, j]
        y_fit = y_fit_all[:, i, j]
        total_error += float(np.linalg.norm(y - y_fit) / (np.linalg.norm(y) + 1e-15))
    avg_error = total_error / len(pairs) if pairs else 0.0
    return k_residues, k0, avg_error


def check_passivity(freq, vf_result, tolerance=1e-6):
    """Test stability of a scalar VF result (all poles strictly in LHP)."""
    if vf_result is None:
        return False
    return np.all(np.real(vf_result.poles) < tolerance)

### ---------------------------------------------- Main fitting pipeline -------------------------------------------- ###

# perform_ulm_fitting() subroutine.
def perform_ulm_fitting(ulm_params, config, verbose=False):
    """Run Yc and H fitting on a pre-processed ULMParameters bundle."""
    freq = ulm_params.freq
    fmax = np.max(freq)
    s = 1j * 2 * np.pi * freq
    nf = ulm_params.nf
    is_freq_dep = ulm_params.is_freq_dependent

    active_modes, mode_groups = merge_modes_by_tau(ulm_params.tau, fmax, config.eps_deg)

    vf_trace = iterative_pole_fitting(
        freq, ulm_params.Yc_trace,
        config.Ymin, config.Ymax, config.epsY,
        config.pole_step, config.max_vf_iterations, False, "tr(Yc)")
    if vf_trace is None:
        raise RuntimeError("vector_fitting::ERROR::tr(Yc) fitting failed")

    k_residues, k0, _ = fit_Yc_matrix_fixed_poles(s, ulm_params.Yc_matrix, vf_trace.poles)

    H_fits = fit_H_modes_pscad_style(
        freq, ulm_params.H_modes, ulm_params.tau,
        mode_groups, ulm_params.D_matrices, config,
        is_freq_dep, ulm_params.H_mode_vf_results,
        H_matrix=ulm_params.H_matrix)

    H_rmse = np.array([fit.rmse for fit in H_fits])

    H_reconstructed = evaluate_H_matrix_from_fit(s, H_fits, mode_groups)
    diff = ulm_params.H_matrix - H_reconstructed
    H_matrix_rmse = float(np.linalg.norm(diff) / (np.linalg.norm(ulm_params.H_matrix) + 1e-15))

    H_reconstruction_metrics = None
    if config.compute_H_reconstruction_metrics and ulm_params.TI_matrix is not None:
        H_reconstruction_metrics = compute_H_reconstruction_rmse(
            freq, ulm_params.H_matrix, ulm_params.TI_matrix)

    is_passive = check_passivity(freq, vf_trace)

    return ULMFittingResult(
        nf=nf, n_active_modes=len(mode_groups),
        active_modes=[g[0] for g in mode_groups],
        mode_groups=mode_groups,
        poles_Yc=vf_trace.poles,
        k_residues=k_residues, k0=k0,
        tau_all=ulm_params.tau,
        D_matrices=ulm_params.D_matrices,
        H_modes_fits=H_fits,
        Yc_trace_rmse=float(vf_trace.rmse),
        H_modes_rmse=H_rmse,
        H_matrix_rmse=H_matrix_rmse,
        is_passive=is_passive,
        is_freq_dependent=is_freq_dep,
        Yc_matrix=ulm_params.Yc_matrix,
        H_modes=ulm_params.H_modes,
        H_matrix=ulm_params.H_matrix,
        freqs_Hz=freq,
        H_reconstruction_metrics=H_reconstruction_metrics,
        H_reconstructed=H_reconstructed,
    )


# ulm_complete_fitting() subroutine.
def ulm_complete_fitting(freq, Z_matrix, Y_matrix, length,
                         velocity_freq=1e6, config=None,
                         use_freq_dependent='auto',
                         pscad_TI_matrix=None,
                         enforce_passivity_flag=True, verbose=False):
    """Top-level ULM fitting entry point.
    Returns (ulm_params, result).
    """
    if config is None:
        config = IterativePoleFindingConfig()

    ulm_params = compute_ulm_parameters(
        freq, Z_matrix, Y_matrix, length,
        velocity_freq, config, use_freq_dependent,
        pscad_TI_matrix=pscad_TI_matrix)

    result = perform_ulm_fitting(ulm_params, config)
    return ulm_params, result

### --------------------------------------- fitULM file export / import --------------------------------------------- ###

def _classify_poles_for_export(poles, tol=1e-12):
    """Keep real poles and upper-half-plane conjugates for export (fitULM
    stores a single representative per conjugate pair).
    """
    unique_poles = []
    is_complex = []
    for p in poles:
        imag_part = np.imag(p)
        if np.abs(imag_part) < tol:
            unique_poles.append(complex(np.real(p), 0.0))
            is_complex.append(False)
        elif imag_part > 0:
            unique_poles.append(p)
            is_complex.append(True)
    return unique_poles, is_complex


def _count_poles_for_header(poles, tol=1e-12):
    """Count poles with conjugate pairs collapsed to a single entry."""
    unique_poles, _ = _classify_poles_for_export(poles, tol)
    return len(unique_poles)


def _write_real(f, x, precision=16):
    f.write(f"{{:+.{precision}e}}\n".format(float(x)))


def _write_integer(f, n):
    f.write(f"{int(n)}\n")


def _write_complex_element(f, z, precision=16):
    _write_real(f, np.real(z), precision)
    _write_real(f, np.imag(z), precision)


def _write_pole(f, p, precision=16):
    """Serialize a pole using the fitULM convention. Returns True if the
    pole is complex (conjugate-paired) and False for a real pole.
    """
    real_part = np.real(p)
    imag_part = np.imag(p)
    if np.abs(imag_part) < 1e-12:
        _write_real(f, real_part, precision)
        return False
    _write_real(f, np.abs(real_part), precision)
    _write_real(f, imag_part, precision)
    return True


def _write_upper_triangular(f, M, is_complex_pole, precision=16):
    """Write the upper-triangular part of a (possibly complex) matrix."""
    nf = M.shape[0]
    for i in range(nf):
        for j in range(i, nf):
            if is_complex_pole:
                _write_complex_element(f, M[i, j], precision)
            else:
                _write_real(f, np.real(M[i, j]), precision)


def _write_full_matrix(f, M, is_complex_pole, precision=16):
    """Write the full (possibly complex) matrix row by row."""
    nf = M.shape[0]
    for i in range(nf):
        for j in range(nf):
            if is_complex_pole:
                _write_complex_element(f, M[i, j], precision)
            else:
                _write_real(f, np.real(M[i, j]), precision)


def _write_k0_upper_triangular(f, k0, precision=16):
    """Write the upper-triangular part of the real k0 matrix."""
    nf = k0.shape[0]
    for i in range(nf):
        for j in range(i, nf):
            _write_real(f, np.real(k0[i, j]), precision)


def write_fitULM(result, filepath, precision=16, verbose=False):
    """Serialize a ULMFittingResult into a fitULM text file.

    Frequency-dependent D matrices collapse to a single snapshot; a non-zero
    constant term d or linear term h on H is discarded (fitULM does not store
    them, which is why H should be fitted with asymp=1).
    """
    nf = result.nf
    H_fits = result.H_modes_fits

    Yc_poles_unique, Yc_poles_is_complex = _classify_poles_for_export(result.poles_Yc)
    pYc = len(Yc_poles_unique)

    pH_list = []
    for fit in H_fits:
        if fit.poles is not None and len(fit.poles) > 0:
            poles_unique, _ = _classify_poles_for_export(fit.poles)
            pH_list.append(len(poles_unique))
        else:
            pH_list.append(0)

    with open(filepath, 'w') as f:
        _write_integer(f, nf)
        _write_integer(f, result.n_active_modes)
        _write_integer(f, pYc)

        for pH in pH_list:
            _write_integer(f, pH)

        for fit in H_fits:
            _write_real(f, fit.tau, precision)

        original_poles = result.poles_Yc
        k_residues = result.k_residues

        for idx_unique, (pole, is_complex) in enumerate(zip(Yc_poles_unique, Yc_poles_is_complex)):
            orig_idx = None
            conj_needed = False
            for i, orig_p in enumerate(original_poles):
                if np.abs(orig_p - pole) < 1e-9:
                    orig_idx = i
                    break
            if orig_idx is None:
                for i, orig_p in enumerate(original_poles):
                    if np.abs(orig_p - np.conj(pole)) < 1e-9:
                        orig_idx = i
                        conj_needed = True
                        break
            if orig_idx is None:
                orig_idx = idx_unique if idx_unique < len(k_residues) else 0

            _write_pole(f, pole, precision)

            if orig_idx < len(k_residues):
                k_matrix = np.conj(k_residues[orig_idx]) if conj_needed else k_residues[orig_idx]
            else:
                k_matrix = np.zeros((nf, nf), dtype=complex)
            _write_upper_triangular(f, k_matrix, is_complex, precision)

        for fit in H_fits:
            if fit.poles is None or len(fit.poles) == 0:
                continue
            H_poles_unique, H_poles_is_complex = _classify_poles_for_export(fit.poles)
            c_residues = fit.c_matrix_residues

            for idx_unique, (pole, is_complex) in enumerate(zip(H_poles_unique, H_poles_is_complex)):
                orig_idx = None
                conj_needed = False
                for i, orig_p in enumerate(fit.poles):
                    if np.abs(orig_p - pole) < 1e-9:
                        orig_idx = i
                        break
                if orig_idx is None:
                    for i, orig_p in enumerate(fit.poles):
                        if np.abs(orig_p - np.conj(pole)) < 1e-9:
                            orig_idx = i
                            conj_needed = True
                            break
                if orig_idx is None:
                    orig_idx = idx_unique if idx_unique < len(c_residues) else 0

                _write_pole(f, pole, precision)

                if orig_idx < len(c_residues):
                    c_matrix = np.conj(c_residues[orig_idx]) if conj_needed else c_residues[orig_idx]
                else:
                    c_matrix = np.zeros((nf, nf), dtype=complex)
                _write_full_matrix(f, c_matrix, is_complex, precision)

        _write_k0_upper_triangular(f, result.k0, precision)

    return filepath


def read_fitULM_header(filepath):
    """Read and parse the header section of a fitULM file.
    Returns a dict with keys nf, m, pYc, pH_list, tau_list, header_lines.
    """
    with open(filepath, 'r') as f:
        lines = f.readlines()
    idx = 0
    nf = int(float(lines[idx].strip())); idx += 1
    m = int(float(lines[idx].strip())); idx += 1
    pYc = int(float(lines[idx].strip())); idx += 1
    pH_list = []
    for _ in range(m):
        pH_list.append(int(float(lines[idx].strip()))); idx += 1
    tau_list = []
    for _ in range(m):
        tau_list.append(float(lines[idx].strip())); idx += 1
    return {
        'nf': nf, 'm': m, 'pYc': pYc,
        'pH_list': pH_list, 'tau_list': tau_list,
        'header_lines': idx,
    }


def verify_fitULM_file(filepath, verbose=False):
    """Verify that a fitULM file has a well-formed header with consistent
    nf, m, pH_list and tau_list lengths. Returns True/False.
    """
    try:
        header = read_fitULM_header(filepath)
        if header['nf'] <= 0 or header['m'] <= 0:
            return False
        if len(header['pH_list']) != header['m'] or len(header['tau_list']) != header['m']:
            return False
        return True
    except Exception:
        return False


def export_ulm_to_fitulm(ulm_params, fitting_result, filepath, precision=16, verbose=False):
    """Convenience wrapper around write_fitULM()."""
    return write_fitULM(fitting_result, filepath, precision, verbose)
