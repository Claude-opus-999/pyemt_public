# vf_core.py module.

"""
*** Vector Fitting adapter over vectfit3 ***

Thin wrapper that exposes a convenient scalar/matrix fitting interface
based on vectfit3.py (Fast Relaxed Vector Fitting for Python).

Rational model fitted by this module:

    F(s) = sum_{n=1..N} r_n / (s - p_n) + d + s*h

    - Original Matlab code author: Bjorn Gustavsen (08/2008)
    - Python vectfit3 port: Sebastian Loaiza (03/2024)

* Exposed API:
    - VectorFitResult        : result data class
    - vector_fitting()       : scalar/vector VF front-end
    - evaluate_rational_fit(), evaluate_rational_fit_s() : model evaluation
    - evaluate_rational_fit_matrix() : matrix-valued model evaluation
    - generate_initial_poles()       : stable initial pole set generator
"""

# Scientific computing modules:
import numpy as np
from dataclasses import dataclass

# Vector Fitting engine:
from .vectfit3 import vectfit, opts as vectfit_opts

### ------------------------------------------------- Data structures ------------------------------------------------- ###

@dataclass
class VectorFitResult:
    """Result of a scalar/vector fitting call.

    Fields:
     - poles        : array of fitted poles [n_poles]
     - residues     : array of residues (scalar case) or residue matrix [Nc, n_poles]
     - d            : constant term  (valid when asymp >= 2)
     - h            : linear term    (valid when asymp == 3, i.e. E*s)
     - rmse         : relative root-mean-square error
     - n_poles      : number of poles
     - n_iterations : pole-relocation iterations executed
     - converged    : True if the loop exited on convergence, not on max_iter
    """
    poles: np.ndarray
    residues: np.ndarray
    d: complex
    h: complex
    rmse: float
    n_poles: int = 0
    n_iterations: int = 1
    converged: bool = True

    def __post_init__(self):
        if self.n_poles == 0:
            self.n_poles = len(self.poles)

### -------------------------------------------------- Functions ----------------------------------------------------- ###

def generate_initial_poles(freq, n_poles, real_ratio=0.0, pole_type='linlog'):
    """Build a stable initial pole set inside the frequency band.

    Arguments.

     - freq       : frequency vector [Hz]
     - n_poles    : requested pole count (rounded up to even for conjugate pairing)
     - real_ratio : fraction of real poles in [0, 1]
     - pole_type  : 'linlog' | 'log' (logarithmic) or 'linear' spacing

    Returns an array of complex poles with negative real parts.
    """
    if n_poles < 2:
        n_poles = 2
    if n_poles % 2 != 0:
        n_poles += 1

    # Avoid pole at zero frequency; fall back to 0.1*f[1] if f[0] == 0
    omega_min = 2 * np.pi * max(freq[0], freq[1] * 0.1)
    omega_max = 2 * np.pi * freq[-1]

    n_real = int(np.round(n_poles * real_ratio))
    if n_real % 2 != 0:
        n_real = max(0, n_real - 1)
    n_complex = n_poles - n_real

    use_log = pole_type in ('log', 'linlog')
    spacing = (lambda a, b, n: np.logspace(np.log10(a), np.log10(b), n)) if use_log \
              else (lambda a, b, n: np.linspace(a, b, n))

    poles = []
    if n_real > 0:
        for omega in spacing(omega_min, omega_max, n_real):
            poles.append(-omega)

    n_pairs = n_complex // 2
    if n_pairs > 0:
        for beta in spacing(omega_min, omega_max, n_pairs):
            # Typical damping: alpha = beta/100 -> Q-factor ~ 100
            alpha = beta / 100.0
            poles.append(-alpha + 1j * beta)
            poles.append(-alpha - 1j * beta)

    return np.array(poles, dtype=np.complex128)

# vector_fitting() subroutine.
def vector_fitting(freq, data, n_poles=10, max_iterations=20, verbose=False,
                   weights=None, initial_poles=None, enforce_stability=True,
                   asymp=2, relaxed=True):
    """Rational approximation of frequency-domain samples via vectfit3.

    Arguments.

     - freq              : frequency vector [Hz], shape [K]
     - data              : samples, shape [K] (scalar) or [Nc, K] (vectorized)
     - n_poles           : pole count
     - max_iterations    : pole-relocation iterations
     - verbose           : forwarded to vectfit3 plotting flags (spy2, errplot)
     - weights           : optional weights, shape [K] or [Nc, K]
     - initial_poles     : optional starting poles; otherwise auto-generated
     - enforce_stability : enforce Re(p_n) < 0
     - asymp             : 1 (D=E=0), 2 (D free), 3 (D and E free)
     - relaxed           : relaxed non-triviality constraint

    Returns.
     - VectorFitResult with the best model across iterations.

    Raises RuntimeError if no valid iteration succeeded.
    """
    s = 1j * 2 * np.pi * freq
    N = len(freq)

    data = np.asarray(data, dtype=np.complex128)
    is_scalar = (data.ndim == 1)
    F = data.reshape(1, N) if is_scalar else data

    if initial_poles is None:
        poles = generate_initial_poles(freq, n_poles)
    else:
        poles = np.asarray(initial_poles, dtype=np.complex128)
        if len(poles) != n_poles:
            poles = generate_initial_poles(freq, n_poles)

    # Weights configuration (common or individual)
    if weights is None:
        weight = np.ones(N, dtype=np.float64)
    else:
        weight = np.asarray(weights, dtype=np.float64)
        if weight.ndim > 1 and is_scalar:
            weight = weight.reshape(-1)

    # vectfit3 options (inherit defaults, override what this adapter cares about)
    vf_opts = vectfit_opts.copy()
    vf_opts.update({
        "stable":    enforce_stability,
        "asymp":     asymp,
        "relax":     relaxed,
        "cmplx_ss":  True,
        "spy1":      False,
        "spy2":      verbose,
        "errplot":   verbose,
        "skip_pole": False,
        "skip_res":  False,
    })

    # Pole-relocation loop: iterate until RMSE stops improving significantly
    current_poles = poles.copy()
    best_rmse = float('inf')
    best_SER = None
    best_fit = None
    iteration = 0

    for iteration in range(max_iterations):
        result = vectfit(F, s, current_poles, weight, vf_opts,
                         graphsTitle=f"VF Iteration {iteration + 1}")
        if result is False:
            continue

        SER, new_poles, rmserr, fit = result
        if rmserr is None or rmserr < 0:
            continue

        if rmserr < best_rmse:
            best_rmse = rmserr
            best_SER = SER
            best_fit = fit

        # Convergence check: negligible pole movement or machine-precision fit
        if iteration > 0:
            pole_change = np.linalg.norm(new_poles - current_poles) \
                          / (np.linalg.norm(current_poles) + 1e-15)
            if pole_change < 1e-6 or rmserr < 1e-10:
                break

        current_poles = new_poles.copy()

    if best_SER is None:
        raise RuntimeError("vf_core::ERROR::vector_fitting failed to produce valid results")

    # Extract final model matrices from SER
    A = best_SER["A"]
    C = best_SER["C"]
    D = best_SER["D"]
    E = best_SER["E"]

    if is_scalar:
        final_poles    = np.diag(A) if A.ndim == 2 else A
        final_residues = C[0, :]   if C.ndim == 2 else C
        d = D[0] if isinstance(D, np.ndarray) else D
        h = E[0] if isinstance(E, np.ndarray) else E
    else:
        final_poles    = np.diag(A) if A.ndim == 2 else A
        final_residues = C
        d = D
        h = E

    # Relative error based on the retained best fit
    if best_fit is not None:
        diff = best_fit - F
        rmse = np.sqrt(np.sum(np.abs(diff) ** 2)) \
               / (np.sqrt(np.sum(np.abs(F) ** 2)) + 1e-15)
    else:
        rmse = best_rmse

    residues_out = final_residues[0, :] if is_scalar and final_residues.ndim == 2 else final_residues

    def _as_complex(x):
        if np.isscalar(x):
            return complex(x)
        if isinstance(x, np.ndarray):
            return complex(x[0])
        return complex(x)

    return VectorFitResult(
        poles=final_poles,
        residues=residues_out,
        d=_as_complex(d),
        h=_as_complex(h),
        rmse=rmse,
        n_poles=len(final_poles),
        n_iterations=iteration + 1,
        converged=(iteration < max_iterations - 1),
    )

def evaluate_rational_fit(freq, poles, residues, d=0.0, h=0.0):
    """Evaluate the rational model on real frequencies f [Hz].

        F(f) = sum_n r_n / (j2 pi f - p_n) + d + j2 pi f * h
    """
    s = 1j * 2 * np.pi * freq
    return evaluate_rational_fit_s(s, poles, residues, d, h)

def evaluate_rational_fit_s(s, poles, residues, d=0.0, h=0.0):
    """Evaluate the rational model on complex frequencies s.

        F(s) = sum_n r_n / (s - p_n) + d + s * h
    """
    poles = np.asarray(poles, dtype=np.complex128)
    residues = np.asarray(residues, dtype=np.complex128)
    s = np.asarray(s, dtype=np.complex128)

    fit = np.full(len(s), complex(d), dtype=np.complex128)
    if h != 0:
        fit = fit + s * complex(h)
    # Broadcast: (K,1) - (1,n) -> (K,n) summed on pole axis
    fit = fit + np.sum(residues[None, :] / (s[:, None] - poles[None, :]), axis=1)
    return fit

def evaluate_rational_fit_matrix(s, poles, residues_matrix, d_matrix, h_matrix=None):
    """Evaluate a matrix-valued rational model.

    Arguments.

     - s                : complex frequency vector [K]
     - poles            : pole vector [n_poles]
     - residues_matrix  : residue matrices [n_poles, nf, nf]
     - d_matrix         : constant matrix [nf, nf]
     - h_matrix         : optional linear matrix [nf, nf]

    Returns fitted samples of shape [K, nf, nf].
    """
    s = np.asarray(s, dtype=np.complex128)
    poles = np.asarray(poles, dtype=np.complex128)

    # Broadcasted pole sum: (K,n) * (n,nf,nf) -> sum over n
    inv_sp = 1.0 / (s[:, None] - poles[None, :])              # [K, n]
    fit = np.einsum('kn,nij->kij', inv_sp, residues_matrix)   # [K, nf, nf]
    fit = fit + d_matrix[None, :, :]
    if h_matrix is not None:
        fit = fit + s[:, None, None] * h_matrix[None, :, :]
    return fit
