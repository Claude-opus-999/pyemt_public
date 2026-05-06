# ulm_atp_zy_deri_semlyen.py module.

"""
*** Overhead transmission line Z/Y matrices (Deri-Semlyen complex depth) ***

Frequency-domain impedance and admittance computation for multi-conductor
overhead transmission lines:
    - Series impedance [Z]: Deri-Semlyen complex-depth formula (Velasco
      internal impedance + ideal-image geometric term + complex-image earth
      correction).
    - Shunt admittance [Y]: classical potential-coefficient method (ideal
      image conductors), with a PSCAD-style shunt conductance on the phase
      diagonal to avoid a purely imaginary Y matrix and model insulator
      leakage.

* Physical model:
    The earth surface is replaced by an ideal-conductor plane at the complex
    depth p = sqrt(rho / (j omega mu_0)). Standard image theory is then
    applied with the complex image located at -(h + 2p):
        Z_ext_self  = (j omega mu_0 / 2 pi) ln(2(h + p) / r_eq)
        Z_ext_mutual = (j omega mu_0 / 2 pi) ln(D'_complex / D)
    where D'_complex = sqrt(dx^2 + (h_i + h_j + 2p)^2).

* Exposed API:
    - SoilParameters, get_constant_soil_params()      : soil model
    - ConductorGeometry, MultiConductorLine           : geometry data classes
    - ImpedanceResult, AdmittanceResult,
      ImpedanceMatrixResult, AdmittanceMatrixResult   : result data classes
    - conductor_internal_impedance_velasco()          : Velasco internal Zi
    - geometric_impedance_self(),
      geometric_impedance_mutual()                    : ideal-image geometric Z
    - deri_semlyen_earth_impedance_self(),
      deri_semlyen_earth_impedance_mutual()           : DS earth correction
    - compute_ohl_self_impedance()                    : single-conductor Z
    - geometric_potential_coeff_self(),
      geometric_potential_coeff_mutual()              : Maxwell coefficients
    - compute_ohl_self_admittance()                   : single-conductor Y
    - compute_impedance_matrix(),
      compute_admittance_matrix()                     : multi-conductor Z, Y

* References:
    [1] Deri, Tevan, Semlyen, Castanheira, "The Complex Ground Return Plane:
        A Simplified Model for Homogeneous and Multi-Layer Earth Return",
        IEEE Trans. PAS, PAS-100(8), 1981.
    [2] Dommel, "EMTP Theory Book", 1986.
    [3] Semlyen, "Ground Return Parameters of Transmission Lines: An
        Asymptotic Analysis for Very High Frequencies", 1981.
"""

# Scientific computing modules:
import numpy as np
from dataclasses import dataclass, field
from typing import List, Union

### -------------------------------------------------- Constants ----------------------------------------------------- ###

MU_0 = 4 * np.pi * 1e-7
EPSILON_0 = 8.854187817e-12
C_LIGHT = 299792458

# PSCAD-style shunt conductance applied to the phase-conductor diagonal of Y
# to avoid a purely imaginary admittance matrix and to model insulator/air
# leakage. Ground-wire diagonals are NOT loaded with this conductance because
# ground wires are directly bonded to the tower.
SHUNT_CONDUCTANCE = 1.0e-11

# Default frequency sweep parameters
DEFAULT_FREQ_START = 1.0
DEFAULT_FREQ_END = 1e7
DEFAULT_POINTS_PER_DECADE = 30

### --------------------------------------------- Frequency utilities ------------------------------------------------ ###

def generate_frequency_vector(f_start=DEFAULT_FREQ_START,
                              f_end=DEFAULT_FREQ_END,
                              points_per_decade=DEFAULT_POINTS_PER_DECADE):
    """Logarithmically spaced frequency vector covering [f_start, f_end]."""
    num_decades = np.log10(f_end / f_start)
    num_points = int(num_decades * points_per_decade) + 1
    return np.logspace(np.log10(f_start), np.log10(f_end), num_points)


def omega_from_freq(freq):
    """Angular frequency: omega = 2 pi f."""
    return 2 * np.pi * freq

### ----------------------------------------------- Soil parameters -------------------------------------------------- ###

@dataclass
class SoilParameters:
    """Homogeneous-resistivity soil with Deri-Semlyen complex penetration depth.

    p_complex(omega) = sqrt(rho / (j omega mu_0)) substitutes for the Carson
    effective depth: the earth surface is treated as an ideal-conductor plane
    located at depth p, which gives Re(p) > 0 (deeper image) and Im(p) < 0
    (loss channel).
    """
    frequency: np.ndarray
    omega: np.ndarray
    rho: float
    sigma: float
    epsilon_r: float
    p_complex: np.ndarray          # Complex penetration depth [m]
    penetration_depth: np.ndarray  # Skin depth delta = sqrt(2 rho / (omega mu_0)) [m]


# get_constant_soil_params() subroutine.
def get_constant_soil_params(freq, rho_0, epsilon_r=10.0):
    """Build a SoilParameters object for constant soil resistivity.

    At omega = 0 the complex depth diverges (ideal conductor at infinity),
    which is handled explicitly with np.inf instead of letting 1/0 propagate.
    """
    freq = np.atleast_1d(np.array(freq, dtype=float))
    omega = omega_from_freq(freq)
    sigma = 1.0 / rho_0

    p_complex = np.zeros_like(freq, dtype=complex)
    penetration_depth = np.zeros_like(freq)
    mask = omega > 0
    p_complex[mask] = np.sqrt(rho_0 / (1j * omega[mask] * MU_0))
    p_complex[~mask] = np.inf
    penetration_depth[mask] = np.sqrt(2 * rho_0 / (omega[mask] * MU_0))
    penetration_depth[~mask] = np.inf

    return SoilParameters(
        frequency=freq, omega=omega, rho=rho_0, sigma=sigma,
        epsilon_r=epsilon_r,
        p_complex=p_complex, penetration_depth=penetration_depth,
    )

### --------------------------------------------- Conductor geometry ------------------------------------------------- ###

@dataclass
class ConductorGeometry:
    """Single-conductor geometry (optionally bundled, with parabolic sag).

    Sag correction follows the PSCAD/ATP-EMTP standard rule
        h_avg = h_tower - (2/3) sag
    which is the parabolic-span average height. Z and Y computations always
    use avg_height, not the tower attachment height.

    Bundle equivalent radius (Markt-Mengele):
        r_eq = (N r R_b^{N-1})^{1/N},  R_b = S / (2 sin(pi/N))
    with the N = 2 special case r_eq = sqrt(r S).
    """
    height: float                 # tower attachment height [m]
    horizontal_pos: float         # [m]
    radius: float                 # sub-conductor radius [m]
    rdc: float                    # sub-conductor DC resistance [Ohm/m]
    mu_r: float = 1.0
    bundle_n: int = 1
    bundle_spacing: float = 0.45
    sag: float = 0.0

    @property
    def avg_height(self):
        """Span-average height h_avg = h_tower - 2/3 sag (sag = 0 disables)."""
        return self.height - (2.0 / 3.0) * self.sag

    @property
    def equivalent_radius(self):
        """Bundle equivalent radius (Markt-Mengele); reverts to r for N=1."""
        if self.bundle_n <= 1:
            return self.radius
        N = self.bundle_n
        r = self.radius
        S = self.bundle_spacing
        if N == 2:
            return np.sqrt(r * S)
        R_b = S / (2 * np.sin(np.pi / N))
        return (N * r * (R_b ** (N - 1))) ** (1 / N)

    @property
    def bundle_radius(self):
        """Bundle circle radius R_b (0 for single conductors)."""
        if self.bundle_n <= 1:
            return 0.0
        if self.bundle_n == 2:
            return self.bundle_spacing / 2
        return self.bundle_spacing / (2 * np.sin(np.pi / self.bundle_n))

    def __post_init__(self):
        if self.bundle_n < 1:
            raise ValueError("ulm_atp_zy::ERROR::bundle_n must be >= 1")
        if self.bundle_n > 1 and self.bundle_spacing <= 0:
            raise ValueError("ulm_atp_zy::ERROR::bundle_spacing must be > 0 for bundled conductors")
        if self.radius <= 0:
            raise ValueError("ulm_atp_zy::ERROR::radius must be > 0")
        if self.height <= 0:
            raise ValueError("ulm_atp_zy::ERROR::height must be > 0")
        if self.sag < 0:
            raise ValueError("ulm_atp_zy::ERROR::sag must be >= 0")
        if self.sag > 0 and self.avg_height <= 0:
            raise ValueError(
                f"ulm_atp_zy::ERROR::sag={self.sag} m gives h_avg={self.avg_height:.2f} m <= 0 "
                f"(h_tower={self.height} m)")

### -------------------------------------- Series impedance (per conductor pair) ------------------------------------ ###

@dataclass
class ImpedanceResult:
    """Single-conductor self-impedance decomposition."""
    Z_int: np.ndarray
    Z_geo: np.ndarray
    Z_earth: np.ndarray
    Z_total: np.ndarray
    R: np.ndarray
    L: np.ndarray


def conductor_internal_impedance_velasco(freq, conductor):
    """Velasco full-frequency approximation of the conductor internal impedance:

        Z_int = sqrt(R_dc^2 + Z_HF^2),   Z_HF = 1 / (2 pi r sigma p_c)
        p_c = 1 / sqrt(j omega mu sigma)

    Bundled conductors are paralleled as Z_sub / N.
    """
    omega = omega_from_freq(freq)
    r = conductor.radius
    rdc_sub = conductor.rdc

    sigma = 1.0 / (np.pi * r ** 2 * rdc_sub)
    mu = MU_0 * conductor.mu_r

    pc = np.empty_like(omega, dtype=complex)
    Z_HF = np.zeros_like(omega, dtype=complex)
    mask = omega > 0
    pc[mask] = 1.0 / np.sqrt(1j * omega[mask] * mu * sigma)
    pc[~mask] = np.inf
    Z_HF[mask] = 1.0 / (2.0 * np.pi * r * sigma * pc[mask])

    Z_sub = np.sqrt(rdc_sub ** 2 + Z_HF ** 2)
    return Z_sub / conductor.bundle_n


def geometric_impedance_self(freq, conductor):
    """Ideal-image geometric self impedance:
        Z_geo_self = j omega mu_0 / (2 pi) ln(2 h_avg / r_eq).
    """
    omega = omega_from_freq(freq)
    L_geo = (MU_0 / (2 * np.pi)) * np.log(2 * conductor.avg_height / conductor.equivalent_radius)
    return 1j * omega * L_geo


def geometric_impedance_mutual(freq, cond_i, cond_j):
    """Ideal-image geometric mutual impedance:
        Z_geo_mutual = j omega mu_0 / (2 pi) ln(D'_ij / D_ij)
    where D_ij and D'_ij are the conductor-conductor and conductor-image
    distances respectively.
    """
    omega = omega_from_freq(freq)
    x_i, h_i = cond_i.horizontal_pos, cond_i.avg_height
    x_j, h_j = cond_j.horizontal_pos, cond_j.avg_height

    D_ij = np.sqrt((x_i - x_j) ** 2 + (h_i - h_j) ** 2)
    D_ij_image = np.sqrt((x_i - x_j) ** 2 + (h_i + h_j) ** 2)
    L_geo_mutual = (MU_0 / (2 * np.pi)) * np.log(D_ij_image / D_ij)
    return 1j * omega * L_geo_mutual


def deri_semlyen_earth_impedance_self(freq, conductor, p_complex):
    """Deri-Semlyen self earth correction (difference between the complex-
    image external impedance and the ideal-image geometric term):

        Z_earth_self = j omega mu_0 / (2 pi) ln((h + p) / h)
    """
    omega = omega_from_freq(freq)
    h = conductor.avg_height
    return (1j * omega * MU_0 / (2 * np.pi)) * np.log((h + p_complex) / h)


def deri_semlyen_earth_impedance_mutual(freq, cond_i, cond_j, p_complex):
    """Deri-Semlyen mutual earth correction:

        Z_earth_mutual = j omega mu_0 / (2 pi) ln(D'_complex / D'_real)
        D'_complex = sqrt(dx^2 + (h_i + h_j + 2 p)^2)
        D'_real    = sqrt(dx^2 + (h_i + h_j)^2)
    """
    omega = omega_from_freq(freq)
    x_i, h_i = cond_i.horizontal_pos, cond_i.avg_height
    x_j, h_j = cond_j.horizontal_pos, cond_j.avg_height
    dx = x_i - x_j

    D_ij_image = np.sqrt(dx ** 2 + (h_i + h_j) ** 2)
    D_ij_complex = np.sqrt(dx ** 2 + (h_i + h_j + 2 * p_complex) ** 2)
    return (1j * omega * MU_0 / (2 * np.pi)) * np.log(D_ij_complex / D_ij_image)


def compute_ohl_self_impedance(freq, conductor, p_complex):
    """Full single-conductor self-impedance:
        Z_total = Z_int + Z_geo_self + Z_earth_self(DS).
    Combined external impedance Z_geo + Z_earth collapses to the compact
    Deri-Semlyen form (j omega mu_0 / 2 pi) ln(2 (h + p) / r_eq).
    """
    omega = omega_from_freq(freq)
    Z_int = conductor_internal_impedance_velasco(freq, conductor)
    Z_geo = geometric_impedance_self(freq, conductor)
    Z_earth = deri_semlyen_earth_impedance_self(freq, conductor, p_complex)

    Z_total = Z_int + Z_geo + Z_earth
    R = np.real(Z_total)
    L = np.imag(Z_total) / omega
    return ImpedanceResult(Z_int=Z_int, Z_geo=Z_geo, Z_earth=Z_earth,
                           Z_total=Z_total, R=R, L=L)

### -------------------------------- Shunt admittance (potential-coefficient method) ------------------------------- ###

@dataclass
class AdmittanceResult:
    """Single-conductor self-admittance decomposition."""
    P_geo: np.ndarray
    P_total: np.ndarray
    Y_total: np.ndarray
    G: np.ndarray
    C: np.ndarray


def geometric_potential_coeff_self(conductor):
    """Self Maxwell potential coefficient (ideal image):
        P_ii = 1/(2 pi eps_0) ln(2 h_avg / r_eq).
    Deri-Semlyen complex-image correction to Y is typically negligible at
    power frequencies and is not applied here.
    """
    return (1.0 / (2 * np.pi * EPSILON_0)) * np.log(
        2 * conductor.avg_height / conductor.equivalent_radius)


def geometric_potential_coeff_mutual(cond_i, cond_j):
    """Mutual Maxwell potential coefficient (ideal image):
        P_ij = 1/(2 pi eps_0) ln(D'_ij / D_ij).
    """
    x_i, h_i = cond_i.horizontal_pos, cond_i.avg_height
    x_j, h_j = cond_j.horizontal_pos, cond_j.avg_height
    D_ij = np.sqrt((x_i - x_j) ** 2 + (h_i - h_j) ** 2)
    D_ij_image = np.sqrt((x_i - x_j) ** 2 + (h_i + h_j) ** 2)
    return (1.0 / (2 * np.pi * EPSILON_0)) * np.log(D_ij_image / D_ij)


def compute_ohl_self_admittance(freq, conductor, shunt_conductance=SHUNT_CONDUCTANCE):
    """Single-conductor self-admittance:
        Y = G_shunt + j omega / P_geo.
    G_shunt is added on the diagonal to avoid a purely imaginary Y.
    """
    omega = omega_from_freq(freq)
    P_geo_scalar = geometric_potential_coeff_self(conductor)
    P_geo = np.full_like(freq, P_geo_scalar, dtype=complex)
    P_total = P_geo.copy()

    Y_total = shunt_conductance + 1j * omega / P_total
    G = np.full_like(freq, shunt_conductance)
    C = np.full_like(freq, 1.0 / P_geo_scalar)
    return AdmittanceResult(P_geo=P_geo, P_total=P_total,
                            Y_total=Y_total, G=G, C=C)

### --------------------------------------- Multi-conductor line assembly ------------------------------------------ ###

@dataclass
class MultiConductorLine:
    """Multi-conductor line: list of ConductorGeometry + naming and
    ground-wire flags.
    """
    conductors: List[ConductorGeometry]
    names: List[str] = field(default_factory=list)
    is_ground_wire: List[bool] = field(default_factory=list)
    bundled: List[int] = field(default_factory=list)
    bundle_spacing: List[float] = field(default_factory=list)

    def __post_init__(self):
        n = len(self.conductors)
        if not self.names:
            self.names = [f'C{i+1}' for i in range(n)]
        if not self.is_ground_wire:
            self.is_ground_wire = [False] * n
        if not self.bundled:
            self.bundled = [1] * n
        if not self.bundle_spacing:
            self.bundle_spacing = [0.0] * n

    @property
    def n_conductors(self):
        return len(self.conductors)

    @property
    def n_phases(self):
        return sum(1 for gw in self.is_ground_wire if not gw)

    def get_distance(self, i, j):
        """Euclidean distance between conductors i and j (avg height)."""
        ci = self.conductors[i]
        cj = self.conductors[j]
        dx = ci.horizontal_pos - cj.horizontal_pos
        dy = ci.avg_height - cj.avg_height
        return np.sqrt(dx ** 2 + dy ** 2)

    def get_image_distance(self, i, j):
        """Distance from conductor i to the ideal-image of conductor j."""
        ci = self.conductors[i]
        cj = self.conductors[j]
        dx = ci.horizontal_pos - cj.horizontal_pos
        dy = ci.avg_height + cj.avg_height
        return np.sqrt(dx ** 2 + dy ** 2)

    def get_complex_image_distance(self, i, j, p_complex):
        """Complex-image distance (Deri-Semlyen):
            D'_complex = sqrt(dx^2 + (h_i + h_j + 2 p)^2).
        """
        ci = self.conductors[i]
        cj = self.conductors[j]
        dx = ci.horizontal_pos - cj.horizontal_pos
        h_sum = ci.avg_height + cj.avg_height
        return np.sqrt(dx ** 2 + (h_sum + 2 * p_complex) ** 2)

### ------------------------------------------ Matrix results and assembly ----------------------------------------- ###

@dataclass
class ImpedanceMatrixResult:
    """Series impedance matrix with physical-term decomposition."""
    Z_matrix: np.ndarray
    Z_int: np.ndarray
    Z_geo: np.ndarray
    Z_earth: np.ndarray
    freq: np.ndarray


@dataclass
class AdmittanceMatrixResult:
    """Shunt admittance matrix with potential-coefficient decomposition."""
    Y_matrix: np.ndarray
    P_geo: np.ndarray
    P_total: np.ndarray
    freq: np.ndarray
    shunt_conductance: float


# compute_impedance_matrix() subroutine.
def compute_impedance_matrix(freq, line, p_complex, verbose=False):
    """Assemble the full [n_freq, n_cond, n_cond] series impedance matrix.

    Diagonal:    Z_ii = Z_int_i + Z_geo_self + Z_earth_self(DS)
    Off-diag:    Z_ij = Z_geo_mutual + Z_earth_mutual(DS)
    """
    n_freq = len(freq)
    n_cond = line.n_conductors
    omega = omega_from_freq(freq)

    Z_matrix = np.zeros((n_freq, n_cond, n_cond), dtype=complex)
    Z_int = np.zeros((n_freq, n_cond, n_cond), dtype=complex)
    Z_geo = np.zeros((n_freq, n_cond, n_cond), dtype=complex)
    Z_earth = np.zeros((n_freq, n_cond, n_cond), dtype=complex)

    for i in range(n_cond):
        ci = line.conductors[i]

        Z_int_i = conductor_internal_impedance_velasco(freq, ci)
        Z_int[:, i, i] = Z_int_i

        Z_geo_self = geometric_impedance_self(freq, ci)
        Z_geo[:, i, i] = Z_geo_self

        Z_earth_self = deri_semlyen_earth_impedance_self(freq, ci, p_complex)
        Z_earth[:, i, i] = Z_earth_self

        Z_matrix[:, i, i] = Z_int_i + Z_geo_self + Z_earth_self

        for j in range(i + 1, n_cond):
            cj = line.conductors[j]

            d_ij = line.get_distance(i, j)
            D_ij = line.get_image_distance(i, j)
            Z_geo_mutual = 1j * omega * (MU_0 / (2 * np.pi)) * np.log(D_ij / d_ij)
            Z_geo[:, i, j] = Z_geo[:, j, i] = Z_geo_mutual

            Z_earth_mutual = deri_semlyen_earth_impedance_mutual(freq, ci, cj, p_complex)
            Z_earth[:, i, j] = Z_earth[:, j, i] = Z_earth_mutual

            Z_matrix[:, i, j] = Z_geo_mutual + Z_earth_mutual
            Z_matrix[:, j, i] = Z_matrix[:, i, j]

    return ImpedanceMatrixResult(Z_matrix=Z_matrix, Z_int=Z_int,
                                 Z_geo=Z_geo, Z_earth=Z_earth, freq=freq)


# compute_admittance_matrix() subroutine.
def compute_admittance_matrix(freq, line, shunt_conductance=SHUNT_CONDUCTANCE, verbose=False):
    """Assemble the full [n_freq, n_cond, n_cond] shunt admittance matrix.

        [Y] = G_shunt I_phase + j omega [P]^-1

    G_shunt is loaded only on phase-conductor diagonals (ground-wire diagonals
    stay at zero). [P] is frequency-independent under the ideal-image model,
    so it is inverted once and broadcast across the frequency axis.
    """
    n_freq = len(freq)
    n_cond = line.n_conductors
    omega = omega_from_freq(freq)

    # Potential-coefficient matrix [P] (frequency-independent)
    P_geo = np.zeros((n_cond, n_cond), dtype=float)
    for i in range(n_cond):
        ci = line.conductors[i]
        P_geo[i, i] = geometric_potential_coeff_self(ci)
        for j in range(i + 1, n_cond):
            cj = line.conductors[j]
            P_geo_mutual = geometric_potential_coeff_mutual(ci, cj)
            P_geo[i, j] = P_geo[j, i] = P_geo_mutual

    try:
        P_inv = np.linalg.inv(P_geo)
    except np.linalg.LinAlgError:
        P_inv = np.linalg.pinv(P_geo)

    # Shunt-conductance diagonal (phases only)
    G_shunt_diag = np.zeros((n_cond, n_cond), dtype=float)
    for i in range(n_cond):
        if not line.is_ground_wire[i]:
            G_shunt_diag[i, i] = shunt_conductance

    P_geo_3d = np.broadcast_to(P_geo, (n_freq, n_cond, n_cond)).astype(complex).copy()
    P_total_3d = P_geo_3d.copy()

    Y_matrix = np.zeros((n_freq, n_cond, n_cond), dtype=complex)
    for k in range(n_freq):
        Y_matrix[k] = G_shunt_diag + 1j * omega[k] * P_inv

    return AdmittanceMatrixResult(Y_matrix=Y_matrix, P_geo=P_geo_3d,
                                  P_total=P_total_3d, freq=freq,
                                  shunt_conductance=shunt_conductance)
