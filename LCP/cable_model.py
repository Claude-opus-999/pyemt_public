# cable_model.py module.

"""
*** Cable impedance and admittance formulation (Ametani 1980) ***

Library routines for single-core armored cable (SC cable), pipe-type cable
(PT cable) and three-core cable. Earth-return impedance uses the exact
Pollaczek integral; pipe internal impedance uses the finite-thickness
formulation (Hoidalen 2013) to avoid the spurious negative self-resistance
of the infinite-thickness formulas at low frequency.

* Exposed API:
    - SoilParameters, ArmoredCableGeometry, InnerConductor,
      PipeTypeCableGeometry, ThreeCoreCableGeometry      : geometry/material
    - solid_conductor_impedance(), tubular_conductor_impedance(),
      insulation_impedance()                              : elementary terms
    - earth_return_impedance_self(), earth_return_impedance_mutual(),
      earth_return_impedance_self_highprecision()         : Pollaczek Zg
    - compute_armored_cable_impedance()                   : 3x3 SC cable
    - compute_pipe_internal_impedance(),
      compute_pipe_connection_impedance(),
      compute_inner_conductor_impedance(),
      compute_pipe_type_cable_impedance(),
      compute_pipe_type_cable_potential()                 : PT cable
    - compute_three_core_cable_impedance()                : three-core cable
    - compute_multi_cable_impedance()                     : multi-cable system
    - compute_sequence_impedance()                        : 012-sequence
    - compute_armored_cable_admittance()                  : SC cable Y-matrix

* References:
    A. Ametani, "A General Formulation of Impedance and Admittance of
    Cables", IEEE Trans. PAS, Vol. PAS-99, No. 3, 1980.

    F. Pollaczek, "Ueber das Feld einer unendlich langen wechselstrom-
    durchflossenen Einfachleitung", ENT, Vol. 3, No. 9, 1926.

    H. Kr. Hoidalen, "Analysis of Pipe-Type Cable Impedance Formulations
    at Low Frequencies", IEEE Trans. Power Delivery, 2013.
"""

# Scientific computing modules:
import numpy as np
from scipy.special import iv, kv, kve
from scipy.integrate import quad
from dataclasses import dataclass, field
from typing import List, Tuple
import warnings

### -------------------------------------------------- Constants ----------------------------------------------------- ###

MU_0 = 4 * np.pi * 1e-7
EPSILON_0 = 8.854187817e-12

### --------------------------------------------- Frequency utilities ------------------------------------------------ ###

def omega_from_freq(freq):
    """Angular frequency from cyclic frequency: omega = 2 pi f."""
    return 2 * np.pi * np.asarray(freq, dtype=float)


def generate_frequency_vector(f_min, f_max, n_points, log_scale=True):
    """Build a frequency vector on a logarithmic or linear grid."""
    if log_scale:
        return np.logspace(np.log10(f_min), np.log10(f_max), n_points)
    return np.linspace(f_min, f_max, n_points)

### -------------------------------------------- Soil and geometries ------------------------------------------------- ###

@dataclass
class SoilParameters:
    """Electrical parameters of the surrounding soil."""
    rho: float
    epsilon_r: float = 10.0
    mu_r: float = 1.0

    def get_gamma(self, freq):
        """Soil propagation constant gamma = sqrt(j omega mu sigma)."""
        omega = omega_from_freq(freq)
        sigma = 1.0 / self.rho
        return np.sqrt(1j * omega * MU_0 * self.mu_r * sigma)

    def get_epsilon_complex(self, freq):
        """Soil complex permittivity epsilon_r*eps0 - j sigma/omega."""
        omega = omega_from_freq(freq)
        sigma = 1.0 / self.rho
        return self.epsilon_r * EPSILON_0 - 1j * sigma / omega


@dataclass
class ArmoredCableGeometry:
    """SC cable geometry with armor (core / insulation / sheath /
    sheath insulation / armor / jacket). Radii follow Ametani 1980 section 2.1.
    """
    core_radius: float
    core_rho: float
    core_mu_r: float = 1.0

    insulation_radius: float = 0.0
    insulation_epsilon_r: float = 2.3
    insulation_tan_delta: float = 0.001
    insulation_mu_r: float = 1.0

    sheath_inner_radius: float = 0.0
    sheath_outer_radius: float = 0.0
    sheath_rho: float = 2.2e-7
    sheath_mu_r: float = 1.0

    sheath_insulation_radius: float = 0.0
    sheath_insulation_epsilon_r: float = 2.3
    sheath_insulation_mu_r: float = 1.0

    armor_inner_radius: float = 0.0
    armor_outer_radius: float = 0.0
    armor_rho: float = 1.4e-7
    armor_mu_r: float = 100.0

    jacket_radius: float = 0.0
    jacket_epsilon_r: float = 2.3
    jacket_tan_delta: float = 0.01
    jacket_mu_r: float = 1.0

    burial_depth: float = 1.5
    horizontal_pos: float = 0.0

    def __post_init__(self):
        if self.sheath_inner_radius == 0.0 and self.insulation_radius > 0:
            self.sheath_inner_radius = self.insulation_radius
        if self.sheath_insulation_radius == 0.0 and self.sheath_outer_radius > 0:
            self.sheath_insulation_radius = self.sheath_outer_radius + 0.002
        if self.armor_inner_radius == 0.0 and self.sheath_insulation_radius > 0:
            self.armor_inner_radius = self.sheath_insulation_radius


@dataclass
class InnerConductor:
    """PT cable inner conductor: core (optionally with sheath/screen).

    distance_from_center and angular_position place the conductor inside the
    enclosing pipe. n_conductors() returns 1 (core only) or 2 (core+sheath).
    """
    core_radius: float
    core_rho: float = 1.72e-8
    core_mu_r: float = 1.0

    insulation_radius: float = 0.0
    insulation_epsilon_r: float = 2.3
    insulation_tan_delta: float = 0.0
    insulation_mu_r: float = 1.0

    has_sheath: bool = False
    sheath_inner_radius: float = 0.0
    sheath_outer_radius: float = 0.0
    sheath_rho: float = 1.72e-8
    sheath_mu_r: float = 1.0

    outer_insulation_radius: float = 0.0
    outer_insulation_epsilon_r: float = 2.3
    outer_insulation_tan_delta: float = 0.0

    distance_from_center: float = 0.0
    angular_position: float = 0.0

    @property
    def outer_radius(self):
        if self.has_sheath and self.outer_insulation_radius > 0:
            return self.outer_insulation_radius
        if self.has_sheath and self.sheath_outer_radius > 0:
            return self.sheath_outer_radius
        if self.insulation_radius > 0:
            return self.insulation_radius
        return self.core_radius

    @property
    def n_conductors(self):
        return 2 if self.has_sheath else 1


@dataclass
class PipeTypeCableGeometry:
    """PT cable geometry: n inner conductors + common pipe + outer jacket.
    Radii follow Ametani 1980 section 2.2 (eqs. 36-39).
    """
    inner_conductors: List[InnerConductor] = field(default_factory=list)

    pipe_inner_radius: float = 0.0
    pipe_outer_radius: float = 0.0
    pipe_rho: float = 1.4e-7
    pipe_mu_r: float = 100.0

    jacket_radius: float = 0.0
    jacket_epsilon_r: float = 2.3

    pipe_inner_insulation_epsilon_r: float = 4.0
    pipe_inner_insulation_tan_delta: float = 0.0

    pipe_outer_insulation_epsilon_r: float = 2.3
    pipe_outer_insulation_tan_delta: float = 0.0

    burial_depth: float = 1.0
    horizontal_pos: float = 0.0

    finite_pipe_thickness: bool = True

    @property
    def n_inner_conductors(self):
        return len(self.inner_conductors)

    @property
    def total_conductors(self):
        n = sum(ic.n_conductors for ic in self.inner_conductors)
        if self.finite_pipe_thickness:
            n += 1
        return n


@dataclass
class ThreeCoreCableGeometry:
    """Three-core cable built on top of the PT cable model.
    Matrix dimension: 4x4 (core only) or 7x7 (core + screen).
    """
    core_radius: float
    core_rho: float = 1.72e-8
    core_mu_r: float = 1.0

    insulation_thickness: float = 0.0
    insulation_epsilon_r: float = 2.3
    insulation_tan_delta: float = 0.001
    insulation_mu_r: float = 1.0

    has_screen: bool = True
    screen_thickness: float = 0.001
    screen_rho: float = 1.72e-8
    screen_mu_r: float = 1.0

    screen_insulation_thickness: float = 0.001
    screen_insulation_epsilon_r: float = 3.0

    filler_epsilon_r: float = 4.0

    bedding_thickness: float = 0.002
    bedding_epsilon_r: float = 4.0

    has_armor: bool = True
    armor_thickness: float = 0.003
    armor_rho: float = 1.4e-7
    armor_mu_r: float = 100.0
    armor_type: str = 'steel_tape'

    jacket_thickness: float = 0.003
    jacket_epsilon_r: float = 2.3
    jacket_tan_delta: float = 0.01

    burial_depth: float = 1.0
    horizontal_pos: float = 0.0

    core_spacing_factor: float = 1.05

    @property
    def insulated_core_radius(self):
        return self.core_radius + self.insulation_thickness

    @property
    def screened_core_radius(self):
        if self.has_screen:
            return self.insulated_core_radius + self.screen_thickness
        return self.insulated_core_radius

    @property
    def inner_conductor_outer_radius(self):
        r = self.screened_core_radius
        if self.has_screen:
            r += self.screen_insulation_thickness
        return r

    @property
    def phase_spacing(self):
        return 2 * self.inner_conductor_outer_radius * self.core_spacing_factor

    @property
    def core_circle_radius(self):
        d = self.phase_spacing
        return d / np.sqrt(3) + self.inner_conductor_outer_radius

    @property
    def pipe_inner_radius(self):
        return self.core_circle_radius + self.bedding_thickness

    @property
    def pipe_outer_radius(self):
        if self.has_armor:
            return self.pipe_inner_radius + self.armor_thickness
        return self.pipe_inner_radius

    @property
    def jacket_outer_radius(self):
        return self.pipe_outer_radius + self.jacket_thickness

    @property
    def overall_diameter(self):
        return 2 * self.jacket_outer_radius

    @property
    def distance_from_center(self):
        return self.phase_spacing / np.sqrt(3)

    @property
    def n_conductors_per_phase(self):
        return 2 if self.has_screen else 1

    @property
    def total_conductors(self):
        n = 3 * self.n_conductors_per_phase
        if self.has_armor:
            n += 1
        return n

    def get_angular_positions(self):
        return [0.0, 2 * np.pi / 3, 4 * np.pi / 3]

    def to_pipe_type_cable(self):
        """Build the equivalent PT cable geometry."""
        angles = self.get_angular_positions()
        d = self.distance_from_center

        inner_conductors = []
        for i in range(3):
            ic = InnerConductor(
                core_radius=self.core_radius,
                core_rho=self.core_rho,
                core_mu_r=self.core_mu_r,
                insulation_radius=self.insulated_core_radius,
                insulation_epsilon_r=self.insulation_epsilon_r,
                insulation_tan_delta=self.insulation_tan_delta,
                insulation_mu_r=self.insulation_mu_r,
                has_sheath=self.has_screen,
                sheath_inner_radius=self.insulated_core_radius if self.has_screen else 0.0,
                sheath_outer_radius=self.screened_core_radius if self.has_screen else 0.0,
                sheath_rho=self.screen_rho,
                sheath_mu_r=self.screen_mu_r,
                outer_insulation_radius=self.inner_conductor_outer_radius if self.has_screen else 0.0,
                outer_insulation_epsilon_r=self.screen_insulation_epsilon_r,
                outer_insulation_tan_delta=0.0,
                distance_from_center=d,
                angular_position=angles[i],
            )
            inner_conductors.append(ic)

        return PipeTypeCableGeometry(
            inner_conductors=inner_conductors,
            pipe_inner_radius=self.pipe_inner_radius,
            pipe_outer_radius=self.pipe_outer_radius,
            pipe_rho=self.armor_rho,
            pipe_mu_r=self.armor_mu_r,
            jacket_radius=self.jacket_outer_radius,
            jacket_epsilon_r=self.jacket_epsilon_r,
            pipe_inner_insulation_epsilon_r=self.filler_epsilon_r,
            pipe_inner_insulation_tan_delta=0.0,
            pipe_outer_insulation_epsilon_r=self.jacket_epsilon_r,
            pipe_outer_insulation_tan_delta=self.jacket_tan_delta,
            burial_depth=self.burial_depth,
            horizontal_pos=self.horizontal_pos,
            finite_pipe_thickness=self.has_armor,
        )

### -------------------------------------------- Elementary impedances ---------------------------------------------- ###

# solid_conductor_impedance() subroutine.
def solid_conductor_impedance(freq, radius, rho, mu_r):
    """Internal impedance of a solid cylindrical conductor (Ametani eq. 1).

        Z_int = (m rho)/(2 pi r) * I0(mr)/I1(mr)

    with m = sqrt(j omega mu / rho). Low-frequency branch uses the DC
    resistance plus internal inductance MU_0 mu_r/(8 pi).
    """
    omega = omega_from_freq(freq)
    R_dc = rho / (np.pi * radius ** 2)

    m = np.sqrt(1j * omega * MU_0 * mu_r / rho)
    mr = m * radius

    Z_int = np.zeros_like(freq, dtype=complex)
    for i, (mr_val, om) in enumerate(zip(mr, omega)):
        if np.abs(mr_val) < 0.1:
            L_int = MU_0 * mu_r / (8 * np.pi)
            Z_int[i] = R_dc + 1j * om * L_int
        else:
            try:
                I0 = iv(0, mr_val)
                I1 = iv(1, mr_val)
                if np.abs(I1) > 1e-20 and not (np.isnan(I0) or np.isnan(I1)):
                    Z_int[i] = (m[i] * rho / (2 * np.pi * radius)) * (I0 / I1)
                else:
                    Z_int[i] = R_dc
            except Exception:
                Z_int[i] = R_dc
    return Z_int


# tubular_conductor_impedance() subroutine.
def tubular_conductor_impedance(freq, r_inner, r_outer, rho, mu_r):
    """Inner-surface, outer-surface and mutual impedances of a tube
    (Ametani eq. 11). Uses DC and skin-effect limits to avoid Bessel overflow.

    Returns the triple (Z_inner, Z_outer, Z_mutual).
    """
    omega = omega_from_freq(freq)
    m = np.sqrt(1j * omega * MU_0 * mu_r / rho)
    m_r_inner = m * r_inner
    m_r_outer = m * r_outer

    Z_inner = np.zeros_like(freq, dtype=complex)
    Z_outer = np.zeros_like(freq, dtype=complex)
    Z_mutual = np.zeros_like(freq, dtype=complex)

    R_dc = rho / (np.pi * (r_outer ** 2 - r_inner ** 2))

    for i, (mr1, mr2, om) in enumerate(zip(m_r_inner, m_r_outer, omega)):
        if np.abs(mr1) < 0.1 and np.abs(mr2) < 0.1:
            L_int = MU_0 * mu_r / (8 * np.pi)
            Z_inner[i] = R_dc + 1j * om * L_int
            Z_outer[i] = Z_inner[i]
            Z_mutual[i] = Z_inner[i]
        elif np.abs(mr1) > 70 or np.abs(mr2) > 70:
            delta = np.sqrt(2 * rho / (om * MU_0 * mu_r))
            Z_inner[i] = (1 + 1j) * rho / (2 * np.pi * r_inner * delta)
            Z_outer[i] = (1 + 1j) * rho / (2 * np.pi * r_outer * delta)
            Z_mutual[i] = 0.0
        else:
            try:
                I0_r1, I1_r1 = iv(0, mr1), iv(1, mr1)
                K0_r1, K1_r1 = kv(0, mr1), kv(1, mr1)
                I0_r2, I1_r2 = iv(0, mr2), iv(1, mr2)
                K0_r2, K1_r2 = kv(0, mr2), kv(1, mr2)

                vals = [I0_r1, I1_r1, K0_r1, K1_r1, I0_r2, I1_r2, K0_r2, K1_r2]
                if np.any(np.isnan(vals)) or np.any(np.isinf(vals)):
                    delta = np.sqrt(2 * rho / (om * MU_0 * mu_r))
                    Z_inner[i] = (1 + 1j) * rho / (2 * np.pi * r_inner * delta)
                    Z_outer[i] = (1 + 1j) * rho / (2 * np.pi * r_outer * delta)
                    Z_mutual[i] = 0.0
                    continue

                D = I1_r2 * K1_r1 - I1_r1 * K1_r2
                if np.abs(D) < 1e-15:
                    Z_inner[i] = Z_outer[i] = Z_mutual[i] = R_dc
                else:
                    Zc_inner = m[i] * rho / (2 * np.pi * r_inner)
                    Zc_outer = m[i] * rho / (2 * np.pi * r_outer)
                    Z_inner[i] = Zc_inner * (I0_r1 * K1_r2 + K0_r1 * I1_r2) / D
                    Z_outer[i] = Zc_outer * (I0_r2 * K1_r1 + K0_r2 * I1_r1) / D
                    Z_mutual[i] = rho / (2 * np.pi * r_inner * r_outer * D)
            except Exception:
                Z_inner[i] = Z_outer[i] = Z_mutual[i] = R_dc

    return Z_inner, Z_outer, Z_mutual


def insulation_impedance(freq, r_inner, r_outer, mu_r=1.0):
    """Series inductive impedance of an insulating layer.

        Z_ins = j omega mu_0 mu_r / (2 pi) * ln(r_outer / r_inner)
    """
    omega = omega_from_freq(freq)
    L_ins = (MU_0 * mu_r / (2 * np.pi)) * np.log(r_outer / r_inner)
    return 1j * omega * L_ins

### ---------------------------------------- Pollaczek earth-return impedance --------------------------------------- ###

def _pollaczek_integrand_self(lam, h, gamma):
    """*!{Integrand: exp(-2 h lambda) / (lambda + sqrt(lambda^2 + gamma^2))}*"""
    sqrt_term = np.sqrt(lam ** 2 + gamma ** 2)
    denom = lam + sqrt_term
    if np.abs(denom) < 1e-30:
        return 0.0 + 0.0j
    return np.exp(-2 * h * lam) / denom


def _pollaczek_integrand_self_real(lam, h, gamma):
    return np.real(_pollaczek_integrand_self(lam, h, gamma))


def _pollaczek_integrand_self_imag(lam, h, gamma):
    return np.imag(_pollaczek_integrand_self(lam, h, gamma))


def _pollaczek_integrand_mutual(lam, h_sum, x, gamma):
    """*!{Integrand: exp(-(h_sum) lambda) cos(lambda x) / (lambda + sqrt(lambda^2 + gamma^2))}*"""
    sqrt_term = np.sqrt(lam ** 2 + gamma ** 2)
    denom = lam + sqrt_term
    if np.abs(denom) < 1e-30:
        return 0.0 + 0.0j
    return np.exp(-h_sum * lam) * np.cos(lam * x) / denom


def _pollaczek_integrand_mutual_real(lam, h_sum, x, gamma):
    return np.real(_pollaczek_integrand_mutual(lam, h_sum, x, gamma))


def _pollaczek_integrand_mutual_imag(lam, h_sum, x, gamma):
    return np.imag(_pollaczek_integrand_mutual(lam, h_sum, x, gamma))


def _compute_pollaczek_integral_self(h, gamma, integration_limit=100.0):
    """Pollaczek self integral J = int_0^inf [exp(-2 h lam) / (lam + sqrt(lam^2 + gamma^2))] dlam.

    Integration upper bound is adapted to the decay factor exp(-2 h lam): for
    2 h lam > 50 the integrand is below 1e-22.
    """
    upper_limit = min(integration_limit, max(50.0 / (2 * h + 1e-10), 10.0))

    result_real, _ = quad(
        _pollaczek_integrand_self_real, 0, upper_limit,
        args=(h, gamma), limit=200, epsabs=1e-12, epsrel=1e-10,
    )
    result_imag, _ = quad(
        _pollaczek_integrand_self_imag, 0, upper_limit,
        args=(h, gamma), limit=200, epsabs=1e-12, epsrel=1e-10,
    )
    return result_real + 1j * result_imag


def _compute_pollaczek_integral_mutual(h_sum, x, gamma, integration_limit=100.0):
    """Pollaczek mutual integral
    J = int_0^inf [exp(-h_sum lam) cos(lam x) / (lam + sqrt(lam^2 + gamma^2))] dlam.

    The upper bound accounts for both the exponential decay and the oscillatory
    factor cos(lam x): at least 20 periods 2 pi/x are covered when x is finite.
    """
    upper_limit = min(integration_limit, max(50.0 / (h_sum + 1e-10), 10.0))
    if np.abs(x) > 0.01:
        upper_limit = min(integration_limit, max(upper_limit, 20 * np.pi / np.abs(x)))

    result_real, _ = quad(
        _pollaczek_integrand_mutual_real, 0, upper_limit,
        args=(h_sum, x, gamma), limit=500, epsabs=1e-12, epsrel=1e-10,
    )
    result_imag, _ = quad(
        _pollaczek_integrand_mutual_imag, 0, upper_limit,
        args=(h_sum, x, gamma), limit=500, epsabs=1e-12, epsrel=1e-10,
    )
    return result_real + 1j * result_imag


# earth_return_impedance_self() subroutine.
def earth_return_impedance_self(freq, burial_depth, outer_radius, gamma_soil):
    """Self earth-return impedance (Pollaczek exact).

        Z_e = (j omega mu_0 / 2 pi) ln(2 h / r)
              + (j omega mu_0 / pi) * J_self(h, gamma)
    """
    omega = omega_from_freq(freq)
    Z_earth = np.zeros_like(freq, dtype=complex)
    h, r = burial_depth, outer_radius

    for i, (om, gs) in enumerate(zip(omega, gamma_soil)):
        Z_geom = (1j * om * MU_0 / (2 * np.pi)) * np.log(2 * h / r)
        J = _compute_pollaczek_integral_self(h, gs)
        Z_earth[i] = Z_geom + (1j * om * MU_0 / np.pi) * J
    return Z_earth


# earth_return_impedance_mutual() subroutine.
def earth_return_impedance_mutual(freq, depth_i, depth_j, pos_i, pos_j, gamma_soil):
    """Mutual earth-return impedance between two buried conductors.

        Z_m = (j omega mu_0 / 2 pi) ln(D' / D)
              + (j omega mu_0 / pi) * J_mutual(h1+h2, |xi-xj|, gamma)

    with D the conductor-to-conductor distance and D' the conductor-to-image
    distance.
    """
    omega = omega_from_freq(freq)

    x_diff = pos_i - pos_j
    D_ij = np.sqrt(x_diff ** 2 + (depth_i - depth_j) ** 2)
    D_ij_image = np.sqrt(x_diff ** 2 + (depth_i + depth_j) ** 2)
    if D_ij < 1e-6:
        D_ij = 1e-6

    h_sum = depth_i + depth_j
    x = np.abs(x_diff)

    Z_earth_mutual = np.zeros_like(freq, dtype=complex)
    for i, (om, gs) in enumerate(zip(omega, gamma_soil)):
        Z_geom = (1j * om * MU_0 / (2 * np.pi)) * np.log(D_ij_image / D_ij)
        J = _compute_pollaczek_integral_mutual(h_sum, x, gs)
        Z_earth_mutual[i] = Z_geom + (1j * om * MU_0 / np.pi) * J
    return Z_earth_mutual

### ------------- High-precision Pollaczek via infinite-interval transform --------------- ###

def _pollaczek_integrand_transformed_real(t, h, gamma):
    """*!{lambda = tan(t) maps [0, inf) to [0, pi/2); dlambda = sec^2(t) dt}*"""
    if t >= np.pi / 2 - 1e-10:
        return 0.0
    lam = np.tan(t)
    sec2 = 1.0 / np.cos(t) ** 2
    sqrt_term = np.sqrt(lam ** 2 + gamma ** 2)
    denom = lam + sqrt_term
    if np.abs(denom) < 1e-30:
        return 0.0
    return np.real(np.exp(-2 * h * lam) * sec2 / denom)


def _pollaczek_integrand_transformed_imag(t, h, gamma):
    if t >= np.pi / 2 - 1e-10:
        return 0.0
    lam = np.tan(t)
    sec2 = 1.0 / np.cos(t) ** 2
    sqrt_term = np.sqrt(lam ** 2 + gamma ** 2)
    denom = lam + sqrt_term
    if np.abs(denom) < 1e-30:
        return 0.0
    return np.imag(np.exp(-2 * h * lam) * sec2 / denom)


def _compute_pollaczek_integral_self_transformed(h, gamma):
    """Pollaczek self integral via the substitution lambda = tan(t) to avoid
    the truncation error of a finite upper bound.
    """
    upper_limit = np.pi / 2 - 1e-8
    result_real, _ = quad(
        _pollaczek_integrand_transformed_real, 0, upper_limit,
        args=(h, gamma), limit=300, epsabs=1e-12, epsrel=1e-10,
    )
    result_imag, _ = quad(
        _pollaczek_integrand_transformed_imag, 0, upper_limit,
        args=(h, gamma), limit=300, epsabs=1e-12, epsrel=1e-10,
    )
    return result_real + 1j * result_imag


def earth_return_impedance_self_highprecision(freq, burial_depth, outer_radius, gamma_soil):
    """Self earth-return impedance using the substitution-based Pollaczek
    integral (removes the finite-upper-bound truncation error).
    """
    omega = omega_from_freq(freq)
    Z_earth = np.zeros_like(freq, dtype=complex)
    h, r = burial_depth, outer_radius

    for i, (om, gs) in enumerate(zip(omega, gamma_soil)):
        Z_geom = (1j * om * MU_0 / (2 * np.pi)) * np.log(2 * h / r)
        J = _compute_pollaczek_integral_self_transformed(h, gs)
        Z_earth[i] = Z_geom + (1j * om * MU_0 / np.pi) * J
    return Z_earth

### ------------------------------------------- SC cable (armored) ------------------------------------------------- ###

@dataclass
class ArmoredCableImpedanceResult:
    """Result of the 3x3 armored cable impedance computation."""
    Z_matrix: np.ndarray
    Z_cc: np.ndarray
    Z_ss: np.ndarray
    Z_aa: np.ndarray
    Z_cs: np.ndarray
    Z_ca: np.ndarray
    Z_sa: np.ndarray
    Z_earth: np.ndarray


# compute_armored_cable_impedance() subroutine.
def compute_armored_cable_impedance(freq, cable, gamma_soil):
    """Build the 3x3 series impedance matrix of a single armored cable
    (core, sheath, armor) using Ametani's loop-impedance assembly.
    """
    n_freq = len(freq)

    Z_11 = solid_conductor_impedance(freq, cable.core_radius,
                                     cable.core_rho, cable.core_mu_r)
    Z_12 = insulation_impedance(freq, cable.core_radius,
                                cable.insulation_radius, cable.insulation_mu_r)

    Z_2i, Z_20, Z_2m = tubular_conductor_impedance(
        freq, cable.sheath_inner_radius, cable.sheath_outer_radius,
        cable.sheath_rho, cable.sheath_mu_r)
    Z_23 = insulation_impedance(freq, cable.sheath_outer_radius,
                                cable.armor_inner_radius, cable.sheath_insulation_mu_r)

    Z_3i, Z_30, Z_3m = tubular_conductor_impedance(
        freq, cable.armor_inner_radius, cable.armor_outer_radius,
        cable.armor_rho, cable.armor_mu_r)
    Z_34 = insulation_impedance(freq, cable.armor_outer_radius,
                                cable.jacket_radius, cable.jacket_mu_r)

    Z_earth = earth_return_impedance_self(freq, cable.burial_depth,
                                          cable.jacket_radius, gamma_soil)

    Z_cs_composite = Z_11 + Z_12 + Z_2i
    Z_sa_composite = Z_20 + Z_23 + Z_3i
    Z_a4_composite = Z_30 + Z_34

    Z_cc = Z_cs_composite + Z_sa_composite + Z_a4_composite - 2 * Z_2m - 2 * Z_3m + Z_earth
    Z_ss = Z_sa_composite + Z_a4_composite - 2 * Z_3m + Z_earth
    Z_aa = Z_a4_composite + Z_earth
    Z_cs = Z_sa_composite + Z_a4_composite - Z_2m - 2 * Z_3m + Z_earth
    Z_ca = Z_a4_composite - Z_3m + Z_earth
    Z_sa = Z_ca.copy()

    Z_matrix = np.zeros((n_freq, 3, 3), dtype=complex)
    Z_matrix[:, 0, 0] = Z_cc
    Z_matrix[:, 1, 1] = Z_ss
    Z_matrix[:, 2, 2] = Z_aa
    Z_matrix[:, 0, 1] = Z_matrix[:, 1, 0] = Z_cs
    Z_matrix[:, 0, 2] = Z_matrix[:, 2, 0] = Z_ca
    Z_matrix[:, 1, 2] = Z_matrix[:, 2, 1] = Z_sa

    return ArmoredCableImpedanceResult(
        Z_matrix=Z_matrix,
        Z_cc=Z_cc, Z_ss=Z_ss, Z_aa=Z_aa,
        Z_cs=Z_cs, Z_ca=Z_ca, Z_sa=Z_sa,
        Z_earth=Z_earth,
    )

### -------------------------------------------------- PT cable ---------------------------------------------------- ###

# compute_pipe_internal_impedance() subroutine.
def compute_pipe_internal_impedance(freq, pt_cable):
    """Pipe internal impedance matrix [Zp] for a PT cable using the finite-
    thickness formulation (Hoidalen 2013 eqs. 8-10).

    - Inner-surface impedance Z_pi follows eq. (8), replacing eq. (2).
    - Proximity correction Z_pSigma follows eq. (9), replacing eq. (3).
    - Outer boundary-condition coefficients D_n, E_n follow eq. (10).

    The low-frequency limit (|x1|<0.05) switches to the DC fallback, the high-
    frequency limit (|x1|>70) uses the infinite-thickness expression, and
    Bessel overflow/NaN is caught and falls back to the infinite-thickness
    expression as well.
    """
    n_freq = len(freq)
    omega = omega_from_freq(freq)

    rp1 = pt_cable.pipe_inner_radius
    rp2 = pt_cable.pipe_outer_radius
    mu_p = pt_cable.pipe_mu_r
    rho_p = pt_cable.pipe_rho

    m_pipe = np.sqrt(1j * omega * MU_0 * mu_p / rho_p)
    x1 = rp1 * m_pipe
    x2 = rp2 * m_pipe

    R_dc_pipe = rho_p / (np.pi * (rp2 ** 2 - rp1 ** 2))

    total_inner_cond = sum(ic.n_conductors for ic in pt_cable.inner_conductors)
    Zp_matrix = np.zeros((n_freq, total_inner_cond, total_inner_cond), dtype=complex)

    for fi in range(n_freq):
        x1_val = x1[fi]
        x2_val = x2[fi]
        om = omega[fi]

        # Inner-surface impedance term bessel_term = (mu_p / x1) * numerator / denominator
        try:
            if np.abs(x1_val) < 0.1 and np.abs(x2_val) < 0.1:
                bessel_term = R_dc_pipe / (1j * om * MU_0 / (2 * np.pi))
            elif np.abs(x1_val) > 70:
                bessel_term = mu_p / x1_val
            else:
                I0_x1 = iv(0, x1_val); I1_x1 = iv(1, x1_val)
                K0_x1 = kv(0, x1_val); K1_x1 = kv(1, x1_val)
                I1_x2 = iv(1, x2_val); K1_x2 = kv(1, x2_val)

                vals = [I0_x1, I1_x1, K0_x1, K1_x1, I1_x2, K1_x2]
                if np.any(np.isnan(vals)) or np.any(np.isinf(vals)):
                    bessel_term = mu_p / x1_val
                else:
                    D0 = I1_x2 * K1_x1 - I1_x1 * K1_x2
                    if np.abs(D0) < 1e-30:
                        bessel_term = mu_p / x1_val
                    else:
                        num0 = I0_x1 * K1_x2 + K0_x1 * I1_x2
                        bessel_term = (mu_p / x1_val) * num0 / D0
        except Exception:
            bessel_term = mu_p / x1_val

        # Precompute Bessel orders for the proximity series
        use_finite_pipe_prox = True
        bessel_cache_x1 = {}
        bessel_cache_x2 = {}

        if np.abs(x1_val) < 0.05 or np.abs(x1_val) > 70:
            use_finite_pipe_prox = False

        if use_finite_pipe_prox:
            try:
                for n_order in range(0, 12):
                    bessel_cache_x1[n_order] = (iv(n_order, x1_val), kv(n_order, x1_val))
                    bessel_cache_x2[n_order] = (iv(n_order, x2_val), kv(n_order, x2_val))
                for n_order in range(0, 12):
                    vals_check = list(bessel_cache_x1[n_order]) + list(bessel_cache_x2[n_order])
                    if np.any(np.isnan(vals_check)) or np.any(np.isinf(vals_check)):
                        use_finite_pipe_prox = False
                        break
            except Exception:
                use_finite_pipe_prox = False

        idx_j = 0
        for j, ic_j in enumerate(pt_cable.inner_conductors):
            d_j = ic_j.distance_from_center
            r_j = ic_j.outer_radius
            theta_j = ic_j.angular_position
            n_cond_j = ic_j.n_conductors

            idx_k = 0
            for k, ic_k in enumerate(pt_cable.inner_conductors):
                d_k = ic_k.distance_from_center
                theta_k = ic_k.angular_position
                n_cond_k = ic_k.n_conductors
                theta_jk = theta_j - theta_k

                # Geometric term Q_jk (Ametani eqs. 4-6)
                if j == k:
                    Q_jk = np.log((rp1 / r_j) * (1 - (d_j / rp1) ** 2))
                else:
                    D_jk_sq = d_j ** 2 + d_k ** 2 - 2 * d_j * d_k * np.cos(theta_jk)
                    D_jk = np.sqrt(max(D_jk_sq, 1e-10))
                    series_sum = 0.0
                    for n in range(1, 21):
                        C_n = ((d_j * d_k / rp1 ** 2) ** n) * np.cos(n * theta_jk)
                        series_sum += C_n / n
                    Q_jk = np.log(rp1 / D_jk) - series_sum

                # Proximity (eddy) correction
                bessel_series = 0.0
                for n in range(1, 11):
                    C_n = ((d_j * d_k / rp1 ** 2) ** n) * np.cos(n * theta_jk)

                    if use_finite_pipe_prox:
                        try:
                            In_m1_x1, Kn_m1_x1 = bessel_cache_x1[n - 1]
                            In_x1, Kn_x1 = bessel_cache_x1[n]
                            In_m1_x2, Kn_m1_x2 = bessel_cache_x2[n - 1]
                            In_x2, Kn_x2 = bessel_cache_x2[n]

                            D_n = x2_val * In_m1_x2 - n * (1 + mu_p) * In_x2
                            E_n = x2_val * Kn_m1_x2 + n * (mu_p - 1) * Kn_x2

                            ratio_num = In_m1_x1 * E_n - Kn_m1_x1 * D_n
                            ratio_den = In_x1 * E_n - Kn_x1 * D_n

                            if np.abs(ratio_den) > 1e-30:
                                finite_ratio = x1_val * ratio_num / ratio_den
                                denom = n * (1 + mu_p) + finite_ratio
                            else:
                                Kn_1e = kve(n - 1, x1_val); Kne = kve(n, x1_val)
                                ratio_Kn = Kn_1e / Kne if np.abs(Kne) > 1e-30 else 1.0
                                denom = n * (1 + mu_p) + x1_val * ratio_Kn
                        except Exception:
                            try:
                                Kn_1e = kve(n - 1, x1_val); Kne = kve(n, x1_val)
                                ratio_Kn = Kn_1e / Kne if np.abs(Kne) > 1e-30 else 1.0
                                denom = n * (1 + mu_p) + x1_val * ratio_Kn
                            except Exception:
                                continue
                    else:
                        try:
                            Kn_1e = kve(n - 1, x1_val); Kne = kve(n, x1_val)
                            if np.abs(Kne) > 1e-30 and not (np.isnan(Kn_1e) or np.isnan(Kne)):
                                ratio_Kn = Kn_1e / Kne
                            else:
                                ratio_Kn = 1.0
                            denom = n * (1 + mu_p) + x1_val * ratio_Kn
                        except Exception:
                            continue

                    if np.abs(denom) > 1e-30:
                        bessel_series += 2 * mu_p * C_n / denom

                Zpjk = (1j * om * MU_0 / (2 * np.pi)) * (bessel_term + Q_jk + bessel_series)

                for ii in range(n_cond_j):
                    for jj in range(n_cond_k):
                        Zp_matrix[fi, idx_j + ii, idx_k + jj] = Zpjk

                idx_k += n_cond_k
            idx_j += n_cond_j

    return Zp_matrix


def compute_pipe_connection_impedance(freq, pt_cable):
    """Pipe connection impedances (Zc1, Zc2, Zc3) built from the tube
    impedances plus the jacket insulation.

    Reusing tubular_conductor_impedance() inherits its low/high/NaN guards
    and avoids the discontinuity caused by Bessel overflow at |x2|~700
    (roughly mu_r=200 steel pipe around 12 kHz).
    """
    rp2 = pt_cable.pipe_outer_radius
    rp3 = pt_cable.jacket_radius

    Z_inner, Z_outer, Z_mutual = tubular_conductor_impedance(
        freq, pt_cable.pipe_inner_radius, rp2, pt_cable.pipe_rho, pt_cable.pipe_mu_r)

    Zp3 = insulation_impedance(freq, rp2, rp3, mu_r=1.0)

    Zc3 = Z_outer + Zp3
    Zc2 = Zc3 - Z_mutual
    Zc1 = Zc3 - 2 * Z_mutual
    return Zc1, Zc2, Zc3


def compute_inner_conductor_impedance(freq, ic):
    """Internal impedance matrix [Zi] of a single inner conductor
    (core, or core + sheath/screen).
    """
    n_freq = len(freq)
    n_cond = ic.n_conductors
    Zi = np.zeros((n_freq, n_cond, n_cond), dtype=complex)

    Z_11 = solid_conductor_impedance(freq, ic.core_radius, ic.core_rho, ic.core_mu_r)
    Z_12 = insulation_impedance(freq, ic.core_radius, ic.insulation_radius, ic.insulation_mu_r)

    if ic.has_sheath:
        Z_2i, Z_20, Z_2m = tubular_conductor_impedance(
            freq, ic.sheath_inner_radius, ic.sheath_outer_radius,
            ic.sheath_rho, ic.sheath_mu_r)

        if ic.outer_insulation_radius > ic.sheath_outer_radius:
            Z_23 = insulation_impedance(freq, ic.sheath_outer_radius,
                                        ic.outer_insulation_radius, mu_r=1.0)
        else:
            Z_23 = np.zeros(n_freq, dtype=complex)

        Z_cs = Z_11 + Z_12 + Z_2i
        Z_s3 = Z_20 + Z_23

        Zi[:, 0, 0] = Z_cs + Z_s3 - 2 * Z_2m
        Zi[:, 1, 1] = Z_s3
        Zi[:, 0, 1] = Zi[:, 1, 0] = Z_s3 - Z_2m
    else:
        Zi[:, 0, 0] = Z_11 + Z_12
    return Zi


# compute_pipe_type_cable_impedance() subroutine.
def compute_pipe_type_cable_impedance(freq, pt_cable, gamma_soil):
    """Full PT cable impedance matrix: inner conductors + pipe internal
    + pipe connection + earth return.
    """
    n_freq = len(freq)
    n_total = pt_cable.total_conductors

    total_inner_cond = sum(ic.n_conductors for ic in pt_cable.inner_conductors)
    Z_matrix = np.zeros((n_freq, n_total, n_total), dtype=complex)

    idx = 0
    for ic in pt_cable.inner_conductors:
        Zi_single = compute_inner_conductor_impedance(freq, ic)
        n_cond = ic.n_conductors
        Z_matrix[:, idx:idx + n_cond, idx:idx + n_cond] = Zi_single
        idx += n_cond

    Zp_matrix = compute_pipe_internal_impedance(freq, pt_cable)
    Z_matrix[:, :total_inner_cond, :total_inner_cond] += Zp_matrix

    if pt_cable.finite_pipe_thickness:
        Zc1, Zc2, Zc3 = compute_pipe_connection_impedance(freq, pt_cable)

        for i in range(total_inner_cond):
            for j in range(total_inner_cond):
                Z_matrix[:, i, j] += Zc1

        pipe_idx = total_inner_cond
        for i in range(total_inner_cond):
            Z_matrix[:, i, pipe_idx] += Zc2
            Z_matrix[:, pipe_idx, i] += Zc2
        Z_matrix[:, pipe_idx, pipe_idx] += Zc3

        # High-precision Pollaczek on the outermost surface (jacket radius)
        Zo = earth_return_impedance_self_highprecision(
            freq, pt_cable.burial_depth, pt_cable.jacket_radius, gamma_soil)
        for i in range(n_total):
            for j in range(n_total):
                Z_matrix[:, i, j] += Zo

    return Z_matrix


def compute_pipe_type_cable_potential(freq, pt_cable):
    """Potential coefficient matrix [P] of a PT cable using complex
    permittivities eps = eps_r eps_0 (1 - j tan_delta) for each dielectric
    layer (consistent with the SC cable admittance computation).
    """
    n_total = pt_cable.total_conductors
    total_inner_cond = sum(ic.n_conductors for ic in pt_cable.inner_conductors)

    P_matrix = np.zeros((n_total, n_total), dtype=complex)

    idx = 0
    for ic in pt_cable.inner_conductors:
        eps_ins = ic.insulation_epsilon_r * (1 - 1j * ic.insulation_tan_delta)

        if ic.has_sheath:
            Pc = (1 / (2 * np.pi * EPSILON_0 * eps_ins)) * \
                 np.log(ic.insulation_radius / ic.core_radius)

            eps_out = ic.outer_insulation_epsilon_r * (1 - 1j * ic.outer_insulation_tan_delta)
            Ps = (1 / (2 * np.pi * EPSILON_0 * eps_out)) * \
                 np.log(ic.outer_insulation_radius / ic.sheath_outer_radius) \
                 if ic.outer_insulation_radius > ic.sheath_outer_radius else 0.0

            P_matrix[idx, idx] = Pc + Ps
            P_matrix[idx + 1, idx + 1] = Ps
            P_matrix[idx, idx + 1] = Ps
            P_matrix[idx + 1, idx] = Ps
            idx += 2
        else:
            Pc = (1 / (2 * np.pi * EPSILON_0 * eps_ins)) * \
                 np.log(ic.insulation_radius / ic.core_radius)
            P_matrix[idx, idx] = Pc
            idx += 1

    rp1 = pt_cable.pipe_inner_radius
    eps_p1 = pt_cable.pipe_inner_insulation_epsilon_r * \
             (1 - 1j * pt_cable.pipe_inner_insulation_tan_delta)

    idx_j = 0
    for j, ic_j in enumerate(pt_cable.inner_conductors):
        d_j = ic_j.distance_from_center
        r_j = ic_j.outer_radius
        theta_j = ic_j.angular_position
        n_cond_j = ic_j.n_conductors

        idx_k = 0
        for k, ic_k in enumerate(pt_cable.inner_conductors):
            d_k = ic_k.distance_from_center
            theta_k = ic_k.angular_position
            n_cond_k = ic_k.n_conductors
            theta_jk = theta_j - theta_k

            if j == k:
                Q_jk = np.log((rp1 / r_j) * (1 - (d_j / rp1) ** 2))
            else:
                D_jk_sq = d_j ** 2 + d_k ** 2 - 2 * d_j * d_k * np.cos(theta_jk)
                D_jk = np.sqrt(max(D_jk_sq, 1e-10))
                series_sum = 0.0
                for n in range(1, 21):
                    C_n = ((d_j * d_k / rp1 ** 2) ** n) * np.cos(n * theta_jk)
                    series_sum += C_n / n
                Q_jk = np.log(rp1 / D_jk) - series_sum

            Ppjk = Q_jk / (2 * np.pi * eps_p1 * EPSILON_0)

            for ii in range(n_cond_j):
                for jj in range(n_cond_k):
                    P_matrix[idx_j + ii, idx_k + jj] += Ppjk

            idx_k += n_cond_k
        idx_j += n_cond_j

    if pt_cable.finite_pipe_thickness:
        rp2 = pt_cable.pipe_outer_radius
        rp3 = pt_cable.jacket_radius
        eps_p2 = pt_cable.pipe_outer_insulation_epsilon_r * \
                 (1 - 1j * pt_cable.pipe_outer_insulation_tan_delta)

        Pc = (1 / (2 * np.pi * eps_p2 * EPSILON_0)) * np.log(rp3 / rp2)
        for i in range(n_total):
            for j in range(n_total):
                P_matrix[i, j] += Pc

    return P_matrix

### ------------------------------------------- Three-core cable --------------------------------------------------- ###

@dataclass
class ThreeCoreCableResult:
    """Result bundle for the three-core cable computation."""
    Z_matrix: np.ndarray
    P_matrix: np.ndarray
    Y_matrix: np.ndarray
    Z_phase: np.ndarray
    Z_0: np.ndarray
    Z_1: np.ndarray
    Z_2: np.ndarray
    freq: np.ndarray
    n_total: int
    n_per_phase: int
    has_armor: bool


# compute_three_core_cable_impedance() subroutine.
def compute_three_core_cable_impedance(freq, cable, gamma_soil):
    """Impedance and admittance of a three-core cable via the PT cable model.
    Phase matrix is extracted from core-conductor rows/columns and transformed
    to 012 sequence quantities.
    """
    n_freq = len(freq)
    omega = omega_from_freq(freq)

    pt_cable = cable.to_pipe_type_cable()

    Z_matrix = compute_pipe_type_cable_impedance(freq, pt_cable, gamma_soil)
    P_matrix = compute_pipe_type_cable_potential(freq, pt_cable)

    n_total = Z_matrix.shape[1]
    Y_matrix = np.zeros((n_freq, n_total, n_total), dtype=complex)
    try:
        P_inv = np.linalg.inv(P_matrix)
        for i, om in enumerate(omega):
            Y_matrix[i] = 1j * om * P_inv
    except np.linalg.LinAlgError:
        warnings.warn("cable_model::WARN::potential coefficient matrix is singular")

    n_per_phase = cable.n_conductors_per_phase
    Z_phase = np.zeros((n_freq, 3, 3), dtype=complex)
    for i in range(3):
        for j in range(3):
            Z_phase[:, i, j] = Z_matrix[:, i * n_per_phase, j * n_per_phase]

    a = np.exp(1j * 2 * np.pi / 3)
    T = np.array([[1, 1, 1], [1, a ** 2, a], [1, a, a ** 2]]) / 3
    T_inv = np.array([[1, 1, 1], [1, a, a ** 2], [1, a ** 2, a]])

    Z_0 = np.zeros(n_freq, dtype=complex)
    Z_1 = np.zeros(n_freq, dtype=complex)
    Z_2 = np.zeros(n_freq, dtype=complex)
    for i in range(n_freq):
        Z_seq = T @ Z_phase[i] @ T_inv
        Z_0[i], Z_1[i], Z_2[i] = Z_seq[0, 0], Z_seq[1, 1], Z_seq[2, 2]

    return ThreeCoreCableResult(
        Z_matrix=Z_matrix,
        P_matrix=P_matrix,
        Y_matrix=Y_matrix,
        Z_phase=Z_phase,
        Z_0=Z_0, Z_1=Z_1, Z_2=Z_2,
        freq=freq,
        n_total=n_total,
        n_per_phase=n_per_phase,
        has_armor=cable.has_armor,
    )

### --------------------------------- Multi-cable assembly and admittance ----------------------------------------- ###

# compute_multi_cable_impedance() subroutine.
def compute_multi_cable_impedance(freq, cables, gamma_soil):
    """Assemble the n x n series impedance matrix of a multi-cable system.
    Diagonal 3x3 blocks come from compute_armored_cable_impedance() and off-
    diagonal blocks are Pollaczek mutual impedances broadcast to full blocks.
    """
    n_freq = len(freq)
    n_cables = len(cables)

    cond_per_cable = [3] * n_cables
    n_total = sum(cond_per_cable)
    Z_matrix = np.zeros((n_freq, n_total, n_total), dtype=complex)

    idx = 0
    for cable in cables:
        result = compute_armored_cable_impedance(freq, cable, gamma_soil)
        Z_matrix[:, idx:idx + 3, idx:idx + 3] = result.Z_matrix
        idx += 3

    idx_i = 0
    for i, cable_i in enumerate(cables):
        n_cond_i = cond_per_cable[i]
        idx_j = 0
        for j, cable_j in enumerate(cables):
            n_cond_j = cond_per_cable[j]
            if i != j:
                Z_mutual = earth_return_impedance_mutual(
                    freq, cable_i.burial_depth, cable_j.burial_depth,
                    cable_i.horizontal_pos, cable_j.horizontal_pos, gamma_soil)
                for ii in range(n_cond_i):
                    for jj in range(n_cond_j):
                        Z_matrix[:, idx_i + ii, idx_j + jj] = Z_mutual
            idx_j += n_cond_j
        idx_i += n_cond_i

    return Z_matrix


def compute_sequence_impedance(Z_phase):
    """Transform a 3x3 phase impedance into zero/positive/negative-sequence
    scalars using the Fortescue transformation.

    Returns the triple (Z_0, Z_1, Z_2).
    """
    n_freq = Z_phase.shape[0]
    a = np.exp(1j * 2 * np.pi / 3)
    T = np.array([[1, 1, 1], [1, a ** 2, a], [1, a, a ** 2]]) / 3
    T_inv = np.array([[1, 1, 1], [1, a, a ** 2], [1, a ** 2, a]])

    Z_0 = np.zeros(n_freq, dtype=complex)
    Z_1 = np.zeros(n_freq, dtype=complex)
    Z_2 = np.zeros(n_freq, dtype=complex)
    for i in range(n_freq):
        Z_seq = T @ Z_phase[i] @ T_inv
        Z_0[i], Z_1[i], Z_2[i] = Z_seq[0, 0], Z_seq[1, 1], Z_seq[2, 2]
    return Z_0, Z_1, Z_2


def compute_armored_cable_admittance(freq, cable):
    """Nodal admittance matrix [Y] of an armored SC cable in
    core / sheath / armor coordinates.
    """
    omega = omega_from_freq(freq)
    n_freq = len(freq)

    eps_1 = cable.insulation_epsilon_r * EPSILON_0 * (1 - 1j * cable.insulation_tan_delta)
    eps_2 = cable.sheath_insulation_epsilon_r * EPSILON_0
    eps_3 = cable.jacket_epsilon_r * EPSILON_0 * (1 - 1j * cable.jacket_tan_delta)

    Y_cs = 1j * omega * 2 * np.pi * eps_1 / np.log(cable.insulation_radius / cable.core_radius)
    Y_sa = 1j * omega * 2 * np.pi * eps_2 / np.log(cable.armor_inner_radius / cable.sheath_outer_radius)
    Y_a4 = 1j * omega * 2 * np.pi * eps_3 / np.log(cable.jacket_radius / cable.armor_outer_radius)

    Y_matrix = np.zeros((n_freq, 3, 3), dtype=complex)
    Y_matrix[:, 0, 0] = Y_cs
    Y_matrix[:, 0, 1] = Y_matrix[:, 1, 0] = -Y_cs
    Y_matrix[:, 1, 1] = Y_cs + Y_sa
    Y_matrix[:, 1, 2] = Y_matrix[:, 2, 1] = -Y_sa
    Y_matrix[:, 2, 2] = Y_sa + Y_a4
    return Y_matrix
