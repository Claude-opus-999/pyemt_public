# ulm_ohl_calculation_deri_semlyen.py module.

"""
*** ULM parameter computation for an overhead line (Deri-Semlyen) ***

Pipeline:
    PSCAD line parameters -> Z/Y matrices -> (optional) Kron reduction ->
    Vector Fitting -> ULM parameters -> (optional) fitULM export.

* Methods:
    - Series impedance [Z]: Deri-Semlyen complex-depth formula with Velasco
      internal-impedance approximation for bundle conductors.
    - Shunt admittance [Y]: classical potential-coefficient method (ideal
      image conductors, zero soil conductance).

* Exposed API:
    - PSCADLineConfig, CalculationConfig     : input parameter data classes
    - create_ohl_geometry_from_pscad()       : line geometry construction
    - compute_ZY_matrices()                  : Z and Y from configuration
    - eliminate_ground_wires()               : Kron reduction (Z_pp -
                                                Z_pg Z_gg^-1 Z_gp)
    - compute_ulm_parameters()               : iterative VF + ULM assembly
    - export_fitulm_file()                   : fitULM writer wrapper

* Required external modules (imported lazily):
    - ulm_atp_zy_deri_semlyen       : Z/Y computation engine
    - vector_fitting_v48_nr_consistent : VF + ULM engine
"""

# Scientific computing modules:
import numpy as np
from dataclasses import dataclass, field
from typing import Tuple, List, Optional, Any
from pathlib import Path
import os
import sys

PROJECT_DIR = Path(__file__).resolve().parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

### -------------------------------------------------- Constants ----------------------------------------------------- ###

C_LIGHT = 299792458.0
MU_0 = 4 * np.pi * 1e-7
EPSILON_0 = 8.854187817e-12

### ------------------------------------------ Configuration data classes ------------------------------------------- ###

@dataclass
class PSCADLineConfig:
    """PSCAD-equivalent line configuration (geometry + materials + soil)."""
    line_name: str = "TLine_2"
    n_conductors: int = 4
    n_phases: int = 2
    n_ground_wires: int = 2

    # Frequency sweep (log-spaced, total points = n_freq_increments + 1)
    freq_start: float = 0.01
    freq_end: float = 100000.0
    n_freq_increments: int = 200

    # Fitting parameters
    max_poles_Yc: int = 20
    max_poles_H: int = 20
    max_error_Yc: float = 0.02
    max_error_H: float = 0.02

    # Soil (constant resistivity)
    ground_resistivity: float = 1000.0
    ground_permeability: float = 1.0
    ground_permittivity: float = 1.0

    # Phase conductors (bundled)
    phase_radius: float = 0.03
    phase_strand_radius: float = 0.0036
    phase_n_outer_strands: int = 48
    phase_n_total_strands: int = 55
    phase_dc_resistance: float = 0.05741
    phase_mu_r: float = 1.0
    phase_bundle_n: int = 4
    phase_bundle_spacing: float = 0.5

    phase_positions: List[Tuple[float, float]] = field(default_factory=lambda: [
        (-11.777, 48.0 - 9.2 * 2 / 3),
        ( 11.777, 48.0 - 9.2 * 2 / 3),
    ])
    phase_sag: float = 9.2

    # Ground wires
    gw_radius: float = 0.00875
    gw_dc_resistance: float = 0.7098
    gw_mu_r: float = 1.0

    gw_positions: List[Tuple[float, float]] = field(default_factory=lambda: [
        (-19.25, 63.0 - 6.1 * 2 / 3),
        ( 19.25, 63.0 - 6.1 * 2 / 3),
    ])
    gw_sag: float = 6.1

    eliminate_ground_wires: bool = False

    def get_average_height(self, tower_height, sag):
        """Average conductor height accounting for sag (2/3 rule)."""
        return tower_height - sag * 2 / 3


@dataclass
class CalculationConfig:
    """High-level calculation configuration."""
    pscad_config: PSCADLineConfig = field(default_factory=PSCADLineConfig)

    line_length: float = 20000.0

    n_freq_increments: int = 200
    freq_min: float = 0.01
    freq_max: float = 1e5

    Yc_poles_min: int = 6
    Yc_poles_max: int = 20
    Yc_target_error: float = 0.002
    H_poles_min: int = 8
    H_poles_max: int = 20
    H_target_error: float = 0.002

    export_fitulm: bool = True
    fitulm_filename: str = "../emtp_simulator_v3_5/ohl_model.fitULM"
    fitulm_precision: int = 16

### ----------------------------------------- Module-level dependency loading --------------------------------------- ###

def _load_zy_module():
    """Import the Z/Y engine. Raises ImportError with a diagnostic message."""
    try:
        import ulm_atp_zy_deri_semlyen as zy_module
        return zy_module
    except ImportError as exc:
        raise ImportError(
            "ulm_ohl::ERROR::cannot import ulm_atp_zy_deri_semlyen") from exc


def _load_vf_module():
    """Import the Vector Fitting / ULM engine."""
    try:
        import vector_fitting_v48_nr_consistent as vf
        return vf
    except ImportError as exc:
        raise ImportError(
            "ulm_ohl::ERROR::cannot import vector_fitting_v48_nr_consistent") from exc

### ----------------------------------------------- Line geometry --------------------------------------------------- ###

# create_ohl_geometry_from_pscad() subroutine.
def create_ohl_geometry_from_pscad(zy_module, pscad_config):
    """Build a MultiConductorLine object from a PSCADLineConfig.

    Conductors are ordered phases-first, then ground wires. DC resistance is
    converted from Ohm/km to Ohm/m. Average conductor heights are corrected
    for sag by the 2/3 rule.
    """
    conductors = []
    names = []
    is_ground_wire = []

    for i, (x, h_tower) in enumerate(pscad_config.phase_positions):
        h_avg = pscad_config.get_average_height(
            h_tower + pscad_config.phase_sag * 2 / 3, pscad_config.phase_sag)
        rdc = pscad_config.phase_dc_resistance / 1000.0

        cond = zy_module.ConductorGeometry(
            height=h_avg,
            horizontal_pos=x,
            radius=pscad_config.phase_radius,
            rdc=rdc,
            mu_r=pscad_config.phase_mu_r,
            bundle_n=pscad_config.phase_bundle_n,
            bundle_spacing=pscad_config.phase_bundle_spacing,
        )
        conductors.append(cond)
        names.append(f"Phase_{i + 1}")
        is_ground_wire.append(False)

    for i, (x, h_tower) in enumerate(pscad_config.gw_positions):
        h_avg = pscad_config.get_average_height(
            h_tower + pscad_config.gw_sag * 2 / 3, pscad_config.gw_sag)
        rdc = pscad_config.gw_dc_resistance / 1000.0

        cond = zy_module.ConductorGeometry(
            height=h_avg,
            horizontal_pos=x,
            radius=pscad_config.gw_radius,
            rdc=rdc,
            mu_r=pscad_config.gw_mu_r,
            bundle_n=1,
            bundle_spacing=0.0,
        )
        conductors.append(cond)
        names.append(f"GW_{i + 1}")
        is_ground_wire.append(True)

    return zy_module.MultiConductorLine(
        conductors=conductors,
        names=names,
        is_ground_wire=is_ground_wire,
    )

### --------------------------------------------- Z/Y matrix computation -------------------------------------------- ###

# compute_ZY_matrices() subroutine.
def compute_ZY_matrices(config, zy_module=None):
    """Compute series impedance Z and shunt admittance Y on a log-spaced
    frequency sweep driven by config.

    Returns the tuple (freq, Z_matrix, Y_matrix, line).
    """
    if zy_module is None:
        zy_module = _load_zy_module()

    pscad = config.pscad_config

    freq = np.logspace(
        np.log10(config.freq_min),
        np.log10(config.freq_max),
        config.n_freq_increments + 1,
    )

    soil = zy_module.get_constant_soil_params(
        freq, pscad.ground_resistivity,
        epsilon_r=pscad.ground_permittivity,
    )

    line = create_ohl_geometry_from_pscad(zy_module, pscad)

    Z_result = zy_module.compute_impedance_matrix(freq, line, soil.p_complex, verbose=False)
    Y_result = zy_module.compute_admittance_matrix(freq, line, verbose=False)

    return freq, Z_result.Z_matrix, Y_result.Y_matrix, line

### ---------------------------------------------- Kron reduction --------------------------------------------------- ###

# eliminate_ground_wires() subroutine.
def eliminate_ground_wires(Z_matrix, Y_matrix, line):
    """Kron reduction: eliminate ground-wire rows/columns by block inversion.

        Z = [[Z_pp, Z_pg], [Z_gp, Z_gg]]
        Z_reduced = Z_pp - Z_pg Z_gg^-1 Z_gp    (same form for Y)

    Falls back to pseudo-inverse when Z_gg is singular.

    Returns the triple (Z_reduced, Y_reduced, phase_names).
    """
    n_total = line.n_conductors
    n_phase = line.n_phases
    n_gw = n_total - n_phase

    if n_gw == 0:
        return Z_matrix, Y_matrix, line.names

    phase_idx = [i for i, is_gw in enumerate(line.is_ground_wire) if not is_gw]
    gw_idx = [i for i, is_gw in enumerate(line.is_ground_wire) if is_gw]

    K = Z_matrix.shape[0]
    Z_reduced = np.zeros((K, n_phase, n_phase), dtype=complex)
    Y_reduced = np.zeros((K, n_phase, n_phase), dtype=complex)

    for k in range(K):
        Z_pp = Z_matrix[k][np.ix_(phase_idx, phase_idx)]
        Z_pg = Z_matrix[k][np.ix_(phase_idx, gw_idx)]
        Z_gp = Z_matrix[k][np.ix_(gw_idx, phase_idx)]
        Z_gg = Z_matrix[k][np.ix_(gw_idx, gw_idx)]

        Y_pp = Y_matrix[k][np.ix_(phase_idx, phase_idx)]
        Y_pg = Y_matrix[k][np.ix_(phase_idx, gw_idx)]
        Y_gp = Y_matrix[k][np.ix_(gw_idx, phase_idx)]
        Y_gg = Y_matrix[k][np.ix_(gw_idx, gw_idx)]

        try:
            Z_gg_inv = np.linalg.inv(Z_gg)
            Y_gg_inv = np.linalg.inv(Y_gg)
        except np.linalg.LinAlgError:
            Z_gg_inv = np.linalg.pinv(Z_gg)
            Y_gg_inv = np.linalg.pinv(Y_gg)

        Z_reduced[k] = Z_pp - Z_pg @ Z_gg_inv @ Z_gp
        Y_reduced[k] = Y_pp - Y_pg @ Y_gg_inv @ Y_gp

    phase_names = [line.names[i] for i in phase_idx]
    return Z_reduced, Y_reduced, phase_names

### ------------------------------------------- Vector Fitting wrapper ---------------------------------------------- ###

# compute_ulm_parameters() subroutine.
def compute_ulm_parameters(freq, Z_matrix, Y_matrix, config, vf_module=None):
    """Run iterative Vector Fitting and assemble ULM parameters.

    Arguments.

     - freq      : frequency vector [Hz]
     - Z_matrix  : series impedance matrix, shape [K, nf, nf]
     - Y_matrix  : shunt admittance matrix, shape [K, nf, nf]
     - config    : CalculationConfig controlling VF thresholds and length
     - vf_module : optional pre-imported VF engine; loaded on demand otherwise

    Returns the tuple (ulm_params, fitting_result).
    """
    if vf_module is None:
        vf_module = _load_vf_module()

    vf_config = vf_module.IterativePoleFindingConfig(
        Ymin=config.Yc_poles_min,
        Ymax=config.Yc_poles_max,
        epsY=config.Yc_target_error,
        Hmin=config.H_poles_min,
        Hmax=config.H_poles_max,
        epsH=config.H_target_error,
        pole_step=2,
        eps_deg=10.0,
        use_pscad_style=True,
        compute_H_reconstruction_metrics=True,
        verbose_H_metrics=False,
    )

    ulm_params, fitting_result = vf_module.ulm_complete_fitting(
        freq=freq,
        Z_matrix=Z_matrix,
        Y_matrix=Y_matrix,
        length=config.line_length,
        velocity_freq=1e5,
        config=vf_config,
        use_freq_dependent='auto',
        enforce_passivity_flag=True,
        verbose=False,
    )
    return ulm_params, fitting_result

### ---------------------------------------------- fitULM export ---------------------------------------------------- ###

def export_fitulm_file(ulm_params, fitting_result, config, vf_module=None):
    """Write the ULM fitting result to a fitULM file.

    Returns the output path on success, None when export is disabled or the
    VF engine does not implement write_fitULM. Raises the underlying IO error
    unchanged when writing fails.
    """
    if not config.export_fitulm:
        return None

    if vf_module is None:
        vf_module = _load_vf_module()

    if not hasattr(vf_module, 'write_fitULM'):
        return None

    fitulm_path = os.path.join(PROJECT_DIR, config.fitulm_filename)

    vf_module.write_fitULM(
        result=fitting_result,
        filepath=fitulm_path,
        precision=config.fitulm_precision,
        verbose=False,
    )

    if hasattr(vf_module, 'verify_fitULM_file'):
        vf_module.verify_fitULM_file(fitulm_path, verbose=False)

    return fitulm_path
