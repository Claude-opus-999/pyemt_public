# -*- coding: utf-8 -*-
"""
Three-phase Y-Delta UMEC transformer lightning impulse simulation.

Adapted for emtp_solver_v3.

Main compatibility changes
--------------------------
1. Import EMTPSolver from emtp_solver_v3 instead of emtp_solver_v2_time.
2. Remove dependency on the old lightning_waveform module.
3. Use EMTPSolver.add_standard_twoexpf_IS() for the ATP-compatible 2/20 us
   lightning current source.
4. Use explicit result units with the latest API:
   get_time(unit="us"), get_node_voltage(node, unit="V").
5. Add batch-friendly plotting controls for script/server execution.

Original physical model retained
--------------------------------
- Transformer: 8.2 MVA, 0.69 kV Y_gnd / 35 kV Delta UMEC three-phase bank.
- Lightning current source injects into primary phase A node.
- Parasitic capacitances:
    C_low_high = 0.00665 uF: low-voltage terminal to high-voltage terminal
    C_low_gnd  = 0.00260 uF: low-voltage side to neutral/ground node
    C_high_gnd = 0.00255 uF: high-voltage side to neutral/ground node
- Delta-side loads: node 4/5/6 to ground through 100 ohm.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Dict, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np


# ---------------------------------------------------------------------------
# Local import path setup
# ---------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent

for _path in (SCRIPT_DIR, PROJECT_DIR, Path.cwd()):
    _path_str = str(_path)
    if _path_str not in sys.path:
        sys.path.insert(0, _path_str)

from emtp_solver_v3 import EMTPSolver  # noqa: E402
from umec_transformer import UMECTransformerData  # noqa: E402


# ---------------------------------------------------------------------------
# Plot / output controls
# ---------------------------------------------------------------------------

RESULT_DIR = "."
SAVE_PLOTS = True
SHOW_PLOTS = True


# ---------------------------------------------------------------------------
# Simulation configuration
# ---------------------------------------------------------------------------

DT = 1e-8
T_END = 20e-6
FREQ = 50.0

LIGHTNING_TYPE = "2/20"
LIGHTNING_PEAK = 10e3
LIGHTNING_TSTART = 0.0
LIGHTNING_PERC = 30

S_RATED = 8.2e6
V1_LL = 690.0
V2_LL = 35000.0

# Parasitic capacitances from the original case description.
C_LOW_HIGH = 0.00665e-6
C_LOW_GND = 0.00260e-6
C_HIGH_GND = 0.00255e-6

R_LOAD = 100.0
R_NEUTRAL = 0.2
R_LIGHTNING = 800.0


def ensure_output_dir(path_like: str | os.PathLike = RESULT_DIR) -> Path:
    path = Path(path_like)
    path.mkdir(parents=True, exist_ok=True)
    return path


def create_transformer_data() -> UMECTransformerData:
    """Create the three-phase-bank UMEC transformer data object."""
    return UMECTransformerData(
        name="T_bank_lightning",
        S_rated=S_RATED,
        freq=FREQ,
        V_rated_LL=[V1_LL, V2_LL],
        winding_types=["Y_gnd", "Delta"],
        X_leak_pu=0.083,
        Im_percent=1.0,
        NLL_pu=0.0005488,
        CL_pu=0.0085,
        enable_saturation=False,
        nodes=[
            [(1, 7), (4, 5)],   # Phase A: primary 1-neutral, secondary delta 4-5
            [(2, 7), (5, 6)],   # Phase B: primary 2-neutral, secondary delta 5-6
            [(3, 7), (6, 4)],   # Phase C: primary 3-neutral, secondary delta 6-4
        ],
    )


def build_solver(verbose: bool = False) -> Tuple[EMTPSolver, object]:
    """Build the latest EMTPSolver model and return (solver, lightning_source)."""
    solver = EMTPSolver(
        dt=DT,
        finish_time=T_END,
        verbose=verbose,

        # This case extracts all node voltages after run().
        record_all_node_voltages=True,

        # Source history is only needed if you call get_source_current().
        record_source_history=True,

        # Branch histories are not needed for the normal output path.
        record_branch_history=False,

        # There are no line models in this case.
        record_line_history=False,

        # Latest solver fast paths.
        pre_sample_sources=True,
        use_rhs_plan=True,
    )

    # Lightning current source: positive current flows from node_from to node_to,
    # so 0 -> 1 injects lightning current into primary-side phase A.
    lightning_source = solver.add_standard_twoexpf_IS(
        name="I_lightning",
        node_from=0,
        node_to=1,
        waveform_type=LIGHTNING_TYPE,
        peak=LIGHTNING_PEAK,
        PERC=LIGHTNING_PERC,
        Tstart=LIGHTNING_TSTART,
        description="2/20 us 10 kA lightning impulse injected into primary phase A",
    )

    solver.add_R("R_lightning", 0, 1, R_LIGHTNING)

    # UMEC transformer.
    solver.add_UMEC_transformer("T1", create_transformer_data())

    # Low-voltage side to neutral/ground node.
    solver.add_C("C_low_A", 1, 7, C_LOW_GND)
    solver.add_C("C_low_B", 2, 7, C_LOW_GND)
    solver.add_C("C_low_C", 3, 7, C_LOW_GND)

    # Low-high inter-winding coupling capacitances.
    solver.add_C("C_lh_A", 1, 4, C_LOW_HIGH)
    solver.add_C("C_lh_B", 2, 5, C_LOW_HIGH)
    solver.add_C("C_lh_C", 3, 6, C_LOW_HIGH)

    # High-voltage side to neutral/ground node.
    solver.add_C("C_high_A", 4, 7, C_HIGH_GND)
    solver.add_C("C_high_B", 5, 7, C_HIGH_GND)
    solver.add_C("C_high_C", 6, 7, C_HIGH_GND)

    # Delta-side loads to ground.
    solver.add_R("R_load_A", 4, 0, R_LOAD)
    solver.add_R("R_load_B", 5, 0, R_LOAD)
    solver.add_R("R_load_C", 6, 0, R_LOAD)

    # Primary neutral grounding resistor.
    solver.add_R("R_neutral", 7, 0, R_NEUTRAL)

    return solver, lightning_source


def extract_results(solver: EMTPSolver) -> Dict[str, np.ndarray]:
    """Extract simulation waveforms."""
    t_us = solver.get_time(unit="us")

    results = {
        "t_us": t_us,
        "I_lightning_A": solver.get_source_current("I_lightning"),
        "V1": solver.get_node_voltage(1, unit="V"),
        "V2": solver.get_node_voltage(2, unit="V"),
        "V3": solver.get_node_voltage(3, unit="V"),
        "V4": solver.get_node_voltage(4, unit="V"),
        "V5": solver.get_node_voltage(5, unit="V"),
        "V6": solver.get_node_voltage(6, unit="V"),
        "V7": solver.get_node_voltage(7, unit="V"),
    }

    results["V_AB"] = results["V4"] - results["V5"]
    results["V_BC"] = results["V5"] - results["V6"]
    results["V_CA"] = results["V6"] - results["V4"]
    results["I_load_A"] = results["V4"] / R_LOAD

    return results


def calculate_metrics(r: Dict[str, np.ndarray]) -> Dict[str, float]:
    """Calculate peak values and key checks."""
    metrics = {
        "I_lightning_peak_A": float(np.max(np.abs(r["I_lightning_A"]))),
        "V1_peak_V": float(np.max(np.abs(r["V1"]))),
        "V2_peak_V": float(np.max(np.abs(r["V2"]))),
        "V3_peak_V": float(np.max(np.abs(r["V3"]))),
        "V4_peak_V": float(np.max(np.abs(r["V4"]))),
        "V5_peak_V": float(np.max(np.abs(r["V5"]))),
        "V6_peak_V": float(np.max(np.abs(r["V6"]))),
        "V_AB_peak_V": float(np.max(np.abs(r["V_AB"]))),
        "I_load_A_peak_A": float(np.max(np.abs(r["I_load_A"]))),
    }

    metrics["t_V1_peak_us"] = float(r["t_us"][int(np.argmax(np.abs(r["V1"])))])

    metrics["no_nan"] = bool(
        not any(np.any(np.isnan(r[key])) for key in ["V1", "V2", "V3", "V4", "V5", "V6"])
    )
    metrics["has_primary_surge"] = bool(metrics["V1_peak_V"] > 1e3)
    metrics["has_secondary_transfer"] = bool(metrics["V4_peak_V"] > 100.0)
    metrics["passed"] = bool(
        metrics["no_nan"]
        and metrics["has_primary_surge"]
        and metrics["has_secondary_transfer"]
    )

    return metrics


def print_header() -> None:
    print("=" * 70)
    print("Three-phase Y-Delta UMEC transformer lightning impulse simulation")
    print(f"  Lightning current: {LIGHTNING_TYPE} us, {LIGHTNING_PEAK / 1e3:.1f} kA")
    print(f"  Transformer: {S_RATED / 1e6:.1f} MVA, {V1_LL/1e3:.2f} kV Y_gnd / {V2_LL/1e3:.1f} kV Delta")
    print("=" * 70)


def print_results(metrics: Dict[str, float], solver: EMTPSolver) -> None:
    stats = solver.get_solver_statistics()

    print("\n" + "=" * 70)
    print("Simulation results")
    print("=" * 70)

    print(f"\n  Time step: {DT * 1e9:.1f} ns")
    print(f"  Total steps: {stats.get('total_steps')}")
    print(f"  MNA size: {stats.get('mna_size')}")
    print(f"  G rebuilds: {stats.get('G_rebuilds')}")
    print(f"  G cache hits: {stats.get('G_cache_hits')}")

    print(
        f"\n  Parasitic capacitances: "
        f"C_low_high={C_LOW_HIGH * 1e6:.5f} uF, "
        f"C_low_gnd={C_LOW_GND * 1e6:.5f} uF, "
        f"C_high_gnd={C_HIGH_GND * 1e6:.5f} uF"
    )

    print("\n  Primary side, 0.69 kV Y_gnd:")
    print(f"    V_A, struck node 1 peak: {metrics['V1_peak_V'] / 1e3:.2f} kV @ t = {metrics['t_V1_peak_us']:.2f} us")
    print(f"    V_B, node 2 peak:       {metrics['V2_peak_V'] / 1e3:.2f} kV")
    print(f"    V_C, node 3 peak:       {metrics['V3_peak_V'] / 1e3:.2f} kV")

    print("\n  Secondary side, 35 kV Delta:")
    print(f"    V_4 to ground peak:     {metrics['V4_peak_V'] / 1e3:.2f} kV")
    print(f"    V_5 to ground peak:     {metrics['V5_peak_V'] / 1e3:.2f} kV")
    print(f"    V_6 to ground peak:     {metrics['V6_peak_V'] / 1e3:.2f} kV")
    print(f"    V_AB line voltage peak: {metrics['V_AB_peak_V'] / 1e3:.2f} kV")
    print(f"    Load current A peak:    {metrics['I_load_A_peak_A']:.2f} A")

    print("\n  Verification:")
    print(f"    Numerical stability, no NaN:      {'PASS' if metrics['no_nan'] else 'FAIL'}")
    print(f"    Primary overvoltage > 1 kV:       {'PASS' if metrics['has_primary_surge'] else 'FAIL'} ({metrics['V1_peak_V'] / 1e3:.1f} kV)")
    print(f"    Secondary transferred > 0.1 kV:   {'PASS' if metrics['has_secondary_transfer'] else 'FAIL'} ({metrics['V4_peak_V'] / 1e3:.1f} kV)")

    print("\n  Result:", "PASS" if metrics["passed"] else "FAIL")
    print("=" * 70)


def plot_primary_side(r: Dict[str, np.ndarray], save_dir: str | os.PathLike | None = None) -> Path:
    if save_dir is None:
        save_dir = RESULT_DIR
    save_path = ensure_output_dir(save_dir) / "primary_side_voltages_latest.png"

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(r["t_us"], r["V1"] / 1e3, label="Node 1 - Phase A, struck", linewidth=1.2)
    ax.plot(r["t_us"], r["V2"] / 1e3, label="Node 2 - Phase B", linewidth=1.2)
    ax.plot(r["t_us"], r["V3"] / 1e3, label="Node 3 - Phase C", linewidth=1.2)

    ax.set_xlabel("Time (us)")
    ax.set_ylabel("Voltage (kV)")
    ax.set_title("Primary Side Node Voltages, 0.69 kV Y_gnd")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()

    if SAVE_PLOTS:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"\n  Saved plot: {save_path}")

    if SHOW_PLOTS:
        plt.show()
    else:
        plt.close(fig)

    return save_path


def plot_secondary_side(r: Dict[str, np.ndarray], save_dir: str | os.PathLike | None = None) -> Path:
    if save_dir is None:
        save_dir = RESULT_DIR
    save_path = ensure_output_dir(save_dir) / "secondary_side_voltages_latest.png"

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(r["t_us"], r["V4"] / 1e3, label="Node 4 - Phase A", linewidth=1.2)
    ax.plot(r["t_us"], r["V5"] / 1e3, label="Node 5 - Phase B", linewidth=1.2)
    ax.plot(r["t_us"], r["V6"] / 1e3, label="Node 6 - Phase C", linewidth=1.2)

    ax.set_xlabel("Time (us)")
    ax.set_ylabel("Voltage (kV)")
    ax.set_title("Secondary Side Node Voltages, 35 kV Delta")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()

    if SAVE_PLOTS:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved plot: {save_path}")

    if SHOW_PLOTS:
        plt.show()
    else:
        plt.close(fig)

    return save_path


def run_lightning_impulse(verbose: bool = False) -> bool:
    """Run the transformer lightning impulse simulation."""
    print_header()

    solver, lightning_source = build_solver(verbose=verbose)

    if verbose:
        solver.print_circuit_summary()

    print("\nStarting simulation...")
    solver.run()
    print("Simulation complete.")

    results = extract_results(solver)
    metrics = calculate_metrics(results)
    print_results(metrics, solver)

    plot_primary_side(results)
    plot_secondary_side(results)

    return bool(metrics["passed"])


def main() -> None:
    passed = run_lightning_impulse(verbose=False)
    if passed:
        print("\nLightning impulse simulation completed successfully.")
    else:
        print("\nLightning impulse simulation completed with failed checks.")


if __name__ == "__main__":
    main()
