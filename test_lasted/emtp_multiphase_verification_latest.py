# -*- coding: utf-8 -*-
"""
Multiphase ULM lightning verification case for emtp.

This script is adapted from emtp_multiphase_verification.py to work with the
latest solver architecture uploaded in this conversation.

Main compatibility changes
--------------------------
1. Import EMTPSolver from emtp instead of emtp_solver_v2_time_NAM.
2. Use the solver's built-in ATP-compatible lightning current source:
   solver.add_standard_twoexpf_IS(...), so the old lightning_waveform module is
   no longer required.
3. Resolve the FitULM file relative to both the current working directory and
   this script's directory.
4. Use the latest result APIs:
   get_time(unit="s") and get_node_voltage(node, unit="V").
5. Keep full node-voltage recording enabled because this verification script
   extracts all sending/receiving-end phase voltages after run().
"""

from __future__ import annotations

import os
import sys
import time as timer
from pathlib import Path
from typing import Dict, Iterable, Tuple

import matplotlib.pyplot as plt
import numpy as np


# ---------------------------------------------------------------------------
# Make local solver modules importable before importing EMTPSolver.
# ---------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent

for _path in (SCRIPT_DIR, PROJECT_DIR, Path.cwd()):
    _path_str = str(_path)
    if _path_str not in sys.path:
        sys.path.insert(0, _path_str)

from emtp import EMTPSolver  # noqa: E402


# ---------------------------------------------------------------------------
# Case configuration
# ---------------------------------------------------------------------------

FITULM_PATH = "ohl_model_900_pscad.fitULM"
LINE_LENGTH = 20000.0        # Same unit expected by your ULMLine implementation.
DT = 1e-8
T_END = 10e-4

LIGHTNING_PEAK = 10e3
LIGHTNING_TYPE = "2/20"      # Must exist in the ATP-compatible standard library.
LIGHTNING_PERC = 30

NC = 4                       # Number of phases/conductors in the FitULM data.
MATCHED_R_TERM = 20.0        # Preserve the original verification setting.

SAVE_PLOTS = True
SHOW_PLOTS = True
RESULT_DIR = "result"


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def resolve_existing_file(path_like: str | os.PathLike) -> Path:
    """Resolve a file path relative to cwd or this script's directory."""
    path = Path(path_like)

    candidates = []
    if path.is_absolute():
        candidates.append(path)
    else:
        candidates.extend([
            Path.cwd() / path,
            SCRIPT_DIR / path,
            PROJECT_DIR / path,
        ])

    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()

    searched = "\n".join(f"  - {candidate}" for candidate in candidates)
    raise FileNotFoundError(
        f"Could not find FitULM file: {path_like}\n"
        f"Searched:\n{searched}"
    )


def phase_nodes(nc: int) -> Tuple[list[int], list[int]]:
    """Return sending-end and receiving-end node lists for an nc-phase line."""
    nodes_k = list(range(1, nc + 1))
    nodes_m = list(range(nc + 1, 2 * nc + 1))
    return nodes_k, nodes_m


def stack_node_voltages(
    solver: EMTPSolver,
    nodes: Iterable[int],
    *,
    unit: str = "V",
) -> np.ndarray:
    """Return a 2-D array with one column per node."""
    return np.column_stack([
        solver.get_node_voltage(node, unit=unit)
        for node in nodes
    ])


def termination_resistance(termination_type: str) -> float:
    """Map a text termination condition to a resistance to ground."""
    key = termination_type.strip().lower()
    if key == "open":
        return 1e8
    if key == "short":
        return 1e-6
    if key == "matched":
        return MATCHED_R_TERM
    raise ValueError(
        f"Unknown termination_type={termination_type!r}. "
        "Use one of: open, short, matched."
    )


# ---------------------------------------------------------------------------
# Solver case
# ---------------------------------------------------------------------------

def run_general_solver_case(termination_type: str = "open"):
    """
    Run the multiphase lightning case with the latest EMTPSolver.

    The current source direction follows the solver convention:
    current flows from node_from to node_to. Therefore add_standard_twoexpf_IS
    with node_from=0 and node_to=nodes_k[0] injects positive current into the
    struck sending-end phase node.
    """
    termination_type = termination_type.strip().lower()
    print(f"\n>>> Running latest EMTPSolver case: {termination_type.upper()} termination")

    t_case_start = timer.perf_counter()

    fitulm_file = resolve_existing_file(FITULM_PATH)

    # 1. Initialize solver.
    t_setup_start = timer.perf_counter()
    solver = EMTPSolver(
        dt=DT,
        finish_time=T_END,
        verbose=False,

        # This script extracts all node voltages after run().
        record_all_node_voltages=True,

        # Keep branch/source history off unless explicitly needed.
        record_line_history=False,
        record_branch_history=False,
        record_source_history=False,

        # Fast paths available in the latest solver.
        pre_sample_sources=True,
        use_rhs_plan=True,
        ulm_batch_mode="auto",
    )

    nodes_k, nodes_m = phase_nodes(NC)

    # 2. Add ULM transmission line.
    line = solver.add_ulm_line(
        name="TL_multiphase",
        nodes_k=nodes_k,
        nodes_m=nodes_m,
        fitulm_file=str(fitulm_file),
        length=LINE_LENGTH,
    )

    line_info = line.get_info()
    zc_equiv = line_info.get("Zc", None)
    tau_equiv = line_info.get("tau", None)
    nc_from_line = line_info.get("nc", NC)

    if int(nc_from_line) != NC:
        raise ValueError(
            f"NC={NC} does not match FitULM data nc={nc_from_line}. "
            "Please update NC to match the FitULM file."
        )

    if zc_equiv is not None:
        print(f"    Line equivalent Zc: {float(zc_equiv):.6g} ohm")
    if tau_equiv is not None:
        print(f"    Line equivalent tau: {float(tau_equiv) * 1e6:.6g} us")

    # 3. Sending end.
    # Struck phase: inject lightning current into phase 1 node.
    # Non-struck phases: solidly grounded through small resistors.
    solver.add_standard_twoexpf_IS(
        name="Lightning_Src",
        node_from=0,
        node_to=nodes_k[0],
        waveform_type=LIGHTNING_TYPE,
        peak=LIGHTNING_PEAK,
        PERC=LIGHTNING_PERC,
    )

    # Struck phase source-side shunt.
    solver.add_R("Rs_Ph1", nodes_k[0], 0, 800.0)

    # Non-struck sending-end phases grounded.
    for phase_idx, node in enumerate(nodes_k[1:], start=2):
        solver.add_R(f"Rs_Ph{phase_idx}", node, 0, 1e-6)

    # 4. Receiving end.
    r_term = termination_resistance(termination_type)

    # Struck phase receiving-end load.
    solver.add_R("Rl_Ph1", nodes_m[0], 0, r_term)

    # Non-struck receiving-end phases grounded.
    for phase_idx, node in enumerate(nodes_m[1:], start=2):
        solver.add_R(f"Rl_Ph{phase_idx}", node, 0, 1e-6)

    t_setup_end = timer.perf_counter()
    print(f"    [Timing] Model setup: {t_setup_end - t_setup_start:.3f} s")

    # 5. Run simulation.
    t_run_start = timer.perf_counter()
    solver.run()
    t_run_end = timer.perf_counter()
    print(f"    [Timing] Solver run: {t_run_end - t_run_start:.3f} s")

    stats = solver.get_solver_statistics()
    print(
        "    [Stats] "
        f"steps={stats.get('total_steps')}, "
        f"MNA size={stats.get('mna_size')}, "
        f"G rebuilds={stats.get('G_rebuilds')}, "
        f"G cache hits={stats.get('G_cache_hits')}"
    )

    # 6. Extract data.
    t_extract_start = timer.perf_counter()
    time_s = solver.get_time(unit="s")
    v_k = stack_node_voltages(solver, nodes_k, unit="V")
    v_m = stack_node_voltages(solver, nodes_m, unit="V")
    t_extract_end = timer.perf_counter()

    print(f"    [Timing] Data extraction: {t_extract_end - t_extract_start:.3f} s")
    print(f"    [Timing] Case total: {t_extract_end - t_case_start:.3f} s")

    return time_s, v_k, v_m


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def ensure_output_dir(save_dir: str | os.PathLike = RESULT_DIR) -> Path:
    path = Path(save_dir)
    path.mkdir(parents=True, exist_ok=True)
    return path


def plot_comparison(results_dict: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]],
                    save_dir: str | os.PathLike = RESULT_DIR) -> None:
    """Plot all termination cases in one comparison figure."""
    if not results_dict:
        return

    save_path = ensure_output_dir(save_dir)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        "Lightning Simulation Comparison - Different Terminations",
        fontsize=16,
        fontweight="bold",
    )

    colors = {"open": "blue", "short": "red", "matched": "green"}
    line_styles = {"open": "-", "short": "--", "matched": "-."}

    plot_specs = [
        (axes[0, 0], 0, "k", "Sending End Voltage (Phase 1)", "Phase 1"),
        (axes[0, 1], 0, "m", "Receiving End Voltage (Phase 1)", "Phase 1"),
        (axes[1, 0], 1, "k", "Induced Voltage Sending End (Phase 2)", "Phase 2"),
        (axes[1, 1], 1, "m", "Induced Voltage Receiving End (Phase 2)", "Phase 2"),
    ]

    for ax, col, side, title, phase_label in plot_specs:
        for term, (t, vk, vm) in results_dict.items():
            data = vk if side == "k" else vm
            ax.plot(
                t * 1e6,
                data[:, col] / 1e3,
                color=colors.get(term),
                linestyle=line_styles.get(term, "-"),
                linewidth=1.5,
                label=f"{term.upper()} ({phase_label})",
            )
        ax.set_title(title)
        ax.set_xlabel("Time (us)")
        ax.set_ylabel("Voltage (kV)")
        ax.grid(True, alpha=0.3)
        ax.legend()

    plt.tight_layout()

    if SAVE_PLOTS:
        filename = save_path / "lightning_simulation_comparison.png"
        plt.savefig(filename, dpi=300, bbox_inches="tight")
        print(f"Saved comparison plot: {filename}")

    if SHOW_PLOTS:
        plt.show()
    else:
        plt.close(fig)


def plot_individual_cases(
    results_dict: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]],
    save_dir: str | os.PathLike = RESULT_DIR,
) -> None:
    """Plot one 2x2 figure for each termination case."""
    if not results_dict:
        return

    save_path = ensure_output_dir(save_dir)
    colors = {"open": "blue", "short": "red", "matched": "green"}

    for term, (t, vk, vm) in results_dict.items():
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(
            f"Lightning Simulation - {term.upper()} Termination",
            fontsize=16,
            fontweight="bold",
        )

        plot_specs = [
            (axes[0, 0], vk[:, 0], f"Sending End Voltage (Phase 1) - {term.upper()}",
             "Phase 1 (Struck)"),
            (axes[0, 1], vm[:, 0], f"Receiving End Voltage (Phase 1) - {term.upper()}",
             "Phase 1 (Struck)"),
            (axes[1, 0], vk[:, 1], f"Induced Voltage Sending End (Phase 2) - {term.upper()}",
             "Phase 2 (Induced)"),
            (axes[1, 1], vm[:, 1], f"Induced Voltage Receiving End (Phase 2) - {term.upper()}",
             "Phase 2 (Induced)"),
        ]

        for ax, y, title, label in plot_specs:
            ax.plot(
                t * 1e6,
                y / 1e3,
                color=colors.get(term),
                linewidth=2,
                label=label,
            )
            ax.set_title(title)
            ax.set_xlabel("Time (us)")
            ax.set_ylabel("Voltage (kV)")
            ax.grid(True, alpha=0.3)
            ax.legend()

        plt.tight_layout()

        if SAVE_PLOTS:
            filename = save_path / f"lightning_simulation_{term}.png"
            plt.savefig(filename, dpi=300, bbox_inches="tight")
            print(f"Saved: {filename}")

        if SHOW_PLOTS:
            plt.show()
        else:
            plt.close(fig)


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_case_peaks(term: str, t: np.ndarray, vk: np.ndarray, vm: np.ndarray) -> None:
    vk0_abs = np.abs(vk[:, 0])
    vm0_abs = np.abs(vm[:, 0])

    v_k_peak_idx = int(np.argmax(vk0_abs))
    v_m_peak_idx = int(np.argmax(vm0_abs))

    print(
        f"  {term.upper()} Result: "
        f"Vk_max={vk0_abs[v_k_peak_idx] / 1e3:.6f} kV "
        f"@ t={t[v_k_peak_idx] * 1e6:.6f} us, "
        f"Vm_max={vm0_abs[v_m_peak_idx] / 1e3:.6f} kV "
        f"@ t={t[v_m_peak_idx] * 1e6:.6f} us"
    )

    if term == "short":
        v_k_neg_idx = int(np.argmin(vk[:, 0]))
        v_m_neg_idx = int(np.argmin(vm[:, 0]))
        print(
            "  SHORT Negative Peak: "
            f"Vk_min={vk[v_k_neg_idx, 0] / 1e3:.6f} kV "
            f"@ t={t[v_k_neg_idx] * 1e6:.6f} us, "
            f"Vm_min={vm[v_m_neg_idx, 0] / 1e3:.6f} kV "
            f"@ t={t[v_m_neg_idx] * 1e6:.6f} us"
        )


def print_short_case_snapshot(
    results: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]],
    *,
    t_query: float = 2.9e-6,
) -> None:
    if "short" not in results:
        return

    t_s, vk_s, vm_s = results["short"]
    idx = int(np.argmin(np.abs(t_s - t_query)))
    t_actual_us = t_s[idx] * 1e6

    print(f"\n{'=' * 55}")
    print(f"  SHORT termination @ query time t={t_query * 1e6:.6f} us")
    print(f"  Nearest time step: t={t_actual_us:.6f} us  (index={idx})")
    print(f"{'=' * 55}")
    print(f"  {'Phase':>6}  {'Sending Vk (kV)':>20}  {'Receiving Vm (kV)':>20}")
    print(f"  {'-' * 51}")
    for ph in range(NC):
        print(
            f"  Phase {ph + 1:>1}  "
            f"{vk_s[idx, ph] / 1e3:>20.6f}  "
            f"{vm_s[idx, ph] / 1e3:>20.6f}"
        )
    print(f"{'=' * 55}\n")


def print_summary(results: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]]) -> None:
    print("\n=== Simulation Summary ===")

    for term, (t, vk, vm) in results.items():
        vk0_abs = np.abs(vk[:, 0])
        vm0_abs = np.abs(vm[:, 0])
        vk1_abs = np.abs(vk[:, 1])
        vm1_abs = np.abs(vm[:, 1])

        vk_peak_idx = int(np.argmax(vk0_abs))
        vm_peak_idx = int(np.argmax(vm0_abs))
        vk_ind_idx = int(np.argmax(vk1_abs))
        vm_ind_idx = int(np.argmax(vm1_abs))

        print(f"\n{term.upper()} termination:")
        print(
            f"  Sending struck-phase peak:    {vk0_abs[vk_peak_idx] / 1e3:.6f} kV "
            f"@ t={t[vk_peak_idx] * 1e6:.6f} us"
        )
        print(
            f"  Receiving struck-phase peak:  {vm0_abs[vm_peak_idx] / 1e3:.6f} kV "
            f"@ t={t[vm_peak_idx] * 1e6:.6f} us"
        )
        print(
            f"  Sending induced peak:         {vk1_abs[vk_ind_idx] / 1e3:.6f} kV "
            f"@ t={t[vk_ind_idx] * 1e6:.6f} us"
        )
        print(
            f"  Receiving induced peak:       {vm1_abs[vm_ind_idx] / 1e3:.6f} kV "
            f"@ t={t[vm_ind_idx] * 1e6:.6f} us"
        )

        if term == "short":
            vk_neg_idx = int(np.argmin(vk[:, 0]))
            vm_neg_idx = int(np.argmin(vm[:, 0]))
            vk1_neg_idx = int(np.argmin(vk[:, 1]))
            vm1_neg_idx = int(np.argmin(vm[:, 1]))

            print("  --- Negative voltage peaks ---")
            print(
                f"  Sending struck-phase negative:   {vk[vk_neg_idx, 0] / 1e3:.6f} kV "
                f"@ t={t[vk_neg_idx] * 1e6:.6f} us"
            )
            print(
                f"  Receiving struck-phase negative: {vm[vm_neg_idx, 0] / 1e3:.6f} kV "
                f"@ t={t[vm_neg_idx] * 1e6:.6f} us"
            )
            print(
                f"  Sending induced negative:        {vk[vk1_neg_idx, 1] / 1e3:.6f} kV "
                f"@ t={t[vk1_neg_idx] * 1e6:.6f} us"
            )
            print(
                f"  Receiving induced negative:      {vm[vm1_neg_idx, 1] / 1e3:.6f} kV "
                f"@ t={t[vm1_neg_idx] * 1e6:.6f} us"
            )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    try:
        resolve_existing_file(FITULM_PATH)
    except FileNotFoundError as exc:
        print(f"Error: {exc}")
        return

    results: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    t_total_start = timer.perf_counter()

    for term in ("open", "short", "matched"):
        try:
            t, vk, vm = run_general_solver_case(term)
            results[term] = (t, vk, vm)
            print_case_peaks(term, t, vk, vm)
        except Exception as exc:
            print(f"Simulation failed for {term.upper()}: {exc}")
            import traceback
            traceback.print_exc()

    print_short_case_snapshot(results)

    if results:
        print("\nGenerating and saving plots...")
        t_plot_start = timer.perf_counter()
        plot_comparison(results)
        print("\nGenerating individual termination plots...")
        plot_individual_cases(results)
        t_plot_end = timer.perf_counter()
        print(f"\n[Timing] Total plotting time: {t_plot_end - t_plot_start:.3f} s")

        print_summary(results)

    t_total_end = timer.perf_counter()
    print(f"\n{'=' * 55}")
    print(f"  Total workflow time: {t_total_end - t_total_start:.3f} s")
    print(f"{'=' * 55}")


if __name__ == "__main__":
    main()
