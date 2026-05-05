# -*- coding: utf-8 -*-
"""
DC cable lightning case adapted for emtp_solver_v3.

Compatibility changes from DC_CABLE_TEST_0201.py
------------------------------------------------
1. Import EMTPSolver from emtp_solver_v3 instead of emtp_solver_v2_time_NAM.
2. Remove the old lightning_waveform dependency and use the solver's
   ATP-compatible current-source API: add_standard_twoexpf_IS().
3. Resolve the PCH/FitULM file relative to cwd, this script directory, and its
   parent directory.
4. Use latest result APIs with explicit units:
   get_time(unit="s"), get_node_voltage(node, unit="V"),
   get_line_current_m(name, unit="A", phase=0).
5. Keep record_line_history=True because this case exports port-7 line current.
6. Use non-blocking plotting controls so the script can run in batch mode.
"""

from __future__ import annotations

import os
import sys
import time as timer
from pathlib import Path
from typing import Iterable, Tuple

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

from emtp_solver_v3 import EMTPSolver  # noqa: E402


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

FITULM_PATH = "cable_model_100km.pch"

LINE_LENGTH = 100e3
DT = 1e-8
T_END = 1e-3

LIGHTNING_PEAK = 10e3
LIGHTNING_TYPE = "2/20"
LIGHTNING_PERC = 30

NC = 6

RESULT_DIR = "result"
PORT7_DATA_FILENAME = "port7_data.txt"

SAVE_PLOTS = True
SHOW_PLOTS = True


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def resolve_existing_file(path_like: str | os.PathLike) -> Path:
    """Resolve a model file from cwd, script directory, or project directory."""
    path = Path(path_like)

    if path.is_absolute():
        candidates = [path]
    else:
        candidates = [
            Path.cwd() / path,
            SCRIPT_DIR / path,
            PROJECT_DIR / path,
        ]

    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()

    searched = "\n".join(f"  - {candidate}" for candidate in candidates)
    raise FileNotFoundError(
        f"Could not find line model file: {path_like}\n"
        f"Searched:\n{searched}"
    )


def phase_nodes(nc: int) -> Tuple[list[int], list[int]]:
    """Return sending-end and receiving-end node lists."""
    return list(range(1, nc + 1)), list(range(nc + 1, 2 * nc + 1))


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


def ensure_dir(path_like: str | os.PathLike) -> Path:
    path = Path(path_like)
    path.mkdir(parents=True, exist_ok=True)
    return path


# ---------------------------------------------------------------------------
# Solver case
# ---------------------------------------------------------------------------

def run_general_solver_case(termination_type: str = "open"):
    """
    Run the DC cable lightning case with the latest EMTPSolver.

    The current source direction follows the solver convention:
    current flows from node_from to node_to. Therefore node_from=0 and
    node_to=1 injects positive lightning current into node 1.
    """
    print(f"\n>>> Running latest EMTPSolver DC cable case: {termination_type.upper()}")

    t_case_start = timer.perf_counter()
    model_file = resolve_existing_file(FITULM_PATH)

    # 1. Initialize solver.
    t_setup_start = timer.perf_counter()
    solver = EMTPSolver(
        dt=DT,
        finish_time=T_END,
        verbose=False,

        # Required because this script extracts all node voltages after run().
        record_all_node_voltages=True,

        # Required by save_port7_data(), which calls get_line_current_m().
        record_line_history=True,

        # Branch/source histories are not required by the normal path.
        record_branch_history=False,
        record_source_history=False,

        # Latest solver fast paths.
        pre_sample_sources=True,
        use_rhs_plan=True,
        ulm_batch_mode="auto",
    )

    nodes_k, nodes_m = phase_nodes(NC)

    # 2. Add ULM cable line.
    line = solver.add_ulm_line(
        name="TL_6ph",
        nodes_k=nodes_k,
        nodes_m=nodes_m,
        fitulm_file=str(model_file),
        length=LINE_LENGTH,
    )

    line_info = line.get_info()
    nc_from_line = int(line_info.get("nc", NC))
    if nc_from_line != NC:
        raise ValueError(
            f"NC={NC} does not match model data nc={nc_from_line}. "
            "Please update NC."
        )

    zc_equiv = line_info.get("Zc")
    tau_equiv = line_info.get("tau")
    if zc_equiv is not None:
        print(f"    Line equivalent Zc: {float(zc_equiv):.6g} ohm")
    if tau_equiv is not None:
        print(f"    Line equivalent tau: {float(tau_equiv) * 1e6:.6g} us")

    # 3. Lightning source at sending-end phase 1.
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

    # Preserve original shunt-load logic for nodes 2..12.
    # Nodes 4, 7, 10 use 10 ohm; all others use 0.5 ohm.
    for node in range(2, 2 * NC + 1):
        resistance = 10.0 if node in [4, NC + 1, NC + 4] else 0.5
        solver.add_R(f"Rl_Ph{node}", node, 0, resistance)

    t_setup_end = timer.perf_counter()
    print(f"    [Timing] Model setup: {t_setup_end - t_setup_start:.3f} s")

    # 4. Run simulation.
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

    # 5. Extract data.
    t_extract_start = timer.perf_counter()
    time_s = solver.get_time(unit="s")
    v_k = stack_node_voltages(solver, nodes_k, unit="V")
    v_m = stack_node_voltages(solver, nodes_m, unit="V")
    t_extract_end = timer.perf_counter()

    print(f"    [Timing] Data extraction: {t_extract_end - t_extract_start:.3f} s")
    print(f"    [Timing] Case total: {t_extract_end - t_case_start:.3f} s")

    return time_s, v_k, v_m, solver


# ---------------------------------------------------------------------------
# Data export
# ---------------------------------------------------------------------------

def save_port7_data(time_s: np.ndarray, v_m: np.ndarray, solver: EMTPSolver,
                    save_dir: str | os.PathLike = ".") -> Path:
    """
    Save port-7 data: time, receiving-end phase-1 voltage, and line current.

    Port 7 is receiving-end phase 1, i.e. node NC+1.
    """
    save_path = ensure_dir(save_dir)
    v_port7 = v_m[:, 0]

    try:
        i_m = solver.get_line_current_m("TL_6ph", unit="A", phase=0)
    except Exception as exc:
        print(f"Warning: could not get line-end current; using V/R fallback: {exc}")
        r_port7 = 10.0
        i_m = v_port7 / r_port7

    i_m = np.asarray(i_m, dtype=float)
    n_points = len(time_s)

    if len(i_m) != n_points:
        # Avoid np.resize repeating data silently.
        aligned = np.full(n_points, np.nan, dtype=float)
        aligned[:min(n_points, len(i_m))] = i_m[:min(n_points, len(i_m))]
        i_m = aligned

    filename = save_path / PORT7_DATA_FILENAME

    with filename.open("w", encoding="utf-8") as f:
        f.write("=" * 70 + "\n")
        f.write("Port 7 data: receiving-end phase 1, node 7\n")
        f.write("=" * 70 + "\n")
        f.write(f"Number of samples: {n_points}\n")
        f.write(f"Time range: {time_s[0] * 1e6:.4f} to {time_s[-1] * 1e6:.4f} us\n")
        f.write(f"Voltage peak: {np.nanmax(np.abs(v_port7)) / 1e3:.4f} kV\n")
        f.write(f"Current peak: {np.nanmax(np.abs(i_m)):.4f} A\n")
        f.write("=" * 70 + "\n\n")
        f.write(
            f"{'Time (s)':<20} "
            f"{'Time (us)':<15} "
            f"{'Voltage (V)':<20} "
            f"{'Voltage (kV)':<15} "
            f"{'Current (A)':<15}\n"
        )
        f.write("-" * 85 + "\n")

        for t_s, v_v, i_a in zip(time_s, v_port7, i_m):
            f.write(
                f"{t_s:<20.10e} "
                f"{t_s * 1e6:<15.6f} "
                f"{v_v:<20.6e} "
                f"{v_v / 1e3:<15.6f} "
                f"{i_a:<15.6e}\n"
            )

    print(f"\nPort 7 data saved to: {filename}")
    print(f"  - Samples: {n_points}")
    print(f"  - Voltage peak: {np.nanmax(np.abs(v_port7)) / 1e3:.4f} kV")
    print(f"  - Current peak: {np.nanmax(np.abs(i_m)):.4f} A")

    return filename


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_all_ports_voltage(
    time_s: np.ndarray,
    v_k: np.ndarray,
    v_m: np.ndarray,
    save_dir: str | os.PathLike = RESULT_DIR,
) -> Path | None:
    """Plot all 12 port voltages in one 4x3 figure."""
    save_path = ensure_dir(save_dir)

    fig, axes = plt.subplots(4, 3, figsize=(18, 16))
    fig.suptitle(
        "Lightning Simulation - All 12 Port Voltages (Dual-Circuit Cable)",
        fontsize=16,
        fontweight="bold",
    )

    # Sending-end voltages: nodes 1..6.
    for i in range(NC):
        row = i // 3
        col = i % 3
        ax = axes[row, col]

        ax.plot(time_s * 1e6, v_k[:, i] / 1e3, linewidth=1.5, label=f"Node {i + 1}")

        peak_idx = int(np.argmax(np.abs(v_k[:, i])))
        peak_val = v_k[peak_idx, i] / 1e3
        peak_time = time_s[peak_idx] * 1e6
        ax.plot(peak_time, peak_val, marker="o", markersize=5)
        ax.annotate(
            f"Peak: {peak_val:.2f} kV\n@ {peak_time:.2f} us",
            xy=(peak_time, peak_val),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=8,
        )

        ax.set_title(f"Sending End - Node {i + 1} (Phase {i + 1})", fontsize=10)
        ax.set_xlabel("Time (us)")
        ax.set_ylabel("Voltage (kV)")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right", fontsize=8)

    # Receiving-end voltages: nodes 7..12.
    for i in range(NC):
        row = 2 + i // 3
        col = i % 3
        ax = axes[row, col]

        node_id = i + NC + 1
        ax.plot(time_s * 1e6, v_m[:, i] / 1e3, linewidth=1.5, label=f"Node {node_id}")

        peak_idx = int(np.argmax(np.abs(v_m[:, i])))
        peak_val = v_m[peak_idx, i] / 1e3
        peak_time = time_s[peak_idx] * 1e6
        ax.plot(peak_time, peak_val, marker="o", markersize=5)
        ax.annotate(
            f"Peak: {peak_val:.2f} kV\n@ {peak_time:.2f} us",
            xy=(peak_time, peak_val),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=8,
        )

        ax.set_title(f"Receiving End - Node {node_id} (Phase {i + 1})", fontsize=10)
        ax.set_xlabel("Time (us)")
        ax.set_ylabel("Voltage (kV)")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right", fontsize=8)

    plt.tight_layout()

    filename = save_path / "lightning_simulation_all_ports.png"
    if SAVE_PLOTS:
        plt.savefig(filename, dpi=300, bbox_inches="tight")
        print(f"Saved: {filename}")

    if SHOW_PLOTS:
        plt.show()
    else:
        plt.close(fig)

    return filename if SAVE_PLOTS else None


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_peak_info(time_s: np.ndarray, v_k: np.ndarray, v_m: np.ndarray) -> None:
    """Print peak voltage and peak time for all 12 ports."""
    print("\n" + "=" * 70)
    print("                    Simulation result: peak voltage and time")
    print("=" * 70)

    print("\nSending End - Nodes 1 to 6")
    print("-" * 58)
    print(f"{'Node':<8} {'Phase':<8} {'Peak Voltage (kV)':<20} {'Peak Time (us)':<15}")
    print("-" * 58)

    for i in range(NC):
        peak_idx = int(np.argmax(np.abs(v_k[:, i])))
        peak_val = v_k[peak_idx, i] / 1e3
        peak_time = time_s[peak_idx] * 1e6
        print(f"Node {i + 1:<4} Phase {i + 1:<4} {peak_val:>+16.2f}      {peak_time:>10.2f}")

    print("\nReceiving End - Nodes 7 to 12")
    print("-" * 58)
    print(f"{'Node':<8} {'Phase':<8} {'Peak Voltage (kV)':<20} {'Peak Time (us)':<15}")
    print("-" * 58)

    for i in range(NC):
        peak_idx = int(np.argmax(np.abs(v_m[:, i])))
        peak_val = v_m[peak_idx, i] / 1e3
        peak_time = time_s[peak_idx] * 1e6
        print(f"Node {i + NC + 1:<4} Phase {i + 1:<4} {peak_val:>+16.2f}      {peak_time:>10.2f}")

    print("=" * 70)

    vk_max_idx = np.unravel_index(np.argmax(np.abs(v_k)), v_k.shape)
    vm_max_idx = np.unravel_index(np.argmax(np.abs(v_m)), v_m.shape)

    vk_max = v_k[vk_max_idx] / 1e3
    vm_max = v_m[vm_max_idx] / 1e3
    vk_max_time = time_s[vk_max_idx[0]] * 1e6
    vm_max_time = time_s[vm_max_idx[0]] * 1e6

    print("\nSummary")
    print(
        f"  Sending-end maximum: {vk_max:+.2f} kV "
        f"@ Node {vk_max_idx[1] + 1}, t = {vk_max_time:.2f} us"
    )
    print(
        f"  Receiving-end maximum: {vm_max:+.2f} kV "
        f"@ Node {vm_max_idx[1] + NC + 1}, t = {vm_max_time:.2f} us"
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

    try:
        t_total_start = timer.perf_counter()

        # Original case runs only OPEN condition.
        time_s, v_k, v_m, solver = run_general_solver_case("open")

        print_peak_info(time_s, v_k, v_m)

        t_save_start = timer.perf_counter()
        save_port7_data(time_s, v_m, solver, save_dir=".")
        t_save_end = timer.perf_counter()
        print(f"\n[Timing] Data saving: {t_save_end - t_save_start:.3f} s")

        print("\nGenerating and saving plot...")
        t_plot_start = timer.perf_counter()
        plot_all_ports_voltage(time_s, v_k, v_m)
        t_plot_end = timer.perf_counter()
        print(f"\n[Timing] Plotting: {t_plot_end - t_plot_start:.3f} s")

        t_total_end = timer.perf_counter()
        print(f"\n{'=' * 55}")
        print(f"  Total workflow time: {t_total_end - t_total_start:.3f} s")
        print(f"{'=' * 55}")
        print("\nSimulation complete.")

    except Exception as exc:
        print(f"Simulation failed: {exc}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
