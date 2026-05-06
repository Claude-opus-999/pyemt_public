# -*- coding: utf-8 -*-
"""
Pipe-type 3-core cable lightning simulation adapted for emtp.

Original file: DC_CABLE_TEST_3CORE.py

Compatibility changes
---------------------
1. Import EMTPSolver from emtp instead of emtp_solver_v2_time.
2. Remove dependency on the old lightning_waveform module.
3. Use EMTPSolver.add_standard_twoexpf_IS() for the ATP-compatible 2/20 us
   lightning current source.
4. Resolve the FitULM file relative to cwd, this script directory, and parent.
5. Use latest result APIs with explicit units:
   get_time(unit="s"), get_node_voltage(node, unit="V"),
   get_line_current_m(name, unit="A", phase=...).
6. Keep record_line_history=True because this case exports S3 receiving-end
   line current.
7. Use batch/script-friendly plotting controls.

Important topology note
-----------------------
The original code comments say lightning is injected into S3 sheath, but the
actual original implementation injected the source into node 5 and added the
800 ohm source shunt at node 5. In the conductor mapping, node 5 is C3 core
(index 4), while S3 sheath is node 6 (index 5). This adapted script preserves
the original executable topology by default:

    LIGHTNING_INJECTION_NODE = 5

To inject into the S3 sheath instead, set:

    LIGHTNING_INJECTION_NODE = 6
    LIGHTNING_SOURCE_SHUNT_NODE = 6
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
# Local import path setup
# ---------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent

for _path in (SCRIPT_DIR, PROJECT_DIR, Path.cwd()):
    _path_str = str(_path)
    if _path_str not in sys.path:
        sys.path.insert(0, _path_str)

from emtp import EMTPSolver  # noqa: E402


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

FITULM_PATH = "cable14.fitULM"    # Pipe-type 3-core cable ULM fit file.
LINE_LENGTH = 5e3                 # Cable length, same unit expected by ULMLine.
DT = 1e-8                         # 10 ns
T_END = 3e-4                      # 0.3 ms

LIGHTNING_PEAK = 10e3             # 10 kA
LIGHTNING_TYPE = "2/20"
LIGHTNING_PERC = 30
LIGHTNING_TSTART = 0.0

NC = 7                            # C1, S1, C2, S2, C3, S3, P

# Preserve the original executable topology:
# original code: solver.add_IS(..., 0, 5, ...) and solver.add_R("Rs_S3", 5, 0, 800)
LIGHTNING_INJECTION_NODE = 5
LIGHTNING_SOURCE_SHUNT_NODE = 5
LIGHTNING_SOURCE_SHUNT_R = 800.0

RESULT_DIR = "result"
SAVE_PLOTS = True
SHOW_PLOTS = True


# ---------------------------------------------------------------------------
# Conductor definitions
# ---------------------------------------------------------------------------

CONDUCTOR_NAMES = {
    0: "C1 (Core 1)",
    1: "S1 (Sheath 1)",
    2: "C2 (Core 2)",
    3: "S2 (Sheath 2)",
    4: "C3 (Core 3)",
    5: "S3 (Sheath 3)",
    6: "P  (Pipe)",
}

PROBE_POINTS = {
    "EaP1": {"end": "sending", "phase_idx": 4, "node": 5,  "desc": "Sending-end C3 core voltage"},
    "EaP2": {"end": "sending", "phase_idx": 2, "node": 3,  "desc": "Sending-end C2 core voltage"},
    "EaN1": {"end": "receiving", "phase_idx": 4, "node": 12, "desc": "Receiving-end C3 core voltage"},
    "EaN2": {"end": "receiving", "phase_idx": 2, "node": 10, "desc": "Receiving-end C2 core voltage"},
}


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def resolve_existing_file(path_like: str | os.PathLike) -> Path:
    """Resolve a FitULM/PCH file from cwd, script dir, or project dir."""
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
        f"Could not find ULM fit file: {path_like}\n"
        f"Searched:\n{searched}"
    )


def ensure_dir(path_like: str | os.PathLike) -> Path:
    path = Path(path_like)
    path.mkdir(parents=True, exist_ok=True)
    return path


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


def align_array_length(values: np.ndarray, n_points: int) -> np.ndarray:
    """Align an array length without repeating data."""
    values = np.asarray(values, dtype=float)
    if len(values) == n_points:
        return values

    aligned = np.full(n_points, np.nan, dtype=float)
    n = min(n_points, len(values))
    aligned[:n] = values[:n]
    return aligned


def conductor_label_from_node(node: int) -> str:
    """Return conductor label for external node id."""
    if 1 <= node <= NC:
        idx = node - 1
        return f"sending {CONDUCTOR_NAMES[idx]}"
    if NC + 1 <= node <= 2 * NC:
        idx = node - NC - 1
        return f"receiving {CONDUCTOR_NAMES[idx]}"
    return f"node {node}"


# ---------------------------------------------------------------------------
# Solver case
# ---------------------------------------------------------------------------

def run_3core_cable_simulation(termination_type: str = "open"):
    """
    Run the pipe-type 3-core cable lightning simulation.

    Returns
    -------
    time_s : ndarray
        Time in seconds.
    V_k : ndarray
        Sending-end node voltages, shape (n_steps, 7), in V.
    V_m : ndarray
        Receiving-end node voltages, shape (n_steps, 7), in V.
    solver : EMTPSolver
        Completed solver object.
    """
    print(f"\n{'=' * 70}")
    print("  Pipe-Type 3-Core Cable Lightning Simulation")
    print(f"  Termination: {termination_type.upper()}")
    print(f"  Conductors NC={NC}: C1, S1, C2, S2, C3, S3, P")
    print(f"{'=' * 70}")

    t_case_start = timer.perf_counter()
    fitulm_file = resolve_existing_file(FITULM_PATH)

    solver = EMTPSolver(
        dt=DT,
        finish_time=T_END,
        verbose=False,

        # This script extracts all node voltages after run().
        record_all_node_voltages=True,

        # Required by save_probe_data() for get_line_current_m().
        record_line_history=True,

        # Not required by the normal output path.
        record_branch_history=False,
        record_source_history=False,

        # Latest solver fast paths.
        pre_sample_sources=True,
        use_rhs_plan=True,
        ulm_batch_mode="auto",
    )

    nodes_k, nodes_m = phase_nodes(NC)

    print(f"\n  Sending-end nodes:   {nodes_k}")
    print(f"  Receiving-end nodes: {nodes_m}")

    line = solver.add_ulm_line(
        name="TL_3core",
        nodes_k=nodes_k,
        nodes_m=nodes_m,
        fitulm_file=str(fitulm_file),
        length=LINE_LENGTH,
    )

    line_info = line.get_info()
    nc_from_line = int(line_info.get("nc", NC))
    if nc_from_line != NC:
        raise ValueError(
            f"NC={NC} does not match the ULM model nc={nc_from_line}. "
            "Please update NC or use the matching fit file."
        )

    zc_equiv = line_info.get("Zc")
    tau_equiv = line_info.get("tau")
    if zc_equiv is not None:
        print(f"  Line equivalent Zc: {float(zc_equiv):.6g} ohm")
    if tau_equiv is not None:
        print(f"  Line equivalent tau: {float(tau_equiv) * 1e6:.6g} us")

    print(
        f"  Lightning source injection node: {LIGHTNING_INJECTION_NODE} "
        f"({conductor_label_from_node(LIGHTNING_INJECTION_NODE)})"
    )

    # Lightning current source. Positive current flows from node_from to node_to.
    solver.add_standard_twoexpf_IS(
        name="Lightning_Src",
        node_from=0,
        node_to=LIGHTNING_INJECTION_NODE,
        waveform_type=LIGHTNING_TYPE,
        peak=LIGHTNING_PEAK,
        PERC=LIGHTNING_PERC,
        Tstart=LIGHTNING_TSTART,
    )

    # Source shunt / internal resistance.
    solver.add_R("Rs_lightning", LIGHTNING_SOURCE_SHUNT_NODE, 0, LIGHTNING_SOURCE_SHUNT_R)

    # Sending-end grounding / termination network.
    solver.add_R("Rg_S3_send", 6, 0, 5.0)
    solver.add_R("Rg_S2_send", 4, 0, 5.0)
    solver.add_R("Rg_S1_send", 2, 0, 5.0)
    solver.add_R("Rg_P_send", 7, 0, 5.0)
    solver.add_R("Rg_C2_send", 3, 0, 15.0)
    solver.add_R("Rg_C1_send", 1, 0, 15.0)

    # Receiving-end grounding / termination network.
    solver.add_R("Rg_S3_recv", 13, 0, 5.0)
    solver.add_R("Rg_S2_recv", 11, 0, 5.0)
    solver.add_R("Rg_S1_recv", 9, 0, 5.0)
    solver.add_R("Rg_C3_recv", 12, 0, 15.0)
    solver.add_R("Rg_C2_recv", 10, 0, 15.0)
    solver.add_R("Rg_C1_recv", 8, 0, 15.0)
    solver.add_R("Rg_P_recv", 14, 0, 5.0)

    print("\n  Circuit setup complete. Starting simulation...")

    t_run_start = timer.perf_counter()
    solver.run()
    t_run_end = timer.perf_counter()

    stats = solver.get_solver_statistics()
    print(f"  Simulation complete in {t_run_end - t_run_start:.3f} s.")
    print(
        "  Solver stats: "
        f"steps={stats.get('total_steps')}, "
        f"MNA size={stats.get('mna_size')}, "
        f"G rebuilds={stats.get('G_rebuilds')}, "
        f"G cache hits={stats.get('G_cache_hits')}"
    )

    time_s = solver.get_time(unit="s")
    V_k = stack_node_voltages(solver, nodes_k, unit="V")
    V_m = stack_node_voltages(solver, nodes_m, unit="V")

    print(f"  Data extracted. Samples: {len(time_s)}")
    print(f"  Case total time: {timer.perf_counter() - t_case_start:.3f} s")

    return time_s, V_k, V_m, solver


# ---------------------------------------------------------------------------
# Data export
# ---------------------------------------------------------------------------

def save_probe_data(
    time_s: np.ndarray,
    V_k: np.ndarray,
    V_m: np.ndarray,
    solver: EMTPSolver,
    save_dir: str | os.PathLike = ".",
) -> None:
    """
    Save PSCAD probe data EaP1, EaP2, EaN1, EaN2 to TXT files.

    Also saves S3 receiving-end voltage and current data.
    """
    save_path = ensure_dir(save_dir)
    n_points = len(time_s)

    for probe_name, info in PROBE_POINTS.items():
        idx = info["phase_idx"]
        node = info["node"]

        V_probe = V_k[:, idx] if info["end"] == "sending" else V_m[:, idx]
        filename = save_path / f"{probe_name}_data_latest.txt"

        with filename.open("w", encoding="utf-8") as f:
            f.write("=" * 70 + "\n")
            f.write(f"Probe {probe_name}: {info['desc']} (node {node})\n")
            f.write("=" * 70 + "\n")
            f.write(f"Samples: {n_points}\n")
            f.write(f"Time range: {time_s[0] * 1e6:.4f} to {time_s[-1] * 1e6:.4f} us\n")
            f.write(f"Voltage peak: {np.max(np.abs(V_probe)) / 1e3:.4f} kV\n")
            f.write("=" * 70 + "\n\n")
            f.write(f"{'Time (s)':<20} {'Time (us)':<15} {'Voltage (V)':<20} {'Voltage (kV)':<15}\n")
            f.write("-" * 70 + "\n")

            for t_s, v_v in zip(time_s, V_probe):
                f.write(
                    f"{t_s:<20.10e} "
                    f"{t_s * 1e6:<15.6f} "
                    f"{v_v:<20.6e} "
                    f"{v_v / 1e3:<15.6f}\n"
                )

        print(f"  Probe {probe_name} data saved: {filename}")

    # S3 receiving-end data: node 13, phase index 5 at receiving end.
    V_S3_recv = V_m[:, 5]
    try:
        I_S3_recv = solver.get_line_current_m("TL_3core", unit="A", phase=5)
    except Exception as exc:
        print(f"  Warning: could not get S3 receiving-end line current: {exc}")
        # Fallback: current through the 5 ohm receiving-end shunt.
        I_S3_recv = V_S3_recv / 5.0

    I_S3_recv = align_array_length(I_S3_recv, n_points)

    filename = save_path / "S3_recv_data_latest.txt"
    with filename.open("w", encoding="utf-8") as f:
        f.write("=" * 70 + "\n")
        f.write("S3 receiving-end data: node 13, lightning-struck sheath conductor\n")
        f.write("=" * 70 + "\n")
        f.write(f"Samples: {n_points}\n")
        f.write(f"Time range: {time_s[0] * 1e6:.4f} to {time_s[-1] * 1e6:.4f} us\n")
        f.write(f"Voltage peak: {np.max(np.abs(V_S3_recv)) / 1e3:.4f} kV\n")
        f.write(f"Current peak: {np.nanmax(np.abs(I_S3_recv)):.4f} A\n")
        f.write("=" * 70 + "\n\n")
        f.write(
            f"{'Time (s)':<20} {'Time (us)':<15} {'Voltage (V)':<20} "
            f"{'Voltage (kV)':<15} {'Current (A)':<15}\n"
        )
        f.write("-" * 85 + "\n")

        for t_s, v_v, i_a in zip(time_s, V_S3_recv, I_S3_recv):
            f.write(
                f"{t_s:<20.10e} "
                f"{t_s * 1e6:<15.6f} "
                f"{v_v:<20.6e} "
                f"{v_v / 1e3:<15.6f} "
                f"{i_a:<15.6e}\n"
            )

    print(f"  S3 receiving-end data saved: {filename}")


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_all_ports_voltage(
    time_s: np.ndarray,
    V_k: np.ndarray,
    V_m: np.ndarray,
    save_dir: str | os.PathLike = RESULT_DIR,
) -> Path:
    """Plot all 14 port voltages in one figure."""
    save_path = ensure_dir(save_dir)

    fig, axes = plt.subplots(4, 4, figsize=(22, 18))
    fig.suptitle(
        "Lightning Simulation - Pipe-Type 3-Core Cable\n"
        "All 14 Port Voltages (C1, S1, C2, S2, C3, S3, P)",
        fontsize=16,
        fontweight="bold",
    )

    send_layout = [
        (0, 0, 0), (0, 1, 1), (0, 2, 2), (0, 3, 3),
        (1, 0, 4), (1, 1, 5), (1, 2, 6),
    ]

    recv_layout = [
        (2, 0, 0), (2, 1, 1), (2, 2, 2), (2, 3, 3),
        (3, 0, 4), (3, 1, 5), (3, 2, 6),
    ]

    for row, col, ph_idx in send_layout:
        ax = axes[row, col]
        node = ph_idx + 1
        name = CONDUCTOR_NAMES[ph_idx]

        ax.plot(time_s * 1e6, V_k[:, ph_idx] / 1e3, linewidth=1.5, label=f"Node {node}")

        peak_idx = int(np.argmax(np.abs(V_k[:, ph_idx])))
        peak_val = V_k[peak_idx, ph_idx] / 1e3
        peak_time = time_s[peak_idx] * 1e6
        ax.plot(peak_time, peak_val, marker="o", markersize=5)
        ax.annotate(
            f"{peak_val:.2f} kV\n@ {peak_time:.2f} us",
            xy=(peak_time, peak_val),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=7,
        )

        suffix = " - injection node" if node == LIGHTNING_INJECTION_NODE else ""
        ax.set_title(f"Send - Node {node}: {name}{suffix}", fontsize=9)
        ax.set_xlabel("Time (us)", fontsize=8)
        ax.set_ylabel("Voltage (kV)", fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right", fontsize=7)

    axes[1, 3].set_visible(False)

    for row, col, ph_idx in recv_layout:
        ax = axes[row, col]
        node = ph_idx + NC + 1
        name = CONDUCTOR_NAMES[ph_idx]

        ax.plot(time_s * 1e6, V_m[:, ph_idx] / 1e3, linewidth=1.5, label=f"Node {node}")

        peak_idx = int(np.argmax(np.abs(V_m[:, ph_idx])))
        peak_val = V_m[peak_idx, ph_idx] / 1e3
        peak_time = time_s[peak_idx] * 1e6
        ax.plot(peak_time, peak_val, marker="o", markersize=5)
        ax.annotate(
            f"{peak_val:.2f} kV\n@ {peak_time:.2f} us",
            xy=(peak_time, peak_val),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=7,
        )

        ax.set_title(f"Recv - Node {node}: {name}", fontsize=9)
        ax.set_xlabel("Time (us)", fontsize=8)
        ax.set_ylabel("Voltage (kV)", fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right", fontsize=7)

    axes[3, 3].set_visible(False)

    plt.tight_layout()

    filename = save_path / "lightning_3core_all_ports_latest.png"
    if SAVE_PLOTS:
        plt.savefig(filename, dpi=300, bbox_inches="tight")
        print(f"  Plot saved: {filename}")

    if SHOW_PLOTS:
        plt.show()
    else:
        plt.close(fig)

    return filename


def plot_probe_comparison(
    time_s: np.ndarray,
    V_k: np.ndarray,
    V_m: np.ndarray,
    save_dir: str | os.PathLike = RESULT_DIR,
) -> Path:
    """Plot key measurement points: EaP1, EaP2, EaN1, EaN2, S3, and C3."""
    save_path = ensure_dir(save_dir)

    fig, axes = plt.subplots(3, 2, figsize=(14, 12))
    fig.suptitle(
        "Lightning Simulation - Key Measurement Points\n"
        "PSCAD Probes: EaP1, EaP2, EaN1, EaN2",
        fontsize=14,
        fontweight="bold",
    )

    plot_items = [
        (axes[0, 0], V_k[:, 4], "EaP1 - Sending End C3 Core, Node 5"),
        (axes[0, 1], V_k[:, 2], "EaP2 - Sending End C2 Core, Node 3"),
        (axes[1, 0], V_m[:, 4], "EaN1 - Receiving End C3 Core, Node 12"),
        (axes[1, 1], V_m[:, 2], "EaN2 - Receiving End C2 Core, Node 10"),
    ]

    for ax, voltage, title in plot_items:
        ax.plot(time_s * 1e6, voltage / 1e3, linewidth=1.5)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Time (us)")
        ax.set_ylabel("Voltage (kV)")
        ax.grid(True, alpha=0.3)

    ax = axes[2, 0]
    ax.plot(time_s * 1e6, V_k[:, 5] / 1e3, linewidth=1.5, label="S3 Sending, Node 6")
    ax.plot(time_s * 1e6, V_m[:, 5] / 1e3, linewidth=1.5, label="S3 Receiving, Node 13")
    ax.set_title("S3 Sheath - Sending vs Receiving", fontsize=10)
    ax.set_xlabel("Time (us)")
    ax.set_ylabel("Voltage (kV)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[2, 1]
    ax.plot(time_s * 1e6, V_k[:, 4] / 1e3, linewidth=1.5, label="C3 Sending, Node 5")
    ax.plot(time_s * 1e6, V_m[:, 4] / 1e3, linewidth=1.5, label="C3 Receiving, Node 12")
    ax.set_title("C3 Core - Sending vs Receiving", fontsize=10)
    ax.set_xlabel("Time (us)")
    ax.set_ylabel("Voltage (kV)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    filename = save_path / "lightning_3core_probes_latest.png"
    if SAVE_PLOTS:
        plt.savefig(filename, dpi=300, bbox_inches="tight")
        print(f"  Probe comparison plot saved: {filename}")

    if SHOW_PLOTS:
        plt.show()
    else:
        plt.close(fig)

    return filename


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_peak_info(time_s: np.ndarray, V_k: np.ndarray, V_m: np.ndarray) -> None:
    """Print peak voltage and peak time for all 14 ports."""
    print("\n" + "=" * 75)
    print("          Pipe-Type 3-Core Cable Lightning Result: Peaks")
    print("=" * 75)

    print("\nSending End - Nodes 1 to 7")
    print("-" * 70)
    print(f"{'Node':<8} {'Conductor':<18} {'Peak Voltage (kV)':<20} {'Peak Time (us)':<15}")
    print("-" * 70)

    for i in range(NC):
        peak_idx = int(np.argmax(np.abs(V_k[:, i])))
        peak_val = V_k[peak_idx, i] / 1e3
        peak_time = time_s[peak_idx] * 1e6
        marker = " *" if (i + 1) == LIGHTNING_INJECTION_NODE else ""
        print(
            f"Node {i + 1:<4} {CONDUCTOR_NAMES[i]:<18} "
            f"{peak_val:>+16.2f}      {peak_time:>10.2f}{marker}"
        )

    print("\nReceiving End - Nodes 8 to 14")
    print("-" * 70)
    print(f"{'Node':<8} {'Conductor':<18} {'Peak Voltage (kV)':<20} {'Peak Time (us)':<15}")
    print("-" * 70)

    for i in range(NC):
        peak_idx = int(np.argmax(np.abs(V_m[:, i])))
        peak_val = V_m[peak_idx, i] / 1e3
        peak_time = time_s[peak_idx] * 1e6
        print(
            f"Node {i + NC + 1:<4} {CONDUCTOR_NAMES[i]:<18} "
            f"{peak_val:>+16.2f}      {peak_time:>10.2f}"
        )

    print("=" * 75)

    vk_max_idx = np.unravel_index(np.argmax(np.abs(V_k)), V_k.shape)
    vm_max_idx = np.unravel_index(np.argmax(np.abs(V_m)), V_m.shape)

    vk_max = V_k[vk_max_idx] / 1e3
    vm_max = V_m[vm_max_idx] / 1e3
    vk_max_time = time_s[vk_max_idx[0]] * 1e6
    vm_max_time = time_s[vm_max_idx[0]] * 1e6

    print("\nSummary")
    print(
        f"  Sending-end maximum: {vk_max:+.2f} kV @ Node {vk_max_idx[1] + 1} "
        f"({CONDUCTOR_NAMES[vk_max_idx[1]]}), t = {vk_max_time:.2f} us"
    )
    print(
        f"  Receiving-end maximum: {vm_max:+.2f} kV @ Node {vm_max_idx[1] + NC + 1} "
        f"({CONDUCTOR_NAMES[vm_max_idx[1]]}), t = {vm_max_time:.2f} us"
    )

    print("\nPSCAD probe readings")
    for probe_name, info in PROBE_POINTS.items():
        idx = info["phase_idx"]
        voltage = V_k[:, idx] if info["end"] == "sending" else V_m[:, idx]
        peak_idx = int(np.argmax(np.abs(voltage)))
        peak_val = voltage[peak_idx] / 1e3
        peak_time = time_s[peak_idx] * 1e6
        print(
            f"  {probe_name}: {peak_val:+.2f} kV @ {peak_time:.2f} us "
            f"({info['desc']})"
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
        t, vk, vm, solver = run_3core_cable_simulation("open")

        print_peak_info(t, vk, vm)

        print("\nSaving measurement probe data...")
        save_probe_data(t, vk, vm, solver, save_dir=".")

        print("\nGenerating all-port voltage plot...")
        plot_all_ports_voltage(t, vk, vm, save_dir=RESULT_DIR)

        print("\nGenerating probe comparison plot...")
        plot_probe_comparison(t, vk, vm, save_dir=RESULT_DIR)

        print("\n" + "=" * 70)
        print("  Pipe-type 3-core cable lightning simulation complete.")
        print("=" * 70)

    except Exception as exc:
        print(f"\nSimulation failed: {exc}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
