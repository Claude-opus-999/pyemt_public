# -*- coding: utf-8 -*-
"""
Lightning surge intrusion simulation with PSCAD-style segmented MOA.

Adapted for emtp.

Compatibility changes
---------------------
1. Import EMTPSolver from emtp instead of emtp_solver_v2_time.
2. Remove old dependencies:
   - emtp_components.CurrentSource
   - lightning_waveform.LightningWaveform / create_lightning_waveform
3. Use EMTPSolver.add_standard_twoexpf_IS() for the ATP-compatible 2/20 us
   lightning current source.
4. Keep add_MOA_from_file() for the PSCAD segmented linear MOA model.
5. Enable the histories required by this analysis:
   - record_source_history=True
   - record_branch_history=True
   - record_all_node_voltages=True
6. Resolve the V-I file robustly, including filenames with spaces.
7. Save plots and CSV outputs to RESULT_DIR by default.
"""

from __future__ import annotations

import csv
import os
import sys
import warnings
from pathlib import Path
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# ---------------------------------------------------------------------------
# Matplotlib configuration
# ---------------------------------------------------------------------------

plt.rcParams["font.sans-serif"] = [
    "DejaVu Sans",
    "Arial Unicode MS",
    "SimHei",
    "Microsoft YaHei",
]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 120
plt.rcParams["savefig.dpi"] = 150

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")


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
# Global output controls
# ---------------------------------------------------------------------------

RESULT_DIR = "result_pscad_latest"
SAVE_PLOTS = True
SHOW_PLOTS = False


def ensure_dir(path_like: str | os.PathLike) -> Path:
    path = Path(path_like)
    path.mkdir(parents=True, exist_ok=True)
    return path


def resolve_existing_file(path_like: str | os.PathLike) -> Path:
    """Resolve input file relative to cwd, script dir, and project dir."""
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
        f"Could not find file: {path_like}\n"
        f"Searched:\n{searched}"
    )


class LightningSimulationAnalyzer:
    """Lightning intrusion analyzer using PSCAD segmented MOA method."""

    def __init__(self):
        self.solver: EMTPSolver | None = None
        self.lightning_source = None
        self.moa_model = None
        self.results: Dict[str, np.ndarray] = {}
        self.metrics: Dict[str, float] = {}

        self.config = {
            "dt": 1e-8,
            "finish_time": 1e-4,
            "lightning_type": "2/20",
            "lightning_peak": 10e3,
            "lightning_t_start": 0.0,
            "lightning_PERC": 30,
            "line_impedance": 400.0,
            "moa_rated_voltage": 96e3,
            "moa_vi_file": "V_I _old.txt",
            "transformer_capacitance": 5e-9,
        }

    # ---------------------------------------------------------------------
    # Model setup and run
    # ---------------------------------------------------------------------

    def setup_circuit(self) -> None:
        """Build the circuit model."""
        print("=" * 70)
        print("Lightning intrusion simulation with PSCAD segmented MOA")
        print("=" * 70)

        vi_file = resolve_existing_file(self.config["moa_vi_file"])

        self.solver = EMTPSolver(
            dt=self.config["dt"],
            finish_time=self.config["finish_time"],
            verbose=True,

            # Required by get_node_voltage().
            record_all_node_voltages=True,

            # Required by get_branch_current() for MOA/R/C/R_line histories.
            record_branch_history=True,

            # Required by get_source_current("IS_lightning").
            record_source_history=True,

            # No transmission line histories are used in this lumped case.
            record_line_history=False,

            # Latest solver fast paths.
            pre_sample_sources=True,
            use_rhs_plan=True,
        )

        # Lightning current source.
        # Current direction follows EMTPSolver convention: node_from -> node_to.
        # Therefore 0 -> 1 injects positive lightning current into node 1.
        self.lightning_source = self.solver.add_standard_twoexpf_IS(
            name="IS_lightning",
            node_from=0,
            node_to=1,
            waveform_type=self.config["lightning_type"],
            peak=self.config["lightning_peak"],
            PERC=self.config.get("lightning_PERC", 30),
            Tstart=self.config["lightning_t_start"],
            atp_compatible=True,
            description="Substation incoming lightning current",
        )

        # Source-side line surge impedance.
        self.solver.add_R(
            "R_line",
            1,
            2,
            self.config["line_impedance"],
        )

        # PSCAD-style segmented linear MOA model from V-I file.
        self.solver.add_MOA_from_file(
            name="MOA1",
            node_from=2,
            node_to=0,
            file_path=str(vi_file),
            rated_voltage=self.config["moa_rated_voltage"],
            voltage_is_pu=True,
        )

        # Keep model reference for V-I plotting.
        self.moa_model = self.solver.branches["MOA1"].nonlinear_model

        # Transformer entrance equivalent capacitance.
        self.solver.add_C(
            "C_transformer",
            2,
            0,
            self.config["transformer_capacitance"],
        )

        self.solver.print_circuit_summary()

    def run_simulation(self) -> None:
        """Run simulation and extract waveforms."""
        if self.solver is None:
            raise RuntimeError("Call setup_circuit() before run_simulation().")

        print("\n[Running simulation...]")
        self.solver.run()
        print("[Simulation complete.]")

        self.results = {
            "t_us": self.solver.get_time(unit="us"),
            "t_s": self.solver.get_time(unit="s"),

            "I_lightning_A": self.solver.get_source_current("IS_lightning"),
            "I_lightning_kA": self.solver.get_source_current("IS_lightning") / 1e3,

            "V_bus_V": self.solver.get_node_voltage(1, unit="V"),
            "V_bus_kV": self.solver.get_node_voltage(1, unit="kV"),

            "V_moa_V": self.solver.get_node_voltage(2, unit="V"),
            "V_moa_kV": self.solver.get_node_voltage(2, unit="kV"),

            "I_moa_A": self.solver.get_branch_current("MOA1", unit="A"),
            "I_moa_kA": self.solver.get_branch_current("MOA1", unit="kA"),

            "I_cap_A": self.solver.get_branch_current("C_transformer", unit="A"),
            "I_cap_kA": self.solver.get_branch_current("C_transformer", unit="kA"),

            "I_line_A": self.solver.get_branch_current("R_line", unit="A"),
            "I_line_kA": self.solver.get_branch_current("R_line", unit="kA"),
        }

        self._calculate_metrics()

    def _calculate_metrics(self) -> None:
        """Calculate key metrics."""
        r = self.results

        i_light_abs = np.abs(r["I_lightning_kA"])
        v_bus_abs = np.abs(r["V_bus_kV"])
        v_moa_abs = np.abs(r["V_moa_kV"])
        i_moa_abs = np.abs(r["I_moa_kA"])
        i_cap_abs = np.abs(r["I_cap_A"])

        i_peak_idx = int(np.argmax(i_light_abs))
        v_bus_peak_idx = int(np.argmax(v_bus_abs))
        v_moa_peak_idx = int(np.argmax(v_moa_abs))
        i_moa_peak_idx = int(np.argmax(i_moa_abs))
        i_cap_peak_idx = int(np.argmax(i_cap_abs))

        self.metrics = {
            "I_lightning_peak_kA": float(i_light_abs[i_peak_idx]),
            "t_I_peak_us": float(r["t_us"][i_peak_idx]),

            "V_bus_peak_kV": float(v_bus_abs[v_bus_peak_idx]),
            "t_V_bus_peak_us": float(r["t_us"][v_bus_peak_idx]),

            "V_moa_peak_kV": float(v_moa_abs[v_moa_peak_idx]),
            "t_V_moa_peak_us": float(r["t_us"][v_moa_peak_idx]),

            "I_moa_peak_kA": float(i_moa_abs[i_moa_peak_idx]),
            "t_I_moa_peak_us": float(r["t_us"][i_moa_peak_idx]),

            "I_cap_peak_A": float(i_cap_abs[i_cap_peak_idx]),
            "t_I_cap_peak_us": float(r["t_us"][i_cap_peak_idx]),
        }

    # ---------------------------------------------------------------------
    # Reporting
    # ---------------------------------------------------------------------

    def print_results(self) -> None:
        """Print result summary."""
        if self.solver is None:
            raise RuntimeError("No solver available.")

        m = self.metrics
        stats = self.solver.get_solver_statistics()

        print("\n" + "=" * 70)
        print("Simulation result summary")
        print("=" * 70)

        print("\n[Lightning source]")
        print(f"  Waveform type:     {self.config['lightning_type']} us")
        print(f"  Peak current:      {m['I_lightning_peak_kA']:.3f} kA")
        print(f"  Peak time:         {m['t_I_peak_us']:.3f} us")
        print(f"  Start time:        {self.config['lightning_t_start'] * 1e6:.3f} us")

        print("\n[Node voltage]")
        print(f"  Node 1 bus peak:   {m['V_bus_peak_kV']:.3f} kV @ {m['t_V_bus_peak_us']:.3f} us")
        print(f"  Node 2 MOA peak:   {m['V_moa_peak_kV']:.3f} kV @ {m['t_V_moa_peak_us']:.3f} us")

        print("\n[MOA]")
        print(f"  Rated voltage:     {self.config['moa_rated_voltage'] / 1e3:.1f} kV")
        print(f"  Peak current:      {m['I_moa_peak_kA']:.3f} kA @ {m['t_I_moa_peak_us']:.3f} us")
        print(f"  Limiting effect:   {m['I_lightning_peak_kA']:.2f} kA -> {m['V_moa_peak_kV']:.2f} kV")

        print("\n[Transformer entrance capacitance]")
        print(f"  Capacitance:       {self.config['transformer_capacitance'] * 1e9:.3f} nF")
        print(f"  Peak current:      {m['I_cap_peak_A']:.3f} A @ {m['t_I_cap_peak_us']:.3f} us")

        print("\n[Solver statistics]")
        print(f"  Total steps:       {stats.get('total_steps')}")
        print(f"  MNA size:          {stats.get('mna_size')}")
        print(f"  G rebuilds:        {stats.get('G_rebuilds')}")
        print(f"  G cache hits:      {stats.get('G_cache_hits')}")
        print(f"  Segment switches:  {stats.get('segment_switches', 0)}")
        print(f"  Segment ratio:     {stats.get('segment_switch_ratio', 0.0) * 100:.3f}%")
        print(f"  Segment resolves:  {stats.get('segment_resolves', 0)}")

        print("=" * 70)

    # ---------------------------------------------------------------------
    # Plot helpers
    # ---------------------------------------------------------------------

    def _save_or_show(self, fig, save_path: str | os.PathLike) -> Path:
        path = Path(save_path)
        if SAVE_PLOTS:
            fig.savefig(path, bbox_inches="tight")
            print(f"  Saved: {path}")

        if SHOW_PLOTS:
            plt.show()
        else:
            plt.close(fig)

        return path

    def _moa_static_curve(self, n_points: int = 300):
        if self.moa_model is None:
            raise RuntimeError("MOA model not available.")

        breakpoints = self.moa_model.get_all_breakpoints()
        v_data = np.array([bp[0] for bp in breakpoints], dtype=float) / 1e3
        i_data = np.array([bp[1] for bp in breakpoints], dtype=float) / 1e3

        valid = (v_data > 0) & (i_data > 0)
        v_data = v_data[valid]
        i_data = i_data[valid]

        if v_data.size < 2:
            raise RuntimeError("MOA V-I curve does not contain enough valid points.")

        v_interp = np.linspace(v_data.min(), v_data.max(), n_points)
        i_interp = []
        for v_kv in v_interp:
            v_v = float(v_kv * 1e3)
            try:
                self.moa_model.update_segment(v_v)
            except Exception:
                pass
            i_interp.append(abs(self.moa_model.get_current(v_v)) / 1e3)

        return v_data, i_data, v_interp, np.asarray(i_interp, dtype=float)

    # ---------------------------------------------------------------------
    # Plots
    # ---------------------------------------------------------------------

    def plot_lightning_waveform(self, save_path="01_lightning_waveform.png") -> Path:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle("Figure 1: Lightning Current Source Waveform", fontsize=14, fontweight="bold")

        t = self.results["t_us"]
        current = self.results["I_lightning_kA"]

        ax1 = axes[0]
        ax1.plot(t, current, linewidth=2, label="Lightning Current")
        ax1.fill_between(t, 0, current, alpha=0.3)
        ax1.set_xlabel("Time (us)")
        ax1.set_ylabel("Current (kA)")
        ax1.set_title(f"(a) Full Waveform ({self.config['lightning_type']} us)")
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim([0, max(t)])
        ax1.set_ylim([0, max(current) * 1.1])
        ax1.legend(loc="upper right")

        i_peak = self.metrics["I_lightning_peak_kA"]
        t_peak = self.metrics["t_I_peak_us"]
        ax1.annotate(
            f"Peak: {i_peak:.2f} kA\n@ {t_peak:.2f} us",
            xy=(t_peak, i_peak),
            xytext=(t_peak + 8, i_peak * 0.85),
            arrowprops=dict(arrowstyle="->", lw=1.5),
            fontsize=10,
            fontweight="bold",
        )

        ax2 = axes[1]
        try:
            t_front_end = float(str(self.config["lightning_type"]).split("/")[0]) * 3.0
        except Exception:
            t_front_end = 30.0

        mask = t <= t_front_end
        ax2.plot(t[mask], current[mask], linewidth=2)
        ax2.fill_between(t[mask], 0, current[mask], alpha=0.3)
        ax2.set_xlabel("Time (us)")
        ax2.set_ylabel("Current (kA)")
        ax2.set_title("(b) Wavefront Detail")
        ax2.grid(True, alpha=0.3)

        ax2.axhline(y=i_peak * 0.1, linestyle=":", alpha=0.7)
        ax2.axhline(y=i_peak * 0.9, linestyle=":", alpha=0.7)
        ax2.text(0.1, i_peak * 0.1, "10%", fontsize=9)
        ax2.text(0.1, i_peak * 0.9, "90%", fontsize=9)

        plt.tight_layout()
        return self._save_or_show(fig, save_path)

    def plot_moa_vi_characteristic(self, save_path="02_moa_vi_characteristic.png") -> Path:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle("Figure 2: MOA V-I Characteristic", fontsize=14, fontweight="bold")

        v_data, i_data, v_interp, i_interp = self._moa_static_curve()

        ax1 = axes[0]
        ax1.semilogy(v_data, i_data, "o-", markersize=7, linewidth=2, label="V-I Data Points")
        ax1.semilogy(v_interp, np.maximum(i_interp, 1e-12), "-", linewidth=1.5, alpha=0.8, label="Segmented Curve")
        ax1.set_xlabel("Voltage (kV)")
        ax1.set_ylabel("Current (kA), log scale")
        ax1.set_title("(a) Static V-I Characteristic")
        ax1.grid(True, alpha=0.3, which="both")
        ax1.legend()

        vref_kv = self.config["moa_rated_voltage"] / 1e3
        ax1.axvline(x=vref_kv, linestyle="--", alpha=0.7)
        ax1.text(vref_kv * 1.02, max(i_data.min(), 1e-6), f"Vref = {vref_kv:.0f} kV", fontsize=9)

        ax2 = axes[1]
        v_moa = np.abs(self.results["V_moa_kV"])
        i_moa = np.abs(self.results["I_moa_kA"])
        t = self.results["t_us"]

        scatter = ax2.scatter(np.maximum(i_moa, 1e-12), v_moa, c=t, cmap="plasma", s=5, alpha=0.7)
        cbar = plt.colorbar(scatter, ax=ax2)
        cbar.set_label("Time (us)")

        ax2.plot(np.maximum(i_interp, 1e-12), v_interp, "--", linewidth=1.5, alpha=0.8, label="Static Curve")
        ax2.set_xlabel("MOA Current (kA)")
        ax2.set_ylabel("MOA Voltage (kV)")
        ax2.set_title("(b) Dynamic V-I Trajectory")
        ax2.set_xscale("log")
        xmax = max(float(np.nanmax(np.maximum(i_moa, 1e-9))) * 2, 1e-3)
        ax2.set_xlim([1e-6, xmax])
        ax2.grid(True, alpha=0.3, which="both")
        ax2.legend(loc="lower right")

        plt.tight_layout()
        return self._save_or_show(fig, save_path)

    def plot_moa_voltage_current(self, save_path="03_moa_voltage_current.png") -> Path:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle("Figure 3: MOA Voltage and Current Waveforms", fontsize=14, fontweight="bold")

        t = self.results["t_us"]
        v_moa = self.results["V_moa_kV"]
        i_moa = self.results["I_moa_kA"]
        i_lightning = self.results["I_lightning_kA"]

        ax1 = axes[0, 0]
        ax1.plot(t, v_moa, linewidth=2, label="MOA Voltage")
        ax1.fill_between(t, 0, v_moa, alpha=0.2)
        ax1.set_xlabel("Time (us)")
        ax1.set_ylabel("Voltage (kV)")
        ax1.set_title("(a) MOA Voltage")
        ax1.grid(True, alpha=0.3)
        ax1.legend()

        ax2 = axes[0, 1]
        ax2.plot(t, i_moa, linewidth=2, label="MOA Current")
        ax2.fill_between(t, 0, i_moa, alpha=0.2)
        ax2.set_xlabel("Time (us)")
        ax2.set_ylabel("Current (kA)")
        ax2.set_title("(b) MOA Current")
        ax2.grid(True, alpha=0.3)
        ax2.legend()

        ax3 = axes[1, 0]
        line1, = ax3.plot(t, v_moa, linewidth=2, label="Voltage")
        ax3.set_xlabel("Time (us)")
        ax3.set_ylabel("Voltage (kV)")
        ax3_twin = ax3.twinx()
        line2, = ax3_twin.plot(t, i_moa, linewidth=2, linestyle="--", label="Current")
        ax3_twin.set_ylabel("Current (kA)")
        ax3.set_title("(c) Voltage vs Current")
        ax3.grid(True, alpha=0.3)
        ax3.legend([line1, line2], ["Voltage", "Current"], loc="upper right")

        ax4 = axes[1, 1]
        ax4.plot(t, i_lightning, linewidth=2, label="Lightning Current")
        ax4.plot(t, i_moa, linewidth=2, label="MOA Current")
        ax4.set_xlabel("Time (us)")
        ax4.set_ylabel("Current (kA)")
        ax4.set_title("(d) Lightning Current vs MOA Current")
        ax4.grid(True, alpha=0.3)
        ax4.legend()

        denom = max(self.metrics["I_lightning_peak_kA"], 1e-12)
        ax4.text(
            max(t) * 0.55,
            max(i_lightning) * 0.5,
            f"MOA peak share\n{self.metrics['I_moa_peak_kA'] / denom * 100:.1f}%",
            fontsize=10,
            bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8),
        )

        plt.tight_layout()
        return self._save_or_show(fig, save_path)

    def plot_transformer_capacitor(self, save_path="04_transformer_capacitor.png") -> Path:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle("Figure 4: Transformer Equivalent Capacitor", fontsize=14, fontweight="bold")

        t = self.results["t_us"]
        v_cap = self.results["V_moa_kV"]
        i_cap = self.results["I_cap_A"]

        ax1 = axes[0, 0]
        ax1.plot(t, v_cap, linewidth=2, label="Capacitor Voltage")
        ax1.fill_between(t, 0, v_cap, alpha=0.2)
        ax1.set_xlabel("Time (us)")
        ax1.set_ylabel("Voltage (kV)")
        ax1.set_title("(a) Capacitor Voltage")
        ax1.grid(True, alpha=0.3)
        ax1.legend()

        ax2 = axes[0, 1]
        ax2.plot(t, i_cap, linewidth=2, label="Capacitor Current")
        ax2.axhline(y=0, linewidth=0.8, alpha=0.6)
        ax2.fill_between(t, 0, i_cap, where=(i_cap >= 0), alpha=0.2, label="Charging")
        ax2.fill_between(t, 0, i_cap, where=(i_cap < 0), alpha=0.2, label="Discharging")
        ax2.set_xlabel("Time (us)")
        ax2.set_ylabel("Current (A)")
        ax2.set_title("(b) Capacitor Current")
        ax2.grid(True, alpha=0.3)
        ax2.legend()

        ax3 = axes[1, 0]
        line1, = ax3.plot(t, v_cap, linewidth=2, label="Voltage")
        ax3.set_xlabel("Time (us)")
        ax3.set_ylabel("Voltage (kV)")
        ax3_twin = ax3.twinx()
        line2, = ax3_twin.plot(t, i_cap, linewidth=2, linestyle="--", label="Current")
        ax3_twin.set_ylabel("Current (A)")
        ax3.set_title("(c) Voltage vs Current")
        ax3.grid(True, alpha=0.3)
        ax3.legend([line1, line2], ["Voltage", "Current"], loc="upper right")

        ax4 = axes[1, 1]
        cap = self.config["transformer_capacitance"]
        energy_mJ = 0.5 * cap * self.results["V_moa_V"] ** 2 * 1e3
        ax4.plot(t, energy_mJ, linewidth=2, label="Stored Energy")
        ax4.fill_between(t, 0, energy_mJ, alpha=0.2)
        ax4.set_xlabel("Time (us)")
        ax4.set_ylabel("Energy (mJ)")
        ax4.set_title("(d) Capacitor Stored Energy")
        ax4.grid(True, alpha=0.3)
        ax4.legend()

        if energy_mJ.size:
            idx = int(np.argmax(energy_mJ))
            ax4.annotate(
                f"Wpeak = {energy_mJ[idx]:.2f} mJ",
                xy=(t[idx], energy_mJ[idx]),
                xytext=(t[idx] + 10, energy_mJ[idx] * 0.8 if energy_mJ[idx] else 0.1),
                arrowprops=dict(arrowstyle="->"),
                fontsize=10,
                fontweight="bold",
            )

        plt.tight_layout()
        return self._save_or_show(fig, save_path)

    def plot_comprehensive_summary(self, save_path="05_comprehensive_summary.png") -> Path:
        fig = plt.figure(figsize=(16, 12))
        fig.suptitle("Figure 5: Comprehensive Simulation Summary", fontsize=14, fontweight="bold")

        gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.3)
        t = self.results["t_us"]

        ax1 = fig.add_subplot(gs[0, :2])
        ax1.plot(t, self.results["I_lightning_kA"], linewidth=2, label="Lightning Source")
        ax1.plot(t, self.results["I_moa_kA"], linewidth=2, label="MOA")
        ax1.plot(t, self.results["I_line_kA"], linestyle="--", linewidth=1.5, label="Line Impedance")
        ax1.plot(t, self.results["I_cap_kA"], linestyle=":", linewidth=1.5, label="Transformer Cap")
        ax1.set_xlabel("Time (us)")
        ax1.set_ylabel("Current (kA)")
        ax1.set_title("(a) Current Distribution")
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc="upper right", fontsize=9)

        stats = self.solver.get_solver_statistics() if self.solver else {}

        ax2 = fig.add_subplot(gs[0, 2])
        ax2.axis("off")
        param_text = f"""Circuit Parameters
{'=' * 30}
Lightning:
  Type: {self.config['lightning_type']} us
  Peak: {self.config['lightning_peak'] / 1e3:.1f} kA

Line:
  Z = {self.config['line_impedance']:.0f} ohm

MOA:
  Rated: {self.config['moa_rated_voltage'] / 1e3:.0f} kV
  Method: PSCAD segmented

Transformer Cap:
  C = {self.config['transformer_capacitance'] * 1e9:.1f} nF

Solver:
  Steps: {stats.get('total_steps')}
  G rebuilds: {stats.get('G_rebuilds')}
  Seg. switches: {stats.get('segment_switches', 0)}
  Resolves: {stats.get('segment_resolves', 0)}
{'=' * 30}"""
        ax2.text(
            0.06,
            0.95,
            param_text,
            transform=ax2.transAxes,
            fontsize=10,
            va="top",
            fontfamily="monospace",
            bbox=dict(boxstyle="round", facecolor="lightblue", alpha=0.45),
        )

        ax3 = fig.add_subplot(gs[1, 0])
        ax3.plot(t, self.results["V_moa_kV"], linewidth=2)
        ax3.fill_between(t, 0, self.results["V_moa_kV"], alpha=0.2)
        ax3.set_xlabel("Time (us)")
        ax3.set_ylabel("Voltage (kV)")
        ax3.set_title("(b) MOA Bus Voltage")
        ax3.grid(True, alpha=0.3)

        ax4 = fig.add_subplot(gs[1, 1])
        v_moa = np.abs(self.results["V_moa_kV"])
        i_moa = np.abs(self.results["I_moa_kA"])
        scatter = ax4.scatter(np.maximum(i_moa, 1e-12), v_moa, c=t, cmap="viridis", s=3, alpha=0.6)
        plt.colorbar(scatter, ax=ax4, label="Time (us)")
        ax4.set_xlabel("Current (kA)")
        ax4.set_ylabel("Voltage (kV)")
        ax4.set_title("(c) MOA Dynamic V-I")
        ax4.set_xscale("log")
        ax4.set_xlim([1e-6, max(float(np.nanmax(np.maximum(i_moa, 1e-9))) * 2, 1e-3)])
        ax4.grid(True, alpha=0.3, which="both")

        ax5 = fig.add_subplot(gs[1, 2])
        ax5.axis("off")
        result_text = f"""Simulation Results
{'=' * 30}
Lightning Peak:
  {self.metrics['I_lightning_peak_kA']:.2f} kA
  @ {self.metrics['t_I_peak_us']:.2f} us

MOA Voltage Peak:
  {self.metrics['V_moa_peak_kV']:.2f} kV
  @ {self.metrics['t_V_moa_peak_us']:.2f} us

MOA Current Peak:
  {self.metrics['I_moa_peak_kA']:.2f} kA

Cap Current Peak:
  {self.metrics['I_cap_peak_A']:.2f} A
{'=' * 30}"""
        ax5.text(
            0.06,
            0.95,
            result_text,
            transform=ax5.transAxes,
            fontsize=10,
            va="top",
            fontfamily="monospace",
            bbox=dict(boxstyle="round", facecolor="lightgreen", alpha=0.45),
        )

        ax6 = fig.add_subplot(gs[2, :])
        p_moa = np.abs(self.results["V_moa_V"]) * np.abs(self.results["I_moa_A"]) / 1e6
        p_line = (self.results["V_bus_V"] - self.results["V_moa_V"]) ** 2 / self.config["line_impedance"] / 1e6
        p_cap = np.abs(self.results["V_moa_V"]) * np.abs(self.results["I_cap_A"]) / 1e6

        ax6.plot(t, p_moa, linewidth=2, label="MOA Power")
        ax6.plot(t, p_line, linestyle="--", linewidth=1.5, label="Line Power")
        ax6.plot(t, p_cap, linestyle=":", linewidth=1.5, label="Capacitor Power")
        ax6.fill_between(t, 0, p_moa, alpha=0.2)
        ax6.set_xlabel("Time (us)")
        ax6.set_ylabel("Power (MW)")
        ax6.set_title("(d) Instantaneous Power Distribution")
        ax6.grid(True, alpha=0.3)
        ax6.legend(loc="upper right", fontsize=9)

        return self._save_or_show(fig, save_path)

    def generate_all_plots(self, output_dir: str | os.PathLike = RESULT_DIR) -> List[Path]:
        """Generate all analysis plots."""
        print("\n[Generating analysis plots...]")
        out_dir = ensure_dir(output_dir)

        plots = [
            self.plot_lightning_waveform(out_dir / "01_lightning_waveform.png"),
            self.plot_moa_vi_characteristic(out_dir / "02_moa_vi_characteristic.png"),
            self.plot_moa_voltage_current(out_dir / "03_moa_voltage_current.png"),
            self.plot_transformer_capacitor(out_dir / "04_transformer_capacitor.png"),
            self.plot_comprehensive_summary(out_dir / "05_comprehensive_summary.png"),
        ]
        return plots

    def save_data_to_csv(self, filepath: str | os.PathLike = "simulation_data_latest.csv") -> Path:
        """Save simulation waveforms to CSV."""
        path = Path(filepath)
        if path.parent and str(path.parent) != ".":
            path.parent.mkdir(parents=True, exist_ok=True)

        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "Time_us",
                "I_lightning_kA",
                "V_bus_kV",
                "V_moa_kV",
                "I_moa_kA",
                "I_cap_A",
                "I_line_kA",
            ])

            for i in range(len(self.results["t_us"])):
                writer.writerow([
                    self.results["t_us"][i],
                    self.results["I_lightning_kA"][i],
                    self.results["V_bus_kV"][i],
                    self.results["V_moa_kV"][i],
                    self.results["I_moa_kA"][i],
                    self.results["I_cap_A"][i],
                    self.results["I_line_kA"][i],
                ])

        print(f"  Saved data: {path}")
        return path


def main() -> LightningSimulationAnalyzer:
    print("=" * 70)
    print("Lightning intrusion simulation - PSCAD segmented MOA, emtp")
    print("=" * 70)

    analyzer = LightningSimulationAnalyzer()
    analyzer.setup_circuit()
    analyzer.run_simulation()
    analyzer.print_results()
    analyzer.generate_all_plots(RESULT_DIR)
    analyzer.save_data_to_csv(Path(RESULT_DIR) / "simulation_data_latest.csv")

    print("\n" + "=" * 70)
    print("Simulation analysis complete.")
    print("=" * 70)
    return analyzer


if __name__ == "__main__":
    analyzer = main()
