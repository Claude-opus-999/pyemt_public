"""PR-0: Baseline tests — LCP modules are importable and key APIs exist."""

import py_compile
import sys
from pathlib import Path

import numpy as np
import pytest

# Ensure project root is on sys.path for LCP imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


class TestLCPModulesImportable:
    def test_cable_model_importable(self):
        from LCP import cable_model
        assert hasattr(cable_model, "compute_multi_cable_impedance")
        assert hasattr(cable_model, "compute_armored_cable_admittance")

    def test_ulm_atp_zy_deri_semlyen_importable(self):
        from LCP import ulm_atp_zy_deri_semlyen as ohl
        assert hasattr(ohl, "compute_impedance_matrix")
        assert hasattr(ohl, "compute_admittance_matrix")
        assert hasattr(ohl, "MultiConductorLine")

    def test_vf_core_importable(self):
        from LCP import vf_core
        assert hasattr(vf_core, "vector_fitting")
        assert hasattr(vf_core, "VectorFitResult")

    def test_vector_fitting_v411_importable(self):
        from LCP import vector_fitting_v411_independent as vf

    def test_vectfit3_importable(self):
        from LCP import vectfit3
        assert hasattr(vectfit3, "vectfit")


class TestFitULMAPIExists:
    def test_ulm_complete_fitting_exists(self):
        from LCP.vector_fitting_v411_independent import ulm_complete_fitting
        assert callable(ulm_complete_fitting)

    def test_write_fitULM_exists(self):
        from LCP.vector_fitting_v411_independent import write_fitULM
        assert callable(write_fitULM)

    def test_verify_fitULM_file_exists(self):
        from LCP.vector_fitting_v411_independent import verify_fitULM_file
        assert callable(verify_fitULM_file)

    def test_read_fitULM_header_exists(self):
        from LCP.vector_fitting_v411_independent import read_fitULM_header
        assert callable(read_fitULM_header)

    def test_iterative_pole_finding_config_exists(self):
        from LCP.vector_fitting_v411_independent import IterativePoleFindingConfig
        cfg = IterativePoleFindingConfig()
        assert cfg.Ymin == 10
        assert cfg.Hmin == 20


class TestFitULMFileRoundtrip:
    def test_write_and_verify_minimal_fitulm(self, tmp_path):
        """Write a minimal fitULM file and verify it passes validation."""
        from LCP.vector_fitting_v411_independent import (
            write_fitULM, verify_fitULM_file, ULMFittingResult,
        )
        fitulm_path = tmp_path / "test_minimal.fitULM"

        class MinimalHModeFit:
            poles = np.array([-1e6 + 0j])
            c_matrix_residues = np.array([[[1.0 + 0j]]])
            tau = 1e-6

        result = ULMFittingResult(
            nf=1, n_active_modes=1, active_modes=[],
            mode_groups=[[0]],
            poles_Yc=np.array([-1e5 + 0j]),
            k_residues=np.array([[[1.0 + 0j]]]),
            k0=np.array([[0.0]]),
            tau_all=np.array([1e-6]),
            D_matrices=None,
            H_modes_fits=[MinimalHModeFit()],
            Yc_trace_rmse=0.001,
            H_modes_rmse=[0.001],
            H_matrix_rmse=0.001,
            is_passive=True,
            is_freq_dependent=False,
            H_reconstruction_metrics=None,
            H_reconstructed=None,
        )

        write_fitULM(result, str(fitulm_path), precision=16, verbose=False)
        assert fitulm_path.exists()
        assert fitulm_path.stat().st_size > 0

        ok = verify_fitULM_file(str(fitulm_path), verbose=False)
        assert ok, "verify_fitULM_file returned False for a freshly written file"


class TestEMTPULMPipeline:
    def test_fitulm_reader_can_read_valid_file(self, tmp_path):
        """End to end: write a fitULM via VF module, read via EMTP's FitULMReader."""
        from LCP.vector_fitting_v411_independent import (
            write_fitULM, ULMFittingResult,
        )
        from ulm_transmission_line_PARA import FitULMReader

        fitulm_path = tmp_path / "e2e.fitULM"

        class MinimalHModeFit:
            poles = np.array([-1e6 + 0j])
            c_matrix_residues = np.array([[[1.0 + 0j]]])
            tau = 1e-6

        result = ULMFittingResult(
            nf=1, n_active_modes=1, active_modes=[],
            mode_groups=[[0]],
            poles_Yc=np.array([-1e5 + 0j]),
            k_residues=np.array([[[1.0 + 0j]]]),
            k0=np.array([[0.0]]),
            tau_all=np.array([1e-6]),
            D_matrices=None,
            H_modes_fits=[MinimalHModeFit()],
            Yc_trace_rmse=0.001,
            H_modes_rmse=[0.001],
            H_matrix_rmse=0.001,
            is_passive=True,
            is_freq_dependent=False,
            H_reconstruction_metrics=None,
            H_reconstructed=None,
        )

        write_fitULM(result, str(fitulm_path), precision=16, verbose=False)
        reader = FitULMReader(str(fitulm_path))
        fit_data = reader.read()
        assert fit_data.nf == 1


class TestAllNewFilesCompile:
    def test_lcp_and_resolver_python_files_compile(self):
        """Every .py file in pylcp/ and fitulm_resolver.py must compile clean."""
        from pathlib import Path

        files = [
            *sorted(Path("pylcp").glob("*.py")),
            *sorted(Path("pylcp/generation").glob("*.py")),
            Path("emtp/lines/fitulm_resolver.py"),
        ]

        for file in files:
            py_compile.compile(str(file), doraise=True)
