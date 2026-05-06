"""PR-6+7: Cache + end-to-end integration tests."""

from pathlib import Path

import numpy as np
import pytest

from emtp.lines.fitulm_resolver import FitULMSpec, FitULMResolver
from pylcp import LCPLineType, LCPFitULMSpec
from pylcp.cache import compute_cache_key, get_cache_path


class TestCacheKeys:
    def test_same_spec_same_key(self):
        s1 = LCPFitULMSpec(
            line_type=LCPLineType.OHL_DERI_SEMLYEN, name="t",
            length=1000.0, freq=np.array([1.0, 10.0, 100.0]),
            geometry_config={"h": 30.0},
        )
        s2 = LCPFitULMSpec(
            line_type=LCPLineType.OHL_DERI_SEMLYEN, name="t",
            length=1000.0, freq=np.array([1.0, 10.0, 100.0]),
            geometry_config={"h": 30.0},
        )
        assert compute_cache_key(s1) == compute_cache_key(s2)

    def test_different_length_different_key(self):
        s1 = LCPFitULMSpec(
            line_type=LCPLineType.OHL_DERI_SEMLYEN, name="t",
            length=1000.0, freq=np.array([1.0, 10.0]), geometry_config={},
        )
        s2 = LCPFitULMSpec(
            line_type=LCPLineType.OHL_DERI_SEMLYEN, name="t",
            length=2000.0, freq=np.array([1.0, 10.0]), geometry_config={},
        )
        assert compute_cache_key(s1) != compute_cache_key(s2)

    def test_different_freq_different_key(self):
        s1 = LCPFitULMSpec(
            line_type=LCPLineType.OHL_DERI_SEMLYEN, name="t",
            length=1000.0, freq=np.array([1.0, 10.0]), geometry_config={},
        )
        s2 = LCPFitULMSpec(
            line_type=LCPLineType.OHL_DERI_SEMLYEN, name="t",
            length=1000.0, freq=np.array([1.0, 20.0]), geometry_config={},
        )
        assert compute_cache_key(s1) != compute_cache_key(s2)

    def test_cache_path_naming(self):
        spec = LCPFitULMSpec(
            line_type=LCPLineType.PIPE_TYPE_CABLE, name="cable14",
            length=5000.0, freq=np.logspace(0, 5, 51),
            geometry_config={"cable": "test"},
            cache_dir=Path(".lcp_cache"),
        )
        path = get_cache_path(spec)
        assert path.parent.name == ".lcp_cache"
        assert path.name.startswith("cable14")
        assert path.name.endswith(".fitULM")


def _write_minimal_valid_fitulm(path):
    """Write a real valid fitULM file that passes verify_fitULM_file."""
    import numpy as np
    from LCP.vector_fitting_v411_independent import write_fitULM, ULMFittingResult

    class MinimalHModeFit:
        poles = np.array([-1e6 + 0j])
        c_matrix_residues = np.array([[[1.0 + 0j]]])
        tau = 1e-6

    result = ULMFittingResult(
        nf=1, n_active_modes=1, active_modes=[], mode_groups=[[0]],
        poles_Yc=np.array([-1e5 + 0j]),
        k_residues=np.array([[[1.0 + 0j]]]),
        k0=np.array([[0.0]]),
        tau_all=np.array([1e-6]),
        D_matrices=None, H_modes_fits=[MinimalHModeFit()],
        Yc_trace_rmse=0.001, H_modes_rmse=[0.001], H_matrix_rmse=0.001,
        is_passive=True, is_freq_dependent=False,
        H_reconstruction_metrics=None, H_reconstructed=None,
    )
    write_fitULM(result, str(path), precision=16, verbose=False)


class TestResolverCacheReuse:
    def test_cache_reuse_skips_generation(self, tmp_path, monkeypatch):
        """When a cached file exists and force_recompute=False, skip generation."""
        cached_file = tmp_path / "cached.fitULM"
        _write_minimal_valid_fitulm(cached_file)

        class FakeSpec:
            line_type = LCPLineType.OHL_DERI_SEMLYEN
            output_path = cached_file
            name = "cached"

        fitulm_spec = FitULMSpec(
            name="cached", generate_fitulm=True,
            lcp_spec=FakeSpec(),
        )

        generated = []
        def mock_generate(self, spec):
            generated.append(True)
            return cached_file

        monkeypatch.setattr(
            "pylcp.lcp_fitulm_generator.LCPFitULMGenerator.generate",
            mock_generate,
        )

        result = FitULMResolver().resolve(fitulm_spec)
        assert result == cached_file
        assert len(generated) == 0  # cached, not regenerated

    def test_force_recompute_regenerates(self, tmp_path, monkeypatch):
        """force_recompute=True regenerates even if cache exists."""
        cached_file = tmp_path / "cached.fitULM"
        _write_minimal_valid_fitulm(cached_file)

        class FakeSpec:
            line_type = LCPLineType.OHL_DERI_SEMLYEN
            output_path = cached_file
            name = "cached"

        fitulm_spec = FitULMSpec(
            name="cached", generate_fitulm=True,
            lcp_spec=FakeSpec(),
            force_recompute=True,
        )

        generated = []
        def mock_generate(self, spec):
            generated.append(True)
            return cached_file

        monkeypatch.setattr(
            "pylcp.lcp_fitulm_generator.LCPFitULMGenerator.generate",
            mock_generate,
        )

        FitULMResolver().resolve(fitulm_spec)
        assert len(generated) == 1  # regenerated


class TestE2ESolverWithExternalFitULM:
    def test_add_ulm_line_and_run_minimal_simulation(self, tmp_path):
        """End-to-end: generate fitULM, load as ULM line, run solver."""
        from LCP.vector_fitting_v411_independent import write_fitULM, ULMFittingResult
        from emtp import EMTPSolver

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
            Yc_trace_rmse=0.001, H_modes_rmse=[0.001], H_matrix_rmse=0.001,
            is_passive=True, is_freq_dependent=False,
            H_reconstruction_metrics=None, H_reconstructed=None,
        )
        write_fitULM(result, str(fitulm_path), precision=16, verbose=False)

        solver = EMTPSolver(dt=1e-6, finish_time=50e-6, verbose=False)
        solver.add_VS("Vs", 1, 0, 1.0)
        solver.add_R("Rload", 2, 0, 50.0)
        solver.add_ULM_line(
            name="ulm_line",
            nodes_send=1, nodes_recv=2,
            length=1000.0,
            generate_fitulm=False,
            fitulm_path=str(fitulm_path),
        )
        solver.add_voltage_probe("V_recv", 2, 0)
        solver.run()

        v = solver.get_voltage_probe("V_recv", "V")
        assert len(v) > 0
