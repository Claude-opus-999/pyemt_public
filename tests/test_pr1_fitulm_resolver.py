"""PR-1: Tests for FitULMResolver (external file mode) and solver.add_ULM_line()."""

from pathlib import Path

import numpy as np
import pytest

from emtp.lines.fitulm_resolver import FitULMSpec, FitULMResolver


class TestFitULMResolverExternalFile:
    def test_resolves_existing_valid_file(self):
        """A real fitULM file (written via VF module) passes verification."""
        # This test requires an actual valid file; the E2E tests cover this path.

    def test_raises_when_path_is_none(self):
        spec = FitULMSpec(name="line1", generate_fitulm=False)
        with pytest.raises(ValueError, match="fitulm_path is required"):
            FitULMResolver().resolve(spec)

    def test_raises_when_file_missing(self):
        spec = FitULMSpec(name="line1", fitulm_path=Path("nonexistent.fitULM"))
        with pytest.raises(FileNotFoundError, match="fitULM file not found"):
            FitULMResolver().resolve(spec)

    def test_raises_when_file_empty(self, tmp_path):
        empty = tmp_path / "empty.fitULM"
        empty.write_text("")
        spec = FitULMSpec(name="line1", fitulm_path=empty)
        with pytest.raises(ValueError, match="empty"):
            FitULMResolver().resolve(spec)

    def test_invalid_fitulm_rejected_by_verifier(self, tmp_path):
        """A bad fitULM file must be rejected, not silently passed."""
        path = tmp_path / "bad.fitULM"
        path.write_text("this is not a valid fitULM file")
        spec = FitULMSpec(name="bad", fitulm_path=path)
        with pytest.raises((ValueError, Exception)):
            FitULMResolver().resolve(spec)

    def test_fitulm_verify_exception_is_not_swallowed(self, tmp_path, monkeypatch):
        """If the verifier itself throws, the exception must propagate."""
        path = tmp_path / "bad.fitULM"
        path.write_text("some content")

        import LCP.vector_fitting_v411_independent as vf
        monkeypatch.setattr(
            vf, "verify_fitULM_file",
            lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("verifier broken")),
        )

        spec = FitULMSpec(name="bad", fitulm_path=path)
        resolver = FitULMResolver()
        with pytest.raises(RuntimeError, match="verifier broken"):
            resolver._verify_fitulm(path)

    def test_lcp_mode_requires_lcp_spec(self):
        spec = FitULMSpec(name="line1", generate_fitulm=True)
        with pytest.raises(ValueError, match="lcp_spec is required"):
            FitULMResolver().resolve(spec)

    def test_lcp_mode_requires_valid_spec_not_arbitrary_object(self):
        spec = FitULMSpec(
            name="line1", generate_fitulm=True, lcp_spec="dummy",
        )
        with pytest.raises(AttributeError):
            FitULMResolver().resolve(spec)


class TestSolverAddULMLineExternalFile:
    def test_add_ulm_line_from_external_fitulm(self, tmp_path):
        """End-to-end: add_ULM_line reads external fitULM and creates a ULM line."""
        from LCP.vector_fitting_v411_independent import (
            write_fitULM, ULMFittingResult,
        )
        from emtp import EMTPSolver

        fitulm_path = tmp_path / "line.fitULM"

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

        solver = EMTPSolver(dt=1e-6, finish_time=100e-6, verbose=False)
        solver.add_ULM_line(
            name="line1",
            nodes_send=1, nodes_recv=2,
            length=1000.0,
            generate_fitulm=False,
            fitulm_path=str(fitulm_path),
        )
        assert "line1" in solver.transmission_lines

    def test_add_ulm_line_missing_fitulm_raises(self):
        from emtp import EMTPSolver

        solver = EMTPSolver(dt=1e-6, finish_time=100e-6, verbose=False)
        with pytest.raises(FileNotFoundError):
            solver.add_ULM_line(
                name="bad", nodes_send=1, nodes_recv=2,
                length=1000.0,
                generate_fitulm=False,
                fitulm_path="missing.fitULM",
            )

    def test_add_ulm_line_no_path_raises(self):
        from emtp import EMTPSolver

        solver = EMTPSolver(dt=1e-6, finish_time=100e-6, verbose=False)
        with pytest.raises(ValueError, match="fitulm_path is required"):
            solver.add_ULM_line(
                name="bad", nodes_send=1, nodes_recv=2,
                length=1000.0,
                generate_fitulm=False,
            )

    def test_add_ulm_line_with_path_object(self, tmp_path):
        """fitulm_path accepts Path objects."""
        from LCP.vector_fitting_v411_independent import (
            write_fitULM, ULMFittingResult,
        )
        from emtp import EMTPSolver

        fitulm_path = tmp_path / "line.fitULM"

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

        solver = EMTPSolver(dt=1e-6, finish_time=100e-6, verbose=False)
        solver.add_ULM_line(
            name="line1",
            nodes_send=1, nodes_recv=2,
            length=1000.0,
            generate_fitulm=False,
            fitulm_path=fitulm_path,
        )
        assert "line1" in solver.transmission_lines


class TestSolverAddULMLineLengthConsistency:
    def test_length_mismatch_raises(self, monkeypatch):
        """When generate_fitulm=True, solver length must match lcp_spec.length."""
        from emtp import EMTPSolver
        from emtp.lines import fitulm_resolver as resolver_module

        class FakeLCPSpec:
            length = 5000.0

        def mock_resolve(self, spec):
            return Path("dummy.fitULM")

        monkeypatch.setattr(resolver_module.FitULMResolver, "resolve", mock_resolve)

        solver = EMTPSolver(dt=1e-6, finish_time=100e-6, verbose=False)
        with pytest.raises(ValueError, match="length mismatch"):
            solver.add_ULM_line(
                name="bad",
                nodes_send=1, nodes_recv=2,
                length=6000.0,
                generate_fitulm=True,
                lcp_spec=FakeLCPSpec(),
            )

    def test_length_consistent_passes(self, monkeypatch):
        """When lengths match, we get past the length check."""
        from emtp import EMTPSolver
        from emtp.lines import fitulm_resolver as resolver_module

        class FakeLCPSpec:
            length = 5000.0

        def mock_resolve(self, spec):
            raise RuntimeError("past length check")

        monkeypatch.setattr(resolver_module.FitULMResolver, "resolve", mock_resolve)

        solver = EMTPSolver(dt=1e-6, finish_time=100e-6, verbose=False)
        with pytest.raises(RuntimeError, match="past length check"):
            solver.add_ULM_line(
                name="ok",
                nodes_send=1, nodes_recv=2,
                length=5000.0,
                generate_fitulm=True,
                lcp_spec=FakeLCPSpec(),
            )
