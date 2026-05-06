"""PR7: element_builder ulm_line support."""

import pytest
from emtp import EMTPSolver
from emtp.builders.element_builder import add_element_to_solver


class TestElementBuilderULMLine:
    def test_ulm_line_external_file_registered(self, tmp_path):
        """Element builder can create a ULM line from a fitULM file."""
        # Write minimal valid fitULM
        import numpy as np
        from LCP.vector_fitting_v411_independent import write_fitULM, ULMFittingResult

        fitulm_path = tmp_path / "line.fitULM"

        class MinimalHModeFit:
            poles = np.array([-1e6 + 0j])
            c_matrix_residues = np.array([[[1.0 + 0j]]])
            tau = 1e-6

        result = ULMFittingResult(
            nf=1, n_active_modes=1, active_modes=[], mode_groups=[[0]],
            poles_Yc=np.array([-1e5 + 0j]),
            k_residues=np.array([[[1.0 + 0j]]]),
            k0=np.array([[0.0]]), tau_all=np.array([1e-6]),
            D_matrices=None, H_modes_fits=[MinimalHModeFit()],
            Yc_trace_rmse=0.001, H_modes_rmse=[0.001], H_matrix_rmse=0.001,
            is_passive=True, is_freq_dependent=False,
            H_reconstruction_metrics=None, H_reconstructed=None,
        )
        write_fitULM(result, str(fitulm_path), precision=16, verbose=False)

        s = EMTPSolver(dt=1e-6, finish_time=100e-6, verbose=False)
        element = {
            "kind": "ulm_line",
            "name": "line1",
            "nodes_send": [1],
            "nodes_recv": [2],
            "length": 1000.0,
            "generate_fitulm": False,
            "fitulm_path": str(fitulm_path),
        }
        add_element_to_solver(s, element)
        assert "line1" in s.transmission_lines

    def test_ulm_line_raises_for_unsupported(self):
        s = EMTPSolver(dt=1e-6, finish_time=10e-6, verbose=False)
        with pytest.raises(ValueError, match="Unsupported element kind"):
            add_element_to_solver(s, {"kind": "unknown_thing", "name": "x"})
