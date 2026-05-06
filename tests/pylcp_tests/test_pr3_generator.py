"""PR-3: Tests for LCPFitULMGenerator."""

from pathlib import Path

import numpy as np
import pytest

from pylcp import (
    LCPLineType, LCPFitULMSpec, LCPFitULMGenerator, LCPInputError,
)


class TestGeneratorDispatch:
    def test_raises_on_unsupported_line_type(self):
        spec = LCPFitULMSpec(
            line_type="unsupported",
            name="bad",
            length=1.0,
            freq=np.array([1.0, 10.0]),
            geometry_config={},
        )
        with pytest.raises((LCPInputError, KeyError, AttributeError)):
            LCPFitULMGenerator().generate(spec)


class TestGeneratorMocked:
    def test_generate_calls_zy_and_fit_and_write(self, tmp_path, monkeypatch):
        """Mock the heavy VF step — verify the pipeline is called in order."""
        from pylcp import lcp_fitulm_generator as gen_module

        calls = []

        def mock_compute_zy(self, spec):
            calls.append("compute_zy")
            K, n = 2, 1
            freq = np.array([1.0, 10.0])
            Z = np.ones((K, n, n), dtype=complex)
            Y = np.ones((K, n, n), dtype=complex)
            return freq, Z, Y, {}

        def mock_fit_ulm(self, spec, freq, Z, Y):
            calls.append("fit_ulm")
            return None, None

        def mock_write_fitulm(self, spec, fitting_result):
            calls.append("write_fitulm")
            out = tmp_path / "test.fitULM"
            out.write_text("nf=1\n")
            return out

        monkeypatch.setattr(gen_module.LCPFitULMGenerator, "_compute_zy", mock_compute_zy)
        monkeypatch.setattr(gen_module.LCPFitULMGenerator, "_fit_ulm", mock_fit_ulm)
        monkeypatch.setattr(gen_module.LCPFitULMGenerator, "_write_fitulm", mock_write_fitulm)

        spec = LCPFitULMSpec(
            line_type=LCPLineType.OHL_DERI_SEMLYEN,
            name="test",
            length=1000.0,
            freq=np.array([1.0, 10.0]),
            geometry_config={},
        )
        path = LCPFitULMGenerator().generate(spec)
        assert path == tmp_path / "test.fitULM"
        assert calls == ["compute_zy", "fit_ulm", "write_fitulm"]
