"""PR-2: Tests for pylcp Z/Y generation modules."""

import numpy as np
import pytest

from pylcp import LCPLineType, LCPFitULMSpec
from pylcp.validation import validate_frequency_vector, validate_zy_matrices


class TestValidation:
    def test_valid_freq_vector_passes(self):
        validate_frequency_vector(np.array([1.0, 10.0, 100.0]))

    def test_empty_freq_raises(self):
        with pytest.raises(ValueError):
            validate_frequency_vector(np.array([]))

    def test_non_positive_freq_raises(self):
        with pytest.raises(ValueError):
            validate_frequency_vector(np.array([0.0, 10.0]))

    def test_2d_freq_raises(self):
        with pytest.raises(ValueError):
            validate_frequency_vector(np.array([[1.0, 10.0]]))

    def test_valid_zy_passes(self):
        K, n = 2, 3
        f = np.array([1.0, 10.0])
        Z = np.ones((K, n, n), dtype=complex)
        Y = np.ones((K, n, n), dtype=complex)
        validate_zy_matrices(f, Z, Y)

    def test_nan_in_Z_raises(self):
        K, n = 2, 3
        f = np.array([1.0, 10.0])
        Z = np.ones((K, n, n), dtype=complex)
        Z[0, 0, 0] = np.nan
        Y = np.ones((K, n, n), dtype=complex)
        with pytest.raises(ValueError, match="NaN"):
            validate_zy_matrices(f, Z, Y)

    def test_shape_mismatch_raises(self):
        f = np.array([1.0, 10.0])
        Z = np.ones((2, 3, 3), dtype=complex)
        Y = np.ones((2, 4, 4), dtype=complex)
        with pytest.raises(ValueError, match="conductor counts"):
            validate_zy_matrices(f, Z, Y)


class TestLCPFitULMSpec:
    def test_minimal_spec(self):
        spec = LCPFitULMSpec(
            line_type=LCPLineType.OHL_DERI_SEMLYEN,
            name="test",
            length=1000.0,
            freq=np.logspace(0, 5, 51),
            geometry_config={"dummy": True},
        )
        assert spec.line_type == LCPLineType.OHL_DERI_SEMLYEN
        assert spec.length == 1000.0

    def test_spec_defaults(self):
        spec = LCPFitULMSpec(
            line_type=LCPLineType.PIPE_TYPE_CABLE,
            name="cable",
            length=5000.0,
            freq=np.array([1.0, 10.0]),
            geometry_config={},
        )
        assert spec.use_freq_dependent == "auto"
        assert spec.enforce_passivity is True
        assert spec.precision == 16
        assert spec.verbose is False


class TestOHLGenerationSmoke:
    def test_ohl_zy_basic_2conductor(self):
        """Smoke test: 2-conductor OHL with tiny freq sweep produces valid Z/Y."""
        from LCP.ulm_atp_zy_deri_semlyen import (
            MultiConductorLine, ConductorGeometry,
        )
        from pylcp.generation.ohl_deri_semlyen import compute_ohl_zy

        conductors = [
            ConductorGeometry(30.0, -5.0, 0.02, 0.05, 1.0),
            ConductorGeometry(30.0, 5.0, 0.02, 0.05, 1.0),
        ]
        line = MultiConductorLine(
            conductors=conductors,
            names=["PhaseA", "PhaseB"],
            is_ground_wire=[False, False],
        )

        freq = np.logspace(1, 4, 5)
        f, Z, Y, meta = compute_ohl_zy(freq, line, verbose=False)

        assert len(f) == 5
        assert Z.shape == (5, 2, 2)
        assert Y.shape == (5, 2, 2)
        assert meta["n_original"] == 2
        validate_zy_matrices(f, Z, Y)
