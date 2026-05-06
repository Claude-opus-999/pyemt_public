"""LCPFitULMGenerator — chain Z/Y computation → VF fitting → fitULM export."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .specs import LCPFitULMSpec, LCPLineType
from .validation import validate_frequency_vector, validate_zy_matrices
from .exceptions import (
    LCPInputError, LCPGenerationError, LCPFittingError, FitULMExportError,
)


class LCPFitULMGenerator:
    """Generate a fitULM file from line geometry and soil parameters.

    Usage::

        spec = LCPFitULMSpec(
            line_type=LCPLineType.OHL_DERI_SEMLYEN,
            name="ohl_test", length=1000.0,
            freq=np.logspace(0, 5, 51),
            geometry_config=line_geometry,
        )
        path = LCPFitULMGenerator().generate(spec)
    """

    def generate(self, spec: LCPFitULMSpec) -> Path:
        """Run the full pipeline and return the path to the generated fitULM file."""
        validate_frequency_vector(spec.freq)

        freq, Z, Y, metadata = self._compute_zy(spec)
        validate_zy_matrices(freq, Z, Y)

        ulm_params, fitting_result = self._fit_ulm(spec, freq, Z, Y)

        path = self._write_fitulm(spec, fitting_result)

        return path

    # -----------------------------------------------------------------
    # Z/Y computation dispatcher
    # -----------------------------------------------------------------

    def _compute_zy(self, spec: LCPFitULMSpec):
        if spec.line_type == LCPLineType.OHL_DERI_SEMLYEN:
            from .generation.ohl_deri_semlyen import compute_ohl_zy
            return compute_ohl_zy(spec.freq, spec.geometry_config,
                                soil_config=spec.soil_config,
                                verbose=spec.verbose)

        if spec.line_type == LCPLineType.PIPE_TYPE_CABLE:
            from .generation.pipe_type_cable import compute_pipe_type_cable_zy
            return compute_pipe_type_cable_zy(spec.freq, spec.geometry_config,
                                            soil_config=spec.soil_config,
                                            verbose=spec.verbose)

        if spec.line_type == LCPLineType.MULTI_ARMORED_CABLE:
            from .generation.multi_armored_cable import compute_multi_armored_cable_zy
            return compute_multi_armored_cable_zy(spec.freq, spec.geometry_config,
                                                soil_config=spec.soil_config,
                                                verbose=spec.verbose)

        raise LCPInputError(f"Unsupported line_type: {spec.line_type}")

    # -----------------------------------------------------------------
    # Vector Fitting
    # -----------------------------------------------------------------

    def _fit_ulm(self, spec: LCPFitULMSpec, freq, Z_matrix, Y_matrix):
        from LCP.vector_fitting_v411_independent import (
            ulm_complete_fitting, IterativePoleFindingConfig,
        )

        vf_config = spec.vf_config or IterativePoleFindingConfig(
            Ymin=12, Ymax=20, epsY=0.002,
            Hmin=12, Hmax=20, epsH=0.002,
            pole_step=2, eps_deg=10.0,
            compute_H_reconstruction_metrics=True,
            verbose_H_metrics=spec.verbose,
        )

        try:
            return ulm_complete_fitting(
                freq=freq,
                Z_matrix=Z_matrix,
                Y_matrix=Y_matrix,
                length=spec.length,
                velocity_freq=1e5,
                config=vf_config,
                use_freq_dependent=spec.use_freq_dependent,
                enforce_passivity_flag=spec.enforce_passivity,
                verbose=spec.verbose,
            )
        except Exception as exc:
            raise LCPFittingError("ULM fitting failed") from exc

    # -----------------------------------------------------------------
    # fitULM export
    # -----------------------------------------------------------------

    def _write_fitulm(self, spec: LCPFitULMSpec, fitting_result) -> Path:
        from LCP.vector_fitting_v411_independent import (
            write_fitULM, verify_fitULM_file,
        )
        from .cache import get_cache_path

        output_path = (
            Path(spec.output_path) if spec.output_path
            else get_cache_path(spec)
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            write_fitULM(
                fitting_result, str(output_path),
                precision=spec.precision, verbose=spec.verbose,
            )
        except Exception as exc:
            raise FitULMExportError(
                f"Failed to write fitULM: {output_path}"
            ) from exc

        ok = verify_fitULM_file(str(output_path), verbose=spec.verbose)
        if ok is False:
            raise FitULMExportError(
                f"Generated invalid fitULM file: {output_path}"
            )

        return output_path
