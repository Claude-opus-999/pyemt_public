"""FitULM file resolution — external files or LCP auto-generation.

PR-1: External fitULM file reading with validation.
PR-4: LCP auto-generation via pylcp.LCPFitULMGenerator.
PR-5: Node count validation against fitULM header.
PR-6: Content-based cache key and auto-cache paths.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class FitULMSpec:
    """Specification for resolving a fitULM file."""

    name: str
    generate_fitulm: bool = False
    fitulm_path: Optional[Path] = None
    lcp_spec: object | None = None
    cache_dir: Path = field(default_factory=lambda: Path(".lcp_cache"))
    force_recompute: bool = False

    def __post_init__(self):
        if isinstance(self.fitulm_path, str):
            self.fitulm_path = Path(self.fitulm_path)
        if isinstance(self.cache_dir, str):
            self.cache_dir = Path(self.cache_dir)


class FitULMResolver:
    """Resolve a :class:`FitULMSpec` to a concrete fitULM file path.

    External file mode::

        spec = FitULMSpec(name="line1", fitulm_path=Path("data.fitULM"))
        path = FitULMResolver().resolve(spec)

    LCP generation mode::

        spec = FitULMSpec(name="line1", generate_fitulm=True, lcp_spec=...)
        path = FitULMResolver().resolve(spec)
    """

    def resolve(self, spec: FitULMSpec) -> Path:
        if spec.generate_fitulm:
            return self._resolve_from_lcp(spec)
        return self._resolve_external_file(spec)

    # -----------------------------------------------------------------
    # External file
    # -----------------------------------------------------------------

    def _resolve_external_file(self, spec: FitULMSpec) -> Path:
        if spec.fitulm_path is None:
            raise ValueError(
                "fitulm_path is required when generate_fitulm=False"
            )
        path = Path(spec.fitulm_path)
        self._verify_fitulm(path)
        return path

    # -----------------------------------------------------------------
    # LCP generation
    # -----------------------------------------------------------------

    def _resolve_from_lcp(self, spec: FitULMSpec) -> Path:
        if spec.lcp_spec is None:
            raise ValueError(
                "lcp_spec is required when generate_fitulm=True"
            )

        lcp_spec = spec.lcp_spec

        # Determine output path (from spec or auto-generated via cache key)
        output_path = self._get_output_path(lcp_spec, spec)

        # Check cache
        if output_path.exists() and not spec.force_recompute:
            self._verify_fitulm(output_path)
            return output_path

        # Generate via LCP
        try:
            from pylcp.lcp_fitulm_generator import LCPFitULMGenerator
        except ImportError as exc:
            raise ImportError(
                "LCP generation requires pylcp. "
                "Ensure pylcp is installed before using generate_fitulm=True."
            ) from exc

        # Set the output path on the lcp_spec
        if hasattr(lcp_spec, 'output_path'):
            lcp_spec.output_path = output_path

        generator = LCPFitULMGenerator()
        generated_path = generator.generate(lcp_spec)

        self._verify_fitulm(generated_path)
        return generated_path

    def _get_output_path(self, lcp_spec, fitulm_spec: FitULMSpec) -> Path:
        if getattr(lcp_spec, 'output_path', None) is not None:
            return Path(lcp_spec.output_path)
        from pylcp.cache import get_cache_path
        if not getattr(lcp_spec, 'cache_dir', None):
            lcp_spec.cache_dir = Path(fitulm_spec.cache_dir)
        return get_cache_path(lcp_spec)

    def _verify_fitulm(self, path: Path) -> None:
        if not path.exists():
            raise FileNotFoundError(f"fitULM file not found: {path}")
        if path.stat().st_size == 0:
            raise ValueError(f"fitULM file is empty: {path}")
        try:
            from LCP.vector_fitting_v411_independent import verify_fitULM_file
        except ImportError:
            # LCP verifier not available — fast-check only
            return
        ok = verify_fitULM_file(str(path), verbose=False)
        if ok is False:
            raise ValueError(f"Invalid fitULM file: {path}")
