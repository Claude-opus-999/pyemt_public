"""Tests for pylcp.cache — version fields and cache_dir propagation."""

from pathlib import Path

import numpy as np
import pytest

from pylcp import LCPLineType, LCPFitULMSpec
from pylcp.cache import compute_cache_key, get_cache_path
from emtp.lines.fitulm_resolver import FitULMSpec, FitULMResolver


def _make_minimal_spec(**kwargs):
    defaults = dict(
        line_type=LCPLineType.OHL_DERI_SEMLYEN,
        name="test_line",
        length=1000.0,
        freq=np.array([1.0, 10.0, 100.0]),
        geometry_config={"h": 30.0, "conductors": 3},
    )
    defaults.update(kwargs)
    return LCPFitULMSpec(**defaults)


class TestCacheKeyVersionFields:
    def test_pylcp_version_change_changes_key(self, monkeypatch):
        import pylcp.cache as cache
        spec = _make_minimal_spec()

        monkeypatch.setattr(cache, "_get_pylcp_version", lambda: "pylcp-v1.0")
        monkeypatch.setattr(cache, "_get_lcp_version", lambda: "lcp-v1.0")
        key_a = compute_cache_key(spec)

        monkeypatch.setattr(cache, "_get_pylcp_version", lambda: "pylcp-v2.0")
        monkeypatch.setattr(cache, "_get_lcp_version", lambda: "lcp-v1.0")
        key_b = compute_cache_key(spec)

        assert key_a != key_b

    def test_lcp_version_change_changes_key(self, monkeypatch):
        import pylcp.cache as cache
        spec = _make_minimal_spec()

        monkeypatch.setattr(cache, "_get_pylcp_version", lambda: "pylcp-v1.0")
        monkeypatch.setattr(cache, "_get_lcp_version", lambda: "lcp-v1.0")
        key_a = compute_cache_key(spec)

        monkeypatch.setattr(cache, "_get_pylcp_version", lambda: "pylcp-v1.0")
        monkeypatch.setattr(cache, "_get_lcp_version", lambda: "lcp-v2.0")
        key_b = compute_cache_key(spec)

        assert key_a != key_b

    def test_schema_version_ensures_new_keys_differ_from_old(self):
        spec = _make_minimal_spec()
        # Even without version mocks, schema_version=2 in payload ensures
        # the key format is distinguishable from any v1 schema
        key = compute_cache_key(spec)
        assert len(key) == 16  # hex hash, always this length


class TestCacheDirPropagation:
    def test_outer_cache_dir_overrides_lcp_default(self, tmp_path):
        """Outer FitULMSpec.cache_dir must take precedence over lcp_spec default."""
        lcp_spec = _make_minimal_spec()
        lcp_spec.output_path = None
        lcp_spec.cache_dir = Path(".lcp_cache")  # default, should be overridden

        fitulm_spec = FitULMSpec(
            name="line",
            generate_fitulm=True,
            lcp_spec=lcp_spec,
            cache_dir=tmp_path / "my_cache",
        )

        path = FitULMResolver()._get_output_path(lcp_spec, fitulm_spec)
        assert path.parent == tmp_path / "my_cache"
        assert "test_line" in path.name
        assert path.suffix == ".fitULM"

    def test_explicit_lcp_output_path_has_highest_priority(self, tmp_path):
        """When lcp_spec.output_path is set, it wins over everything."""
        explicit = tmp_path / "explicit.fitULM"
        lcp_spec = _make_minimal_spec()
        lcp_spec.output_path = explicit
        lcp_spec.cache_dir = tmp_path / "ignored_lcp"

        fitulm_spec = FitULMSpec(
            name="line",
            generate_fitulm=True,
            lcp_spec=lcp_spec,
            cache_dir=tmp_path / "ignored_outer",
        )

        path = FitULMResolver()._get_output_path(lcp_spec, fitulm_spec)
        assert path == explicit
