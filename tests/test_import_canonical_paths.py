"""Verify canonical import paths and single-implementation guarantees.

After the PR-1–PR-9 migration:
- ``emtp/runtime/`` is a package, not a single-file module.
- ``emtp/results/`` is a package, not a single-file module.
- ``DynamicDeviceRuntime`` exists in exactly one place.
- ``EMTPSolver`` is served from ``emtp/solver.py``.
"""

import os

import pytest


class TestRuntimePackage:
    def test_imports_as_package(self):
        import emtp.runtime
        path = emtp.runtime.__file__.replace("\\", "/")
        assert path.endswith("emtp/runtime/__init__.py"), f"got {path}"

    def test_dynamic_runtime_from_package(self):
        from emtp.runtime import DynamicDeviceRuntime
        assert DynamicDeviceRuntime.__module__ == "emtp.runtime"

    def test_resolve_manager_from_package(self):
        from emtp.runtime.resolve import ResolveManager
        assert ResolveManager.__module__ == "emtp.runtime.resolve"


class TestResultsPackage:
    def test_imports_as_package(self):
        import emtp.results
        path = emtp.results.__file__.replace("\\", "/")
        assert path.endswith("emtp/results/__init__.py"), f"got {path}"

    def test_result_store_from_package(self):
        from emtp.results.store import ResultStore
        assert ResultStore.__module__ == "emtp.results.store"


class TestSingleImplementations:
    def test_dynamic_runtime_single_source(self):
        """DynamicDeviceRuntime must have exactly one class definition."""
        import glob
        import emtp
        pkg_root = os.path.dirname(emtp.__file__)
        project_root = os.path.dirname(pkg_root)

        matches = []
        for pyfile in glob.glob(os.path.join(project_root, "**", "*.py"), recursive=True):
            if "__pycache__" in pyfile or "tests" + os.sep in pyfile:
                continue
            with open(pyfile, encoding="utf-8", errors="ignore") as f:
                for line in f:
                    if "class DynamicDeviceRuntime" in line:
                        matches.append(pyfile)
                        break

        canonical = [f for f in matches
                     if f.replace("\\", "/").endswith("emtp/runtime/__init__.py")]
        assert len(canonical) == 1, f"canonical not found in {matches}"
        others = [f for f in matches
                  if not f.replace("\\", "/").endswith("emtp/runtime/__init__.py")]
        assert len(others) == 0, f"stale definitions in {others}"

    def test_emtp_solver_from_canonical_source(self):
        from emtp import EMTPSolver as A
        from emtp.solver import EMTPSolver as B
        assert A is B

    def test_no_runtime_py_file(self):
        """emtp/runtime.py must not exist next to emtp/runtime/ package."""
        import emtp
        pkg_dir = os.path.dirname(emtp.__file__)
        runtime_py = os.path.join(pkg_dir, "runtime.py")
        results_py = os.path.join(pkg_dir, "results.py")
        assert not os.path.isfile(runtime_py), f"{runtime_py} still exists"
        assert not os.path.isfile(results_py), f"{results_py} still exists"
