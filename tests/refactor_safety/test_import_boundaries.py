"""PR0: Import boundary tests — enforce layered dependency architecture.

These tests prevent future code from introducing forbidden imports.
Some are currently expected to fail (xfail) because the current codebase
has not yet been fully refactored (PR2-PR7).  By PR7 all xfail markers
should be removed.
"""

import ast
import os
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EMTP_DIR = PROJECT_ROOT / "emtp"
L0_FILES = [
    "transmission_line_emtp_v2",
    "ulm_transmission_line_PARA",
    "nonlinear_models_pscad",
    "umec_transformer",
    "atp_lightning_current_generator_simplified",
]


def _py_files_under(path: Path):
    return sorted(
        f for f in path.rglob("*.py") if "__pycache__" not in str(f)
    )


def _imports_in_file(filepath: Path):
    """Parse a .py file and return (imported_module_names, from_import_names)."""
    text = filepath.read_text(encoding="utf-8", errors="ignore")
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return set(), set()

    import_names = set()
    from_names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                import_names.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                from_names.add(node.module.split(".")[0])
    return import_names, from_names


# =========================================================================
# Rule: Layer 2 modules must not depend on case_runner (Layer 3)
# =========================================================================

LAYER2_DIRS = ["devices", "assembly", "runtime", "results",
               "lines", "transformers", "sources", "nonlinear"]


class TestLayer2DoesNotImportCaseRunner:
    def test_layer2_no_case_runner_import(self):
        """Layer 2 subpackages should not import from emtp.case_runner."""
        violations = []
        for dirname in LAYER2_DIRS:
            layer_dir = EMTP_DIR / dirname
            if not layer_dir.is_dir():
                continue
            for pyfile in _py_files_under(layer_dir):
                _, from_names = _imports_in_file(pyfile)
                bad = {n for n in from_names if "case_runner" in n}
                if bad:
                    violations.append(f"{pyfile.relative_to(PROJECT_ROOT)}: {bad}")
        assert not violations, (
            "Layer 2 must not import case_runner (Layer 3).  Violations:\n"
            + "\n".join(violations)
        )


# =========================================================================
# Rule: solver.py must not directly import Layer 0 physics models (PR7 goal)
# =========================================================================

class TestSolverDoesNotImportLayer0:
    @pytest.mark.xfail(
        reason="solver.py still imports Layer 0 directly.  "
               "Expected to pass by PR7."
    )
    def test_solver_no_layer0_import(self):
        """solver.py must not directly import top-level physics modules."""
        solver_py = EMTP_DIR / "solver.py"
        import_names, from_names = _imports_in_file(solver_py)
        violations = set()
        for l0 in L0_FILES:
            if l0 in import_names:
                violations.add(f"import {l0}")
            if l0 in from_names:
                violations.add(f"from {l0}")
        assert not violations, (
            f"solver.py imports Layer 0 directly: {violations}"
        )


# =========================================================================
# Rule: No module should import emtp.case_runner at init time from core
# =========================================================================

class TestNoReverseDependencies:
    def test_devices_do_not_import_solver(self):
        """devices/ must not import solver or case_runner."""
        violations = []
        for pyfile in _py_files_under(EMTP_DIR / "devices"):
            import_names, from_names = _imports_in_file(pyfile)
            if "solver" in from_names and "emtp.solver" in str(from_names):
                violations.append(str(pyfile.relative_to(PROJECT_ROOT)))
            if "case_runner" in from_names:
                violations.append(str(pyfile.relative_to(PROJECT_ROOT)))
        assert not violations, (
            "devices/ must not import solver or case_runner: " + str(violations)
        )

    def test_assembly_does_not_import_case_runner(self):
        """assembly/ must not import case_runner."""
        violations = []
        for pyfile in _py_files_under(EMTP_DIR / "assembly"):
            _, from_names = _imports_in_file(pyfile)
            if any("case_runner" in n for n in from_names):
                violations.append(str(pyfile.relative_to(PROJECT_ROOT)))
        assert not violations

    def test_runtime_does_not_import_case_runner(self):
        """runtime/ must not import case_runner."""
        violations = []
        for pyfile in _py_files_under(EMTP_DIR / "runtime"):
            _, from_names = _imports_in_file(pyfile)
            if any("case_runner" in n for n in from_names):
                violations.append(str(pyfile.relative_to(PROJECT_ROOT)))
        assert not violations
