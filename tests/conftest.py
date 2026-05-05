"""Shared pytest setup for the EMTP P1 verification suite."""
from __future__ import annotations

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: integration tests that run a larger transient case")
    config.addinivalue_line("markers", "validation: physical validation tests")
    config.addinivalue_line("markers", "external: requires external ATP/PSCAD/EMTP-RV reference data")
