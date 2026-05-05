"""P5 validation test configuration — ensure validation package is importable."""

import sys
from pathlib import Path


def pytest_configure(config):
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
