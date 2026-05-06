"""Probe management — registration, sampling, and result retrieval.

PR3: ProbeManager handles probe registration and metadata.
Storage arrays remain in ResultStore (emtp/results/store.py).
"""

from .probe_manager import ProbeManager, ProbeSpec

__all__ = ["ProbeManager", "ProbeSpec"]
