"""ReproBench: a CVE-only IoT-firmware vulnerability reproduction benchmark.

The package is intentionally light-weight at import time: no network, no model
SDKs and no Docker are required to import ``reprobench``.  Heavy optional
dependencies (openai/anthropic, python-magic, docker CLI) are imported lazily by
the modules that need them.
"""

from __future__ import annotations

__all__ = [
    "__version__",
    "BENCHMARK_NAME",
    "PAPER_TITLE",
    "TRIALS",
    "CVE_COUNT",
    "MODEL_COUNT",
    "TOTAL_RUNS",
]

__version__ = "1.0.0"

BENCHMARK_NAME = "ReproBench"
PAPER_TITLE = (
    "ReproBench: Benchmarking LLM Agents on Reproducing Vulnerability From Scratch"
)

#: Paper run matrix (30 CVEs x 5 models x 3 trials = 450 runs).
TRIALS = 3
CVE_COUNT = 30
MODEL_COUNT = 5
TOTAL_RUNS = CVE_COUNT * MODEL_COUNT * TRIALS
