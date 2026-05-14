"""PyTorch 版 Hartree ポテンシャル時間計測 API。"""

from __future__ import annotations

from src.approximation.timing.core import benchmark
from src.approximation.timing.core import get_torch_metadata
from src.approximation.timing.core import require_torch
from src.approximation.timing.core import with_cuda_sync
from src.approximation.timing.inputs import (
    prepare_timing_benchmark_inputs,
)
from src.approximation.timing.inputs import validate_rank
from src.approximation.timing.io import load_exp_sum_arrays
from src.approximation.timing.io import load_k_1d_list
from src.approximation.timing.targets import iter_timing_targets
from src.approximation.timing.types import TimingBenchmarkInputs
from src.approximation.timing.types import TimingTarget
from src.approximation.timing.types import TorchMetadata

__all__ = [
    "TimingBenchmarkInputs",
    "TimingTarget",
    "TorchMetadata",
    "benchmark",
    "get_torch_metadata",
    "iter_timing_targets",
    "load_exp_sum_arrays",
    "load_k_1d_list",
    "prepare_timing_benchmark_inputs",
    "require_torch",
    "validate_rank",
    "with_cuda_sync",
]
