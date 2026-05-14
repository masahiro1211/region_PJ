"""Exponential-sum approximation utilities.

1/r ≈ Σ_k w_k exp(-α_k r²) の VARPRO フィットと、分離 Gaussian
カーネルを使った 3D ポテンシャル適用を提供する。
"""

from .benchmark import BenchmarkRunner
from .grid import LogUniformGrid
from .models import ExponentialSum
from .separable import (
    _apply_1d_kernel_along_axis,
    apply_3d_kernel,
    apply_1d_kernel_along_axis,
    apply_exp_sum_potential_3d,
    apply_separable_gaussian_3d,
)
from .varpro import VarproOptimizer

__all__ = [
    "ExponentialSum",
    "LogUniformGrid",
    "VarproOptimizer",
    "BenchmarkRunner",
    "apply_3d_kernel",
    "apply_1d_kernel_along_axis",
    "_apply_1d_kernel_along_axis",
    "apply_separable_gaussian_3d",
    "apply_exp_sum_potential_3d",
]
