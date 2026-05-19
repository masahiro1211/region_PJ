"""Separable Gaussian application for exponential-sum potentials."""

from __future__ import annotations

import numpy as np

from .models import ExponentialSum


def apply_1d_kernel_along_axis(
    kernel: np.ndarray,
    rho: np.ndarray,
    axis: int,
) -> np.ndarray:
    """3D配列の指定軸に沿って1Dカーネル行列を作用させる。"""
    return np.moveaxis(np.tensordot(kernel, rho, axes=([1], [axis])), 0, axis)


def apply_3d_kernel(
    kernel_1d: np.ndarray,
    rho_grid: np.ndarray,
) -> np.ndarray:
    """1D カーネルを 3 軸すべてに順次作用させる。

    Parameters
    ----------
    kernel_1d
        各軸に作用させる 1D カーネル行列。shape は ``(N, N)``。
    rho_grid
        入力密度グリッド。shape は ``(N, N, N)``。

    Returns
    -------
    np.ndarray
        3 軸すべてに ``kernel_1d`` を作用させた配列。
        shape は ``(N, N, N)``。
    """
    result = apply_1d_kernel_along_axis(kernel_1d, rho_grid, axis=0)
    result = apply_1d_kernel_along_axis(kernel_1d, result, axis=1)
    return apply_1d_kernel_along_axis(kernel_1d, result, axis=2)


def apply_separable_gaussian_3d(
    alpha: float, x_axis: np.ndarray, rho: np.ndarray
) -> np.ndarray:
    """exp(-α|r1-r2|²) の分離性を利用して 3D ρ に作用させる."""
    diff = x_axis[:, None] - x_axis[None, :]
    K_1d = np.exp(-alpha * diff**2)
    return apply_3d_kernel(K_1d, rho)


def apply_exp_sum_potential_3d(
    fit: ExponentialSum,
    x_axis: np.ndarray,
    rho: np.ndarray,
    dx: float,
) -> np.ndarray:
    """指数和近似 K(r) ≈ Σ_k w_k exp(-α_k r²) で V = (K * ρ) dx³ を計算する."""
    V = np.zeros_like(rho)
    for w_k, alpha_k in zip(fit.weights, fit.alphas):
        V += w_k * apply_separable_gaussian_3d(alpha_k, x_axis, rho)
    return V * dx**3
