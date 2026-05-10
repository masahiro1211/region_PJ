"""Gaussian 密度の分離表現。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from src.approximation.exp_sum import ExponentialSum


@dataclass(frozen=True)
class SeparableDensityTerm:
    """CP 形式の1項を表すデータ構造。

    Attributes
    ----------
    coefficient : float
        外積項全体に掛かる係数。
    fx : np.ndarray, shape (N,)
        x 方向の1D因子。
    fy : np.ndarray, shape (N,)
        y 方向の1D因子。
    fz : np.ndarray, shape (N,)
        z 方向の1D因子。
    """

    coefficient: float
    fx: np.ndarray
    fy: np.ndarray
    fz: np.ndarray


def make_gaussian_density_terms(
    x_axis: np.ndarray,
    alpha: float,
    centers: np.ndarray | None = None,
    weights: Sequence[float] | np.ndarray | None = None,
) -> list[SeparableDensityTerm]:
    """Gaussian 密度を1D因子の外積和として表す。

    Parameters
    ----------
    x_axis : np.ndarray, shape (N,)
        各方向で共通に使う1Dグリッド点の座標配列。
    alpha : float
        Gaussian の幅パラメータ。
    centers : np.ndarray, shape (M, 3), optional
        各 Gaussian の中心座標。None の場合は原点中心の1項を使う。
    weights : Sequence[float] or np.ndarray, shape (M,), optional
        各 Gaussian に掛ける係数。None の場合はすべて1とする。

    Returns
    -------
    terms : list[SeparableDensityTerm]
        ``Σ_l c_l fx_l(x) fy_l(y) fz_l(z)`` で密度を表す項のリスト。

    Raises
    ------
    ValueError
        ``centers`` または ``weights`` の形状が不正な場合。

    Notes
    -----
    各項は
    ``w * exp(-alpha * (x-cx)^2) * exp(-alpha * (y-cy)^2)
    * exp(-alpha * (z-cz)^2)`` を表す。
    """
    x_arr = np.asarray(x_axis, dtype=float)
    if centers is None:
        centers_arr = np.zeros((1, 3), dtype=float)
    else:
        centers_arr = np.asarray(centers, dtype=float)
        if centers_arr.ndim != 2 or centers_arr.shape[1] != 3:
            raise ValueError(
                "centers must have shape (n_terms, 3); "
                f"got shape={centers_arr.shape}."
            )

    if weights is None:
        weights_arr = np.ones(len(centers_arr), dtype=float)
    else:
        weights_arr = np.asarray(weights, dtype=float)
        if weights_arr.shape != (len(centers_arr),):
            raise ValueError(
                "weights must have shape (n_terms,); "
                f"got shape={weights_arr.shape}."
            )

    terms: list[SeparableDensityTerm] = []
    for center, weight in zip(centers_arr, weights_arr):
        fx = np.exp(-alpha * (x_arr - center[0]) ** 2)
        fy = np.exp(-alpha * (x_arr - center[1]) ** 2)
        fz = np.exp(-alpha * (x_arr - center[2]) ** 2)
        terms.append(SeparableDensityTerm(float(weight), fx, fy, fz))
    return terms


def outer3(fx: np.ndarray, fy: np.ndarray, fz: np.ndarray) -> np.ndarray:
    """3本の1Dベクトルから3D外積テンソルを構築する。

    Parameters
    ----------
    fx : np.ndarray, shape (N,)
        x 方向の1Dベクトル。
    fy : np.ndarray, shape (N,)
        y 方向の1Dベクトル。
    fz : np.ndarray, shape (N,)
        z 方向の1Dベクトル。

    Returns
    -------
    tensor : np.ndarray, shape (N, N, N)
        ``fx[:, None, None] * fy[None, :, None] * fz[None, None, :]``。
    """
    return fx[:, None, None] * fy[None, :, None] * fz[None, None, :]


def materialize_density_terms(
    terms: Sequence[SeparableDensityTerm],
) -> np.ndarray:
    """分離表現の密度を dense な3Dテンソルへ戻す。

    Parameters
    ----------
    terms : Sequence[SeparableDensityTerm]
        CP 形式で表した密度項の列。

    Returns
    -------
    rho : np.ndarray, shape (N, N, N)
        dense な電荷密度テンソル。

    Raises
    ------
    ValueError
        ``terms`` が空の場合。
    """
    if not terms:
        raise ValueError("terms must contain at least one density term.")

    n = len(terms[0].fx)
    rho = np.zeros((n, n, n), dtype=float)
    for term in terms:
        rho += term.coefficient * outer3(term.fx, term.fy, term.fz)
    return rho


def build_gaussian_kernel_1d(alpha: float, x_axis: np.ndarray) -> np.ndarray:
    """1D Gaussian カーネル行列を構築する。

    Parameters
    ----------
    alpha : float
        Gaussian カーネルの幅パラメータ。
    x_axis : np.ndarray, shape (N,)
        1Dグリッド点の座標配列。

    Returns
    -------
    kernel : np.ndarray, shape (N, N)
        ``K[i, j] = exp(-alpha * (x_i - x_j)^2)``。
    """
    x_arr = np.asarray(x_axis, dtype=float)
    diff = x_arr[:, None] - x_arr[None, :]
    return np.exp(-alpha * diff**2)


def apply_exp_sum_to_separable_density(
    fit: ExponentialSum,
    x_axis: np.ndarray,
    density_terms: Sequence[SeparableDensityTerm],
    dx: float,
    diag_coeff: float = 0.0,
) -> np.ndarray:
    """指数和 Coulomb 近似を分離表現の密度に作用させる。

    Parameters
    ----------
    fit : ExponentialSum
        ``1/r ≈ Σ_k w_k exp(-alpha_k r^2)`` の指数和近似。
    x_axis : np.ndarray, shape (N,)
        各方向で共通に使う1Dグリッド点の座標配列。
    density_terms : Sequence[SeparableDensityTerm]
        CP 形式で表した電荷密度。
    dx : float
        グリッド幅。返り値には体積要素 ``dx^3`` が掛かる。
    diag_coeff : float, default=0.0
        対角補正として密度に掛けて足す係数。

    Returns
    -------
    potential : np.ndarray, shape (N, N, N)
        指数和カーネルを密度に作用させたポテンシャル。

    Raises
    ------
    ValueError
        ``dx <= 0`` の場合、または ``density_terms`` が空の場合。

    Notes
    -----
    Gaussian カーネルは各方向に分離できるため、各密度項
    ``fx * fy * fz`` に対して ``(K fx) * (K fy) * (K fz)`` を計算する。
    """
    if dx <= 0:
        raise ValueError("dx must be positive.")
    if not density_terms:
        raise ValueError("density_terms must contain at least one term.")

    x_arr = np.asarray(x_axis, dtype=float)
    n = len(x_arr)
    potential = np.zeros((n, n, n), dtype=float)

    for weight, alpha in zip(fit.weights, fit.alphas):
        kernel = build_gaussian_kernel_1d(float(alpha), x_arr)
        for term in density_terms:
            gx = kernel @ term.fx
            gy = kernel @ term.fy
            gz = kernel @ term.fz
            potential += weight * term.coefficient * outer3(gx, gy, gz)

    if diag_coeff != 0.0:
        potential += diag_coeff * materialize_density_terms(density_terms)

    return potential * dx**3
