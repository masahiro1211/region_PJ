"""Gaussian ポテンシャル計算の離散化誤差評価。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

from src.approximation.exp_sum.separable import apply_3d_kernel
from src.approximation.exp_sum.models import ExponentialSum
from src.approximation.low_rank import reconstruct_svd
from src.potential.separable_density import (
    apply_exp_sum_to_separable_density,
    make_gaussian_density_terms,
    materialize_density_terms,
)
from src.utils.grid import build_xyz
from src.utils.metrics import hartree_energy, v_e_errors


@dataclass(frozen=True)
class ExpSumEnergyError:
    """指数和近似エネルギーの誤差評価。

    Attributes
    ----------
    rank : int
        指数和近似のrank。
    N : int
        各軸のグリッド点数。
    dx : float
        グリッド幅。
    energy_exp_sum : float
        指数和近似で得たポテンシャルから計算したエネルギー。
    err_vs_cont_exact : float
        連続系の解析エネルギーに対する相対誤差。
    """

    rank: int
    N: int
    dx: float
    energy_exp_sum: float
    err_vs_cont_exact: float


def analytic_gaussian_hartree_energy(alpha: float) -> float:
    """原点中心 Gaussian 密度の連続系 Hartree 自己エネルギーを返す。

    Parameters
    ----------
    alpha : float
        密度 ``rho(r) = exp(-alpha * |r|^2)`` の幅パラメータ。

    Returns
    -------
    energy : float
        ``1/2 ∫∫ rho(r) rho(r') / |r-r'| dr dr'`` の解析値。

    Raises
    ------
    ValueError
        ``alpha <= 0`` の場合。
    """
    if alpha <= 0:
        raise ValueError("alpha must be positive.")
    return float(np.pi**2.5 / (np.sqrt(2.0) * alpha**2.5))


def compute_exp_sum_energy_error(
    fit: ExponentialSum,
    N: int,
    L: float,
    alpha: float,
    cell_int_const: float = 2.38,
) -> ExpSumEnergyError:
    """指数和近似ポテンシャルのエネルギー誤差を指定Nで評価する。

    Parameters
    ----------
    fit : ExponentialSum
        ``1/r`` の指数和近似。
    N : int
        各軸のグリッド点数。``build_xyz`` と同じく奇数を想定する。
    L : float
        計算領域の一辺の長さ。
    alpha : float
        Gaussian 密度の幅パラメータ。
    cell_int_const : float, default=2.38
        立方体セル内の ``1/r`` 積分に対応する対角補正定数。

    Returns
    -------
    error : ExpSumEnergyError
        連続系解析値に対する誤差評価。
    """
    dx = L / N
    xyz = build_xyz(N, L)
    x_axis = xyz[0, :, 0, 0]
    density_terms = make_gaussian_density_terms(x_axis, alpha)
    rho = materialize_density_terms(density_terms)
    energy_exact = analytic_gaussian_hartree_energy(alpha)
    diag_coeff = cell_int_const / dx - float(np.sum(fit.weights))
    potential = apply_exp_sum_to_separable_density(
        fit=fit,
        x_axis=x_axis,
        density_terms=density_terms,
        dx=dx,
        diag_coeff=diag_coeff,
    )
    energy_exp_sum = hartree_energy(rho, potential, dx)

    return ExpSumEnergyError(
        rank=fit.rank,
        N=N,
        dx=dx,
        energy_exp_sum=energy_exp_sum,
        err_vs_cont_exact=abs(energy_exp_sum - energy_exact)
        / abs(energy_exact),
    )


def compare_exp_sum_discretization(
    N_values: Sequence[int],
    L: float,
    alpha: float,
    fits: Mapping[int, ExponentialSum],
    ranks: Sequence[int],
    cell_int_const: float = 2.38,
) -> dict[int, list[ExpSumEnergyError]]:
    """複数Nで指数和近似エネルギーの連続系解析値との誤差を比較する。

    Parameters
    ----------
    N_values : Sequence[int]
        比較するグリッド点数の列。
    L : float
        計算領域の一辺の長さ。
    alpha : float
        Gaussian 密度の幅パラメータ。
    fits : Mapping[int, ExponentialSum]
        rank をキー、指数和近似を値とする辞書。
    ranks : Sequence[int]
        比較対象とする rank の列。
    cell_int_const : float, default=2.38
        立方体セル内の ``1/r`` 積分に対応する対角補正定数。

    Returns
    -------
    rank_rows : dict[int, list[ExpSumEnergyError]]
        rank ごとの指数和近似エネルギー誤差。
    """
    rank_rows: dict[int, list[ExpSumEnergyError]] = {
        rank: [] for rank in ranks
    }

    for rank in ranks:
        fit = fits[rank]
        for N in N_values:
            rank_rows[rank].append(
                compute_exp_sum_energy_error(
                    fit=fit,
                    N=N,
                    L=L,
                    alpha=alpha,
                    cell_int_const=cell_int_const,
                )
            )

    return rank_rows


def compute_rpca_error_sweep(
    *,
    fit: ExponentialSum,
    rpca_1d_list: list[dict[str, np.ndarray]],
    rho_grid: np.ndarray,
    v_analytic: np.ndarray,
    dx: float,
    k_diag_true: float,
    svd_ranks: list[int],
    thresholds: list[float],
) -> dict[str, np.ndarray | dict[str, np.ndarray]]:
    """RPCA / SVD 1D カーネル近似の 3D 誤差を sweep する。

    Parameters
    ----------
    fit : ExponentialSum
        ``1/r`` の指数和近似。
    rpca_1d_list : list[dict[str, np.ndarray]]
        各指数和項に対応する 1D カーネル分解。各要素は
        ``S_1d``, ``U_L``, ``S_L``, ``Vt_L``, ``U_s``, ``S_s``,
        ``Vt_s`` を持つ。
    rho_grid : np.ndarray, shape (N, N, N)
        3D 密度グリッド。
    v_analytic : np.ndarray, shape (N, N, N)
        比較対象の解析ポテンシャル。
    dx : float
        グリッド幅。
    k_diag_true : float
        立方体セル積分に基づく真の対角補正係数。
    svd_ranks : list[int]
        評価する SVD / RPCA 低ランク成分の rank。
    thresholds : list[float]
        RPCA sparse 成分をしきい値処理する値。

    Returns
    -------
    dict[str, np.ndarray | dict[str, np.ndarray]]
        SVD only、RPCA no threshold、RPCA threshold ごとの
        ポテンシャル相対誤差と Hartree エネルギー相対誤差。
    """
    errors_v_svd_only = []
    errors_e_svd_only = []
    errors_v_rpca = []
    errors_e_rpca = []
    errors_v_rpca_thresh = {thresh: [] for thresh in thresholds}
    errors_e_rpca_thresh = {thresh: [] for thresh in thresholds}

    diag_coeff = k_diag_true - float(np.sum(fit.weights))

    def _correct(V_raw: np.ndarray) -> np.ndarray:
        return (V_raw + diag_coeff * rho_grid) * dx**3

    for rank in svd_ranks:
        print(f"[rank {rank}] evaluating SVD / RPCA error sweep", flush=True)

        V_svd = np.zeros_like(rho_grid)
        V_rpca_no_thresh = np.zeros_like(rho_grid)
        V_rpca_th = {thresh: np.zeros_like(rho_grid) for thresh in thresholds}

        for w_k, k_data in zip(fit.weights, rpca_1d_list):
            K_svd_1d = reconstruct_svd(
                k_data["U_s"][:, :rank],
                k_data["S_s"][:rank],
                k_data["Vt_s"][:rank, :],
            )
            V_svd += w_k * apply_3d_kernel(K_svd_1d, rho_grid)

            L_r_1d = reconstruct_svd(
                k_data["U_L"][:, :rank],
                k_data["S_L"][:rank],
                k_data["Vt_L"][:rank, :],
            )
            # Compute L_r_1d contribution once; reuse across all thresholds
            # via linearity: apply_3d_kernel(L+S) = apply_3d_kernel(L) + apply_3d_kernel(S)
            V_L = w_k * apply_3d_kernel(L_r_1d, rho_grid)
            V_rpca_no_thresh += V_L + w_k * apply_3d_kernel(
                k_data["S_1d"],
                rho_grid,
            )

            abs_S_1d = np.abs(k_data["S_1d"])
            for thresh in thresholds:
                S_thresh = np.where(abs_S_1d > thresh, k_data["S_1d"], 0.0)
                V_rpca_th[thresh] += V_L + w_k * apply_3d_kernel(
                    S_thresh,
                    rho_grid,
                )

        err_v_s, err_e_s = v_e_errors(_correct(V_svd), v_analytic, rho_grid, dx)
        errors_v_svd_only.append(err_v_s)
        errors_e_svd_only.append(err_e_s)

        err_v_rpca_no, err_e_rpca_no = v_e_errors(
            _correct(V_rpca_no_thresh),
            v_analytic,
            rho_grid,
            dx,
        )
        errors_v_rpca.append(err_v_rpca_no)
        errors_e_rpca.append(err_e_rpca_no)

        for thresh in thresholds:
            err_v_th, err_e_th = v_e_errors(
                _correct(V_rpca_th[thresh]),
                v_analytic,
                rho_grid,
                dx,
            )
            errors_v_rpca_thresh[thresh].append(err_v_th)
            errors_e_rpca_thresh[thresh].append(err_e_th)

    return {
        "errors_v_svd_only": np.asarray(errors_v_svd_only),
        "errors_e_svd_only": np.asarray(errors_e_svd_only),
        "errors_v_rpca": np.asarray(errors_v_rpca),
        "errors_e_rpca": np.asarray(errors_e_rpca),
        "errors_v_rpca_thresh": {
            f"{thresh:.0e}": np.asarray(values)
            for thresh, values in errors_v_rpca_thresh.items()
        },
        "errors_e_rpca_thresh": {
            f"{thresh:.0e}": np.asarray(values)
            for thresh, values in errors_e_rpca_thresh.items()
        },
    }
