"""Gaussian ポテンシャル計算の離散化誤差評価。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

from src.approximation.exp_sum.models import ExponentialSum
from src.potential.charge_potential import v_analytic_gaussian
from src.potential.separable_density import (
    apply_exp_sum_to_separable_density,
    make_gaussian_density_terms,
    materialize_density_terms,
)
from src.utils.grid import build_xyz
from src.utils.metrics import hartree_energy


@dataclass(frozen=True)
class GridEnergyReference:
    """各Nの解析ポテンシャルを使ったグリッド上のエネルギー評価。

    Attributes
    ----------
    N : int
        各軸のグリッド点数。
    dx : float
        グリッド幅。
    energy_grid : float
        同じグリッド上で解析ポテンシャルを積分したエネルギー。
    energy_exact : float
        連続系での解析的な自己エネルギー。
    abs_error : float
        ``abs(energy_grid - energy_exact)``。
    rel_error : float
        ``abs_error / abs(energy_exact)``。
    """

    N: int
    dx: float
    energy_grid: float
    energy_exact: float
    abs_error: float
    rel_error: float


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
    err_vs_grid_ref : float
        同じグリッド上の解析ポテンシャル積分に対する相対誤差。
    err_vs_cont_exact : float
        連続系の解析エネルギーに対する相対誤差。
    """

    rank: int
    N: int
    dx: float
    energy_exp_sum: float
    err_vs_grid_ref: float
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


def compute_grid_energy_reference(
    N: int,
    L: float,
    alpha: float,
) -> GridEnergyReference:
    """指定Nで解析ポテンシャルをグリッド積分したエネルギーを計算する。

    Parameters
    ----------
    N : int
        各軸のグリッド点数。``build_xyz`` と同じく奇数を想定する。
    L : float
        計算領域の一辺の長さ。
    alpha : float
        Gaussian 密度の幅パラメータ。

    Returns
    -------
    reference : GridEnergyReference
        グリッド上のエネルギーと連続系解析値との差分をまとめた結果。
    """
    dx = L / N
    xyz = build_xyz(N, L)
    radius = np.sqrt(xyz[0] ** 2 + xyz[1] ** 2 + xyz[2] ** 2)
    rho = np.exp(-alpha * radius**2)
    # 解析ポテンシャルは、厳密解を使用してはいけないのでは？
    potential_exact = v_analytic_gaussian(radius, alpha)
    energy_grid = hartree_energy(rho, potential_exact, dx)
    energy_exact = analytic_gaussian_hartree_energy(alpha)
    abs_error = abs(energy_grid - energy_exact)
    rel_error = abs_error / abs(energy_exact)

    return GridEnergyReference(
        N=N,
        dx=dx,
        energy_grid=energy_grid,
        energy_exact=energy_exact,
        abs_error=abs_error,
        rel_error=rel_error,
    )


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
        グリッド参照値と連続系解析値に対する誤差評価。
    """
    dx = L / N
    xyz = build_xyz(N, L)
    x_axis = xyz[0, :, 0, 0]
    density_terms = make_gaussian_density_terms(x_axis, alpha)
    rho = materialize_density_terms(density_terms)
    reference = compute_grid_energy_reference(N=N, L=L, alpha=alpha)
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
        err_vs_grid_ref=abs(energy_exp_sum - reference.energy_grid)
        / abs(reference.energy_grid),
        err_vs_cont_exact=abs(energy_exp_sum - reference.energy_exact)
        / abs(reference.energy_exact),
    )


def compare_exp_sum_discretization(
    N_values: Sequence[int],
    L: float,
    alpha: float,
    fits: Mapping[int, ExponentialSum],
    ranks: Sequence[int],
    cell_int_const: float = 2.38,
) -> tuple[list[GridEnergyReference], dict[int, list[ExpSumEnergyError]]]:
    """複数Nで離散化誤差と指数和近似誤差を比較する。

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
    grid_rows : list[GridEnergyReference]
        各Nの純粋なグリッド積分誤差。
    rank_rows : dict[int, list[ExpSumEnergyError]]
        rank ごとの指数和近似エネルギー誤差。
    """
    grid_rows = [
        compute_grid_energy_reference(N=N, L=L, alpha=alpha)
        for N in N_values
    ]
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

    return grid_rows, rank_rows
