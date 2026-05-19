"""時間計測用 PyTorch 入力テンソルの準備。"""

from __future__ import annotations

import numpy as np

from src.approximation.timing.types import TimingBenchmarkInputs
from src.approximation.torch_kernels import to_float64_tensor
from src.potential.separable_density import (
    make_gaussian_density_terms,
    materialize_density_terms,
)
from src.utils.cache import Rpca1dComponents
from src.utils.grid import build_coords_centered


def validate_rank(
    rpca_1d_list: list[Rpca1dComponents],
    rank: int,
) -> None:
    """要求 rank が読み込み済みデータで利用可能か確認する。

    Parameters
    ----------
    rpca_1d_list : list[Rpca1dComponents]
        RPCA / SVD 分解済み 1D カーネルデータ。
    rank : int
        使用したい SVD / RPCA rank。

    Raises
    ------
    ValueError
        ``rank <= 0`` または読み込み済みデータの rank を超える場合。
    """
    if rank <= 0:
        raise ValueError("rank must be positive.")
    available_rank = min(
        min(k_data["S_s"].shape[0], k_data["S_L"].shape[0])
        for k_data in rpca_1d_list
    )
    if rank > available_rank:
        raise ValueError(
            f"requested rank {rank} exceeds available rank "
            f"{available_rank}."
        )


def prepare_timing_benchmark_inputs(
    *,
    n_grid: int,
    length: float,
    density_alpha: float,
    weights: np.ndarray,
    k_1d_list: list[np.ndarray],
    rpca_1d_list: list[Rpca1dComponents],
    r_bench: int,
    tau_bench: float,
) -> TimingBenchmarkInputs:
    """PyTorch 時間計測用の入力テンソル一式を作る。

    Parameters
    ----------
    n_grid : int
        1 軸あたりの格子点数。
    length : float
        計算領域の一辺の長さ。
    density_alpha : float
        Gaussian 密度の幅パラメータ。
    weights : np.ndarray, shape (K,)
        指数和の重み。
    k_1d_list : list[np.ndarray]
        full 1D Gaussian カーネル行列。
    rpca_1d_list : list[Rpca1dComponents]
        RPCA / SVD 分解済み 1D カーネルデータ。
    r_bench : int
        SVD / RPCA のベンチマーク用 rank。
    tau_bench : float
        RPCA の S 成分をゼロ化する閾値。

    Returns
    -------
    inputs : TimingBenchmarkInputs
        10 手法の時間計測に必要な PyTorch 入力一式。
    """
    validate_rank(rpca_1d_list, r_bench)
    dx, x_axis = build_coords_centered(n_grid, length)
    rho_terms = make_gaussian_density_terms(x_axis, density_alpha)
    rho_grid = materialize_density_terms(rho_terms)
    rho_pt = to_float64_tensor(rho_grid)
    rho_terms_pt = [
        (
            term.coefficient,
            to_float64_tensor(term.fx),
            to_float64_tensor(term.fy),
            to_float64_tensor(term.fz),
        )
        for term in rho_terms
    ]

    rpca_dense_data_pt = []
    rpca_lowrank_only_data_pt = []
    lowrank_data_pt = []
    full_kernels_pt = []
    lowrank_only_count = 0
    dense_count = 0

    for weight, kernel, k_data in zip(weights, k_1d_list, rpca_1d_list):
        w_k = float(weight)

        ur = k_data["U_s"][:, :r_bench]
        sr = k_data["S_s"][:r_bench]
        vtr = k_data["Vt_s"][:r_bench, :]

        ur_l = k_data["U_L"][:, :r_bench]
        sr_l = k_data["S_L"][:r_bench]
        vtr_l = k_data["Vt_L"][:r_bench, :]

        s_1d = k_data["S_1d"]
        s_thr = np.where(np.abs(s_1d) > tau_bench, s_1d, 0.0)
        nnz = np.count_nonzero(s_thr)

        lowrank_data_pt.append(
            (
                w_k,
                to_float64_tensor(ur),
                to_float64_tensor(sr),
                to_float64_tensor(vtr),
            )
        )
        full_kernels_pt.append((w_k, to_float64_tensor(kernel)))

        ur_l_pt = to_float64_tensor(ur_l)
        sr_l_pt = to_float64_tensor(sr_l)
        vtr_l_pt = to_float64_tensor(vtr_l)
        if nnz == 0:
            rpca_lowrank_only_data_pt.append(
                (w_k, ur_l_pt, sr_l_pt, vtr_l_pt)
            )
            lowrank_only_count += 1
        else:
            rpca_dense_data_pt.append(
                (
                    w_k,
                    ur_l_pt,
                    sr_l_pt,
                    vtr_l_pt,
                    to_float64_tensor(s_thr),
                )
            )
            dense_count += 1

    kernel_list_pt = [
        (float(w), to_float64_tensor(kernel))
        for w, kernel in zip(weights, k_1d_list)
    ]

    return TimingBenchmarkInputs(
        dx=dx,
        rho_pt=rho_pt,
        rho_terms_pt=rho_terms_pt,
        full_kernels_pt=full_kernels_pt,
        lowrank_data_pt=lowrank_data_pt,
        rpca_lowrank_only_data_pt=rpca_lowrank_only_data_pt,
        rpca_dense_data_pt=rpca_dense_data_pt,
        kernel_list_pt=kernel_list_pt,
        rpca_lowrank_only_count=lowrank_only_count,
        rpca_dense_count=dense_count,
    )
