"""時間計測ヘルパーで共有する型定義。"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, TypedDict


@dataclass(frozen=True)
class TimingBenchmarkInputs:
    """時間計測対象 callable の構築に必要な PyTorch 入力一式。

    Attributes
    ----------
    dx : float
        グリッド間隔。
    rho_pt : torch.Tensor, shape (N, N, N)
        dense 3D 密度テンソル。
    rho_terms_pt : list
        CP 形式密度項。各項は ``(coefficient, fx, fy, fz)``。
    full_kernels_pt : list
        full 実装用の ``(weight, K)`` リスト。
    lowrank_data_pt : list
        SVD low-rank 実装用の ``(weight, U, s, Vt)`` リスト。
    rpca_lowrank_only_data_pt : list
        rPCA で S 成分がゼロの項の ``(weight, U_L, s_L, Vt_L)``。
    rpca_dense_data_pt : list
        rPCA で S 成分が非ゼロの項。S は dense Tensor。
    rpca_sparse_data_pt : list
        rPCA で S 成分が非ゼロの項。S は sparse CSR Tensor。
    kernel_list_pt : list
        CP-ρ full 実装用の ``(weight, K)`` リスト。
    rpca_lowrank_only_count : int
        rPCA で S 成分がゼロの項数。
    rpca_dense_count : int
        rPCA で S 成分が非ゼロの項数。
    rpca_sparse_count : int
        sparse-S 版 rPCA で S 成分が非ゼロの項数。
    s_total_size : int
        threshold 後 S 成分の総要素数。
    s_nnz : int
        threshold 後 S 成分の非ゼロ要素数。
    s_zero_rate_percent : float
        threshold 後 S 成分のゼロ要素率 [%]。
    s_nonzero_rate_percent : float
        threshold 後 S 成分の非ゼロ要素率 [%]。
    """

    dx: float
    rho_pt: Any
    rho_terms_pt: list
    full_kernels_pt: list
    lowrank_data_pt: list
    rpca_lowrank_only_data_pt: list
    rpca_dense_data_pt: list
    rpca_sparse_data_pt: list
    kernel_list_pt: list
    rpca_lowrank_only_count: int
    rpca_dense_count: int
    rpca_sparse_count: int
    s_total_size: int
    s_nnz: int
    s_zero_rate_percent: float
    s_nonzero_rate_percent: float


@dataclass(frozen=True)
class TimingTarget:
    """1 つの時間計測対象を表す。

    Attributes
    ----------
    name : str
        保存ファイル名 ``t_{name}.npy`` に使う手法名。
    fn : Callable[[], Any]
        引数なしで実行できる計測対象関数。
    """

    name: str
    fn: Callable[[], Any]


class TorchMetadata(TypedDict):
    """PyTorch 実行環境の記録用メタデータ。"""

    version: str
    cuda_available: bool
    cuda_device_count: int


def timing_sweep_label(
    n_grid: int,
    length: float,
    density_alpha: float,
    exp_sum_rank: int,
    r_bench: int,
    tau_bench: float,
    n_warmup: int,
    n_inner: int,
    n_repeat: int,
) -> str:
    """タイミング計測条件を一意に表すディレクトリ名を返す。

    Parameters
    ----------
    n_grid : int
        グリッド点数 N。
    length : float
        ボックス長 L。
    density_alpha : float
        密度ガウシアンの指数係数 α。
    exp_sum_rank : int
        使用する指数和のランク R。
    r_bench : int
        SVD/RPCA のランク r。
    tau_bench : float
        RPCA の S 閾値 τ。
    n_warmup : int
        ウォームアップ回数。
    n_inner : int
        内側ループ回数。
    n_repeat : int
        繰り返し回数。

    Returns
    -------
    label : str
        ``run_timing_benchmark.py`` が出力ディレクトリ名に使うラベルと同一の文字列。
    """
    return (
        f"N{n_grid}_L{length:g}_alpha{density_alpha:g}"
        f"_R{exp_sum_rank:02d}_r{r_bench:02d}"
        f"_tau{tau_bench:.0e}"
        f"_warm{n_warmup}_inner{n_inner}"
        f"_repeat{n_repeat}"
    )
