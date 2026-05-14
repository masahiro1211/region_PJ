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
        rPCA で S 成分が非ゼロの項。
    kernel_list_pt : list
        CP-ρ full 実装用の ``(weight, K)`` リスト。
    rpca_lowrank_only_count : int
        rPCA で S 成分がゼロの項数。
    rpca_dense_count : int
        rPCA で S 成分が非ゼロの項数。
    """

    dx: float
    rho_pt: Any
    rho_terms_pt: list
    full_kernels_pt: list
    lowrank_data_pt: list
    rpca_lowrank_only_data_pt: list
    rpca_dense_data_pt: list
    kernel_list_pt: list
    rpca_lowrank_only_count: int
    rpca_dense_count: int


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
