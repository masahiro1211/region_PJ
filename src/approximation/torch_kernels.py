"""PyTorch 版の1Dカーネル作用と低ランク作用。"""

from __future__ import annotations

from typing import Any

import numpy as np

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    torch = None  # type: ignore[assignment]


def _require_torch() -> Any:
    """PyTorch が利用可能な場合に torch モジュールを返す。"""
    if torch is None:
        raise ImportError(
            "PyTorch is required for src.approximation.torch_kernels. "
            "Install the notebook optional dependencies to use this module."
        )
    return torch


def to_float64_tensor(array: np.ndarray) -> Any:
    """NumPy配列を float64 の PyTorch Tensor に変換する。

    Parameters
    ----------
    array : np.ndarray
        変換対象の配列。

    Returns
    -------
    tensor : torch.Tensor
        ``dtype=torch.float64`` の Tensor。
    """
    torch_mod = _require_torch()
    return torch_mod.as_tensor(array, dtype=torch_mod.float64)


def apply_dense_axis(
    kernel: Any,
    rho: Any,
    axis: int,
) -> Any:
    """denseな1Dカーネル行列を指定軸に作用させる。

    Parameters
    ----------
    kernel : torch.Tensor, shape (N, N)
        指定軸に作用させる1Dカーネル行列。
    rho : torch.Tensor
        カーネルを作用させるテンソル。
    axis : int
        ``rho`` のうち ``kernel`` を作用させる軸。

    Returns
    -------
    result : torch.Tensor
        指定軸に ``kernel`` を作用させたテンソル。
    """
    torch_mod = _require_torch()
    tmp = torch_mod.tensordot(kernel, rho, dims=([1], [axis]))
    return torch_mod.movedim(tmp, 0, axis)


def apply_low_rank_axis(
    u: Any,
    singular_values: Any,
    vt: Any,
    rho: Any,
    axis: int,
) -> Any:
    """低ランクSVD因子を指定軸に再構成せず作用させる。

    Parameters
    ----------
    u : torch.Tensor, shape (N, r)
        左特異ベクトル。
    singular_values : torch.Tensor, shape (r,)
        特異値。
    vt : torch.Tensor, shape (r, N)
        右特異ベクトルの転置。
    rho : torch.Tensor
        作用対象のテンソル。
    axis : int
        ``rho`` のうち低ランク行列を作用させる軸。

    Returns
    -------
    result : torch.Tensor
        ``u @ diag(singular_values) @ vt`` を指定軸に作用させた結果。
    """
    torch_mod = _require_torch()
    tmp = torch_mod.tensordot(vt, rho, dims=([1], [axis]))
    tmp = torch_mod.movedim(tmp, 0, axis)
    shape = [1] * rho.ndim
    shape[axis] = -1
    tmp = tmp * singular_values.view(shape)
    out = torch_mod.tensordot(u, tmp, dims=([1], [axis]))
    return torch_mod.movedim(out, 0, axis)


def make_sparse_csr_tensor(
    indptr: np.ndarray,
    indices: np.ndarray,
    data: np.ndarray,
    shape: tuple[int, int],
) -> Any:
    """CSR表現から PyTorch sparse CSR Tensor を構築する。

    Parameters
    ----------
    indptr : np.ndarray, shape (N + 1,)
        CSR の行ポインタ。
    indices : np.ndarray, shape (nnz,)
        CSR の列インデックス。
    data : np.ndarray, shape (nnz,)
        CSR の非ゼロ値。
    shape : tuple[int, int]
        sparse行列の形状。

    Returns
    -------
    sparse_tensor : torch.Tensor
        ``dtype=torch.float64`` の sparse CSR Tensor。
    """
    torch_mod = _require_torch()
    indptr_tensor = torch_mod.as_tensor(
        indptr.copy(),
        dtype=torch_mod.int64,
    ).contiguous()
    indices_tensor = torch_mod.as_tensor(
        indices.copy(),
        dtype=torch_mod.int64,
    ).contiguous()
    data_tensor = torch_mod.as_tensor(
        data.copy(),
        dtype=torch_mod.float64,
    ).contiguous()

    # PyTorch の invariant check は nnz=0 の CSR で
    # contiguous 判定に失敗することがあるため、空行列だけ明示的に外す。
    check_invariants = data_tensor.numel() > 0
    return torch_mod.sparse_csr_tensor(
        indptr_tensor,
        indices_tensor,
        data_tensor,
        size=shape,
        check_invariants=check_invariants,
    )


def apply_exp_sum_3d_full(
    rho: Any,
    kernel_list: list,
) -> Any:
    """フルカーネルを3軸に密行列 tensordot で作用させる。

    各指数和項で N×N の密行列カーネルを3軸すべてに適用する。
    計算量は O(3 K N^4)。速度比較の基準実装。

    Parameters
    ----------
    rho : torch.Tensor, shape (N, N, N)
        電荷密度テンソル。
    kernel_list : list of (float, torch.Tensor shape (N, N))
        各指数和項の重みと N×N フル行列カーネルのリスト。

    Returns
    -------
    out : torch.Tensor, shape (N, N, N)
        ポテンシャルテンソル（dx^3 スケーリングなし）。
    """
    _torch = _require_torch()
    out = _torch.zeros_like(rho)
    for w_k, K in kernel_list:
        tmp = apply_dense_axis(K, rho, axis=0)
        tmp = apply_dense_axis(K, tmp, axis=1)
        tmp = apply_dense_axis(K, tmp, axis=2)
        out = out + w_k * tmp
    return out


def apply_exp_sum_3d_lowrank_naive(
    rho: Any,
    kernel_list: list,
) -> Any:
    """低ランク因子を再構成してから3軸に作用させる（ナイーブ実装）。

    K_r = (U_r Σ_r) @ Vt_r と N×N に再構成してから tensordot する
    ため O(N^4) のまま。``apply_exp_sum_3d_lowrank`` との速度比較用。

    Parameters
    ----------
    rho : torch.Tensor, shape (N, N, N)
        電荷密度テンソル。
    kernel_list : list of (float, U, s, Vt)
        各指数和項の重みと低ランク SVD 因子のリスト。

    Returns
    -------
    out : torch.Tensor, shape (N, N, N)
        ポテンシャルテンソル（dx^3 スケーリングなし）。

    Notes
    -----
    計算量は O(K N^4)。``apply_exp_sum_3d_lowrank`` が O(K r N^3)
    であることと対比するベンチマーク用途のみを想定した実装。
    """
    _torch = _require_torch()
    out = _torch.zeros_like(rho)
    for w_k, U_r, s_r, Vt_r in kernel_list:
        K_r = (U_r * s_r) @ Vt_r
        tmp = apply_dense_axis(K_r, rho, axis=0)
        tmp = apply_dense_axis(K_r, tmp, axis=1)
        tmp = apply_dense_axis(K_r, tmp, axis=2)
        out = out + w_k * tmp
    return out


def apply_exp_sum_3d_lowrank(
    rho: Any,
    kernel_list: list,
) -> Any:
    """低ランク因子を mode-n product の順序で3軸に作用させる。

    K_r を再構成せず Vt_r → Σ_r → U_r の順でテンソルに作用させ、
    中間軸の次元を r に抑える。計算量は O(K r N^3)。

    Parameters
    ----------
    rho : torch.Tensor, shape (N, N, N)
        電荷密度テンソル。
    kernel_list : list of (float, U, s, Vt)
        各指数和項の重みと低ランク SVD 因子のリスト。

    Returns
    -------
    out : torch.Tensor, shape (N, N, N)
        ポテンシャルテンソル（dx^3 スケーリングなし）。
    """
    _torch = _require_torch()
    out = _torch.zeros_like(rho)
    for w_k, U_r, s_r, Vt_r in kernel_list:
        tmp = apply_low_rank_axis(U_r, s_r, Vt_r, rho, axis=0)
        tmp = apply_low_rank_axis(U_r, s_r, Vt_r, tmp, axis=1)
        tmp = apply_low_rank_axis(U_r, s_r, Vt_r, tmp, axis=2)
        out = out + w_k * tmp
    return out


def apply_exp_sum_3d_rpca(
    rho: Any,
    lowrank_only_list: list,
    dense_list: list,
) -> Any:
    """rPCA (L+S) カーネルを3軸に作用させる。

    L 成分は低ランク mode-n product、S 成分は密行列 tensordot で
    それぞれ適用し、各軸で加算する。

    Parameters
    ----------
    rho : torch.Tensor, shape (N, N, N)
        電荷密度テンソル。
    lowrank_only_list : list of (float, U, s, Vt)
        S=0 の項の重みと L 成分の低ランク因子のリスト。
    dense_list : list of (float, U, s, Vt, S_dense)
        S≠0 の項の重みと L 成分の低ランク因子および密行列 S のリスト。

    Returns
    -------
    out : torch.Tensor, shape (N, N, N)
        (L+S) カーネルを3軸に作用させたポテンシャルテンソル
        （dx^3 スケーリングなし）。
    """
    _torch = _require_torch()
    out = _torch.zeros_like(rho)
    for w_k, U_r, s_r, Vt_r in lowrank_only_list:
        tmp = apply_low_rank_axis(U_r, s_r, Vt_r, rho, axis=0)
        tmp = apply_low_rank_axis(U_r, s_r, Vt_r, tmp, axis=1)
        tmp = apply_low_rank_axis(U_r, s_r, Vt_r, tmp, axis=2)
        out = out + w_k * tmp
    for w_k, U_r, s_r, Vt_r, S_dense in dense_list:
        tmp = (
            apply_low_rank_axis(U_r, s_r, Vt_r, rho, axis=0)
            + apply_dense_axis(S_dense, rho, axis=0)
        )
        tmp = (
            apply_low_rank_axis(U_r, s_r, Vt_r, tmp, axis=1)
            + apply_dense_axis(S_dense, tmp, axis=1)
        )
        tmp = (
            apply_low_rank_axis(U_r, s_r, Vt_r, tmp, axis=2)
            + apply_dense_axis(S_dense, tmp, axis=2)
        )
        out = out + w_k * tmp
    return out


def apply_exp_sum_3d_rpca_l_only(
    rho: Any,
    lowrank_only_list: list,
    dense_list: list,
) -> Any:
    """rPCA の L 成分（全項）のみを3軸に作用させる。

    ``lowrank_only_list`` の全項と ``dense_list`` の L 部分を
    低ランク mode-n product で適用する。S 成分は無視。
    計算量は O(K r N^3)。

    Parameters
    ----------
    rho : torch.Tensor, shape (N, N, N)
        電荷密度テンソル。
    lowrank_only_list : list of (float, U, s, Vt)
        S=0 の項の重みと低ランク因子のリスト。
    dense_list : list of (float, U, s, Vt, S_dense)
        S≠0 の項の重みと低ランク因子および密行列 S のリスト。

    Returns
    -------
    out : torch.Tensor, shape (N, N, N)
        L 成分のみを3軸に作用させたポテンシャルテンソル
        （dx^3 スケーリングなし）。
    """
    _torch = _require_torch()
    out = _torch.zeros_like(rho)
    for w_k, U_r, s_r, Vt_r in lowrank_only_list:
        tmp = apply_low_rank_axis(U_r, s_r, Vt_r, rho, axis=0)
        tmp = apply_low_rank_axis(U_r, s_r, Vt_r, tmp, axis=1)
        tmp = apply_low_rank_axis(U_r, s_r, Vt_r, tmp, axis=2)
        out = out + w_k * tmp
    for w_k, U_r, s_r, Vt_r, _ in dense_list:
        tmp = apply_low_rank_axis(U_r, s_r, Vt_r, rho, axis=0)
        tmp = apply_low_rank_axis(U_r, s_r, Vt_r, tmp, axis=1)
        tmp = apply_low_rank_axis(U_r, s_r, Vt_r, tmp, axis=2)
        out = out + w_k * tmp
    return out


def apply_exp_sum_3d_rpca_s_only(
    rho: Any,
    dense_list: list,
) -> Any:
    """rPCA の S 成分（dense 項のみ）を3軸に作用させる。

    ``dense_list`` の S 部分だけを密行列 tensordot で適用する。
    L 成分は無視。S が密のため計算量は O(K N^4)。

    Parameters
    ----------
    rho : torch.Tensor, shape (N, N, N)
        電荷密度テンソル。
    dense_list : list of (float, U, s, Vt, S_dense)
        S≠0 の項の重みと低ランク因子および密行列 S のリスト。

    Returns
    -------
    out : torch.Tensor, shape (N, N, N)
        S 成分のみを3軸に作用させたポテンシャルテンソル
        （dx^3 スケーリングなし）。
    """
    _torch = _require_torch()
    out = _torch.zeros_like(rho)
    for w_k, _, _, _, S_dense in dense_list:
        tmp = apply_dense_axis(S_dense, rho, axis=0)
        tmp = apply_dense_axis(S_dense, tmp, axis=1)
        tmp = apply_dense_axis(S_dense, tmp, axis=2)
        out = out + w_k * tmp
    return out


def apply_sparse_axis(
    sparse_kernel: Any,
    rho: Any,
    axis: int,
) -> Any:
    """sparse CSR の1Dカーネル行列を指定軸に作用させる。

    Parameters
    ----------
    sparse_kernel : torch.Tensor, shape (N, N)
        sparse CSR 形式の1Dカーネル行列。
    rho : torch.Tensor
        カーネルを作用させるテンソル。
    axis : int
        ``rho`` のうち ``sparse_kernel`` を作用させる軸。

    Returns
    -------
    result : torch.Tensor
        指定軸に ``sparse_kernel`` を作用させたテンソル。
    """
    torch_mod = _require_torch()
    rho_moved = torch_mod.movedim(rho, axis, 0)
    n_axis = rho_moved.shape[0]
    rest = rho_moved.shape[1:]
    rho_2d = rho_moved.contiguous().reshape(n_axis, -1)
    out_2d = torch_mod.sparse.mm(sparse_kernel, rho_2d)
    out = out_2d.reshape(n_axis, *rest)
    return torch_mod.movedim(out, 0, axis)
