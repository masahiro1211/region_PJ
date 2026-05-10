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
    return torch_mod.sparse_csr_tensor(
        torch_mod.as_tensor(indptr.copy(), dtype=torch_mod.int64).contiguous(),
        torch_mod.as_tensor(indices.copy(), dtype=torch_mod.int64).contiguous(),
        torch_mod.as_tensor(data.copy(), dtype=torch_mod.float64).contiguous(),
        size=shape,
        check_invariants=True,
    )


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
