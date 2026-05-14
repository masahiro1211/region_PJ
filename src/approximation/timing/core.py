"""時間計測と PyTorch 実行環境に関する小さなヘルパー。"""

from __future__ import annotations

import timeit
from collections.abc import Callable
from typing import Any

import numpy as np

from src.approximation.timing.types import TorchMetadata

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    torch = None  # type: ignore[assignment]


def require_torch() -> Any:
    """PyTorch が利用可能な場合に torch モジュールを返す。

    Returns
    -------
    torch_module : Any
        import 済みの ``torch`` モジュール。

    Raises
    ------
    ImportError
        PyTorch がインストールされていない場合。
    """
    if torch is None:
        raise ImportError(
            "PyTorch is required for timing benchmarks. "
            "Install it with `pip install -e \".[notebook]\"`."
        )
    return torch


def benchmark(
    fn: Callable[[], Any],
    n_warmup: int,
    n_inner: int,
    n_repeat: int,
) -> np.ndarray:
    """関数の実行時間を repeat ごとに測定する。

    Parameters
    ----------
    fn : Callable[[], Any]
        計測対象の引数なし関数。GPU が利用可能な場合、この関数の
        先頭と末尾で ``torch.cuda.synchronize()`` を呼ぶ想定。
    n_warmup : int
        timeit の前に実行するウォームアップ回数。
    n_inner : int
        1 repeat あたりの呼び出し回数。
    n_repeat : int
        独立測定の repeat 数。

    Returns
    -------
    times : np.ndarray, shape (n_repeat,)
        1 呼び出しあたりの実行時間 [秒]。

    Raises
    ------
    ValueError
        ``n_warmup < 0`` または ``n_inner <= 0`` または
        ``n_repeat <= 0`` の場合。
    """
    if n_warmup < 0:
        raise ValueError("n_warmup must be non-negative.")
    if n_inner <= 0:
        raise ValueError("n_inner must be positive.")
    if n_repeat <= 0:
        raise ValueError("n_repeat must be positive.")

    for _ in range(n_warmup):
        fn()

    raw = timeit.repeat(fn, number=n_inner, repeat=n_repeat)
    return np.asarray(raw, dtype=float) / n_inner


def with_cuda_sync(fn: Callable[[], Any]) -> Callable[[], Any]:
    """GPU が利用可能な場合に CUDA 同期を挟む関数へ変換する。

    Parameters
    ----------
    fn : Callable[[], Any]
        計測対象の引数なし関数。

    Returns
    -------
    wrapped : Callable[[], Any]
        GPU 利用時のみ前後に ``torch.cuda.synchronize()`` を挟む関数。
    """
    torch_mod = require_torch()
    use_cuda = torch_mod.cuda.is_available()

    def wrapped() -> Any:
        if use_cuda:
            torch_mod.cuda.synchronize()
        result = fn()
        if use_cuda:
            torch_mod.cuda.synchronize()
        return result

    return wrapped


def get_torch_metadata() -> TorchMetadata:
    """PyTorch 実行環境のメタデータを返す。

    Returns
    -------
    metadata : TorchMetadata
        バージョン、CUDA 利用可否、CUDA device 数。
    """
    torch_mod = require_torch()
    return {
        "version": torch_mod.__version__,
        "cuda_available": torch_mod.cuda.is_available(),
        "cuda_device_count": torch_mod.cuda.device_count(),
    }
