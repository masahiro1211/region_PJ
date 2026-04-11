import numpy as np
from typing import Literal, overload
from ..decomposition.svd import perform_svd
from ..decomposition.tucker import perform_tucker, reconstruct


@overload
def approximate(X: np.ndarray, ranks: int, method: Literal["svd"]) -> np.ndarray: ...


@overload
def approximate(X: np.ndarray, ranks: int | list[int], method: Literal["tucker"] = "tucker") -> np.ndarray: ...


def approximate(X: np.ndarray, ranks: int | list[int], method: str = "tucker") -> np.ndarray:
    """SVD またはタッカー分解による低ランク近似を返す。

    Parameters
    ----------
    X : ndarray
        入力テンソル（SVD の場合は 2 次元行列）
    ranks : int or list of int
        SVD の場合はスカラー、Tucker の場合は各モードのランクリスト
    method : {"tucker", "svd"}

    Returns
    -------
    X_approx : ndarray  近似テンソル
    """
    if method == "svd":
        if X.ndim != 2:
            raise ValueError("SVD は 2 次元行列にのみ適用できます。")
        if not isinstance(ranks, int):
            raise TypeError("SVD の ranks は int を指定してください。")
        A, B = perform_svd(X, rank=ranks)
        return A @ B

    if method == "tucker":
        if isinstance(ranks, int):
            tucker_ranks = [ranks] * X.ndim
        else:
            tucker_ranks = ranks
        G, factors = perform_tucker(X, tucker_ranks)
        return reconstruct(G, factors)

    raise ValueError(f"未知の method: {method!r}。'svd' または 'tucker' を指定してください。")
