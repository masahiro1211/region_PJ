import numpy as np
from typing import Literal, overload
from ..decomposition.svd import perform_svd
from ..decomposition.tucker import perform_tucker, reconstruct


@overload
def approximate(
    X: np.ndarray,
    ranks: int,
    method: Literal["svd"],
) -> np.ndarray: ...


@overload
def approximate(
    X: np.ndarray,
    ranks: int | list[int],
    method: Literal["tucker"] = "tucker",
) -> np.ndarray: ...


def approximate(
    X: np.ndarray,
    ranks: int | list[int],
    method: str = "tucker",
) -> np.ndarray:
    """SVD または Tucker により低ランク近似を計算する。

    Args:
        X: 入力配列。method="svd" の場合は 2 次元行列。
        ranks: 近似ランク。
            method="svd" では int、method="tucker" では int または list[int]。
        method: 近似手法。"svd" または "tucker"。

    Returns:
        近似後の配列。

    Raises:
        ValueError: method が未対応、または SVD で X が 2 次元でない場合。
        TypeError: method="svd" で ranks が int でない場合。
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

    raise ValueError(
        f"未知の method: {method!r}。'svd' または 'tucker' を指定してください。"
    )
