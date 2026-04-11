import numpy as np
from ..decomposition.svd import perform_svd
from ..decomposition.tucker import perform_tucker, reconstruct


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
        A, B = perform_svd(X, rank=ranks)
        return A @ B

    if method == "tucker":
        if isinstance(ranks, int):
            ranks = [ranks] * X.ndim
        G, factors = perform_tucker(X, ranks)
        return reconstruct(G, factors)

    raise ValueError(f"未知の method: {method!r}。'svd' または 'tucker' を指定してください。")
