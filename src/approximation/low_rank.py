import numpy as np
from typing import Literal, overload
from ..decomposition.svd import perform_svd
from ..decomposition.tucker import perform_tucker, reconstruct


def truncated_svd(
    matrix: np.ndarray,
    rank: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """2D行列の上位特異値成分を返す。

    Parameters
    ----------
    matrix : np.ndarray, shape (M, N)
        SVD を適用する2D行列。
    rank : int
        取得する特異値成分の数。

    Returns
    -------
    u : np.ndarray, shape (M, rank)
        左特異ベクトル。
    singular_values : np.ndarray, shape (rank,)
        特異値。
    vt : np.ndarray, shape (rank, N)
        右特異ベクトルの転置。

    Raises
    ------
    ValueError
        ``matrix`` が2Dでない場合、または ``rank`` が不正な場合。
    """
    if matrix.ndim != 2:
        raise ValueError("truncated_svd expects a 2D matrix.")
    if rank < 1 or rank > min(matrix.shape):
        raise ValueError(
            "rank must satisfy 1 <= rank <= min(matrix.shape); "
            f"got rank={rank}, shape={matrix.shape}."
        )

    u, singular_values, vt = np.linalg.svd(matrix, full_matrices=False)
    return u[:, :rank], singular_values[:rank], vt[:rank, :]


def reconstruct_svd(
    u: np.ndarray,
    singular_values: np.ndarray,
    vt: np.ndarray,
) -> np.ndarray:
    """SVD因子から行列を再構成する。

    Parameters
    ----------
    u : np.ndarray, shape (M, r)
        左特異ベクトル。
    singular_values : np.ndarray, shape (r,)
        特異値。
    vt : np.ndarray, shape (r, N)
        右特異ベクトルの転置。

    Returns
    -------
    matrix : np.ndarray, shape (M, N)
        ``u @ diag(singular_values) @ vt`` に対応する行列。
    """
    return (u * singular_values) @ vt


def apply_low_rank_svd(
    u: np.ndarray,
    singular_values: np.ndarray,
    vt: np.ndarray,
    vector: np.ndarray,
) -> np.ndarray:
    """SVD低ランク行列を再構成せずにベクトルへ作用させる。

    Parameters
    ----------
    u : np.ndarray, shape (M, r)
        左特異ベクトル。
    singular_values : np.ndarray, shape (r,)
        特異値。
    vt : np.ndarray, shape (r, N)
        右特異ベクトルの転置。
    vector : np.ndarray, shape (N,)
        作用対象のベクトル。

    Returns
    -------
    result : np.ndarray, shape (M,)
        ``U diag(s) Vt @ vector`` の結果。
    """
    return u @ (singular_values * (vt @ vector))


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

    Parameters
    ----------
    X : np.ndarray
        入力配列。``method="svd"`` の場合は2D行列。
    ranks : int or list[int]
        近似ランク。``method="svd"`` では int、
        ``method="tucker"`` では int または mode ごとの rank リスト。
    method : str, default="tucker"
        近似手法。``"svd"`` または ``"tucker"``。

    Returns
    -------
    X_approx : np.ndarray
        近似後の配列。

    Raises
    ------
    ValueError
        ``method`` が未対応、または SVD で ``X`` が2Dでない場合。
    TypeError
        ``method="svd"`` で ``ranks`` が int でない場合。
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
