import numpy as np


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
    """SVD低ランク行列を再構成せずに1Dベクトルへ作用させる。

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
