import numpy as np
from ..utils.tensor_ops import unfold, mode_n_product


def perform_tucker(X: np.ndarray, ranks: list[int]):
    """HOSVD によるタッカー分解を行う。

    X ≈ G ×_1 U1 ×_2 U2 ... ×_N UN

    Parameters
    ----------
    X : ndarray, shape (I_1, I_2, ..., I_N)
    ranks : list of int, length N
        各モードのランク [r_1, r_2, ..., r_N]

    Returns
    -------
    G : ndarray, shape (r_1, r_2, ..., r_N)  コアテンソル
    factors : list of ndarray
        各モードの因子行列 [U_1, ..., U_N], U_n.shape = (I_n, r_n)
    """
    if len(ranks) != X.ndim:
        raise ValueError(f"ranks の長さ ({len(ranks)}) はテンソルの次元数 ({X.ndim}) と一致する必要があります。")

    factors = []
    for mode, rank in enumerate(ranks):
        X_unfold = unfold(X, mode)           # (I_n, prod of others)
        U, _, _ = np.linalg.svd(X_unfold, full_matrices=False)
        factors.append(U[:, :rank])          # (I_n, r_n)

    # コアテンソル G = X ×_1 U1^T ×_2 U2^T ... ×_N UN^T
    G = X.copy()
    for mode, U in enumerate(factors):
        G = mode_n_product(G, U.T, mode)

    return G, factors


def reconstruct(G: np.ndarray, factors: list[np.ndarray]) -> np.ndarray:
    """タッカー分解結果からテンソルを再構成する。

    Parameters
    ----------
    G : ndarray  コアテンソル
    factors : list of ndarray  因子行列リスト

    Returns
    -------
    X_approx : ndarray  近似テンソル
    """
    X_approx = G.copy()
    for mode, U in enumerate(factors):
        X_approx = mode_n_product(X_approx, U, mode)
    return X_approx
