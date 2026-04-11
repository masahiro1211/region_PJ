import numpy as np
from ..utils.tensor_ops import unfold, mode_n_product


def perform_tucker(X: np.ndarray, ranks: list[int]):
    """HOSVD によりタッカー分解を計算する。

    分解形は X ≈ G ×_1 U1 ×_2 U2 ... ×_N UN である。

    Args:
        X: 入力テンソル。形状は (I_1, I_2, ..., I_N)。
        ranks: 各モードランク [r_1, r_2, ..., r_N]。

    Returns:
        G と factors のタプル。
        G はコアテンソル、factors は各モード因子行列のリスト。

    Raises:
        ValueError: ranks の長さが X.ndim と一致しない場合。
    """
    if len(ranks) != X.ndim:
        raise ValueError(
            f"ranks の長さ ({len(ranks)}) はテンソルの次元数 ({X.ndim}) "
            "と一致する必要があります。"
        )

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
    """タッカー分解結果から元テンソルを再構成する。

    Args:
        G: コアテンソル。
        factors: 因子行列のリスト。

    Returns:
        再構成テンソル。
    """
    X_approx = G.copy()
    for mode, U in enumerate(factors):
        X_approx = mode_n_product(X_approx, U, mode)
    return X_approx
