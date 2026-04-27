import numpy as np
from sklearn.utils.extmath import randomized_svd

from ..utils.tensor_ops import unfold, mode_n_product
from .rpca import randomized_rpca


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


def perform_tucker_rpca(
    X: np.ndarray,
    ranks: list[int],
    rpca_rank: int | None = None,
    lam: float | None = None,
    max_iter: int = 200,
    tol: float = 1e-6,
    random_state: int = 0,
) -> tuple[np.ndarray, list[np.ndarray], list[np.ndarray]]:
    """各 mode unfolding を RPCA で L_n + S_n に分け、L_n の SVD から因子を作る。

    分解形は X ≈ G ×_1 U_1 ×_2 U_2 ... ×_N U_N（コアは X からそのまま縮約）。
    各 U_n は M_n = unfold(X, n) の RPCA による低ランク成分 L_n の左特異ベクトル。
    sparse_components は診断用で、再構成には使わない。

    Args:
        X: 入力テンソル。
        ranks: 各モードの target rank。
        rpca_rank: RPCA 内部の randomized SVD のランク。既定は max(ranks)*2。
        lam: スパース項の重み。None なら 1/sqrt(max(m,n))。
        max_iter: RPCA の最大反復数。
        tol: RPCA の収束閾値（相対残差）。
        random_state: 乱数シード。

    Returns:
        (G, factors, sparse_components) のタプル。
        sparse_components[mode] は mode 展開形のスパース成分 S_n。

    Raises:
        ValueError: ranks の長さが X.ndim と一致しない場合。
    """
    if len(ranks) != X.ndim:
        raise ValueError(
            f"ranks の長さ ({len(ranks)}) はテンソルの次元数 ({X.ndim}) "
            "と一致する必要があります。"
        )

    if rpca_rank is None:
        rpca_rank = max(max(ranks) * 2, 10)

    factors: list[np.ndarray] = []
    sparse_components: list[np.ndarray] = []

    for mode, rank in enumerate(ranks):
        M = unfold(X, mode)
        eff_rpca_rank = min(rpca_rank, min(M.shape) - 1)
        L_mode, S_mode = randomized_rpca(
            M,
            lam=lam,
            rank=eff_rpca_rank,
            max_iter=max_iter,
            tol=tol,
            random_state=random_state,
        )
        sparse_components.append(S_mode)

        eff_rank = min(rank, min(L_mode.shape))
        U, _, _ = randomized_svd(
            L_mode, n_components=eff_rank, random_state=random_state
        )
        factors.append(U)

    G = X.copy()
    for mode, U in enumerate(factors):
        G = mode_n_product(G, U.T, mode)

    return G, factors, sparse_components
