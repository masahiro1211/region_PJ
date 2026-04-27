import numpy as np
from sklearn.utils.extmath import randomized_svd


def rpca(
    M: np.ndarray,
    lam: float | None = None,
    max_iter: int = 1000,
    tol: float = 1e-6,
    verbose: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Robust PCA (ALM) で M = L + S に分解する。

    L は低ランク（核ノルム最小化）、S はスパース（L1 ノルム最小化）。

    Args:
        M: 入力行列。
        lam: スパース項の重み。None なら 1/sqrt(max(m,n))。
        max_iter: 最大反復回数。
        tol: 収束判定の相対残差。
        verbose: 収束時にログを出すかどうか。

    Returns:
        (L, S) のタプル。
    """
    m, n = M.shape
    if lam is None:
        lam = 1.0 / np.sqrt(max(m, n))

    L = np.zeros_like(M)
    S = np.zeros_like(M)
    Y = np.zeros_like(M)
    norm2 = np.linalg.norm(M, ord=2)
    mu = 1.25 / norm2
    mu_inv = 1.0 / mu
    rho = 1.5

    for i in range(max_iter):
        U, s, Vt = np.linalg.svd(M - S + mu_inv * Y, full_matrices=False)
        L = (U * np.maximum(s - mu_inv, 0)) @ Vt

        Z = M - L + mu_inv * Y
        S = np.sign(Z) * np.maximum(np.abs(Z) - lam * mu_inv, 0)

        residual = M - L - S
        Y += mu * residual

        err = np.linalg.norm(residual) / np.linalg.norm(M)
        if err < tol:
            if verbose:
                print(f"converged: iter={i+1}, err={err:.2e}")
            break

        mu = min(rho * mu, mu * 1e7)
        mu_inv = 1.0 / mu

    return L, S


def randomized_rpca(
    M: np.ndarray,
    lam: float | None = None,
    rank: int = 130,
    max_iter: int = 1000,
    tol: float = 1e-6,
    verbose: bool = False,
    random_state: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """各反復で randomized SVD を使う高速版 RPCA。

    Args:
        M: 入力行列。
        lam: スパース項の重み。None なら 1/sqrt(max(m,n))。
        rank: 各反復で使う target rank。
        max_iter: 最大反復回数。
        tol: 収束判定の相対残差。
        verbose: 収束時にログを出すかどうか。
        random_state: randomized SVD の乱数シード。

    Returns:
        (L, S) のタプル。
    """
    m, n = M.shape
    if lam is None:
        lam = 1.0 / np.sqrt(max(m, n))

    L = np.zeros_like(M)
    S = np.zeros_like(M)
    Y = np.zeros_like(M)
    mu = 1.25 / np.linalg.norm(M, ord=2)
    mu_inv = 1.0 / mu
    rho = 1.5

    for i in range(max_iter):
        U, s, Vt = randomized_svd(
            M - S + mu_inv * Y,
            n_components=rank,
            random_state=random_state,
        )
        L = (U * np.maximum(s - mu_inv, 0)) @ Vt

        Z = M - L + mu_inv * Y
        S = np.sign(Z) * np.maximum(np.abs(Z) - lam * mu_inv, 0)

        residual = M - L - S
        Y += mu * residual

        err = np.linalg.norm(residual) / np.linalg.norm(M)
        if err < tol:
            if verbose:
                print(f"converged: iter={i+1}, err={err:.2e}")
            break

        mu = min(rho * mu, mu * 1e7)
        mu_inv = 1.0 / mu

    return L, S
