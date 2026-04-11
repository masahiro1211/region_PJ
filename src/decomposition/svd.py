import numpy as np
from scipy import linalg


def perform_svd(X: np.ndarray, rank: int):
    """ランク rank の低ランク SVD 分解を行う。

    X ≈ A @ B となる A, B を返す。
    A = U_r @ sqrt(Sigma_r),  B = sqrt(Sigma_r) @ V_r

    Parameters
    ----------
    X : ndarray, shape (m, n)
    rank : int

    Returns
    -------
    A : ndarray, shape (m, rank)
    B : ndarray, shape (rank, n)
    """
    U, s, V = linalg.svd(X, full_matrices=False)
    Ur = U[:, :rank]
    Sr = np.diag(np.sqrt(s[:rank]))
    Vr = V[:rank, :]
    A = Ur @ Sr
    B = Sr @ Vr
    return A, B
