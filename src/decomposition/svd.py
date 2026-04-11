import numpy as np
from scipy import linalg


def perform_svd(X: np.ndarray, rank: int):
    """行列の低ランク SVD 近似因子を返す。

    返り値は X ≈ A @ B を満たす因子であり、
    A = U_r @ sqrt(Sigma_r), B = sqrt(Sigma_r) @ V_r として構成する。

    Args:
        X: 入力行列。形状は (m, n)。
        rank: 近似ランク。

    Returns:
        A, B のタプル。
        A の形状は (m, rank)、B の形状は (rank, n)。
    """
    U, s, V = linalg.svd(X, full_matrices=False)
    Ur = U[:, :rank]
    Sr = np.diag(np.sqrt(s[:rank]))
    Vr = V[:rank, :]
    A = Ur @ Sr
    B = Sr @ Vr
    return A, B
