import numpy as np


def unfold(X: np.ndarray, mode: int) -> np.ndarray:
    """テンソル X を mode-n 展開（行列化）する。"""
    return np.reshape(np.moveaxis(X, mode, 0), (X.shape[mode], -1))


def mode_n_product(X: np.ndarray, M: np.ndarray, mode: int) -> np.ndarray:
    """テンソル X と行列 M の mode-n 積を計算する。

    X: shape (..., I_n, ...), M: shape (J, I_n)
    戻り値: shape (..., J, ...)
    """
    result = np.tensordot(M, X, axes=((1,), (mode,)))
    # tensordot は新軸を先頭に置くので mode の位置に戻す
    return np.moveaxis(result, 0, mode)
