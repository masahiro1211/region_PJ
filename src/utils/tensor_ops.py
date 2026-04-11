import numpy as np


def unfold(X: np.ndarray, mode: int) -> np.ndarray:
    """テンソルを mode-n 展開して行列化する。

    Args:
        X: 入力テンソル。
        mode: 展開対象モード。

    Returns:
        形状 (X.shape[mode], -1) の行列。
    """
    return np.reshape(np.moveaxis(X, mode, 0), (X.shape[mode], -1))


def mode_n_product(X: np.ndarray, M: np.ndarray, mode: int) -> np.ndarray:
    """テンソル X と行列 M の mode-n 積を計算する。

    Args:
        X: 入力テンソル。対象モードの次元は I_n。
        M: 左から掛ける行列。形状は (J, I_n)。
        mode: 積を適用するモード。

    Returns:
        形状 (..., J, ...) のテンソル。
    """
    result = np.tensordot(M, X, axes=((1,), (mode,)))
    # tensordot は新軸を先頭に置くので mode の位置に戻す
    return np.moveaxis(result, 0, mode)
