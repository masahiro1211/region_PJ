import numpy as np


def relative_error(X_true: np.ndarray, X_approx: np.ndarray) -> float:
    """相対近似誤差を計算する。

    定義は ||X_true - X_approx|| / ||X_true|| である。

    Args:
        X_true: 参照配列。
        X_approx: 近似配列。

    Returns:
        相対誤差を Python float で返す。
    """
    return float(np.linalg.norm(X_true - X_approx) / np.linalg.norm(X_true))
