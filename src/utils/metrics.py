import numpy as np


def relative_error(X_true: np.ndarray, X_approx: np.ndarray) -> float:
    """相対近似誤差 ||X - X_approx|| / ||X|| を返す。"""
    return float(np.linalg.norm(X_true - X_approx) / np.linalg.norm(X_true))
