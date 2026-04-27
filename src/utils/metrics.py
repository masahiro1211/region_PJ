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


def l2_error(
    V_numeric: np.ndarray,
    V_analytic: np.ndarray,
    dx: float,
    interior_only: bool = True,
) -> float:
    """格子上の L2 ノルム相対誤差 ||V_num - V_ana||_2 / ||V_ana||_2 を返す。

    L2 ノルムは sqrt(Σ f^2 dx^d) で計算する。

    Args:
        V_numeric: 数値解配列（n 次元）。
        V_analytic: 解析解配列（同じ形状）。
        dx: 格子間隔。
        interior_only: True なら境界 1 層を除外して評価する。

    Returns:
        相対 L2 誤差。
    """
    if interior_only:
        inn = (slice(1, -1),) * V_numeric.ndim
        V_numeric = V_numeric[inn]
        V_analytic = V_analytic[inn]

    d = V_numeric.ndim
    diff = V_numeric - V_analytic
    numerator = np.sqrt(np.sum(diff**2) * dx**d)
    denominator = np.sqrt(np.sum(V_analytic**2) * dx**d)
    return float(numerator / denominator)
