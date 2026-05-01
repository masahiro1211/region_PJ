import numpy as np


def relative_error(
    X_true: np.ndarray,
    X_approx: np.ndarray,
    interior_only: bool = False,
) -> float:
    """相対 L2 誤差 ||X_true - X_approx||_2 / ||X_true||_2 を計算する。

    Args:
        X_true: 参照配列。
        X_approx: 近似配列。
        interior_only: True なら境界 1 層を除外して評価する。

    Returns:
        相対誤差を Python float で返す。
    """
    if interior_only:
        inn = (slice(1, -1),) * X_true.ndim
        X_true = X_true[inn]
        X_approx = X_approx[inn]

    return float(np.linalg.norm(X_true - X_approx) / np.linalg.norm(X_true))


def hartree_energy(
    rho: np.ndarray,
    V: np.ndarray,
    dx: float,
    dim: int | None = None,
) -> float:
    """Hartree エネルギー E = (1/2) ∫ ρ V d³r をミッドポイント則で評価する。

    Args:
        rho: 電荷密度。
        V: ポテンシャル（rho と同形状）。
        dx: 格子間隔。
        dim: 体積要素 dx^dim の次数。未指定時は rho.ndim を使う。
            flatten した配列を渡す場合は明示する必要がある。

    Returns:
        スカラーエネルギー。
    """
    if dim is None:
        dim = rho.ndim
    return 0.5 * float(np.sum(rho * V)) * dx ** dim


def v_e_errors(
    V_approx: np.ndarray,
    V_ref: np.ndarray,
    rho: np.ndarray,
    dx: float,
    dim: int | None = None,
) -> tuple[float, float]:
    """V と Hartree エネルギーの相対誤差をまとめて返す。

    Args:
        V_approx: 近似ポテンシャル。
        V_ref: 参照ポテンシャル。
        rho: 電荷密度（V とは同じ離散化を仮定）。
        dx: 格子間隔。
        dim: エネルギー計算に使う体積要素 dx^dim の次数。
            未指定時は rho.ndim を使う。

    Returns:
        (V 相対誤差, E 相対誤差)。
    """
    err_v = relative_error(V_ref, V_approx)
    E_ref = hartree_energy(rho, V_ref, dx, dim=dim)
    E_approx = hartree_energy(rho, V_approx, dx, dim=dim)
    err_e = abs(E_approx - E_ref) / abs(E_ref)
    return err_v, err_e
