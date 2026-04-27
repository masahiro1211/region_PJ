import numpy as np
from typing import Callable


def compute_monopole_bc(rho: np.ndarray, dx: float) -> np.ndarray:
    """0 次多極子展開（モノポール近似）で境界ポテンシャルを返す。

    全電荷 Q = Σ ρ dx^3 を原点に集中させた点電荷とみなして
    各境界格子点に V = Q / r を割り当てる。

    Args:
        rho: 形状 (N, N, N) の電荷密度グリッド。
        dx: 格子間隔。

    Returns:
        境界点のみ非ゼロな (N, N, N) 配列。
    """
    N = rho.shape[0]
    Q = np.sum(rho) * dx**3

    coords = (np.arange(N) - (N - 1) / 2.0) * dx
    x, y, z = np.meshgrid(coords, coords, coords, indexing="ij")

    r = np.sqrt(x**2 + y**2 + z**2)
    r = np.where(r < 1e-10, 1e-10, r)

    V_mono = Q / r

    V_bc = np.zeros((N, N, N))
    V_bc[0,  :,  :] = V_mono[0,  :,  :]
    V_bc[-1, :,  :] = V_mono[-1, :,  :]
    V_bc[:,  0,  :] = V_mono[:,  0,  :]
    V_bc[:, -1,  :] = V_mono[:, -1,  :]
    V_bc[:,  :,  0] = V_mono[:,  :,  0]
    V_bc[:,  :, -1] = V_mono[:,  :, -1]
    return V_bc


def laplacian_matvec(V: np.ndarray, dx: float) -> np.ndarray:
    """3D の (-∇²) を中央差分で内部点に適用する。

    Args:
        V: 形状 (N, N, N) のスカラー場（境界含む）。
        dx: 格子間隔。

    Returns:
        (-∇² V) の結果。境界点はゼロのまま。
    """
    AV = np.zeros_like(V)
    inn = slice(1, -1)

    AV[inn, inn, inn] = 6.0 * V[inn, inn, inn]
    AV[inn, inn, inn] -= V[0:-2, inn, inn]
    AV[inn, inn, inn] -= V[2:,   inn, inn]
    AV[inn, inn, inn] -= V[inn, 0:-2, inn]
    AV[inn, inn, inn] -= V[inn, 2:,   inn]
    AV[inn, inn, inn] -= V[inn, inn, 0:-2]
    AV[inn, inn, inn] -= V[inn, inn, 2:  ]

    return AV / dx**2


def rhs_with_bc(rho: np.ndarray, V_bc: np.ndarray, dx: float) -> np.ndarray:
    """境界値の寄与を移項した CG 用の右辺ベクトルを構築する。

    解くべき方程式は (-∇²) V_int = 4π ρ - (-∇²) V_bc。

    Args:
        rho: 形状 (N, N, N) の電荷密度。
        V_bc: 境界のみ非ゼロな配列。
        dx: 格子間隔。

    Returns:
        CG に渡す右辺ベクトル。内部点のみ意味を持つ。
    """
    bc_contribution = laplacian_matvec(V_bc, dx)
    b = np.zeros_like(rho)
    inn = slice(1, -1)
    b[inn, inn, inn] = (
        4 * np.pi * rho[inn, inn, inn] - bc_contribution[inn, inn, inn]
    )
    return b


def cg_solve(
    matvec: Callable[[np.ndarray], np.ndarray],
    b: np.ndarray,
    x0: np.ndarray | None = None,
    tol: float = 1e-6,
    max_iter: int = 1000,
) -> tuple[np.ndarray, list[float]]:
    """共役勾配法で対称正定値系 A x = b を解く。

    Args:
        matvec: A x を返す関数。
        b: 右辺ベクトル。
        x0: 初期推定解。None なら零ベクトル。
        tol: 相対残差の収束閾値。
        max_iter: 最大反復回数。

    Returns:
        近似解 x と各反復の相対残差リスト。
    """
    x = np.zeros_like(b) if x0 is None else x0.copy()
    r = b - matvec(x)
    p = r.copy()

    b_norm = np.sqrt(np.sum(b * b))
    if b_norm < 1e-14:
        return x, [0.0]

    residuals: list[float] = []
    for _ in range(max_iter):
        r_norm = np.sqrt(np.sum(r * r))
        residuals.append(r_norm / b_norm)
        if r_norm / b_norm < tol:
            break

        Ap = matvec(p)
        rTr = np.sum(r * r)
        alpha = rTr / np.sum(p * Ap)
        x = x + alpha * p
        r_new = r - alpha * Ap
        beta = np.sum(r_new * r_new) / rTr
        p = r_new + beta * p
        r = r_new

    return x, residuals


def poisson_solve(
    rho: np.ndarray,
    dx: float,
    tol: float = 1e-8,
    max_iter: int = 1000,
) -> tuple[np.ndarray, list[float]]:
    """ポアソン方程式 -∇² V = 4π ρ を CG で解く。

    境界条件はモノポール近似、離散化は中央差分（2 次精度）。

    Args:
        rho: 形状 (N, N, N) の電荷密度。
        dx: 格子間隔。
        tol: CG の収束判定閾値。
        max_iter: CG の最大反復回数。

    Returns:
        境界含む完全解 V と CG の収束履歴のタプル。
    """
    V_bc = compute_monopole_bc(rho, dx)
    b = rhs_with_bc(rho, V_bc, dx)

    def matvec(V: np.ndarray) -> np.ndarray:
        return laplacian_matvec(V, dx)

    V_int, residuals = cg_solve(matvec, b, tol=tol, max_iter=max_iter)
    return V_int + V_bc, residuals
