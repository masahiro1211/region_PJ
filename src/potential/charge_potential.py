import numpy as np
from scipy.special import erf
from typing import TypedDict

from src.decomposition.tucker import perform_tucker, reconstruct
from src.utils.metrics import relative_error


class TuckerRow(TypedDict):
    rank: int
    err_rho: float
    err_v: float
    compression: float


class ChargePotentialResult(TypedDict):
    alpha: float
    N: int
    L: float
    dx: float
    baseline_shift: float
    baseline_error: float
    ref_error: float
    rows: list[TuckerRow]


def create_spatial_grid(
    N: int,
    L: float,
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """空間グリッドと距離グリッドを生成する。

    Args:
        N: 各軸のグリッド点数。
        L: 計算領域の一辺の長さ。

    Returns:
        dx, X, Y, Z, R のタプル。
        dx は格子幅、X/Y/Z は座標グリッド、R は原点からの距離グリッド。
    """
    dx = L / N
    x = np.fft.fftfreq(N, d=1.0 / L)
    X, Y, Z = np.meshgrid(x, x, x, indexing="ij")
    R = np.sqrt(X**2 + Y**2 + Z**2)
    return dx, X, Y, Z, R


def create_k2_grid(N: int, dx: float) -> np.ndarray:
    """ポアソン方程式の FFT 解法に使う k^2 グリッドを生成する。

    Args:
        N: 各軸のグリッド点数。
        dx: 空間格子幅。

    Returns:
        波数ベクトルの二乗和 K2。
        原点成分 K2[0, 0, 0] は 1.0 に設定される。
    """
    kx = 2 * np.pi * np.fft.fftfreq(N, d=dx)
    KX, KY, KZ = np.meshgrid(kx, kx, kx, indexing="ij")
    K2 = KX**2 + KY**2 + KZ**2
    K2[0, 0, 0] = 1.0
    return K2


def compute_potential_fft(rho_grid: np.ndarray, K2: np.ndarray) -> np.ndarray:
    """FFT により電荷密度からポテンシャルを計算する。

    Args:
        rho_grid: 実空間の電荷密度グリッド。
        K2: 波数空間の k^2 グリッド。

    Returns:
        実空間ポテンシャル V。

    Notes:
        k=0 成分はゲージ自由度として 0 に固定する。
    """
    V_k = 4 * np.pi * np.fft.fftn(rho_grid) / K2
    V_k[0, 0, 0] = 0.0
    return np.real(np.fft.ifftn(V_k))


def v_analytic_gaussian(R: np.ndarray, alpha: float) -> np.ndarray:
    """原点中心の単一ガウシアン電荷の解析ポテンシャルを返す。

    Args:
        R: 原点からの距離グリッド。
        alpha: ガウシアン幅パラメータ。

    Returns:
        解析ポテンシャルグリッド。
    """
    V = np.empty_like(R, dtype=float)
    mask = R > 1e-10
    numerator = (np.pi / alpha) ** 1.5 * erf(np.sqrt(alpha) * R)
    np.divide(numerator, R, out=V, where=mask)
    V[~mask] = 2 * np.pi / alpha
    return V


def v_analytic_multi(
    X: np.ndarray,
    Y: np.ndarray,
    Z: np.ndarray,
    centers: np.ndarray,
    weights: np.ndarray,
    alpha: float,
) -> np.ndarray:
    """複数ガウシアン電荷の解析ポテンシャルを線形重ね合わせで計算する。

    Args:
        X: x 座標グリッド。
        Y: y 座標グリッド。
        Z: z 座標グリッド。
        centers: 各ガウシアン中心の配列。
        weights: 各ガウシアンの重み配列。
        alpha: ガウシアン幅パラメータ。

    Returns:
        解析ポテンシャルグリッド。
    """
    V = np.zeros_like(X)
    for c, w in zip(centers, weights):
        R_c = np.sqrt((X - c[0])**2 + (Y - c[1])**2 + (Z - c[2])**2)
        V += w * v_analytic_gaussian(R_c, alpha)
    return V


def build_gaussian_density(
    X: np.ndarray,
    Y: np.ndarray,
    Z: np.ndarray,
    centers: np.ndarray,
    weights: np.ndarray,
    alpha: float,
) -> np.ndarray:
    """ガウシアン混合モデルの電荷密度グリッドを構築する。

    Args:
        X: x 座標グリッド。
        Y: y 座標グリッド。
        Z: z 座標グリッド。
        centers: 各ガウシアン中心の配列。
        weights: 各ガウシアンの重み配列。
        alpha: ガウシアン幅パラメータ。

    Returns:
        電荷密度グリッド。
    """
    rho = np.zeros_like(X)
    for c, w in zip(centers, weights):
        R2_c = (X - c[0])**2 + (Y - c[1])**2 + (Z - c[2])**2
        rho += w * np.exp(-alpha * R2_c)
    return rho


def constant_shift(
    reference: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray,
) -> float:
    """マスク領域で最小二乗となる定数シフトを計算する。

    Args:
        reference: 基準となる配列。
        target: シフト対象の配列。
        mask: 評価に使うブールマスク。

    Returns:
        reference と target の差の平均値。
    """
    return float(np.mean(reference[mask] - target[mask]))


def evaluate_tucker_rows(
    rho: np.ndarray,
    V_exact: np.ndarray,
    K2: np.ndarray,
    mask: np.ndarray,
    ranks: list[int],
) -> list[TuckerRow]:
    """各 Tucker ランクで近似誤差と圧縮率を評価する。

    Args:
        rho: 元の電荷密度グリッド。
        V_exact: 解析ポテンシャルグリッド。
        K2: 波数空間の k^2 グリッド。
        mask: 誤差評価に使うブールマスク。
        ranks: 評価する Tucker ランク一覧。

    Returns:
        ランクごとの指標を保持した辞書のリスト。
    """
    rows: list[TuckerRow] = []
    rho_size = rho.size

    for rank in ranks:
        G, factors = perform_tucker(rho, [rank] * 3)
        rho_approx = reconstruct(G, factors)
        V_approx = compute_potential_fft(rho_approx, K2)

        err_rho = relative_error(rho, rho_approx)
        c_approx = constant_shift(V_exact, V_approx, mask)
        err_v = relative_error(V_exact[mask], V_approx[mask] + c_approx)

        compressed_size = G.size + sum(f.size for f in factors)
        compression = compressed_size / rho_size

        rows.append(
            {
                "rank": rank,
                "err_rho": err_rho,
                "err_v": err_v,
                "compression": compression,
            }
        )

    return rows


def run_charge_potential_demo(
    alpha: float = 1.0,
    N: int = 32,
    L: float = 10.0,
    centers: np.ndarray | None = None,
    weights: np.ndarray | None = None,
    ranks: list[int] | None = None,
    mask_radius_ratio: float = 1.0 / 3.0,
) -> ChargePotentialResult:
    """電荷ポテンシャルのベンチマーク計算を実行して指標を返す。

    Args:
        alpha: ガウシアン幅パラメータ。
        N: 各軸のグリッド点数。
        L: 計算領域の一辺の長さ。
        centers: ガウシアン中心配列。未指定時は既定値を使用。
        weights: ガウシアン重み配列。未指定時は既定値を使用。
        ranks: 評価する Tucker ランク一覧。未指定時は既定値を使用。
        mask_radius_ratio: 誤差評価半径を L に対する比で指定する。

    Returns:
        ベースライン誤差、参照誤差、ランク別指標を含む結果辞書。
    """
    if centers is None:
        centers = np.array(
            [
                [0.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
                [0.0, 2.0, 0.0],
                [-1.5, -1.5, 0.0],
            ]
        )
    if weights is None:
        weights = np.array([1.0, 0.6, 0.5, 0.4])
    if ranks is None:
        ranks = [1, 2, 4, 8, 12, 16]

    dx, X, Y, Z, R = create_spatial_grid(N, L)
    K2 = create_k2_grid(N, dx)

    rho = build_gaussian_density(X, Y, Z, centers, weights, alpha)
    V_ref = compute_potential_fft(rho, K2)
    V_exact = v_analytic_multi(X, Y, Z, centers, weights, alpha)

    V_single_exact = v_analytic_gaussian(R, alpha)
    V_single_numerical = compute_potential_fft(np.exp(-alpha * R**2), K2)

    mask = R < L * mask_radius_ratio

    baseline_shift = constant_shift(V_single_exact, V_single_numerical, mask)
    baseline_error = relative_error(
        V_single_exact[mask],
        V_single_numerical[mask] + baseline_shift,
    )

    ref_shift = constant_shift(V_exact, V_ref, mask)
    ref_error = relative_error(V_exact[mask], V_ref[mask] + ref_shift)

    rows = evaluate_tucker_rows(rho, V_exact, K2, mask, ranks)

    return {
        "alpha": alpha,
        "N": N,
        "L": L,
        "dx": dx,
        "baseline_shift": baseline_shift,
        "baseline_error": baseline_error,
        "ref_error": ref_error,
        "rows": rows,
    }
