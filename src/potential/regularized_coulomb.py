"""ガウシアンポテンシャル計算で使う正則化 Coulomb カーネル。"""

from __future__ import annotations

import numpy as np


def gaussian_density_1d(
    coords: np.ndarray,
    alpha: float,
    center: float = 0.0,
    weight: float = 1.0,
) -> np.ndarray:
    """1D グリッド上の Gaussian 密度を返す。

    Parameters
    ----------
    coords : np.ndarray, shape (N,)
        1D グリッド点の座標配列。
    alpha : float
        Gaussian の幅パラメータ。
    center : float, default=0.0
        Gaussian 中心の座標。
    weight : float, default=1.0
        Gaussian 密度に掛ける係数。

    Returns
    -------
    density : np.ndarray, shape (N,)
        ``weight * exp(-alpha * (x - center)^2)`` で定義される密度。
    """
    coords_arr = np.asarray(coords, dtype=float)
    return weight * np.exp(-alpha * (coords_arr - center) ** 2)


def build_regularized_coulomb_kernel_1d(
    coords: np.ndarray,
    eps: float,
) -> np.ndarray:
    """1D 正則化 Coulomb カーネル行列を構築する。

    Parameters
    ----------
    coords : np.ndarray, shape (N,)
        1D グリッド点の座標配列。
    eps : float
        Coulomb 特異点を正則化する正のパラメータ。

    Returns
    -------
    kernel : np.ndarray, shape (N, N)
        ``K[i, j] = 1 / sqrt((x_i - x_j)^2 + eps^2)``。

    Raises
    ------
    ValueError
        ``eps <= 0`` の場合。
    """
    if eps <= 0:
        raise ValueError("eps must be positive.")

    coords_arr = np.asarray(coords, dtype=float)
    diff = coords_arr[:, None] - coords_arr[None, :]
    return 1.0 / np.sqrt(diff**2 + eps**2)


def flatten_xyz_grid(xyz: np.ndarray) -> np.ndarray:
    """3D 座標グリッドを点列へ変換する。

    Parameters
    ----------
    xyz : np.ndarray, shape (3, N, N, N)
        ``xyz[0]`` が x 座標、``xyz[1]`` が y 座標、
        ``xyz[2]`` が z 座標である座標グリッド。

    Returns
    -------
    points : np.ndarray, shape (N^3, 3)
        各行が1つの3Dグリッド点 ``(x, y, z)`` を表す配列。

    Raises
    ------
    ValueError
        ``xyz`` の形状が ``(3, N, N, N)`` でない場合。
    """
    xyz_arr = np.asarray(xyz, dtype=float)
    if xyz_arr.ndim != 4 or xyz_arr.shape[0] != 3:
        raise ValueError(
            "xyz must have shape (3, N, N, N); "
            f"got shape={xyz_arr.shape}."
        )

    return np.stack(
        [xyz_arr[0].ravel(), xyz_arr[1].ravel(), xyz_arr[2].ravel()],
        axis=1,
    )


def pairwise_distance_matrix(points: np.ndarray) -> np.ndarray:
    """点列からユークリッド距離行列を計算する。

    Parameters
    ----------
    points : np.ndarray, shape (M, d)
        M 個の d 次元点を並べた配列。

    Returns
    -------
    distances : np.ndarray, shape (M, M)
        ``distances[i, j]`` が点 i と点 j の距離である行列。

    Raises
    ------
    ValueError
        ``points`` が2次元配列でない場合。
    """
    points_arr = np.asarray(points, dtype=float)
    if points_arr.ndim != 2:
        raise ValueError(
            "points must be a 2D array with shape (n_points, dim); "
            f"got shape={points_arr.shape}."
        )

    diff = points_arr[:, None, :] - points_arr[None, :, :]
    return np.sqrt(np.sum(diff**2, axis=-1))


def build_regularized_coulomb_kernel_from_distances(
    distances: np.ndarray,
    eps: float,
) -> np.ndarray:
    """距離行列から正則化 Coulomb カーネル行列を構築する。

    Parameters
    ----------
    distances : np.ndarray, shape (M, M)
        点間距離の行列。
    eps : float
        Coulomb 特異点を正則化する正のパラメータ。

    Returns
    -------
    kernel : np.ndarray, shape (M, M)
        ``1 / sqrt(distances^2 + eps^2)`` で定義される行列。

    Raises
    ------
    ValueError
        ``eps <= 0`` の場合。
    """
    if eps <= 0:
        raise ValueError("eps must be positive.")

    distances_arr = np.asarray(distances, dtype=float)
    return 1.0 / np.sqrt(distances_arr**2 + eps**2)


def build_regularized_coulomb_kernel_3d(
    points: np.ndarray,
    eps: float,
) -> np.ndarray:
    """3D 点列に対する正則化 Coulomb カーネル行列を構築する。

    Parameters
    ----------
    points : np.ndarray, shape (M, 3)
        3D グリッド点を並べた配列。
    eps : float
        Coulomb 特異点を正則化する正のパラメータ。

    Returns
    -------
    kernel : np.ndarray, shape (M, M)
        点列に対する正則化 Coulomb カーネル行列。
    """
    distances = pairwise_distance_matrix(points)
    return build_regularized_coulomb_kernel_from_distances(distances, eps)


def smoothstep_window(distances: np.ndarray, threshold: float) -> np.ndarray:
    """C1 連続な近距離窓関数を返す。

    Parameters
    ----------
    distances : np.ndarray
        距離を格納した配列。
    threshold : float
        近距離成分と遠距離成分を切り替える正の距離。

    Returns
    -------
    weights : np.ndarray
        ``distances`` と同じ形状の窓関数値。距離0で1、
        ``threshold`` 以上で0になる。

    Raises
    ------
    ValueError
        ``threshold <= 0`` の場合。
    """
    if threshold <= 0:
        raise ValueError("threshold must be positive.")

    t = np.clip(np.abs(distances) / threshold, 0.0, 1.0)
    return 1.0 - 3.0 * t**2 + 2.0 * t**3


def split_kernel_near_far(
    kernel: np.ndarray,
    distances: np.ndarray,
    threshold: float,
    smooth: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """カーネル行列を近距離成分と遠距離成分に分割する。

    Parameters
    ----------
    kernel : np.ndarray, shape (M, M)
        分割対象のカーネル行列。
    distances : np.ndarray, shape (M, M)
        ``kernel`` と同じ形状の距離行列。
    threshold : float
        近距離成分の半径。
    smooth : bool, default=False
        True の場合は smoothstep 窓を使う。False の場合は hard mask を使う。

    Returns
    -------
    near : np.ndarray, shape (M, M)
        近距離成分のカーネル行列。
    far : np.ndarray, shape (M, M)
        遠距離成分のカーネル行列。
    weights : np.ndarray, shape (M, M)
        近距離成分に掛けた重み。

    Raises
    ------
    ValueError
        ``kernel`` と ``distances`` の形状が異なる場合、または
        ``threshold <= 0`` の場合。

    Notes
    -----
    ``near + far`` は丸め誤差を除いて元の ``kernel`` に一致する。
    """
    kernel_arr = np.asarray(kernel, dtype=float)
    distances_arr = np.asarray(distances, dtype=float)
    if kernel_arr.shape != distances_arr.shape:
        raise ValueError(
            "kernel and distances must have the same shape; "
            f"got {kernel_arr.shape} and {distances_arr.shape}."
        )

    if smooth:
        weights = smoothstep_window(distances_arr, threshold)
    else:
        if threshold <= 0:
            raise ValueError("threshold must be positive.")
        weights = (distances_arr <= threshold).astype(float)

    near = kernel_arr * weights
    far = kernel_arr * (1.0 - weights)
    return near, far, weights
