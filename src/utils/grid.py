import numpy as np


def build_xyz(N: int, L: float) -> np.ndarray:
    """中心対称な座標グリッドを (3, N, N, N) で返す。

    N は奇数であること。原点を格子点に乗せるため、
    呼び出し側は N = 2n + 1 の形で渡す。

    Args:
        N: 各軸の格子点数（奇数）。
        L: 計算領域の一辺の長さ。

    Returns:
        座標グリッド。xyz[0] が x, xyz[1] が y, xyz[2] が z。

    Raises:
        ValueError: N が偶数の場合。
    """
    if N % 2 == 0:
        raise ValueError(
            f"N は奇数である必要があります（N={N}）。2*n+1 の形で渡してください。"
        )

    dx = L / N
    coords = (np.arange(N) - (N - 1) // 2) * dx
    x, y, z = np.meshgrid(coords, coords, coords, indexing="ij")
    return np.array([x, y, z])


def build_coords_centered(N: int, L: float) -> tuple[float, np.ndarray]:
    """中心対称な 1D 座標配列と格子間隔を返す。

    Args:
        N: 格子点数。
        L: 領域長。

    Returns:
        dx と coords のタプル。coords[i] = (i - (N-1)/2) * dx。
    """
    dx = L / N
    coords = (np.arange(N) - (N - 1) / 2.0) * dx
    return dx, coords
