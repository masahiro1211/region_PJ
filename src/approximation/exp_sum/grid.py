"""Sampling grids for exponential-sum fitting."""

from __future__ import annotations

import numpy as np


class LogUniformGrid:
    """[r_min, r_max] 上の log-uniform サンプルグリッド."""

    def __init__(self, r_min: float, r_max: float, n_points: int) -> None:
        if r_min <= 0:
            raise ValueError(
                "r_min must be positive (avoids 1/r singularity)."
            )
        if r_max <= r_min:
            raise ValueError("r_max must be greater than r_min.")
        self.r_min = r_min
        self.r_max = r_max
        self.n_points = n_points

    @property
    def points(self) -> np.ndarray:
        return np.logspace(
            np.log10(self.r_min),
            np.log10(self.r_max),
            self.n_points,
        )

    def alpha_range(self) -> tuple[float, float]:
        """1/r の積分表現から導かれる有効 s の範囲を返す."""
        return 1.0 / self.r_max**2, 1.0 / self.r_min**2
