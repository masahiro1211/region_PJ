"""VARPRO による 1/r ≈ Σ_k w_k exp(-α_k r²) の指数和近似.

次元分離可能性 (dimension separability):
    exp(-α (x1-x2)² - α (y1-y2)² - α (z1-z2)²)
    = exp(-α (x1-x2)²) · exp(-α (y1-y2)²) · exp(-α (z1-z2)²)

これにより 3D Coulomb 積分を Tucker ベースで O(R · r_rank³ · N log N) で
扱うことが可能になる。

Optimization strategy (VARPRO):
    Outer loop : L-BFGS-B on log(α_k)       — nonlinear, R variables
    Inner loop : NNLS on w_k given α_k       — linear, closed-form

References:
    Hackbusch (2019), "Tucker Approximation of Operators"
    Beylkin & Monzón (2005), "On approximation of functions by exponential sums"
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
from scipy.optimize import minimize, nnls


@dataclass
class ExponentialSum:
    """指数和近似 f(r) ≈ Σ_k w_k exp(-α_k r²) のパラメータ."""

    weights: np.ndarray    # w_k,  shape (R,)
    alphas: np.ndarray     # α_k,  shape (R,), 昇順ソート
    l2_error: float = 0.0
    linf_error: float = 0.0

    @property
    def rank(self) -> int:
        return len(self.weights)

    def evaluate(self, r: np.ndarray) -> np.ndarray:
        """指定された r で近似を評価する."""
        A = np.exp(-np.outer(r ** 2, self.alphas))
        return A @ self.weights

    def to_dict(self) -> dict:
        return {
            "rank": self.rank,
            "weights": self.weights.tolist(),
            "alphas": self.alphas.tolist(),
            "L2_error": self.l2_error,
            "Linf_error": self.linf_error,
        }


class LogUniformGrid:
    """[r_min, r_max] 上の log-uniform サンプルグリッド."""

    def __init__(self, r_min: float, r_max: float, n_points: int) -> None:
        if r_min <= 0:
            raise ValueError("r_min must be positive (avoids 1/r singularity).")
        if r_max <= r_min:
            raise ValueError("r_max must be greater than r_min.")
        self.r_min = r_min
        self.r_max = r_max
        self.n_points = n_points

    @property
    def points(self) -> np.ndarray:
        return np.logspace(np.log10(self.r_min), np.log10(self.r_max), self.n_points)

    def alpha_range(self) -> tuple[float, float]:
        """1/r の積分表現 1/r = (1/√π) ∫ s^{-1/2} exp(-r² s) ds から
        導かれる有効 s の範囲 [1/r_max², 1/r_min²] を返す."""
        return 1.0 / self.r_max ** 2, 1.0 / self.r_min ** 2


class VarproOptimizer:
    """1/r ≈ Σ w_k exp(-α_k r²) の VARPRO 最適化器.

    Args:
        fit_grid: 最適化に用いるグリッド (典型的には 200 点).
        eval_grid: 誤差評価用の高密度グリッド (典型的には 2000 点).
        nonneg: True なら NNLS で w_k ≥ 0 を強制する (推奨).
        max_iter: L-BFGS-B の最大反復回数.
    """

    def __init__(
        self,
        fit_grid: LogUniformGrid,
        eval_grid: LogUniformGrid,
        nonneg: bool = True,
        max_iter: int = 2000,
    ) -> None:
        self.fit_grid = fit_grid
        self.eval_grid = eval_grid
        self.nonneg = nonneg
        self.max_iter = max_iter

        self._r_fit = fit_grid.points
        self._f_fit = 1.0 / self._r_fit
        self._r_eval = eval_grid.points
        self._f_eval = 1.0 / self._r_eval

    def _build_matrix(self, alphas: np.ndarray, r: np.ndarray) -> np.ndarray:
        return np.exp(-np.outer(r ** 2, alphas))

    def _solve_weights(self, alphas: np.ndarray) -> np.ndarray:
        A = self._build_matrix(alphas, self._r_fit)
        if self.nonneg:
            w, _ = nnls(A, self._f_fit)
        else:
            w, _, _, _ = np.linalg.lstsq(A, self._f_fit, rcond=None)
        return w

    def _objective(self, log_alphas: np.ndarray) -> float:
        alphas = np.exp(log_alphas)
        w = self._solve_weights(alphas)
        A = self._build_matrix(alphas, self._r_fit)
        residual = self._f_fit - A @ w
        return float(np.dot(residual, residual))

    def _initial_log_alphas(self, rank: int) -> np.ndarray:
        alpha_min, alpha_max = self.fit_grid.alpha_range()
        return np.linspace(np.log(alpha_min), np.log(alpha_max), rank)

    def _run_lbfgsb(self, rank: int) -> np.ndarray:
        x0 = self._initial_log_alphas(rank)
        result = minimize(
            self._objective,
            x0,
            method="L-BFGS-B",
            options={"maxiter": self.max_iter, "ftol": 1e-15, "gtol": 1e-10},
        )
        return result.x

    def _compute_errors(self, fit: ExponentialSum) -> tuple[float, float]:
        f_approx = fit.evaluate(self._r_eval)
        rel_err = np.abs((f_approx - self._f_eval) / self._f_eval)
        l2 = float(np.sqrt(np.mean(rel_err ** 2)))
        linf = float(np.max(rel_err))
        return l2, linf

    def fit(self, rank: int) -> ExponentialSum:
        """指定ランクで VARPRO を実行し、フィット済みの ExponentialSum を返す."""
        log_alphas_opt = self._run_lbfgsb(rank)
        alphas_opt = np.exp(np.sort(log_alphas_opt))
        weights_opt = self._solve_weights(alphas_opt)

        fit = ExponentialSum(weights=weights_opt, alphas=alphas_opt)
        fit.l2_error, fit.linf_error = self._compute_errors(fit)
        return fit


class BenchmarkRunner:
    """ランクのリストに対して VarproOptimizer を実行し結果を集約する."""

    def __init__(self, optimizer: VarproOptimizer, ranks: Sequence[int]) -> None:
        self.optimizer = optimizer
        self.ranks = list(ranks)
        self.results: dict[int, ExponentialSum] = {}

    def run(self) -> dict[int, ExponentialSum]:
        header = f"{'R':>4}  {'L2 rel err':>14}  {'L∞ rel err':>14}"
        print(header)
        print("-" * len(header))

        for rank in self.ranks:
            fit = self.optimizer.fit(rank)
            self.results[rank] = fit
            print(f"{rank:>4}  {fit.l2_error:>14.3e}  {fit.linf_error:>14.3e}")

        return self.results

    def save_json(self, path: str | Path) -> None:
        payload = {str(r): fit.to_dict() for r, fit in self.results.items()}
        with open(path, "w") as fp:
            json.dump(payload, fp, indent=2)
        print(f"Saved → {path}")


def _apply_1d_kernel_along_axis(K: np.ndarray, rho: np.ndarray, axis: int) -> np.ndarray:
    """3D 配列 rho の指定軸に沿って (N, N) の行列 K を作用させる."""
    return np.moveaxis(np.tensordot(K, rho, axes=([1], [axis])), 0, axis)


def apply_separable_gaussian_3d(
    alpha: float, x_axis: np.ndarray, rho: np.ndarray
) -> np.ndarray:
    """exp(-α|r1-r2|²) の分離性を利用して 3D ρ に作用させる.

    exp(-α|r1-r2|²) = exp(-α(x1-x2)²) · exp(-α(y1-y2)²) · exp(-α(z1-z2)²)
    なので、各軸に 1D Gaussian カーネルを順次適用すればよい。
    計算量 O(N⁴) (full kernel の O(N⁶) に対して N² 倍速い)。
    """
    diff = x_axis[:, None] - x_axis[None, :]
    K_1d = np.exp(-alpha * diff ** 2)
    result = _apply_1d_kernel_along_axis(K_1d, rho, axis=0)
    result = _apply_1d_kernel_along_axis(K_1d, result, axis=1)
    result = _apply_1d_kernel_along_axis(K_1d, result, axis=2)
    return result


def apply_exp_sum_potential_3d(
    fit: ExponentialSum,
    x_axis: np.ndarray,
    rho: np.ndarray,
    dx: float,
) -> np.ndarray:
    """指数和近似 K(r) ≈ Σ_k w_k exp(-α_k r²) を用いて V = (K * ρ) dx³ を計算する.

    分離性により O(R · N⁴) で評価可能 (R は展開項数)。

    Args:
        fit: 1/r をフィットした ExponentialSum.
        x_axis: 1D 座標軸 (shape (N,)).
        rho: 3D 電荷密度 (shape (N, N, N)).
        dx: グリッド間隔.

    Returns:
        V (shape (N, N, N)).
    """
    V = np.zeros_like(rho)
    for w_k, alpha_k in zip(fit.weights, fit.alphas):
        V += w_k * apply_separable_gaussian_3d(alpha_k, x_axis, rho)
    return V * dx ** 3
