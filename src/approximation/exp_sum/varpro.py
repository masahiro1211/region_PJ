"""VARPRO optimizer for 1/r exponential-sum approximations."""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize, nnls

from .grid import LogUniformGrid
from .models import ExponentialSum


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
        return np.exp(-np.outer(r**2, alphas))

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
        l2 = float(np.sqrt(np.mean(rel_err**2)))
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
