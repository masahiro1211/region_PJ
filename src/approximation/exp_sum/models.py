"""Data model for exponential-sum approximations."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ExponentialSum:
    """指数和近似 f(r) ≈ Σ_k w_k exp(-α_k r²) のパラメータ."""

    weights: np.ndarray
    alphas: np.ndarray
    l2_error: float = 0.0
    linf_error: float = 0.0

    @property
    def rank(self) -> int:
        return len(self.weights)

    def evaluate(self, r: np.ndarray) -> np.ndarray:
        """指定された r で近似を評価する."""
        A = np.exp(-np.outer(r**2, self.alphas))
        return A @ self.weights

    def to_dict(self) -> dict:
        return {
            "rank": self.rank,
            "weights": self.weights.tolist(),
            "alphas": self.alphas.tolist(),
            "L2_error": self.l2_error,
            "Linf_error": self.linf_error,
        }
