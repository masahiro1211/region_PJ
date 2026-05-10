"""Benchmark helpers for exponential-sum fitting."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

from .models import ExponentialSum
from .varpro import VarproOptimizer


class BenchmarkRunner:
    """ランクのリストに対して VarproOptimizer を実行し結果を集約する."""

    def __init__(
        self,
        optimizer: VarproOptimizer,
        ranks: Sequence[int],
    ) -> None:
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
        print(f"Saved -> {path}")
