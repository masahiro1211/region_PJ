"""VARPRO による指数和フィッティングを実行し、結果を .npy に保存する。

ノートブックの重いフィッティングセルをスクリプトとして実行できるように
したもの。既存の pkl キャッシュがあれば再計算せず、キャッシュ内容から
weights / alphas だけを numpy 配列として書き出す。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from src.approximation.exp_sum.benchmark import BenchmarkRunner
from src.approximation.exp_sum.grid import LogUniformGrid
from src.approximation.exp_sum.varpro import VarproOptimizer
from src.utils.cache import cache_path
from src.utils.cache import exp_sum_label
from src.utils.cache import load_or_compute

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _parse_rank_list(value: str) -> list[int]:
    """`1:30` 形式（両端含む）または comma-separated rank list を読む。"""
    stripped = value.strip()
    if ":" in stripped:
        parts = [part.strip() for part in stripped.split(":")]
        if len(parts) not in (2, 3):
            raise argparse.ArgumentTypeError(
                "rank range must be start:stop or start:stop:step."
            )
        start = int(parts[0])
        stop = int(parts[1])
        step = int(parts[2]) if len(parts) == 3 else 1
        if step == 0:
            raise argparse.ArgumentTypeError(
                "rank range step must be non-zero."
            )
        end = stop + (1 if step > 0 else -1)
        ranks = list(range(start, end, step))
    else:
        ranks = [
            int(item.strip()) for item in stripped.split(",") if item.strip()
        ]

    if not ranks:
        raise argparse.ArgumentTypeError("at least one rank is required.")
    if min(ranks) < 1:
        raise argparse.ArgumentTypeError("all ranks must be positive.")
    return ranks


parser = argparse.ArgumentParser(
    description="指数和フィッティング結果を .npy に保存する。"
)
parser.add_argument(
    "--compute",
    action="store_true",
    help="cache miss 時に重い VARPRO 計算を実行する。",
)
parser.add_argument("--L", type=float, default=20.0)
parser.add_argument("--ranks", type=_parse_rank_list, default="1:30")
parser.add_argument(
    "--nonneg",
    action=argparse.BooleanOptionalAction,
    default=True,
)
parser.add_argument("--max-iter", type=int, default=200000)
parser.add_argument("--n-points", type=int, default=2000)
parser.add_argument("--r-min", type=float, default=1e-2)
parser.add_argument("--r-max", type=float, default=None)
args = parser.parse_args()


L = args.L
ranks = args.ranks
nonneg = args.nonneg
max_iter = args.max_iter
n_points = args.n_points
r_min = args.r_min
r_max = np.sqrt(3) * L if args.r_max is None else args.r_max

out_label = exp_sum_label(
    L=L,
    ranks=ranks,
    nonneg=nonneg,
    max_iter=max_iter,
    n_points=n_points,
    r_min=r_min,
    r_max=r_max,
)


# ---- VARPRO フィッティング（既存の pkl キャッシュを利用） ----
fit_grid = LogUniformGrid(
    r_min=r_min,
    r_max=r_max,
    n_points=n_points,
)
eval_grid = LogUniformGrid(
    r_min=r_min,
    r_max=r_max,
    n_points=n_points,
)

optimizer = VarproOptimizer(
    fit_grid=fit_grid,
    eval_grid=eval_grid,
    nonneg=nonneg,
    max_iter=max_iter,
)
runner = BenchmarkRunner(optimizer=optimizer, ranks=ranks)

fits_cache_params = {
    "fit_grid": (fit_grid.r_min, fit_grid.r_max, fit_grid.n_points),
    "eval_grid": (eval_grid.r_min, eval_grid.r_max, eval_grid.n_points),
    "nonneg": nonneg,
    "max_iter": max_iter,
    "ranks": tuple(ranks),
}

fits_cache_path = cache_path("exp_sum_fits", fits_cache_params)
if not fits_cache_path.exists() and not args.compute:
    raise FileNotFoundError(
        "exp_sum_fits の cache が見つかりません。"
        "この PC で重い計算を避けるため停止します。"
        "計算用 PC / Slurm では `--compute` を付けて実行してください: "
        "`python scripts/run_exp_sum_fitting.py --compute`"
    )

fits = load_or_compute(
    namespace="exp_sum_fits",
    params=fits_cache_params,
    compute=runner.run,
)

# ---- ログ表示（キャッシュヒット時にも精度を確認できるようにする） ----
print(f"{'R':>4}  {'L2 rel err':>14}  {'Linf rel err':>14}")
print("-" * 38)
for rank in sorted(fits):
    fit = fits[rank]
    print(f"{rank:>4}  {fit.l2_error:>14.3e}  {fit.linf_error:>14.3e}")

# ---- notebook / 他スクリプトから np.load しやすい形式で保存 ----
out_dir = PROJECT_ROOT / "data" / "npy" / "exp_sum" / out_label
out_dir.mkdir(parents=True, exist_ok=True)
for rank, fit in fits.items():
    np.save(out_dir / f"R{rank:02d}_weights.npy", fit.weights)
    np.save(out_dir / f"R{rank:02d}_alphas.npy", fit.alphas)

params_json = {
    "L": L,
    "out_label": out_label,
    "output_dir": str(out_dir.relative_to(PROJECT_ROOT)),
    "ranks": ranks,
    "nonneg": nonneg,
    "max_iter": max_iter,
    "n_points": n_points,
    "r_min": r_min,
    "r_max": r_max,
    "cache_namespace": "exp_sum_fits",
    "cache_params": fits_cache_params,
}
with open(out_dir / "params.json", "w", encoding="utf-8") as fp:
    json.dump(params_json, fp, ensure_ascii=False, indent=2)

print(f"Saved {2 * len(fits)} files -> {out_dir}")
print(f"Saved params -> {out_dir / 'params.json'}")
