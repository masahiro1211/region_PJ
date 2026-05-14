"""VARPRO による指数和フィッティングを実行し、結果を .npy に保存する。

ノートブックの重いフィッティングセルをスクリプトとして実行できるように
したもの。既存の pkl キャッシュがあれば再計算せず、キャッシュ内容から
weights / alphas だけを numpy 配列として書き出す。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.approximation.exp_sum.benchmark import BenchmarkRunner  # noqa: E402
from src.approximation.exp_sum.grid import LogUniformGrid  # noqa: E402
from src.approximation.exp_sum.varpro import VarproOptimizer  # noqa: E402
from src.utils.cache import cache_path  # noqa: E402
from src.utils.cache import load_or_compute  # noqa: E402
from scripts._io import exp_sum_label  # noqa: E402


parser = argparse.ArgumentParser(
    description="指数和フィッティング結果を .npy に保存する。"
)
parser.add_argument(
    "--compute",
    action="store_true",
    help="cache miss 時に重い VARPRO 計算を実行する。",
)
args = parser.parse_args()


# ---- パラメータ（必要ならここだけ変更する） ----
L = 20
ranks = list(range(1, 31))
nonneg = True
max_iter = 200000
n_points = 2000
r_min = 1e-2
r_max = 2 * np.sqrt(3) * L

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
