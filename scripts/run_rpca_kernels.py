"""1D Gaussian カーネルの RPCA / SVD 分解を実行し、.npy に保存する。

ノートブックの 3D ポテンシャル誤差ループには入らず、その前段で必要な
1D カーネルと分解済み行列だけを作る。既存の pkl キャッシュがあれば
再計算せず、キャッシュ内容を numpy 配列として書き出す。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from src.decomposition.rpca import randomized_rpca
from src.potential.separable_density import (
    build_gaussian_kernel_1d,
)
from src.utils.cache import cache_path
from src.utils.cache import exp_sum_label as make_exp_sum_label
from src.utils.cache import load_or_compute
from src.utils.cache import rpca_label
from src.utils.grid import build_xyz

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
    description="1D Gaussian カーネル分解結果を .npy に保存する。"
)
parser.add_argument(
    "--compute",
    action="store_true",
    help="cache miss 時に重い RPCA / SVD 計算を実行する。",
)
parser.add_argument("--N", type=int, default=201)
parser.add_argument("--L", type=float, default=20.0)
parser.add_argument("--exp-sum-rank", type=int, default=11)
parser.add_argument("--rpca-rank", type=int, default=None)
parser.add_argument("--rpca-max-iter", type=int, default=2000)
parser.add_argument("--rpca-tol", type=float, default=1e-6)
parser.add_argument(
    "--exp-sum-ranks",
    type=_parse_rank_list,
    default="1:30",
)
parser.add_argument(
    "--exp-sum-nonneg",
    action=argparse.BooleanOptionalAction,
    default=True,
)
parser.add_argument("--exp-sum-max-iter", type=int, default=200000)
parser.add_argument("--exp-sum-n-points", type=int, default=2000)
parser.add_argument("--exp-sum-r-min", type=float, default=1e-2)
parser.add_argument("--exp-sum-r-max", type=float, default=None)
args = parser.parse_args()


N = args.N
L = args.L
exp_sum_R = args.exp_sum_rank
rpca_rank = N // 4 if args.rpca_rank is None else args.rpca_rank
rpca_max_iter = args.rpca_max_iter
rpca_tol = args.rpca_tol
exp_sum_ranks = args.exp_sum_ranks
exp_sum_nonneg = args.exp_sum_nonneg
exp_sum_max_iter = args.exp_sum_max_iter
exp_sum_n_points = args.exp_sum_n_points
exp_sum_r_min = args.exp_sum_r_min
exp_sum_r_max = args.exp_sum_r_max
if exp_sum_r_max is None:
    exp_sum_r_max = np.sqrt(3) * L
exp_sum_label = make_exp_sum_label(
    L=L,
    ranks=exp_sum_ranks,
    nonneg=exp_sum_nonneg,
    max_iter=exp_sum_max_iter,
    n_points=exp_sum_n_points,
    r_min=exp_sum_r_min,
    r_max=exp_sum_r_max,
)

out_label = rpca_label(
    N=N,
    L=L,
    exp_sum_R=exp_sum_R,
    rpca_rank=rpca_rank,
    rpca_max_iter=rpca_max_iter,
    rpca_tol=rpca_tol,
)


# ---- exp-sum フィットを .npy から読み込む ----
exp_sum_dir = PROJECT_ROOT / "data" / "npy" / "exp_sum" / exp_sum_label
weights_path = exp_sum_dir / f"R{exp_sum_R:02d}_weights.npy"
alphas_path = exp_sum_dir / f"R{exp_sum_R:02d}_alphas.npy"
if not weights_path.exists() or not alphas_path.exists():
    raise FileNotFoundError(
        "exp-sum の .npy ファイルが見つかりません。先に "
        "`python scripts/run_exp_sum_fitting.py` を実行してください。"
    )

weights = np.load(weights_path)
alphas = np.load(alphas_path)
x_axis = build_xyz(N, L)[0, :, 0, 0]


def _compute_kernels():
    """load_or_compute に渡す、1D カーネル分解のローカル計算本体。"""
    K_1d_list_local = []
    RPCA_1d_list_local = []

    for k, alpha_k in enumerate(alphas):
        print(f"[{k + 1}/{len(alphas)}] factoring 1D Gaussian kernel")

        K_1D = build_gaussian_kernel_1d(float(alpha_k), x_axis)
        K_1d_list_local.append(K_1D)

        L_1d, S_1d = randomized_rpca(
            K_1D,
            rank=rpca_rank,
            max_iter=rpca_max_iter,
            tol=rpca_tol,
        )
        U_L, S_L, Vt_L = np.linalg.svd(L_1d)
        U_s, S_s, Vt_s = np.linalg.svd(K_1D)

        RPCA_1d_list_local.append(
            {
                "S_1d": S_1d,
                "U_L": U_L,
                "S_L": S_L,
                "Vt_L": Vt_L,
                "U_s": U_s,
                "S_s": S_s,
                "Vt_s": Vt_s,
            }
        )

    return {"K_1d_list": K_1d_list_local, "RPCA_1d_list": RPCA_1d_list_local}


# ---- RPCA / SVD 分解（既存の pkl キャッシュを利用） ----
kernels_cache_params = {
    "N": N,
    "L": L,
    "exp_sum_R": exp_sum_R,
    "alphas": np.asarray(alphas, dtype=np.float64),
    "rpca_rank": rpca_rank,
    "rpca_max_iter": rpca_max_iter,
    "rpca_tol": rpca_tol,
}

kernels_cache_path = cache_path("rpca_svd_1d_kernels", kernels_cache_params)
if not kernels_cache_path.exists() and not args.compute:
    raise FileNotFoundError(
        "rpca_svd_1d_kernels の cache が見つかりません。"
        "この PC で重い計算を避けるため停止します。"
        "計算用 PC / Slurm では `--compute` を付けて実行してください: "
        "`python scripts/run_rpca_kernels.py --compute`"
    )

kernels_data = load_or_compute(
    namespace="rpca_svd_1d_kernels",
    params=kernels_cache_params,
    compute=_compute_kernels,
)
K_1d_list = kernels_data["K_1d_list"]
RPCA_1d_list = kernels_data["RPCA_1d_list"]

if len(K_1d_list) != len(weights) or len(RPCA_1d_list) != len(weights):
    raise ValueError(
        "キャッシュされた RPCA データの個数が exp-sum の項数と一致しません: "
        f"len(K_1d_list)={len(K_1d_list)}, "
        f"len(RPCA_1d_list)={len(RPCA_1d_list)}, len(weights)={len(weights)}."
    )

# ---- notebook / 他スクリプトから np.load しやすい形式で保存 ----
out_dir = (
    PROJECT_ROOT
    / "data"
    / "npy"
    / "rpca_kernels"
    / exp_sum_label
    / out_label
)
out_dir.mkdir(parents=True, exist_ok=True)
for k, (K1d, rpca) in enumerate(zip(K_1d_list, RPCA_1d_list)):
    np.save(out_dir / f"k{k:02d}_K1d.npy", K1d)
    np.save(out_dir / f"k{k:02d}_UL.npy", rpca["U_L"])
    np.save(out_dir / f"k{k:02d}_SL.npy", rpca["S_L"])
    np.save(out_dir / f"k{k:02d}_VtL.npy", rpca["Vt_L"])
    np.save(out_dir / f"k{k:02d}_Ssparse.npy", rpca["S_1d"])
    np.save(out_dir / f"k{k:02d}_Us.npy", rpca["U_s"])
    np.save(out_dir / f"k{k:02d}_Ss.npy", rpca["S_s"])
    np.save(out_dir / f"k{k:02d}_Vts.npy", rpca["Vt_s"])

params_json = {
    "N": N,
    "L": L,
    "out_label": out_label,
    "output_dir": str(out_dir.relative_to(PROJECT_ROOT)),
    "exp_sum_R": exp_sum_R,
    "exp_sum_label": exp_sum_label,
    "rpca_rank": rpca_rank,
    "rpca_max_iter": rpca_max_iter,
    "rpca_tol": rpca_tol,
    "exp_sum_dir": str(exp_sum_dir.relative_to(PROJECT_ROOT)),
    "alphas": alphas.tolist(),
    "weights": weights.tolist(),
    "cache_namespace": "rpca_svd_1d_kernels",
    "cache_params": {
        "N": N,
        "L": L,
        "exp_sum_R": exp_sum_R,
        "alphas": alphas.tolist(),
        "rpca_rank": rpca_rank,
        "rpca_max_iter": rpca_max_iter,
        "rpca_tol": rpca_tol,
    },
}
with open(out_dir / "params.json", "w", encoding="utf-8") as fp:
    json.dump(params_json, fp, ensure_ascii=False, indent=2)

print(f"Saved {8 * len(K_1d_list)} files -> {out_dir}")
print(f"Saved params -> {out_dir / 'params.json'}")
