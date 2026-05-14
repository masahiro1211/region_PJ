"""RPCA / SVD 1D カーネルの 3D ポテンシャル誤差 sweep を実行する。

事前に scripts/run_exp_sum_fitting.py と scripts/run_rpca_kernels.py で保存した
.npy を読み込み、ノートブック上の重い rank / threshold sweep をスクリプトで
再現して保存する。
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

from src.approximation.exp_sum.models import ExponentialSum  # noqa: E402
from src.approximation.exp_sum.separable import (  # noqa: E402
    apply_1d_kernel_along_axis,
)
from src.potential.charge_potential import v_analytic_gaussian  # noqa: E402
from src.utils.grid import build_xyz  # noqa: E402
from src.utils.metrics import v_e_errors  # noqa: E402


DEFAULT_THRESHOLDS = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0]


def _parse_float_list(value: str) -> list[float]:
    """Comma-separated float list を読む。"""
    items = [item.strip() for item in value.split(",") if item.strip()]
    if not items:
        raise argparse.ArgumentTypeError(
            "at least one float value is required."
        )
    return [float(item) for item in items]


def _parse_rank_list(value: str) -> list[int]:
    """`1:25` 形式（両端含む）または comma-separated rank list を読む。"""
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


def _exp_sum_label(
    *,
    L: float,
    ranks: list[int],
    nonneg: bool,
    max_iter: int,
    n_points: int,
    r_min: float,
    r_max: float,
) -> str:
    return (
        f"L{L:g}_rmin{r_min:.0e}_rmax{r_max:.6g}"
        f"_n{n_points}_nonneg{int(nonneg)}"
        f"_iter{max_iter}"
        f"_R{min(ranks):02d}-{max(ranks):02d}"
        f"_count{len(ranks)}"
    )


def _rpca_label(
    *,
    N: int,
    L: float,
    exp_sum_R: int,
    rpca_rank: int,
    rpca_max_iter: int,
    rpca_tol: float,
) -> str:
    return (
        f"N{N}_L{L:g}_R{exp_sum_R:02d}_rank{rpca_rank}"
        f"_iter{rpca_max_iter}_tol{rpca_tol:.0e}"
    )


def _load_rpca_1d_list(
    rpca_dir: Path,
    n_terms: int,
) -> list[dict[str, np.ndarray]]:
    rpca_1d_list = []
    for k in range(n_terms):
        rpca_1d_list.append(
            {
                "S_1d": np.load(rpca_dir / f"k{k:02d}_Ssparse.npy"),
                "U_L": np.load(rpca_dir / f"k{k:02d}_UL.npy"),
                "S_L": np.load(rpca_dir / f"k{k:02d}_SL.npy"),
                "Vt_L": np.load(rpca_dir / f"k{k:02d}_VtL.npy"),
                "U_s": np.load(rpca_dir / f"k{k:02d}_Us.npy"),
                "S_s": np.load(rpca_dir / f"k{k:02d}_Ss.npy"),
                "Vt_s": np.load(rpca_dir / f"k{k:02d}_Vts.npy"),
            }
        )
    return rpca_1d_list


def _apply_3d_kernel(
    kernel_1d: np.ndarray,
    rho_grid: np.ndarray,
) -> np.ndarray:
    result = apply_1d_kernel_along_axis(kernel_1d, rho_grid, axis=0)
    result = apply_1d_kernel_along_axis(kernel_1d, result, axis=1)
    result = apply_1d_kernel_along_axis(kernel_1d, result, axis=2)
    return result


def _truncated_svd_kernel(
    U: np.ndarray,
    singular_values: np.ndarray,
    Vt: np.ndarray,
    rank: int,
) -> np.ndarray:
    eff_rank = min(rank, singular_values.shape[0])
    return (U[:, :eff_rank] * singular_values[:eff_rank]) @ Vt[:eff_rank, :]


def compute_rpca_error_sweep(
    *,
    fit: ExponentialSum,
    rpca_1d_list: list[dict[str, np.ndarray]],
    rho_grid: np.ndarray,
    v_analytic: np.ndarray,
    dx: float,
    k_diag_true: float,
    svd_ranks: list[int],
    thresholds: list[float],
) -> dict[str, np.ndarray | dict[str, np.ndarray]]:
    """SVD only / RPCA no threshold / RPCA threshold の V, E 誤差を計算する。"""
    errors_v_svd_only = []
    errors_e_svd_only = []
    errors_v_rpca = []
    errors_e_rpca = []
    errors_v_rpca_thresh = {thresh: [] for thresh in thresholds}
    errors_e_rpca_thresh = {thresh: [] for thresh in thresholds}

    diag_coeff = k_diag_true - float(np.sum(fit.weights))

    for rank in svd_ranks:
        print(f"[rank {rank}] evaluating SVD / RPCA error sweep", flush=True)

        V_svd = np.zeros_like(rho_grid)
        V_rpca_no_thresh = np.zeros_like(rho_grid)
        V_rpca_th = {thresh: np.zeros_like(rho_grid) for thresh in thresholds}

        for w_k, k_data in zip(fit.weights, rpca_1d_list):
            K_svd_1d = _truncated_svd_kernel(
                k_data["U_s"],
                k_data["S_s"],
                k_data["Vt_s"],
                rank,
            )
            V_svd += w_k * _apply_3d_kernel(K_svd_1d, rho_grid)

            L_r_1d = _truncated_svd_kernel(
                k_data["U_L"],
                k_data["S_L"],
                k_data["Vt_L"],
                rank,
            )
            K_rpca_1d = L_r_1d + k_data["S_1d"]
            V_rpca_no_thresh += w_k * _apply_3d_kernel(K_rpca_1d, rho_grid)

            for thresh in thresholds:
                S_thresh = np.where(
                    np.abs(k_data["S_1d"]) > thresh,
                    k_data["S_1d"],
                    0.0,
                )
                K_rpca_th_1d = L_r_1d + S_thresh
                V_rpca_th[thresh] += w_k * _apply_3d_kernel(
                    K_rpca_th_1d,
                    rho_grid,
                )

        V_svd_corr = (V_svd + diag_coeff * rho_grid) * dx**3
        V_rpca_no_corr = (V_rpca_no_thresh + diag_coeff * rho_grid) * dx**3

        err_v_s, err_e_s = v_e_errors(V_svd_corr, v_analytic, rho_grid, dx)
        errors_v_svd_only.append(err_v_s)
        errors_e_svd_only.append(err_e_s)

        err_v_rpca_no, err_e_rpca_no = v_e_errors(
            V_rpca_no_corr,
            v_analytic,
            rho_grid,
            dx,
        )
        errors_v_rpca.append(err_v_rpca_no)
        errors_e_rpca.append(err_e_rpca_no)

        for thresh in thresholds:
            V_rpca_th_corr = (
                V_rpca_th[thresh] + diag_coeff * rho_grid
            ) * dx**3
            err_v_th, err_e_th = v_e_errors(
                V_rpca_th_corr,
                v_analytic,
                rho_grid,
                dx,
            )
            errors_v_rpca_thresh[thresh].append(err_v_th)
            errors_e_rpca_thresh[thresh].append(err_e_th)

    return {
        "errors_v_svd_only": np.asarray(errors_v_svd_only),
        "errors_e_svd_only": np.asarray(errors_e_svd_only),
        "errors_v_rpca": np.asarray(errors_v_rpca),
        "errors_e_rpca": np.asarray(errors_e_rpca),
        "errors_v_rpca_thresh": {
            f"{thresh:.0e}": np.asarray(values)
            for thresh, values in errors_v_rpca_thresh.items()
        },
        "errors_e_rpca_thresh": {
            f"{thresh:.0e}": np.asarray(values)
            for thresh, values in errors_e_rpca_thresh.items()
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="RPCA / SVD 1D kernels から 3D ポテンシャル誤差を sweep する。"
    )
    parser.add_argument("--N", type=int, default=201)
    parser.add_argument("--L", type=float, default=20.0)
    parser.add_argument("--density-alpha", type=float, default=1.0)
    parser.add_argument("--cell-int-const", type=float, default=2.38)
    parser.add_argument("--exp-sum-rank", type=int, default=11)
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
    parser.add_argument("--rpca-rank", type=int, default=None)
    parser.add_argument("--rpca-max-iter", type=int, default=2000)
    parser.add_argument("--rpca-tol", type=float, default=1e-6)
    parser.add_argument("--svd-ranks", type=_parse_rank_list, default="1:25")
    parser.add_argument(
        "--thresholds",
        type=_parse_float_list,
        default=",".join(str(v) for v in DEFAULT_THRESHOLDS),
    )
    parser.add_argument(
        "--exp-sum-label",
        default=None,
        help="既存の exp_sum 出力ディレクトリ名を明示する場合に指定。",
    )
    parser.add_argument(
        "--rpca-label",
        default=None,
        help="既存の rpca_kernels 出力ディレクトリ名を明示する場合に指定。",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="保存先を明示する場合に指定。未指定なら data/npy/rpca_error_sweep 配下。",
    )
    args = parser.parse_args()

    exp_sum_r_max = args.exp_sum_r_max
    if exp_sum_r_max is None:
        exp_sum_r_max = 2 * np.sqrt(3) * args.L
    exp_sum_label = args.exp_sum_label or _exp_sum_label(
        L=args.L,
        ranks=args.exp_sum_ranks,
        nonneg=args.exp_sum_nonneg,
        max_iter=args.exp_sum_max_iter,
        n_points=args.exp_sum_n_points,
        r_min=args.exp_sum_r_min,
        r_max=exp_sum_r_max,
    )
    rpca_rank = args.N // 4 if args.rpca_rank is None else args.rpca_rank
    rpca_label = args.rpca_label or _rpca_label(
        N=args.N,
        L=args.L,
        exp_sum_R=args.exp_sum_rank,
        rpca_rank=rpca_rank,
        rpca_max_iter=args.rpca_max_iter,
        rpca_tol=args.rpca_tol,
    )

    exp_sum_dir = PROJECT_ROOT / "data" / "npy" / "exp_sum" / exp_sum_label
    rpca_dir = (
        PROJECT_ROOT
        / "data"
        / "npy"
        / "rpca_kernels"
        / exp_sum_label
        / rpca_label
    )
    weights_path = exp_sum_dir / f"R{args.exp_sum_rank:02d}_weights.npy"
    alphas_path = exp_sum_dir / f"R{args.exp_sum_rank:02d}_alphas.npy"

    if not weights_path.exists() or not alphas_path.exists():
        raise FileNotFoundError(
            "exp-sum の .npy ファイルが見つかりません。先に "
            "`python scripts/run_exp_sum_fitting.py` を実行してください: "
            f"{exp_sum_dir}"
        )
    if not rpca_dir.exists():
        raise FileNotFoundError(
            "RPCA カーネルの .npy ディレクトリが見つかりません。先に "
            "`python scripts/run_rpca_kernels.py` を実行してください: "
            f"{rpca_dir}"
        )

    weights = np.load(weights_path)
    alphas = np.load(alphas_path)
    fit = ExponentialSum(weights=weights, alphas=alphas)
    rpca_1d_list = _load_rpca_1d_list(rpca_dir, n_terms=fit.rank)

    max_requested_rank = max(args.svd_ranks)
    available_rank = min(
        min(k_data["S_s"].shape[0], k_data["S_L"].shape[0])
        for k_data in rpca_1d_list
    )
    if max_requested_rank > available_rank:
        raise ValueError(
            f"requested max rank {max_requested_rank} exceeds available rank "
            f"{available_rank}."
        )

    dx = args.L / args.N
    xyz = build_xyz(args.N, args.L)
    radius = np.sqrt(xyz[0] ** 2 + xyz[1] ** 2 + xyz[2] ** 2)
    rho_grid = np.exp(-args.density_alpha * radius**2)
    v_analytic = v_analytic_gaussian(radius, args.density_alpha)
    k_diag_true = args.cell_int_const / dx

    results = compute_rpca_error_sweep(
        fit=fit,
        rpca_1d_list=rpca_1d_list,
        rho_grid=rho_grid,
        v_analytic=v_analytic,
        dx=dx,
        k_diag_true=k_diag_true,
        svd_ranks=args.svd_ranks,
        thresholds=args.thresholds,
    )

    out_dir = args.output_dir
    if out_dir is None:
        rank_label = f"r{min(args.svd_ranks):02d}-{max(args.svd_ranks):02d}"
        out_dir = (
            PROJECT_ROOT
            / "data"
            / "npy"
            / "rpca_error_sweep"
            / exp_sum_label
            / rpca_label
            / rank_label
        )
    elif not out_dir.is_absolute():
        out_dir = PROJECT_ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    np.save(out_dir / "svd_ranks.npy", np.asarray(args.svd_ranks, dtype=int))
    np.save(
        out_dir / "thresholds.npy",
        np.asarray(args.thresholds, dtype=float),
    )
    np.save(out_dir / "errors_v_svd_only.npy", results["errors_v_svd_only"])
    np.save(out_dir / "errors_e_svd_only.npy", results["errors_e_svd_only"])
    np.save(out_dir / "errors_v_rpca.npy", results["errors_v_rpca"])
    np.save(out_dir / "errors_e_rpca.npy", results["errors_e_rpca"])
    for key, values in results["errors_v_rpca_thresh"].items():
        np.save(out_dir / f"errors_v_rpca_thresh_{key}.npy", values)
    for key, values in results["errors_e_rpca_thresh"].items():
        np.save(out_dir / f"errors_e_rpca_thresh_{key}.npy", values)

    params_json = {
        "N": args.N,
        "L": args.L,
        "dx": dx,
        "density_alpha": args.density_alpha,
        "cell_int_const": args.cell_int_const,
        "K_diag_true": k_diag_true,
        "exp_sum_rank": args.exp_sum_rank,
        "exp_sum_label": exp_sum_label,
        "rpca_label": rpca_label,
        "svd_ranks": args.svd_ranks,
        "thresholds": args.thresholds,
        "exp_sum_dir": str(exp_sum_dir.relative_to(PROJECT_ROOT)),
        "rpca_dir": str(rpca_dir.relative_to(PROJECT_ROOT)),
        "output_dir": str(out_dir.relative_to(PROJECT_ROOT)),
    }
    with open(out_dir / "params.json", "w", encoding="utf-8") as fp:
        json.dump(params_json, fp, ensure_ascii=False, indent=2)

    print(f"Saved RPCA / SVD error sweep -> {out_dir}")
    print(f"Saved params -> {out_dir / 'params.json'}")


if __name__ == "__main__":
    main()
