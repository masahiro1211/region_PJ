"""PyTorch カーネル実装の時間計測を行い、.npy に保存する。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# スクリプト直実行時にも repo root から import できるよう sys.path を整える。
from src.approximation.timing import benchmark  # noqa: E402
from src.approximation.timing import get_torch_metadata  # noqa: E402
from src.approximation.timing import iter_timing_targets  # noqa: E402
from src.approximation.timing import load_exp_sum_arrays  # noqa: E402
from src.approximation.timing import load_k_1d_list  # noqa: E402
from src.approximation.timing import (  # noqa: E402
    prepare_timing_benchmark_inputs,
)
from src.approximation.timing import with_cuda_sync  # noqa: E402
from src.utils.cache import exp_sum_label as make_exp_sum_label  # noqa: E402
from src.utils.cache import load_rpca_1d_list  # noqa: E402
from src.utils.cache import rpca_label as make_rpca_label  # noqa: E402


DEFAULT_EXP_SUM_RANKS = list(range(1, 31))
DEFAULT_EXP_SUM_NONNEG = True
DEFAULT_EXP_SUM_MAX_ITER = 200000
DEFAULT_EXP_SUM_N_POINTS = 2000
DEFAULT_EXP_SUM_R_MIN = 1e-2
DEFAULT_RPCA_MAX_ITER = 2000
DEFAULT_RPCA_TOL = 1e-6


def _parse_args() -> argparse.Namespace:
    """コマンドライン引数を読む。"""
    parser = argparse.ArgumentParser(
        description="PyTorch 実装の時間計測値を repeat ごとに .npy 保存する。"
    )
    parser.add_argument("--N", type=int, default=201)
    parser.add_argument("--L", type=float, default=20.0)
    parser.add_argument("--density-alpha", type=float, default=1.0)
    parser.add_argument("--exp-sum-rank", type=int, default=11)
    parser.add_argument("--r-bench", type=int, default=15)
    parser.add_argument("--tau-bench", type=float, default=1e-2)
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
    parser.add_argument("--n-warmup", type=int, default=5)
    parser.add_argument("--n-inner", type=int, default=10)
    parser.add_argument("--n-repeat", type=int, default=20)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/npy/timing"),
        help="保存先の基準ディレクトリ。",
    )
    return parser.parse_args()


def _make_exp_sum_label(args: argparse.Namespace) -> str:
    """既定設定から exp-sum の保存ラベルを作る。"""
    r_max = np.sqrt(3) * args.L
    return make_exp_sum_label(
        L=args.L,
        ranks=DEFAULT_EXP_SUM_RANKS,
        nonneg=DEFAULT_EXP_SUM_NONNEG,
        max_iter=DEFAULT_EXP_SUM_MAX_ITER,
        n_points=DEFAULT_EXP_SUM_N_POINTS,
        r_min=DEFAULT_EXP_SUM_R_MIN,
        r_max=r_max,
    )


def _make_rpca_label(args: argparse.Namespace) -> str:
    """既定設定から RPCA カーネルの保存ラベルを作る。"""
    return make_rpca_label(
        N=args.N,
        L=args.L,
        exp_sum_R=args.exp_sum_rank,
        rpca_rank=args.N // 4,
        rpca_max_iter=DEFAULT_RPCA_MAX_ITER,
        rpca_tol=DEFAULT_RPCA_TOL,
    )


def _sweep_label(args: argparse.Namespace) -> str:
    """タイミング計測条件を表す保存ラベルを作る。"""
    return (
        f"N{args.N}_L{args.L:g}_alpha{args.density_alpha:g}"
        f"_R{args.exp_sum_rank:02d}_r{args.r_bench:02d}"
        f"_tau{args.tau_bench:.0e}"
        f"_warm{args.n_warmup}_inner{args.n_inner}"
        f"_repeat{args.n_repeat}"
    )


def _resolve_output_base(path: Path) -> Path:
    """保存先の基準ディレクトリを絶対パスに変換する。"""
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def _relative_or_str(path: Path) -> str:
    """プロジェクト相対パスにできる場合は相対表記を返す。"""
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def _print_result(name: str, values: np.ndarray) -> None:
    """測定結果の要約を表示する。"""
    mean = values.mean()
    std = values.std(ddof=1) if values.size > 1 else 0.0
    print(f"{name:12s}: mean={mean:.6e} s, std={std:.6e} s")


def main() -> None:
    """時間計測を実行し、各 repeat の秒数を .npy に保存する。"""
    args = _parse_args()

    exp_sum_label = args.exp_sum_label or _make_exp_sum_label(args)
    rpca_label = args.rpca_label or _make_rpca_label(args)
    exp_sum_dir = (
        PROJECT_ROOT / "data" / "npy" / "exp_sum" / exp_sum_label
    )
    rpca_dir = (
        PROJECT_ROOT
        / "data"
        / "npy"
        / "rpca_kernels"
        / exp_sum_label
        / rpca_label
    )
    if not rpca_dir.exists():
        raise FileNotFoundError(
            "RPCA カーネルの .npy ディレクトリが見つかりません。先に "
            "`python scripts/run_rpca_kernels.py` を実行してください: "
            f"{rpca_dir}"
        )

    weights, alphas = load_exp_sum_arrays(exp_sum_dir, args.exp_sum_rank)
    if len(weights) != args.exp_sum_rank or len(alphas) != args.exp_sum_rank:
        raise ValueError(
            "exp-sum rank と読み込んだ weights / alphas の長さが "
            f"一致しません: rank={args.exp_sum_rank}, "
            f"len(weights)={len(weights)}, len(alphas)={len(alphas)}."
        )

    rpca_1d_list = load_rpca_1d_list(rpca_dir, n_terms=args.exp_sum_rank)
    k_1d_list = load_k_1d_list(rpca_dir, n_terms=args.exp_sum_rank)
    inputs = prepare_timing_benchmark_inputs(
        n_grid=args.N,
        length=args.L,
        density_alpha=args.density_alpha,
        weights=weights,
        k_1d_list=k_1d_list,
        rpca_1d_list=rpca_1d_list,
        r_bench=args.r_bench,
        tau_bench=args.tau_bench,
    )

    print(
        f"rpca path: dense={inputs.rpca_dense_count}, "
        f"lowrank_only={inputs.rpca_lowrank_only_count} "
        "(backend = PyTorch)"
    )

    output_base = _resolve_output_base(args.output_dir)
    out_dir = output_base / exp_sum_label / rpca_label / _sweep_label(args)
    out_dir.mkdir(parents=True, exist_ok=True)

    results: dict[str, np.ndarray] = {}
    for target in iter_timing_targets(inputs):
        print(f"benchmarking {target.name} ...")
        values = benchmark(
            with_cuda_sync(target.fn),
            n_warmup=args.n_warmup,
            n_inner=args.n_inner,
            n_repeat=args.n_repeat,
        )
        results[target.name] = values
        np.save(out_dir / f"t_{target.name}.npy", values)
        _print_result(target.name, values)

    params_json = {
        "N": args.N,
        "L": args.L,
        "dx": inputs.dx,
        "density_alpha": args.density_alpha,
        "exp_sum_rank": args.exp_sum_rank,
        "r_bench": args.r_bench,
        "tau_bench": args.tau_bench,
        "n_warmup": args.n_warmup,
        "n_inner": args.n_inner,
        "n_repeat": args.n_repeat,
        "exp_sum_label": exp_sum_label,
        "rpca_label": rpca_label,
        "sweep_label": _sweep_label(args),
        "exp_sum_dir": _relative_or_str(exp_sum_dir),
        "rpca_dir": _relative_or_str(rpca_dir),
        "output_dir": _relative_or_str(out_dir),
        "rpca_dense_count": inputs.rpca_dense_count,
        "rpca_lowrank_only_count": inputs.rpca_lowrank_only_count,
        "exp_sum_label_defaults": {
            "ranks": DEFAULT_EXP_SUM_RANKS,
            "nonneg": DEFAULT_EXP_SUM_NONNEG,
            "max_iter": DEFAULT_EXP_SUM_MAX_ITER,
            "n_points": DEFAULT_EXP_SUM_N_POINTS,
            "r_min": DEFAULT_EXP_SUM_R_MIN,
            "r_max": np.sqrt(3) * args.L,
        },
        "rpca_label_defaults": {
            "rpca_rank": args.N // 4,
            "rpca_max_iter": DEFAULT_RPCA_MAX_ITER,
            "rpca_tol": DEFAULT_RPCA_TOL,
        },
        "torch": get_torch_metadata(),
        "timing_files": {
            name: f"t_{name}.npy"
            for name in results
        },
    }
    with open(out_dir / "params.json", "w", encoding="utf-8") as fp:
        json.dump(params_json, fp, ensure_ascii=False, indent=2)

    print(f"Saved timing benchmark -> {out_dir}")
    print(f"Saved params -> {out_dir / 'params.json'}")


if __name__ == "__main__":
    main()
