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
from src.approximation.timing import timing_sweep_label  # noqa: E402
from src.approximation.timing import with_cuda_sync  # noqa: E402
from src.experiment_config import density_alpha as _cfg_density_alpha  # noqa: E402
from src.experiment_config import exp_sum_max_iter as _cfg_exp_sum_max_iter  # noqa: E402
from src.experiment_config import exp_sum_n_points as _cfg_exp_sum_n_points  # noqa: E402
from src.experiment_config import exp_sum_nonneg as _cfg_exp_sum_nonneg  # noqa: E402
from src.experiment_config import exp_sum_r_min as _cfg_exp_sum_r_min  # noqa: E402
from src.experiment_config import exp_sum_rank as _cfg_exp_sum_rank  # noqa: E402
from src.experiment_config import exp_sum_ranks as _cfg_exp_sum_ranks  # noqa: E402
from src.experiment_config import L as _cfg_L  # noqa: E402
from src.experiment_config import N as _cfg_N  # noqa: E402
from src.experiment_config import rpca_max_iter as _cfg_rpca_max_iter  # noqa: E402
from src.experiment_config import rpca_tol as _cfg_rpca_tol  # noqa: E402
from src.experiment_config import timing_n_inner as _cfg_timing_n_inner  # noqa: E402
from src.experiment_config import timing_n_repeat as _cfg_timing_n_repeat  # noqa: E402
from src.experiment_config import timing_n_warmup as _cfg_timing_n_warmup  # noqa: E402
from src.experiment_config import timing_r_bench as _cfg_timing_r_bench  # noqa: E402
from src.experiment_config import timing_tau_bench as _cfg_timing_tau_bench  # noqa: E402
from src.utils.cache import exp_sum_label as make_exp_sum_label  # noqa: E402
from src.utils.cache import load_rpca_1d_list  # noqa: E402
from src.utils.cache import rpca_label as make_rpca_label  # noqa: E402


def _parse_args() -> argparse.Namespace:
    """コマンドライン引数を読む。デフォルト値は experiment_config.sh から取得。"""
    parser = argparse.ArgumentParser(
        description="PyTorch 実装の時間計測値を repeat ごとに .npy 保存する。"
    )
    parser.add_argument("--N", type=int, default=_cfg_N())
    parser.add_argument("--L", type=float, default=_cfg_L())
    parser.add_argument("--density-alpha", type=float, default=_cfg_density_alpha())
    parser.add_argument("--exp-sum-rank", type=int, default=_cfg_exp_sum_rank())
    parser.add_argument("--r-bench", type=int, default=_cfg_timing_r_bench())
    parser.add_argument("--tau-bench", type=float, default=_cfg_timing_tau_bench())
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
    parser.add_argument("--n-warmup", type=int, default=_cfg_timing_n_warmup())
    parser.add_argument("--n-inner", type=int, default=_cfg_timing_n_inner())
    parser.add_argument("--n-repeat", type=int, default=_cfg_timing_n_repeat())
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/npy/timing"),
        help="保存先の基準ディレクトリ。",
    )
    return parser.parse_args()


def _make_exp_sum_label(args: argparse.Namespace) -> str:
    """既定設定から exp-sum の保存ラベルを作る。"""
    return make_exp_sum_label(
        L=args.L,
        ranks=_cfg_exp_sum_ranks(),
        nonneg=_cfg_exp_sum_nonneg(),
        max_iter=_cfg_exp_sum_max_iter(),
        n_points=_cfg_exp_sum_n_points(),
        r_min=_cfg_exp_sum_r_min(),
        r_max=np.sqrt(3) * args.L,
    )


def _make_rpca_label(args: argparse.Namespace) -> str:
    """既定設定から RPCA カーネルの保存ラベルを作る。"""
    return make_rpca_label(
        N=args.N,
        L=args.L,
        exp_sum_R=args.exp_sum_rank,
        rpca_rank=args.N // 4,
        rpca_max_iter=_cfg_rpca_max_iter(),
        rpca_tol=_cfg_rpca_tol(),
    )


def _sweep_label(args: argparse.Namespace) -> str:
    """タイミング計測条件を表す保存ラベルを作る。"""
    return timing_sweep_label(
        n_grid=args.N,
        length=args.L,
        density_alpha=args.density_alpha,
        exp_sum_rank=args.exp_sum_rank,
        r_bench=args.r_bench,
        tau_bench=args.tau_bench,
        n_warmup=args.n_warmup,
        n_inner=args.n_inner,
        n_repeat=args.n_repeat,
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
            "ranks": _cfg_exp_sum_ranks(),
            "nonneg": _cfg_exp_sum_nonneg(),
            "max_iter": _cfg_exp_sum_max_iter(),
            "n_points": _cfg_exp_sum_n_points(),
            "r_min": _cfg_exp_sum_r_min(),
            "r_max": np.sqrt(3) * args.L,
        },
        "rpca_label_defaults": {
            "rpca_rank": args.N // 4,
            "rpca_max_iter": _cfg_rpca_max_iter(),
            "rpca_tol": _cfg_rpca_tol(),
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
