"""PyTorch カーネル実装の時間計測を行い、.npy に保存する。"""

from __future__ import annotations

import argparse
import json
import sys
import timeit
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    torch = None  # type: ignore[assignment]

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# スクリプト直実行時にも repo root から import できるよう sys.path を整える。
from scripts._io import exp_sum_label as make_exp_sum_label  # noqa: E402
from scripts._io import load_rpca_1d_list  # noqa: E402
from scripts._io import rpca_label as make_rpca_label  # noqa: E402
from src.approximation.cp_coulomb import (
    apply_cp_rho_E,
    apply_cp_rho_E_rpca,
    apply_cp_rho_V,
    apply_cp_rho_V_rpca,
)  # noqa: E402
from src.approximation.torch_kernels import (
    apply_exp_sum_3d_full,
    apply_exp_sum_3d_lowrank,
    apply_exp_sum_3d_lowrank_naive,
    apply_exp_sum_3d_rpca,
    apply_exp_sum_3d_rpca_l_only,
    apply_exp_sum_3d_rpca_s_only,
    to_float64_tensor,
)  # noqa: E402
from src.potential.separable_density import (
    make_gaussian_density_terms,
    materialize_density_terms,
)  # noqa: E402
from src.utils.grid import build_coords_centered  # noqa: E402


DEFAULT_EXP_SUM_RANKS = list(range(1, 31))
DEFAULT_EXP_SUM_NONNEG = True
DEFAULT_EXP_SUM_MAX_ITER = 200000
DEFAULT_EXP_SUM_N_POINTS = 2000
DEFAULT_EXP_SUM_R_MIN = 1e-2
DEFAULT_RPCA_MAX_ITER = 2000
DEFAULT_RPCA_TOL = 1e-6


def _require_torch() -> Any:
    """PyTorch が利用可能な場合に torch モジュールを返す。"""
    if torch is None:
        raise ImportError(
            "PyTorch is required for timing benchmarks. "
            "Install it with `pip install -e \".[notebook]\"`."
        )
    return torch


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


def benchmark(
    fn: Callable[[], Any],
    n_warmup: int,
    n_inner: int,
    n_repeat: int,
) -> np.ndarray:
    """関数の実行時間を repeat ごとに測定する。

    Parameters
    ----------
    fn : Callable[[], Any]
        計測対象の引数なし関数。GPU が利用可能な場合、この関数の
        先頭と末尾で ``torch.cuda.synchronize()`` を呼ぶ想定。
    n_warmup : int
        timeit の前に実行するウォームアップ回数。
    n_inner : int
        1 repeat あたりの呼び出し回数。
    n_repeat : int
        独立測定の repeat 数。

    Returns
    -------
    times : np.ndarray, shape (n_repeat,)
        1 呼び出しあたりの実行時間 [秒]。

    Raises
    ------
    ValueError
        ``n_warmup < 0`` または ``n_inner <= 0`` または
        ``n_repeat <= 0`` の場合。
    """
    if n_warmup < 0:
        raise ValueError("n_warmup must be non-negative.")
    if n_inner <= 0:
        raise ValueError("n_inner must be positive.")
    if n_repeat <= 0:
        raise ValueError("n_repeat must be positive.")

    for _ in range(n_warmup):
        fn()

    raw = timeit.repeat(fn, number=n_inner, repeat=n_repeat)
    return np.asarray(raw, dtype=float) / n_inner


def _with_cuda_sync(fn: Callable[[], Any]) -> Callable[[], Any]:
    """GPU が利用可能な場合に CUDA 同期を挟む関数へ変換する。"""
    torch_mod = _require_torch()
    use_cuda = torch_mod.cuda.is_available()

    def wrapped() -> Any:
        if use_cuda:
            torch_mod.cuda.synchronize()
        result = fn()
        if use_cuda:
            torch_mod.cuda.synchronize()
        return result

    return wrapped


def _make_exp_sum_label(args: argparse.Namespace) -> str:
    """既定設定から exp-sum の保存ラベルを作る。"""
    r_max = 2 * np.sqrt(3) * args.L
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


def _load_exp_sum_arrays(
    exp_sum_dir: Path,
    exp_sum_rank: int,
) -> tuple[np.ndarray, np.ndarray]:
    """exp-sum の weights / alphas を .npy から読み込む。

    Parameters
    ----------
    exp_sum_dir : pathlib.Path
        ``run_exp_sum_fitting.py`` の出力ディレクトリ。
    exp_sum_rank : int
        読み込む指数和 rank。

    Returns
    -------
    weights : np.ndarray, shape (exp_sum_rank,)
        指数和の重み。
    alphas : np.ndarray, shape (exp_sum_rank,)
        Gaussian 幅パラメータ。

    Raises
    ------
    FileNotFoundError
        必要な .npy が存在しない場合。
    """
    weights_path = exp_sum_dir / f"R{exp_sum_rank:02d}_weights.npy"
    alphas_path = exp_sum_dir / f"R{exp_sum_rank:02d}_alphas.npy"
    if not weights_path.exists() or not alphas_path.exists():
        raise FileNotFoundError(
            "exp-sum の .npy ファイルが見つかりません。先に "
            "`python scripts/run_exp_sum_fitting.py` を実行してください: "
            f"{exp_sum_dir}"
        )
    return np.load(weights_path), np.load(alphas_path)


def _load_k_1d_list(rpca_dir: Path, n_terms: int) -> list[np.ndarray]:
    """RPCA 出力ディレクトリから full 1D カーネルを読み込む。

    Parameters
    ----------
    rpca_dir : pathlib.Path
        ``run_rpca_kernels.py`` の出力ディレクトリ。
    n_terms : int
        読み込む指数和の項数。

    Returns
    -------
    kernels : list[np.ndarray]
        各項の 1D Gaussian カーネル行列。

    Raises
    ------
    FileNotFoundError
        必要な .npy が存在しない場合。
    """
    kernels = []
    for k in range(n_terms):
        kernels.append(np.load(rpca_dir / f"k{k:02d}_K1d.npy"))
    return kernels


def _validate_rank(
    rpca_1d_list: list[dict[str, np.ndarray]],
    rank: int,
) -> None:
    """要求 rank が読み込み済みデータで利用可能か確認する。"""
    if rank <= 0:
        raise ValueError("r_bench must be positive.")
    available_rank = min(
        min(k_data["S_s"].shape[0], k_data["S_L"].shape[0])
        for k_data in rpca_1d_list
    )
    if rank > available_rank:
        raise ValueError(
            f"requested r_bench {rank} exceeds available rank "
            f"{available_rank}."
        )


def _prepare_torch_inputs(
    weights: np.ndarray,
    k_1d_list: list[np.ndarray],
    rpca_1d_list: list[dict[str, np.ndarray]],
    r_bench: int,
    tau_bench: float,
) -> tuple[list, list, list, list, int, int]:
    """notebook と同じ変換で PyTorch 入力データを作る。

    Parameters
    ----------
    weights : np.ndarray, shape (K,)
        指数和の重み。
    k_1d_list : list[np.ndarray]
        full 1D Gaussian カーネル行列。
    rpca_1d_list : list[dict[str, np.ndarray]]
        RPCA / SVD 分解済み 1D カーネルデータ。
    r_bench : int
        SVD / RPCA のベンチマーク用 rank。
    tau_bench : float
        RPCA の S 成分をゼロ化する閾値。

    Returns
    -------
    full_kernels_pt : list
        full 実装用の PyTorch 入力。
    lowrank_data_pt : list
        naive / lowrank 実装用の PyTorch 入力。
    rpca_lowrank_only_data_pt : list
        S がゼロの RPCA 項。
    rpca_dense_data_pt : list
        S が非ゼロの RPCA 項。
    lowrank_only_count : int
        S がゼロの項数。
    dense_count : int
        S が非ゼロの項数。
    """
    rpca_dense_data_pt = []
    rpca_lowrank_only_data_pt = []
    lowrank_data_pt = []
    full_kernels_pt = []
    lowrank_only_count = 0
    dense_count = 0

    for weight, kernel, k_data in zip(weights, k_1d_list, rpca_1d_list):
        w_k = float(weight)

        ur = k_data["U_s"][:, :r_bench]
        sr = k_data["S_s"][:r_bench]
        vtr = k_data["Vt_s"][:r_bench, :]

        ur_l = k_data["U_L"][:, :r_bench]
        sr_l = k_data["S_L"][:r_bench]
        vtr_l = k_data["Vt_L"][:r_bench, :]

        s_1d = k_data["S_1d"]
        s_thr = np.where(np.abs(s_1d) > tau_bench, s_1d, 0.0)
        nnz = np.count_nonzero(s_thr)

        lowrank_data_pt.append(
            (
                w_k,
                to_float64_tensor(ur),
                to_float64_tensor(sr),
                to_float64_tensor(vtr),
            )
        )
        full_kernels_pt.append((w_k, to_float64_tensor(kernel)))

        ur_l_pt = to_float64_tensor(ur_l)
        sr_l_pt = to_float64_tensor(sr_l)
        vtr_l_pt = to_float64_tensor(vtr_l)
        if nnz == 0:
            rpca_lowrank_only_data_pt.append(
                (w_k, ur_l_pt, sr_l_pt, vtr_l_pt)
            )
            lowrank_only_count += 1
        else:
            rpca_dense_data_pt.append(
                (
                    w_k,
                    ur_l_pt,
                    sr_l_pt,
                    vtr_l_pt,
                    to_float64_tensor(s_thr),
                )
            )
            dense_count += 1

    return (
        full_kernels_pt,
        lowrank_data_pt,
        rpca_lowrank_only_data_pt,
        rpca_dense_data_pt,
        lowrank_only_count,
        dense_count,
    )


def _print_result(name: str, values: np.ndarray) -> None:
    """測定結果の要約を表示する。"""
    mean = values.mean()
    std = values.std(ddof=1) if values.size > 1 else 0.0
    print(f"{name:12s}: mean={mean:.6e} s, std={std:.6e} s")


def main() -> None:
    """時間計測を実行し、各 repeat の秒数を .npy に保存する。"""
    args = _parse_args()
    _require_torch()

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

    weights, alphas = _load_exp_sum_arrays(exp_sum_dir, args.exp_sum_rank)
    if len(weights) != args.exp_sum_rank or len(alphas) != args.exp_sum_rank:
        raise ValueError(
            "exp-sum rank と読み込んだ weights / alphas の長さが "
            f"一致しません: rank={args.exp_sum_rank}, "
            f"len(weights)={len(weights)}, len(alphas)={len(alphas)}."
        )

    rpca_1d_list = load_rpca_1d_list(rpca_dir, n_terms=args.exp_sum_rank)
    k_1d_list = _load_k_1d_list(rpca_dir, n_terms=args.exp_sum_rank)
    _validate_rank(rpca_1d_list, args.r_bench)

    dx, x_axis = build_coords_centered(args.N, args.L)
    rho_terms = make_gaussian_density_terms(
        x_axis,
        args.density_alpha,
    )
    rho_grid = materialize_density_terms(rho_terms)
    rho_pt = to_float64_tensor(rho_grid)
    rho_terms_pt = [
        (
            term.coefficient,
            to_float64_tensor(term.fx),
            to_float64_tensor(term.fy),
            to_float64_tensor(term.fz),
        )
        for term in rho_terms
    ]

    (
        full_kernels_pt,
        lowrank_data_pt,
        rpca_lowrank_only_data_pt,
        rpca_dense_data_pt,
        lowrank_only_count,
        dense_count,
    ) = _prepare_torch_inputs(
        weights=weights,
        k_1d_list=k_1d_list,
        rpca_1d_list=rpca_1d_list,
        r_bench=args.r_bench,
        tau_bench=args.tau_bench,
    )
    kernel_list_pt = [
        (float(w), to_float64_tensor(kernel))
        for w, kernel in zip(weights, k_1d_list)
    ]

    print(
        f"rpca path: dense={dense_count}, "
        f"lowrank_only={lowrank_only_count} (backend = PyTorch)"
    )

    targets: dict[str, Callable[[], Any]] = {
        "full": lambda: apply_exp_sum_3d_full(rho_pt, full_kernels_pt),
        "naive": lambda: apply_exp_sum_3d_lowrank_naive(
            rho_pt,
            lowrank_data_pt,
        ),
        "lowrank": lambda: apply_exp_sum_3d_lowrank(
            rho_pt,
            lowrank_data_pt,
        ),
        "rpca": lambda: apply_exp_sum_3d_rpca(
            rho_pt,
            rpca_lowrank_only_data_pt,
            rpca_dense_data_pt,
        ),
        "rpca_l": lambda: apply_exp_sum_3d_rpca_l_only(
            rho_pt,
            rpca_lowrank_only_data_pt,
            rpca_dense_data_pt,
        ),
        "rpca_s": lambda: apply_exp_sum_3d_rpca_s_only(
            rho_pt,
            rpca_dense_data_pt,
        ),
        "cp_v": lambda: apply_cp_rho_V(rho_terms_pt, kernel_list_pt),
        "cp_e": lambda: apply_cp_rho_E(rho_terms_pt, kernel_list_pt, dx),
        "cp_rpca_v": lambda: apply_cp_rho_V_rpca(
            rho_terms_pt,
            rpca_lowrank_only_data_pt,
            rpca_dense_data_pt,
        ),
        "cp_rpca_e": lambda: apply_cp_rho_E_rpca(
            rho_terms_pt,
            rpca_lowrank_only_data_pt,
            rpca_dense_data_pt,
            dx,
        ),
    }

    output_base = _resolve_output_base(args.output_dir)
    out_dir = output_base / exp_sum_label / rpca_label / _sweep_label(args)
    out_dir.mkdir(parents=True, exist_ok=True)

    results: dict[str, np.ndarray] = {}
    for name, fn in targets.items():
        print(f"benchmarking {name} ...")
        values = benchmark(
            _with_cuda_sync(fn),
            n_warmup=args.n_warmup,
            n_inner=args.n_inner,
            n_repeat=args.n_repeat,
        )
        results[name] = values
        np.save(out_dir / f"t_{name}.npy", values)
        _print_result(name, values)

    torch_mod = _require_torch()
    params_json = {
        "N": args.N,
        "L": args.L,
        "dx": dx,
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
        "rpca_dense_count": dense_count,
        "rpca_lowrank_only_count": lowrank_only_count,
        "exp_sum_label_defaults": {
            "ranks": DEFAULT_EXP_SUM_RANKS,
            "nonneg": DEFAULT_EXP_SUM_NONNEG,
            "max_iter": DEFAULT_EXP_SUM_MAX_ITER,
            "n_points": DEFAULT_EXP_SUM_N_POINTS,
            "r_min": DEFAULT_EXP_SUM_R_MIN,
            "r_max": 2 * np.sqrt(3) * args.L,
        },
        "rpca_label_defaults": {
            "rpca_rank": args.N // 4,
            "rpca_max_iter": DEFAULT_RPCA_MAX_ITER,
            "rpca_tol": DEFAULT_RPCA_TOL,
        },
        "torch": {
            "version": torch_mod.__version__,
            "cuda_available": torch_mod.cuda.is_available(),
            "cuda_device_count": torch_mod.cuda.device_count(),
        },
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
