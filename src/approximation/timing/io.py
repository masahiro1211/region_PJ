"""時間計測スクリプト用の .npy 入力読み込み。"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def load_exp_sum_arrays(
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


def load_k_1d_list(rpca_dir: Path, n_terms: int) -> list[np.ndarray]:
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
