"""実験スクリプト間で共有する入出力ヘルパー。"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def exp_sum_label(
    *,
    L: float,
    ranks: list[int],
    nonneg: bool,
    max_iter: int,
    n_points: int,
    r_min: float,
    r_max: float,
) -> str:
    """exp-sum フィット結果の出力ディレクトリ名を作る。

    Parameters
    ----------
    L
        計算領域の一辺の長さ。
    ranks
        フィット対象の指数和 rank。
    nonneg
        非負制約付きでフィットしたかどうか。
    max_iter
        VARPRO 最適化の最大反復数。
    n_points
        フィットグリッドの点数。
    r_min
        フィット範囲の下限。
    r_max
        フィット範囲の上限。

    Returns
    -------
    str
        保存先に使う安定したラベル。
    """
    return (
        f"L{L:g}_rmin{r_min:.0e}_rmax{r_max:.6g}"
        f"_n{n_points}_nonneg{int(nonneg)}"
        f"_iter{max_iter}"
        f"_R{min(ranks):02d}-{max(ranks):02d}"
        f"_count{len(ranks)}"
    )


def rpca_label(
    *,
    N: int,
    L: float,
    exp_sum_R: int,
    rpca_rank: int,
    rpca_max_iter: int,
    rpca_tol: float,
) -> str:
    """RPCA カーネル結果の出力ディレクトリ名を作る。

    Parameters
    ----------
    N
        1 軸あたりの格子点数。
    L
        計算領域の一辺の長さ。
    exp_sum_R
        入力に使う指数和 rank。
    rpca_rank
        randomized RPCA の近似 rank。
    rpca_max_iter
        RPCA の最大反復数。
    rpca_tol
        RPCA の収束判定しきい値。

    Returns
    -------
    str
        保存先に使う安定したラベル。
    """
    return (
        f"N{N}_L{L:g}_R{exp_sum_R:02d}_rank{rpca_rank}"
        f"_iter{rpca_max_iter}_tol{rpca_tol:.0e}"
    )


def load_rpca_1d_list(
    rpca_dir: Path,
    n_terms: int,
) -> list[dict[str, np.ndarray]]:
    """RPCA / SVD 1D カーネル分解の .npy 群を読み込む。

    Parameters
    ----------
    rpca_dir
        `run_rpca_kernels.py` が出力したディレクトリ。
    n_terms
        読み込む指数和の項数。

    Returns
    -------
    list[dict[str, np.ndarray]]
        各項の sparse 成分、RPCA 低ランク成分、通常 SVD 成分。
    """
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
