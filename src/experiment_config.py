"""experiment_config.sh を Python から読むためのユーティリティ。

``experiment_config.sh`` を単一の真実の源とし、Python スクリプトや
ノートブックはここ経由で実験条件を取得する。
"""

from __future__ import annotations

import re
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_CONFIG_PATH = _PROJECT_ROOT / "scripts" / "experiment_config.sh"


def _parse_bash_config(path: Path) -> dict[str, str]:
    """KEY=value 形式の bash 変数定義を辞書に読み込む。"""
    result: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        m = re.match(r'^([A-Z_][A-Z0-9_]*)=(.*)$', line)
        if m:
            result[m.group(1)] = m.group(2).strip()
    return result


def _get() -> dict[str, str]:
    return _parse_bash_config(_CONFIG_PATH)


def _float(key: str) -> float:
    return float(_get()[key])


def _int(key: str) -> int:
    return int(_get()[key])


def _str(key: str) -> str:
    return _get()[key]


# ---------------------------------------------------------------------------
# 公開プロパティ (呼び出すたびにファイルを読む → 常に最新値)
# ---------------------------------------------------------------------------

def N() -> int:
    """グリッド点数。"""
    return _int("EXP_N")


def L() -> float:
    """ボックス長。"""
    return _float("EXP_L")


def density_alpha() -> float:
    """密度ガウシアンの指数係数 α。"""
    return _float("EXP_DENSITY_ALPHA")


def cell_int_const() -> float:
    """対角補正係数 (∫_{cube} 1/r d³r / dx²)。"""
    return _float("EXP_CELL_INT_CONST")


def exp_sum_rank() -> int:
    """指数和のランク R。"""
    return _int("EXP_SUM_RANK")


def exp_sum_ranks() -> list[int]:
    """指数和フィッティングに使うランクのリスト。"""
    lo, hi = _str("EXP_SUM_RANKS").split(":")
    return list(range(int(lo), int(hi) + 1))


def exp_sum_nonneg() -> bool:
    """非負制約の有無。"""
    return "nonneg" in _str("EXP_SUM_NONNEG_OPT")


def exp_sum_max_iter() -> int:
    """VARPRO の最大イテレーション数。"""
    return _int("EXP_SUM_MAX_ITER")


def exp_sum_n_points() -> int:
    """フィッティンググリッドの点数。"""
    return _int("EXP_SUM_N_POINTS")


def exp_sum_r_min() -> float:
    """フィッティングの最小半径。"""
    return _float("EXP_SUM_R_MIN")


def rpca_max_iter() -> int:
    """RPCA の最大イテレーション数。"""
    return _int("RPCA_MAX_ITER")


def rpca_tol() -> float:
    """RPCA の収束許容誤差。"""
    return _float("RPCA_TOL")


def svd_ranks() -> list[int]:
    """SVD スイープに使うランクのリスト。"""
    lo, hi = _str("SVD_RANKS").split(":")
    return list(range(int(lo), int(hi) + 1))


def thresholds() -> list[float]:
    """S 閾値のリスト。"""
    return [float(t) for t in _str("THRESHOLDS").split(",")]


def timing_r_bench() -> int:
    """タイミングベンチマークの SVD/RPCA ランク r。"""
    return _int("TIMING_R_BENCH")


def timing_tau_bench() -> float:
    """タイミングベンチマークの RPCA 閾値 τ。"""
    return _float("TIMING_TAU_BENCH")


def timing_n_warmup() -> int:
    return _int("TIMING_N_WARMUP")


def timing_n_inner() -> int:
    return _int("TIMING_N_INNER")


def timing_n_repeat() -> int:
    return _int("TIMING_N_REPEAT")
