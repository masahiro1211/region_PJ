"""pkl ベースの計算結果キャッシュ.

パラメータ辞書から一意なキー (SHA256) を生成し、結果を pkl で保存・再利用する.
重い計算 (VARPRO 指数和フィット, RPCA 分解など) を再実行せずに使い回すことを想定。
"""

from __future__ import annotations

import hashlib
import pickle
from pathlib import Path
from typing import Any, Callable

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CACHE_ROOT = PROJECT_ROOT / "data" / "processed" / "cache"


def _canonical_bytes(value: Any) -> bytes:
    if isinstance(value, np.ndarray):
        return (
            b"ndarray:"
            + str(value.shape).encode()
            + b":"
            + str(value.dtype).encode()
            + b":"
            + value.tobytes()
        )
    if isinstance(value, (list, tuple)):
        kind = b"list:" if isinstance(value, list) else b"tuple:"
        return kind + b"|".join(_canonical_bytes(v) for v in value)
    if isinstance(value, dict):
        return b"dict:" + b"|".join(
            repr(k).encode() + b"=" + _canonical_bytes(v)
            for k, v in sorted(value.items(), key=lambda kv: repr(kv[0]))
        )
    if isinstance(value, (int, float, bool, str, bytes)) or value is None:
        return repr(value).encode()
    return repr(value).encode()


def make_key(params: dict) -> str:
    return hashlib.sha256(_canonical_bytes(params)).hexdigest()[:16]


def cache_path(namespace: str, params: dict, root: Path | None = None) -> Path:
    folder = (root or CACHE_ROOT) / namespace
    folder.mkdir(parents=True, exist_ok=True)
    return folder / f"{make_key(params)}.pkl"


def load_or_compute(
    namespace: str,
    params: dict,
    compute: Callable[[], Any],
    root: Path | None = None,
    verbose: bool = True,
    force: bool = False,
) -> Any:
    path = cache_path(namespace, params, root)
    if path.exists() and not force:
        if verbose:
            print(f"[cache hit ] {namespace}/{path.name}")
        with open(path, "rb") as f:
            return pickle.load(f)
    if verbose:
        print(f"[cache miss] {namespace}/{path.name} ... computing")
    result = compute()
    with open(path, "wb") as f:
        pickle.dump(result, f)
    if verbose:
        print(f"[cache save] {namespace}/{path.name}")
    return result
