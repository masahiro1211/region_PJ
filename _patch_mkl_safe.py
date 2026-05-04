"""cell 76fa9084 の MKL 検出を subprocess 事前検証付きの安全版に差し替える."""
import json
from pathlib import Path

NB_PATH = Path(r"c:\Users\masah\研究\region_PJ\notebooks\calc_potential_exp_expansion_approx.ipynb")
CELL_ID = "76fa9084"

NEW_HEAD = '''import os
import sys
import glob
import subprocess
from scipy import sparse


def _find_mkl_rt() -> str | None:
    env_path = os.environ.get("MKL_RT")
    if env_path and os.path.exists(env_path):
        return env_path

    prefixes: list[str] = []
    for key in ("CONDA_PREFIX", "MKLROOT"):
        val = os.environ.get(key)
        if val:
            prefixes.append(val)
    prefixes.extend([
        sys.prefix,
        sys.base_prefix,
        "/opt/intel/oneapi/mkl/latest",
        "/opt/intel/mkl",
        "/usr",
        "/usr/local",
        r"C:\\Program Files (x86)\\Intel\\oneAPI\\mkl\\latest",
        r"C:\\Program Files\\Intel\\oneAPI\\mkl\\latest",
    ])
    patterns = [
        "lib/libmkl_rt.so*", "lib64/libmkl_rt.so*",
        "lib/intel64/libmkl_rt.so*", "lib/intel64_lin/libmkl_rt.so*",
        "lib/x86_64-linux-gnu/libmkl_rt.so*",
        "lib/libmkl_rt.dylib",
        "Library/bin/mkl_rt.*.dll", "Library/bin/mkl_rt.dll",
        "redist/intel64/mkl_rt.*.dll", "redist/intel64/mkl_rt.dll",
    ]
    for prefix in prefixes:
        for pattern in patterns:
            for hit in glob.glob(os.path.join(prefix, pattern)):
                if os.path.exists(hit):
                    return hit
    return None


def _probe_mkl(mkl_rt: str, timeout: float = 20.0) -> tuple[bool, str]:
    """サブプロセスで sparse_dot_mkl を import + 1 回だけ計算してみる.

    親プロセスの ipykernel を巻き添えにせず、SIGSEGV / OSError / ImportError を
    全て returncode != 0 で検出できる。
    """
    code = (
        "import os, sys; "
        f"os.environ['MKL_RT'] = {mkl_rt!r}; "
        "import numpy as np; import scipy.sparse as sp; "
        "from sparse_dot_mkl import dot_product_mkl; "
        "A = sp.csr_matrix(np.eye(4)); "
        "B = np.ones((4, 4)); "
        "out = dot_product_mkl(A, B); "
        "assert out.shape == (4, 4); "
        "print('OK')"
    )
    try:
        r = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True, text=True, timeout=timeout,
        )
        if r.returncode == 0 and "OK" in r.stdout:
            return True, ""
        msg = (r.stderr or r.stdout or "").strip().splitlines()[-1:] or [f"returncode={r.returncode}"]
        return False, msg[0]
    except subprocess.TimeoutExpired:
        return False, "subprocess timed out"
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


# ---- MKL 検出 + 安全な事前検証 ----
USE_MKL = False
dot_product_mkl = None

if os.environ.get("FORCE_NO_MKL") == "1":
    print("[MKL] FORCE_NO_MKL=1 -> scipy.sparse fallback")
else:
    mkl_rt = _find_mkl_rt()
    if mkl_rt is None:
        print("[MKL] runtime not found -> scipy.sparse fallback")
    else:
        ok, err = _probe_mkl(mkl_rt)
        if not ok:
            print(f"[MKL] probe failed for {mkl_rt} -> scipy.sparse fallback ({err})")
        else:
            os.environ["MKL_RT"] = mkl_rt
            if hasattr(os, "add_dll_directory") and os.name == "nt":
                try:
                    os.add_dll_directory(os.path.dirname(mkl_rt))
                except (OSError, FileNotFoundError):
                    pass
            try:
                from sparse_dot_mkl import dot_product_mkl as _dpm
                dot_product_mkl = _dpm
                USE_MKL = True
                print(f"[MKL] using {mkl_rt}")
            except BaseException as e:
                print(f"[MKL] in-proc import failed -> scipy.sparse fallback ({type(e).__name__}: {e})")
'''


def main() -> None:
    nb = json.loads(NB_PATH.read_text(encoding="utf-8"))
    target = None
    for cell in nb["cells"]:
        if cell.get("id") == CELL_ID:
            target = cell
            break
    if target is None:
        raise RuntimeError(f"cell {CELL_ID} not found")

    src = "".join(target["source"])
    # 既存の "import os" 〜 "[MKL] runtime not found ..." までのヘッダ部を NEW_HEAD で置換
    marker = "# ---- 計測用パラメータ ----"
    if marker not in src:
        raise RuntimeError("marker not found, aborting")
    tail = src[src.index(marker):]
    new_src = NEW_HEAD + "\n\n" + tail

    target["source"] = [ln + "\n" for ln in new_src.split("\n")[:-1]] + (
        [new_src.split("\n")[-1]] if new_src.split("\n")[-1] else []
    )
    target["outputs"] = []
    target["execution_count"] = None
    NB_PATH.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"patched cell {CELL_ID}")


if __name__ == "__main__":
    main()
