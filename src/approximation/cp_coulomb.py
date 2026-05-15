"""CP形式の密度に対する Coulomb 相互作用の PyTorch 実装。"""

from __future__ import annotations

from typing import Any

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    torch = None  # type: ignore[assignment]


def _require_torch() -> Any:
    """PyTorch が利用可能な場合に torch モジュールを返す。"""
    if torch is None:
        raise ImportError(
            "PyTorch is required for src.approximation.cp_coulomb. "
            "Install the notebook optional dependencies to use this module."
        )
    return torch


def apply_cp_rho_V(
    rho_terms_pt: list,
    kernel_list: list,
) -> Any:
    """CP形式の密度から指数和近似によるポテンシャルを計算する。

    各指数和項で 1D matvec を実行し、外積で N^3 の V テンソルを
    具現化して和をとる。計算量は O(K M N^3)。

    Parameters
    ----------
    rho_terms_pt : list of (float, fx, fy, fz)
        CP 形式の密度項リスト。fx, fy, fz は shape (N,) の 1D テンソル。
    kernel_list : list of (float, torch.Tensor shape (N, N))
        各指数和項の重みとフル1Dカーネル行列のリスト。

    Returns
    -------
    V : torch.Tensor, shape (N, N, N)
        ポテンシャルテンソル（dx^3 の積分体積要素・対角補正は含まない）。
    """
    _torch = _require_torch()
    N = rho_terms_pt[0][1].shape[0]
    V = _torch.zeros(N, N, N, dtype=_torch.float64)
    for w_k, K_k in kernel_list:
        for c_m, fx, fy, fz in rho_terms_pt:
            vx = K_k @ fx
            vy = K_k @ fy
            vz = K_k @ fz
            V.add_(_torch.einsum('i,j,k->ijk', vx, vy, vz), alpha=float(w_k * c_m))
    return V


def apply_cp_rho_E(
    rho_terms_pt: list,
    kernel_list: list,
    dx: float,
) -> Any:
    """CP形式の密度からハートリーエネルギーを直接計算する。

    V テンソルを具現化せず、1D 内積の積だけでエネルギーを求める。
    計算量は O(K M^2 N^2)。

    Parameters
    ----------
    rho_terms_pt : list of (float, fx, fy, fz)
        CP 形式の密度項リスト。fx, fy, fz は shape (N,) の 1D テンソル。
    kernel_list : list of (float, torch.Tensor shape (N, N))
        各指数和項の重みとフル1Dカーネル行列のリスト。
    dx : float
        グリッド間隔（積分体積要素 dx^6 の計算に使用）。

    Returns
    -------
    E : torch.Tensor, scalar
        ハートリーエネルギー E = (dx^6/2) Σ_k w_k Σ_{m,n} c_m c_n
        <ρn_x, K_k ρm_x> <ρn_y, K_k ρm_y> <ρn_z, K_k ρm_z>。

    Notes
    -----
    対角補正は含まない。
    """
    _torch = _require_torch()
    E = _torch.tensor(0.0, dtype=_torch.float64)
    for w_k, K_k in kernel_list:
        for c_m, fx_m, fy_m, fz_m in rho_terms_pt:
            vx = K_k @ fx_m
            vy = K_k @ fy_m
            vz = K_k @ fz_m
            for c_n, fx_n, fy_n, fz_n in rho_terms_pt:
                E = E + (
                    w_k * c_m * c_n
                    * (fx_n @ vx)
                    * (fy_n @ vy)
                    * (fz_n @ vz)
                )
    return 0.5 * (dx ** 6) * E


def apply_cp_rho_V_rpca(
    rho_terms_pt: list,
    lowrank_only_list: list,
    dense_list: list,
) -> Any:
    """CP形式の密度から rPCA カーネルによるポテンシャルを計算する。

    1D カーネルに rPCA 分解 K ≈ L+S を適用し、CP-ρ の 1D matvec と
    組み合わせる。L 成分は O(rN)、S 成分は O(N²) の matvec。
    外積の具現化 O(N³) がボトルネック。

    Parameters
    ----------
    rho_terms_pt : list of (float, fx, fy, fz)
        CP 形式の密度項リスト。fx, fy, fz は shape (N,) の 1D テンソル。
    lowrank_only_list : list of (float, U_L, s_L, Vt_L)
        S=0 の項の重みと L 成分の低ランク因子のリスト。
    dense_list : list of (float, U_L, s_L, Vt_L, S_dense)
        S≠0 の項の重みと L 成分の低ランク因子および密行列 S のリスト。

    Returns
    -------
    V : torch.Tensor, shape (N, N, N)
        ポテンシャルテンソル（dx^3 の積分体積要素・対角補正は含まない）。
    """
    _torch = _require_torch()
    N = rho_terms_pt[0][1].shape[0]
    V = _torch.zeros(N, N, N, dtype=_torch.float64)
    for w_k, U_L, s_L, Vt_L in lowrank_only_list:
        for c_m, fx, fy, fz in rho_terms_pt:
            vx = U_L @ (s_L * (Vt_L @ fx))
            vy = U_L @ (s_L * (Vt_L @ fy))
            vz = U_L @ (s_L * (Vt_L @ fz))
            V.add_(_torch.einsum('i,j,k->ijk', vx, vy, vz), alpha=float(w_k * c_m))
    for w_k, U_L, s_L, Vt_L, S_dense in dense_list:
        for c_m, fx, fy, fz in rho_terms_pt:
            vx = U_L @ (s_L * (Vt_L @ fx)) + S_dense @ fx
            vy = U_L @ (s_L * (Vt_L @ fy)) + S_dense @ fy
            vz = U_L @ (s_L * (Vt_L @ fz)) + S_dense @ fz
            V.add_(_torch.einsum('i,j,k->ijk', vx, vy, vz), alpha=float(w_k * c_m))
    return V


def apply_cp_rho_E_rpca(
    rho_terms_pt: list,
    lowrank_only_list: list,
    dense_list: list,
    dx: float,
) -> Any:
    """CP形式の密度から rPCA カーネルを用いてエネルギーを直接計算する。

    1D カーネルに rPCA 分解 K ≈ L+S を適用し、CP-ρ の 1D 内積と
    組み合わせる。V テンソルの具現化は不要。
    計算量は O(K_L r M^2 N + K_S N^2 M^2)。

    Parameters
    ----------
    rho_terms_pt : list of (float, fx, fy, fz)
        CP 形式の密度項リスト。fx, fy, fz は shape (N,) の 1D テンソル。
    lowrank_only_list : list of (float, U_L, s_L, Vt_L)
        S=0 の項の重みと L 成分の低ランク因子のリスト。
    dense_list : list of (float, U_L, s_L, Vt_L, S_dense)
        S≠0 の項の重みと L 成分の低ランク因子および密行列 S のリスト。
    dx : float
        グリッド間隔（積分体積要素 dx^6 の計算に使用）。

    Returns
    -------
    E : torch.Tensor, scalar
        ハートリーエネルギー（対角補正は含まない）。
    """
    _torch = _require_torch()
    E = _torch.tensor(0.0, dtype=_torch.float64)
    for w_k, U_L, s_L, Vt_L in lowrank_only_list:
        for c_m, fx_m, fy_m, fz_m in rho_terms_pt:
            vx = U_L @ (s_L * (Vt_L @ fx_m))
            vy = U_L @ (s_L * (Vt_L @ fy_m))
            vz = U_L @ (s_L * (Vt_L @ fz_m))
            for c_n, fx_n, fy_n, fz_n in rho_terms_pt:
                E = E + (
                    w_k * c_m * c_n
                    * (fx_n @ vx)
                    * (fy_n @ vy)
                    * (fz_n @ vz)
                )
    for w_k, U_L, s_L, Vt_L, S_dense in dense_list:
        for c_m, fx_m, fy_m, fz_m in rho_terms_pt:
            vx = U_L @ (s_L * (Vt_L @ fx_m)) + S_dense @ fx_m
            vy = U_L @ (s_L * (Vt_L @ fy_m)) + S_dense @ fy_m
            vz = U_L @ (s_L * (Vt_L @ fz_m)) + S_dense @ fz_m
            for c_n, fx_n, fy_n, fz_n in rho_terms_pt:
                E = E + (
                    w_k * c_m * c_n
                    * (fx_n @ vx)
                    * (fy_n @ vy)
                    * (fz_n @ vz)
                )
    return 0.5 * (dx ** 6) * E
