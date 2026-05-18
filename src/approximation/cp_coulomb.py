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
) -> list:
    """CP形式の密度から指数和近似によるポテンシャルを CP 形式で返す。

    各指数和項で 1D matvec を実行し、(係数, vx, vy, vz) タプルのリストを
    返す。N³ テンソルの具現化は行わない。計算量は O(K M N²)。

    Parameters
    ----------
    rho_terms_pt : list of (float, fx, fy, fz)
        CP 形式の密度項リスト。fx, fy, fz は shape (N,) の 1D テンソル。
    kernel_list : list of (float, torch.Tensor shape (N, N))
        各指数和項の重みとフル 1D カーネル行列のリスト。

    Returns
    -------
    cp_terms : list of (float, vx, vy, vz)
        CP 形式のポテンシャル項リスト。vx, vy, vz は shape (N,) のテンソル。
        N³ テンソルへの具現化には :func:`materialize_cp_potential` を使う。
    """
    result = []
    for w_k, K_k in kernel_list:
        for c_m, fx, fy, fz in rho_terms_pt:
            result.append((
                float(w_k * c_m),
                K_k @ fx,
                K_k @ fy,
                K_k @ fz,
            ))
    return result


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
        各指数和項の重みとフル 1D カーネル行列のリスト。
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
) -> list:
    """CP形式の密度から rPCA カーネルによるポテンシャルを CP 形式で返す。

    1D カーネルに rPCA 分解 K ≈ L+S を適用し、CP-ρ の 1D matvec と
    組み合わせる。L 成分は O(rN)、S 成分は O(N²) の matvec。
    N³ テンソルの具現化は行わない。

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
    cp_terms : list of (float, vx, vy, vz)
        CP 形式のポテンシャル項リスト。vx, vy, vz は shape (N,) のテンソル。
        L 成分は O(K_L M rN)、S 成分は O(K_S M N²) で計算される。
        N³ テンソルへの具現化には :func:`materialize_cp_potential` を使う。
    """
    result = []
    for w_k, U_L, s_L, Vt_L in lowrank_only_list:
        for c_m, fx, fy, fz in rho_terms_pt:
            result.append((
                float(w_k * c_m),
                U_L @ (s_L * (Vt_L @ fx)),
                U_L @ (s_L * (Vt_L @ fy)),
                U_L @ (s_L * (Vt_L @ fz)),
            ))
    for w_k, U_L, s_L, Vt_L, S_dense in dense_list:
        for c_m, fx, fy, fz in rho_terms_pt:
            result.append((
                float(w_k * c_m),
                U_L @ (s_L * (Vt_L @ fx)) + S_dense @ fx,
                U_L @ (s_L * (Vt_L @ fy)) + S_dense @ fy,
                U_L @ (s_L * (Vt_L @ fz)) + S_dense @ fz,
            ))
    return result


def apply_cp_rho_E_rpca(
    rho_terms_pt: list,
    lowrank_only_list: list,
    dense_list: list,
    dx: float,
) -> Any:
    """CP形式の密度から rPCA カーネルを用いてエネルギーを直接計算する。

    L 成分は r 次元射影をプリコンピュートして r 次元内積に帰着させる。
    N 次元中間ベクトルを経由しないため V テンソルの具現化も不要。
    計算量は O(K_L M rN + K_L M^2 r + K_S M^2 N^2)。

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
        # r次元射影をプリコンピュート: Vt_L @ fx → shape (r,)
        px = [Vt_L @ fx for _, fx, _, _ in rho_terms_pt]
        py = [Vt_L @ fy for _, _, fy, _ in rho_terms_pt]
        pz = [Vt_L @ fz for _, _, _, fz in rho_terms_pt]
        # U_L^T @ fx → shape (r,)（U_L は (N,r) なので転置は (r,N)）
        qx = [U_L.T @ fx for _, fx, _, _ in rho_terms_pt]
        qy = [U_L.T @ fy for _, _, fy, _ in rho_terms_pt]
        qz = [U_L.T @ fz for _, _, _, fz in rho_terms_pt]

        for i, (c_m, _, _, _) in enumerate(rho_terms_pt):
            sp_x = s_L * px[i]  # r次元: s ⊙ (Vt_L fx_m)
            sp_y = s_L * py[i]
            sp_z = s_L * pz[i]
            for j, (c_n, _, _, _) in enumerate(rho_terms_pt):
                # (fx_n · U_L (s ⊙ Vt_L fx_m)) = qx[j] · sp_x: O(r)
                E = E + (
                    w_k * c_m * c_n
                    * (qx[j] @ sp_x)
                    * (qy[j] @ sp_y)
                    * (qz[j] @ sp_z)
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


def materialize_cp_potential(
    cp_terms: list,
    dx: float,
) -> Any:
    """CP形式のポテンシャルを N³ テンソルに具現化する。

    :func:`apply_cp_rho_V` または :func:`apply_cp_rho_V_rpca` の出力を
    受け取り、dx^3 の積分体積要素を掛けた完全な 3D テンソルを返す。
    計算量は O(T N³)（T は CP 項数）。

    Parameters
    ----------
    cp_terms : list of (float, vx, vy, vz)
        CP 形式のポテンシャル項リスト。vx, vy, vz は shape (N,) のテンソル。
    dx : float
        グリッド間隔（積分体積要素 dx^3 の計算に使用）。

    Returns
    -------
    V : torch.Tensor, shape (N, N, N)
        ポテンシャルテンソル（dx^3 の積分体積要素を含む）。
    """
    _torch = _require_torch()
    N = cp_terms[0][1].shape[0]
    V = _torch.zeros(N, N, N, dtype=_torch.float64)
    for coeff, vx, vy, vz in cp_terms:
        V.add_(_torch.einsum('i,j,k->ijk', vx, vy, vz), alpha=coeff)
    return V * (dx ** 3)
