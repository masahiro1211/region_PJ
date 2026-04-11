import numpy as np

from src.potential.charge_potential import (
    ChargePotentialResult,
    run_charge_potential_demo,
)


def print_report(result: ChargePotentialResult) -> None:
    rows = result["rows"]

    print("=" * 55)
    print("  電荷ポテンシャル計算 + Tucker 近似誤差")
    print("=" * 55)
    print(
        f"  alpha={result['alpha']}, N={result['N']}^3, "
        f"L={result['L']}, dx={result['dx']}"
    )

    print("\n[1] 数値積分の精度（単一ガウシアン vs 解析解）")
    print(f"    ゲージ補正量 C : {result['baseline_shift']:.4f}")
    print(f"    相対誤差      : {result['baseline_error']:.4e}")

    print("\n[2] Tucker 近似 rho → V の誤差（複数ガウシアン、解析解比較）")
    print(
        f"    {'Rank':>5} | {'ρ 相対誤差':>12} | "
        f"{'V 相対誤差(解析解比)':>20} | {'圧縮率':>8}"
    )
    print("    " + "-" * 58)
    print(
        f"    {'ref':>5} | {0.0:>12.4e} | "
        f"{result['ref_error']:.4e} | {1.0000:>8.4f}"
    )

    for row in rows:
        print(
            f"    {row['rank']:>5} | {row['err_rho']:>12.4e} | "
            f"{row['err_v']:>20.4e} | {row['compression']:>8.4f}"
        )

    print()
    print("  ※ 圧縮率 < 1 なら Tucker 表現の方がコンパクト")
    print("=" * 55)


def main() -> None:
    alpha = 1.0
    N = 32
    L = 10.0
    centers = np.array([
        [0.0, 0.0, 0.0],
        [2.0, 0.0, 0.0],
        [0.0, 2.0, 0.0],
        [-1.5, -1.5, 0.0],
    ])
    weights = np.array([1.0, 0.6, 0.5, 0.4])
    ranks = [1, 2, 4, 8, 12, 16]

    result = run_charge_potential_demo(
        alpha=alpha,
        N=N,
        L=L,
        centers=centers,
        weights=weights,
        ranks=ranks,
    )
    print_report(result)


if __name__ == "__main__":
    main()
