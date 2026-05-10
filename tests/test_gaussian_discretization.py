import numpy as np

from src.approximation.exp_sum import ExponentialSum
from src.potential.gaussian_discretization import (
    analytic_gaussian_hartree_energy,
    compare_exp_sum_discretization,
    compute_exp_sum_energy_error,
    compute_grid_energy_reference,
)


def test_analytic_gaussian_hartree_energy_matches_closed_form():
    """alpha=1のGaussian自己エネルギーは閉形式の値に一致する。"""
    expected = np.pi**2.5 / np.sqrt(2.0)

    actual = analytic_gaussian_hartree_energy(alpha=1.0)

    assert actual == expected


def test_compute_grid_energy_reference_returns_finite_errors():
    """グリッド参照エネルギーを計算したとき、誤差指標は有限値になる。"""
    reference = compute_grid_energy_reference(N=5, L=6.0, alpha=1.0)

    assert reference.N == 5
    assert reference.dx == 6.0 / 5
    assert np.isfinite(reference.energy_grid)
    assert np.isfinite(reference.energy_exact)
    assert reference.abs_error >= 0.0
    assert reference.rel_error >= 0.0


def test_compute_exp_sum_energy_error_returns_rank_and_finite_errors():
    """指数和近似エネルギーを評価したとき、rankと有限な誤差が返る。"""
    fit = ExponentialSum(
        weights=np.array([1.0, 0.25]),
        alphas=np.array([0.2, 1.5]),
    )

    row = compute_exp_sum_energy_error(fit=fit, N=5, L=6.0, alpha=1.0)

    assert row.rank == 2
    assert row.N == 5
    assert np.isfinite(row.energy_exp_sum)
    assert row.err_vs_grid_ref >= 0.0
    assert row.err_vs_cont_exact >= 0.0


def test_compare_exp_sum_discretization_groups_rows_by_rank():
    """複数Nと複数rankを比較したとき、rankごとに結果がまとまる。"""
    fits = {
        1: ExponentialSum(
            weights=np.array([1.0]),
            alphas=np.array([0.5]),
        ),
        2: ExponentialSum(
            weights=np.array([1.0, 0.25]),
            alphas=np.array([0.2, 1.5]),
        ),
    }

    grid_rows, rank_rows = compare_exp_sum_discretization(
        N_values=[5, 7],
        L=6.0,
        alpha=1.0,
        fits=fits,
        ranks=[1, 2],
    )

    assert [row.N for row in grid_rows] == [5, 7]
    assert set(rank_rows) == {1, 2}
    assert [row.N for row in rank_rows[1]] == [5, 7]
    assert [row.rank for row in rank_rows[2]] == [2, 2]
