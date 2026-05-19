import numpy as np

from src.approximation.exp_sum.models import ExponentialSum
from src.potential.gaussian_discretization import (
    analytic_gaussian_hartree_energy,
    compare_exp_sum_discretization,
    compute_exp_sum_energy_error,
    compute_rpca_error_sweep,
)


def test_analytic_gaussian_hartree_energy_matches_closed_form():
    """alpha=1のGaussian自己エネルギーは閉形式の値に一致する。"""
    expected = np.pi**2.5 / np.sqrt(2.0)

    actual = analytic_gaussian_hartree_energy(alpha=1.0)

    assert actual == expected


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

    rank_rows = compare_exp_sum_discretization(
        N_values=[5, 7],
        L=6.0,
        alpha=1.0,
        fits=fits,
        ranks=[1, 2],
    )

    assert set(rank_rows) == {1, 2}
    assert [row.N for row in rank_rows[1]] == [5, 7]
    assert [row.rank for row in rank_rows[2]] == [2, 2]


def test_compute_rpca_error_sweep_matches_identity_kernel():
    """恒等1Dカーネルでは3Dポテンシャル誤差が0になる。"""
    fit = ExponentialSum(weights=np.array([1.0]), alphas=np.array([1.0]))
    identity = np.eye(1)
    rpca_1d_list = [
        {
            "S_1d": np.zeros((1, 1)),
            "U_L": identity,
            "S_L": np.ones(1),
            "Vt_L": identity,
            "U_s": identity,
            "S_s": np.ones(1),
            "Vt_s": identity,
        }
    ]
    rho_grid = np.ones((1, 1, 1))
    v_analytic = np.ones((1, 1, 1))

    results = compute_rpca_error_sweep(
        fit=fit,
        rpca_1d_list=rpca_1d_list,
        rho_grid=rho_grid,
        v_analytic=v_analytic,
        dx=1.0,
        k_diag_true=1.0,
        svd_ranks=[1],
        thresholds=[1e-3],
    )

    np.testing.assert_allclose(results["errors_v_svd_only"], [0.0])
    np.testing.assert_allclose(results["errors_e_svd_only"], [0.0])
    np.testing.assert_allclose(results["errors_v_rpca"], [0.0])
    np.testing.assert_allclose(results["errors_e_rpca"], [0.0])
    np.testing.assert_allclose(
        results["errors_v_rpca_thresh"]["1e-03"],
        [0.0],
    )
    np.testing.assert_allclose(
        results["errors_e_rpca_thresh"]["1e-03"],
        [0.0],
    )
