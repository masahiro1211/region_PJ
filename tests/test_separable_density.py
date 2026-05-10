import numpy as np

from src.approximation.exp_sum import ExponentialSum, apply_separable_gaussian_3d
from src.potential.separable_density import (
    apply_exp_sum_to_separable_density,
    make_gaussian_density_terms,
    materialize_density_terms,
)
from src.utils.grid import build_xyz


def test_gaussian_density_terms_materialize_to_dense_gaussian():
    """Gaussian密度のCP表現をdense化したとき、直接評価と一致する。"""
    n = 7
    length = 4.0
    alpha = 0.8
    xyz = build_xyz(n, length)
    x_axis = xyz[0, :, 0, 0]
    radius2 = xyz[0] ** 2 + xyz[1] ** 2 + xyz[2] ** 2
    expected = np.exp(-alpha * radius2)

    terms = make_gaussian_density_terms(x_axis, alpha)
    actual = materialize_density_terms(terms)

    np.testing.assert_allclose(actual, expected, rtol=1e-14, atol=1e-14)


def test_separable_density_exp_sum_matches_axis_based_dense_application():
    """分離密度への指数和作用はdenseなmode積の結果と一致する。"""
    n = 7
    length = 4.0
    dx = length / n
    alpha_rho = 0.7
    xyz = build_xyz(n, length)
    x_axis = xyz[0, :, 0, 0]
    radius2 = xyz[0] ** 2 + xyz[1] ** 2 + xyz[2] ** 2
    rho = np.exp(-alpha_rho * radius2)
    fit = ExponentialSum(
        weights=np.array([0.4, 1.1]),
        alphas=np.array([0.2, 1.3]),
    )

    terms = make_gaussian_density_terms(x_axis, alpha_rho)
    actual = apply_exp_sum_to_separable_density(fit, x_axis, terms, dx)

    expected = np.zeros_like(rho)
    for weight, alpha_kernel in zip(fit.weights, fit.alphas):
        expected += weight * apply_separable_gaussian_3d(
            alpha_kernel,
            x_axis,
            rho,
        )
    expected *= dx**3

    np.testing.assert_allclose(actual, expected, rtol=1e-13, atol=1e-13)
