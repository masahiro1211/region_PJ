import numpy as np

from src.approximation.exp_sum.separable import (
    apply_1d_kernel_along_axis,
    apply_separable_gaussian_3d,
)
from src.potential.separable_density import build_gaussian_kernel_1d


def test_apply_1d_kernel_along_axis_matches_explicit_axis_zero():
    """axis=0 の mode product が明示的な einsum と一致する。"""
    kernel = np.array([[1.0, 2.0], [3.0, 4.0]])
    rho = np.arange(8.0).reshape(2, 2, 2)

    actual = apply_1d_kernel_along_axis(kernel, rho, axis=0)
    expected = np.einsum("ia,ajk->ijk", kernel, rho)

    np.testing.assert_allclose(actual, expected)


def test_apply_1d_kernel_along_axis_matches_explicit_axis_one():
    """axis=1 の mode product が明示的な einsum と一致する。"""
    kernel = np.array([[1.0, 2.0], [3.0, 4.0]])
    rho = np.arange(8.0).reshape(2, 2, 2)

    actual = apply_1d_kernel_along_axis(kernel, rho, axis=1)
    expected = np.einsum("jb,ibk->ijk", kernel, rho)

    np.testing.assert_allclose(actual, expected)


def test_apply_1d_kernel_along_axis_matches_explicit_axis_two():
    """axis=2 の mode product が明示的な einsum と一致する。"""
    kernel = np.array([[1.0, 2.0], [3.0, 4.0]])
    rho = np.arange(8.0).reshape(2, 2, 2)

    actual = apply_1d_kernel_along_axis(kernel, rho, axis=2)
    expected = np.einsum("kc,ijc->ijk", kernel, rho)

    np.testing.assert_allclose(actual, expected)


def test_apply_separable_gaussian_3d_matches_three_axis_products():
    """3D Gaussian 適用が1Dカーネルの3軸逐次適用と一致する。"""
    alpha = 0.7
    x_axis = np.linspace(-1.0, 1.0, 4)
    rho = np.arange(64.0).reshape(4, 4, 4)
    kernel = build_gaussian_kernel_1d(alpha, x_axis)

    expected = apply_1d_kernel_along_axis(kernel, rho, axis=0)
    expected = apply_1d_kernel_along_axis(kernel, expected, axis=1)
    expected = apply_1d_kernel_along_axis(kernel, expected, axis=2)
    actual = apply_separable_gaussian_3d(alpha, x_axis, rho)

    np.testing.assert_allclose(actual, expected)
