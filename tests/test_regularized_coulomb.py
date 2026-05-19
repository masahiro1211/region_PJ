import numpy as np
import pytest

from src.potential.regularized_coulomb import (
    build_regularized_coulomb_kernel_1d,
    build_regularized_coulomb_kernel_3d,
    flatten_xyz_grid,
    gaussian_density_1d,
    pairwise_distance_matrix,
    smoothstep_window,
    split_kernel_near_far,
)
from src.utils.grid import build_xyz


def test_regularized_coulomb_1d_is_symmetric_with_expected_diagonal():
    """1D正則化Coulombカーネルは対称で、対角成分が1/epsになる。"""
    coords = np.linspace(-1.0, 1.0, 7)
    eps = 0.25

    kernel = build_regularized_coulomb_kernel_1d(coords, eps)

    np.testing.assert_allclose(kernel, kernel.T)
    np.testing.assert_allclose(
        np.diag(kernel),
        np.full(coords.size, 1.0 / eps),
    )


def test_gaussian_density_1d_matches_formula():
    """中心と重みを指定した1D Gaussian 密度は定義式に一致する。"""
    coords = np.array([-1.0, 0.0, 1.0])

    density = gaussian_density_1d(coords, alpha=2.0, center=0.5, weight=3.0)

    expected = 3.0 * np.exp(-2.0 * (coords - 0.5) ** 2)
    np.testing.assert_allclose(density, expected)


def test_regularized_coulomb_3d_uses_pairwise_distances():
    """3D正則化Coulombカーネルは距離行列からの定義式に一致する。"""
    xyz = build_xyz(3, 3.0)
    points = flatten_xyz_grid(xyz)
    eps = 0.5

    distances = pairwise_distance_matrix(points)
    kernel = build_regularized_coulomb_kernel_3d(points, eps)

    assert kernel.shape == (27, 27)
    np.testing.assert_allclose(kernel, kernel.T)
    np.testing.assert_allclose(kernel, 1.0 / np.sqrt(distances**2 + eps**2))


def test_smoothstep_window_is_one_near_zero_and_zero_outside_threshold():
    """smoothstep窓は距離0で1、閾値以上で0になる。"""
    distances = np.array([0.0, 0.25, 0.5, 0.75])

    weights = smoothstep_window(distances, threshold=0.5)

    assert weights[0] == pytest.approx(1.0)
    assert weights[2] == pytest.approx(0.0)
    assert weights[3] == pytest.approx(0.0)
    assert np.all((0.0 <= weights) & (weights <= 1.0))


def test_split_kernel_near_far_reconstructs_original_kernel():
    """near/far分割後の和は元のカーネルを再構成する。"""
    coords = np.linspace(-1.0, 1.0, 5)
    kernel = build_regularized_coulomb_kernel_1d(coords, eps=0.2)
    distances = np.abs(coords[:, None] - coords[None, :])

    near, far, weights = split_kernel_near_far(
        kernel,
        distances,
        threshold=0.6,
        smooth=True,
    )

    assert near.shape == kernel.shape
    assert far.shape == kernel.shape
    assert weights.shape == kernel.shape
    np.testing.assert_allclose(near + far, kernel)
