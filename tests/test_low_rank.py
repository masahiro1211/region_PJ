import numpy as np
import pytest

from src.approximation.low_rank import (
    apply_low_rank_svd,
    reconstruct_svd,
    truncated_svd,
)


def test_truncated_svd_reconstructs_requested_rank_matrix():
    """2D行列を指定rankでSVD再構成したとき、元の形状が保たれる。"""
    rng = np.random.default_rng(0)
    matrix = rng.normal(size=(10, 8))

    u, singular_values, vt = truncated_svd(matrix, rank=4)
    matrix_rank4 = reconstruct_svd(u, singular_values, vt)

    assert matrix_rank4.shape == matrix.shape


def test_truncated_svd_rejects_invalid_rank():
    """rankが行列サイズの範囲外のとき、ValueErrorが送出される。"""
    matrix = np.ones((3, 2))

    with pytest.raises(ValueError):
        truncated_svd(matrix, rank=3)


def test_apply_low_rank_svd_matches_reconstructed_matrix_product():
    """SVD因子を右から作用させたとき、再構成行列との積に一致する。"""
    rng = np.random.default_rng(0)
    matrix = rng.normal(size=(10, 8))
    vector = rng.normal(size=8)

    u, singular_values, vt = truncated_svd(matrix, rank=4)
    matrix_rank4 = reconstruct_svd(u, singular_values, vt)

    actual = apply_low_rank_svd(u, singular_values, vt, vector)
    expected = matrix_rank4 @ vector

    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)
