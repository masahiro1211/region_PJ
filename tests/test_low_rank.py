import numpy as np
import pytest
from typing import Any, cast

from src.approximation.low_rank import (
    apply_low_rank_svd,
    approximate,
    reconstruct_svd,
    truncated_svd,
)


def test_approximate_svd_shape():
    """2D行列をSVD近似したとき、元の形状が保たれる。"""
    X = np.random.randn(12, 9)
    X_approx = approximate(X, ranks=4, method="svd")
    assert X_approx.shape == X.shape


def test_approximate_tucker_shape_with_scalar_rank():
    """テンソルをスカラーrankでTucker近似したとき、元の形状が保たれる。"""
    X = np.random.randn(5, 4, 3)
    X_approx = approximate(X, ranks=2, method="tucker")
    assert X_approx.shape == X.shape


def test_approximate_tucker_shape_with_rank_list():
    """テンソルをmode別rankでTucker近似したとき、元の形状が保たれる。"""
    X = np.random.randn(5, 4, 3)
    X_approx = approximate(X, ranks=[3, 2, 2], method="tucker")
    assert X_approx.shape == X.shape


def test_approximate_svd_rejects_rank_list():
    """SVD近似でrankリストを渡したとき、TypeErrorが送出される。"""
    X = np.random.randn(8, 6)
    invalid_ranks = cast(Any, [3, 2])
    with pytest.raises(TypeError):
        approximate(X, ranks=invalid_ranks, method="svd")


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
