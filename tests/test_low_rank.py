import numpy as np
import pytest
from typing import Any, cast

from src.approximation.low_rank import approximate


def test_approximate_svd_shape():
    X = np.random.randn(12, 9)
    X_approx = approximate(X, ranks=4, method="svd")
    assert X_approx.shape == X.shape


def test_approximate_tucker_shape_with_scalar_rank():
    X = np.random.randn(5, 4, 3)
    X_approx = approximate(X, ranks=2, method="tucker")
    assert X_approx.shape == X.shape


def test_approximate_tucker_shape_with_rank_list():
    X = np.random.randn(5, 4, 3)
    X_approx = approximate(X, ranks=[3, 2, 2], method="tucker")
    assert X_approx.shape == X.shape


def test_approximate_svd_rejects_rank_list():
    X = np.random.randn(8, 6)
    invalid_ranks = cast(Any, [3, 2])
    with pytest.raises(TypeError):
        approximate(X, ranks=invalid_ranks, method="svd")
