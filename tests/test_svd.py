import numpy as np
import pytest
from src.decomposition.svd import perform_svd


def test_output_shapes():
    X = np.random.randn(10, 8)
    A, B = perform_svd(X, rank=3)
    assert A.shape == (10, 3)
    assert B.shape == (3, 8)


def test_reconstruction_error_decreases_with_rank():
    X = np.random.randn(20, 15)
    errors = []
    for rank in [1, 5, 10]:
        A, B = perform_svd(X, rank=rank)
        errors.append(np.linalg.norm(X - A @ B))
    assert errors[0] > errors[1] > errors[2]


def test_full_rank_reconstruction():
    X = np.random.randn(6, 6)
    rank = min(X.shape)
    A, B = perform_svd(X, rank=rank)
    assert np.allclose(X, A @ B, atol=1e-10)
