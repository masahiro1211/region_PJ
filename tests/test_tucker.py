import numpy as np
import pytest
from src.decomposition.tucker import perform_tucker, reconstruct


def test_output_shapes():
    X = np.random.randn(6, 5, 4)
    ranks = [3, 2, 2]
    G, factors = perform_tucker(X, ranks)
    assert G.shape == tuple(ranks)
    for mode, (U, r) in enumerate(zip(factors, ranks)):
        assert U.shape == (X.shape[mode], r)


def test_reconstruction_error():
    X = np.random.randn(8, 6, 5)
    ranks = [4, 3, 3]
    G, factors = perform_tucker(X, ranks)
    X_approx = reconstruct(G, factors)
    rel_err = np.linalg.norm(X - X_approx) / np.linalg.norm(X)
    assert rel_err < 1.0


def test_full_rank_reconstruction():
    X = np.random.randn(4, 4, 4)
    ranks = [4, 4, 4]
    G, factors = perform_tucker(X, ranks)
    X_approx = reconstruct(G, factors)
    assert np.allclose(X, X_approx, atol=1e-10)


def test_invalid_ranks_length():
    X = np.random.randn(4, 4, 4)
    with pytest.raises(ValueError):
        perform_tucker(X, ranks=[2, 2])
