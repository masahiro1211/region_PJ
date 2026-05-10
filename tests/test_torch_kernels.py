import numpy as np
import pytest

torch = pytest.importorskip("torch")

pytestmark = [
    pytest.mark.filterwarnings("ignore:Sparse invariant checks.*:UserWarning"),
    pytest.mark.filterwarnings(
        "ignore:Sparse CSR tensor support.*:UserWarning"
    ),
]

from src.approximation.torch_kernels import (  # noqa: E402
    apply_dense_axis,
    apply_low_rank_axis,
    apply_sparse_axis,
    make_sparse_csr_tensor,
    to_float64_tensor,
)


def _apply_dense_axis_numpy(
    kernel: np.ndarray,
    rho: np.ndarray,
    axis: int,
) -> np.ndarray:
    return np.moveaxis(np.tensordot(kernel, rho, axes=([1], [axis])), 0, axis)


def _dense_to_csr_arrays(
    matrix: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    indptr = [0]
    indices = []
    data = []
    for row in matrix:
        nz = np.nonzero(row)[0]
        indices.extend(nz.tolist())
        data.extend(row[nz].tolist())
        indptr.append(len(indices))
    return (
        np.array(indptr, dtype=np.int64),
        np.array(indices, dtype=np.int64),
        np.array(data, dtype=np.float64),
    )


def test_apply_dense_axis_matches_numpy_tensordot():
    """PyTorchのdense軸作用はNumPyのtensordot実装と一致する。"""
    kernel_np = np.array(
        [
            [2.0, -1.0, 0.5],
            [0.0, 1.5, 0.25],
            [1.0, 0.0, -0.5],
        ],
    )
    rho_np = np.arange(27, dtype=float).reshape(3, 3, 3)
    kernel = to_float64_tensor(kernel_np)
    rho = to_float64_tensor(rho_np)

    for axis in range(3):
        actual = apply_dense_axis(kernel, rho, axis).numpy()
        expected = _apply_dense_axis_numpy(kernel_np, rho_np, axis)
        np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)


def test_apply_low_rank_axis_matches_reconstructed_kernel():
    """PyTorchの低ランク軸作用は再構成したdenseカーネル作用と一致する。"""
    rng = np.random.default_rng(0)
    kernel_np = rng.normal(size=(4, 4))
    rho_np = rng.normal(size=(4, 4, 4))
    u_np, singular_values_np, vt_np = np.linalg.svd(
        kernel_np,
        full_matrices=False,
    )
    rank = 3
    kernel_rank_np = (u_np[:, :rank] * singular_values_np[:rank]) @ vt_np[
        :rank,
        :,
    ]

    actual = apply_low_rank_axis(
        to_float64_tensor(u_np[:, :rank]),
        to_float64_tensor(singular_values_np[:rank]),
        to_float64_tensor(vt_np[:rank, :]),
        to_float64_tensor(rho_np),
        axis=1,
    ).numpy()
    expected = _apply_dense_axis_numpy(kernel_rank_np, rho_np, axis=1)

    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)


def test_apply_sparse_axis_matches_dense_axis():
    """PyTorchのsparse CSR軸作用はdense行列の軸作用と一致する。"""
    sparse_np = np.array(
        [
            [2.0, 0.0, 0.0, -1.0],
            [0.0, 1.5, 0.0, 0.0],
            [0.25, 0.0, 0.0, 0.5],
            [0.0, 0.0, -0.75, 0.0],
        ],
    )
    rho_np = np.arange(64, dtype=float).reshape(4, 4, 4)
    indptr, indices, data = _dense_to_csr_arrays(sparse_np)
    sparse = make_sparse_csr_tensor(indptr, indices, data, sparse_np.shape)
    dense = to_float64_tensor(sparse_np)
    rho = to_float64_tensor(rho_np)

    actual = apply_sparse_axis(sparse, rho, axis=2).numpy()
    expected = apply_dense_axis(dense, rho, axis=2).numpy()

    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)


def test_apply_sparse_axis_accepts_empty_csr_matrix():
    """全ゼロのCSR行列もPyTorch sparse CSRとして扱える。"""
    sparse_np = np.zeros((4, 4), dtype=float)
    rho_np = np.arange(64, dtype=float).reshape(4, 4, 4)
    indptr, indices, data = _dense_to_csr_arrays(sparse_np)
    sparse = make_sparse_csr_tensor(indptr, indices, data, sparse_np.shape)
    rho = to_float64_tensor(rho_np)

    actual = apply_sparse_axis(sparse, rho, axis=1).numpy()
    expected = np.zeros_like(rho_np)

    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)
