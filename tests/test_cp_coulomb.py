import numpy as np
import pytest

torch = pytest.importorskip("torch")

from src.approximation.cp_coulomb import (  # noqa: E402
    apply_cp_rho_E,
    apply_cp_rho_E_rpca,
    apply_cp_rho_E_rpca_sparse,
    apply_cp_rho_V,
    apply_cp_rho_V_rpca,
    apply_cp_rho_V_rpca_sparse,
    materialize_cp_potential,
)
from src.approximation.torch_kernels import make_sparse_csr_tensor  # noqa: E402


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


def test_apply_cp_rho_v_returns_cp_terms_without_materializing_tensor():
    """CP密度へのV作用はdenseな3Dテンソルではなく1D因子列を返す。"""
    rho_terms = [
        (
            1.25,
            torch.tensor([1.0, 2.0], dtype=torch.float64),
            torch.tensor([0.5, -1.0], dtype=torch.float64),
            torch.tensor([2.0, 0.25], dtype=torch.float64),
        )
    ]
    kernel = torch.tensor(
        [[2.0, -1.0], [0.25, 0.5]],
        dtype=torch.float64,
    )

    cp_terms = apply_cp_rho_V(rho_terms, [(0.4, kernel)])

    assert isinstance(cp_terms, list)
    assert len(cp_terms) == 1
    coeff, vx, vy, vz = cp_terms[0]
    assert coeff == pytest.approx(0.5)
    assert vx.shape == (2,)
    assert vy.shape == (2,)
    assert vz.shape == (2,)


def test_materialize_cp_potential_matches_explicit_outer_sum():
    """CP形式のVを明示的に具現化すると外積和と一致する。"""
    rho_terms = [
        (
            1.25,
            torch.tensor([1.0, 2.0], dtype=torch.float64),
            torch.tensor([0.5, -1.0], dtype=torch.float64),
            torch.tensor([2.0, 0.25], dtype=torch.float64),
        )
    ]
    kernel = torch.tensor(
        [[2.0, -1.0], [0.25, 0.5]],
        dtype=torch.float64,
    )
    dx = 0.3

    cp_terms = apply_cp_rho_V(rho_terms, [(0.4, kernel)])
    actual = materialize_cp_potential(cp_terms, dx).numpy()

    _, fx, fy, fz = rho_terms[0]
    vx = kernel @ fx
    vy = kernel @ fy
    vz = kernel @ fz
    expected = (
        0.4
        * 1.25
        * np.einsum("i,j,k->ijk", vx.numpy(), vy.numpy(), vz.numpy())
        * dx**3
    )

    np.testing.assert_allclose(actual, expected)


def test_apply_cp_rho_e_matches_materialized_potential_energy():
    """CP密度のエネルギーはVを具現化した評価と一致する。"""
    rho_terms = [
        (
            1.25,
            torch.tensor([1.0, 2.0], dtype=torch.float64),
            torch.tensor([0.5, -1.0], dtype=torch.float64),
            torch.tensor([2.0, 0.25], dtype=torch.float64),
        )
    ]
    kernel = torch.tensor(
        [[2.0, -1.0], [0.25, 0.5]],
        dtype=torch.float64,
    )
    dx = 0.3

    cp_terms = apply_cp_rho_V(rho_terms, [(0.4, kernel)])
    potential = materialize_cp_potential(cp_terms, dx)
    _, fx, fy, fz = rho_terms[0]
    rho = 1.25 * torch.einsum("i,j,k->ijk", fx, fy, fz)

    actual = apply_cp_rho_E(rho_terms, [(0.4, kernel)], dx)
    expected = 0.5 * torch.sum(rho * potential) * dx**3

    torch.testing.assert_close(actual, expected)


def test_cp_rpca_sparse_matches_dense_s_path():
    """CP-rPCA の S sparse CSR 経路は dense-S 経路と一致する。"""
    rho_terms = [
        (
            1.25,
            torch.tensor([1.0, 2.0, -0.5], dtype=torch.float64),
            torch.tensor([0.5, -1.0, 0.25], dtype=torch.float64),
            torch.tensor([2.0, 0.25, 1.5], dtype=torch.float64),
        )
    ]
    u = torch.tensor(
        [[1.0, 0.0], [0.5, 1.0], [-0.25, 0.75]],
        dtype=torch.float64,
    )
    s = torch.tensor([0.8, 0.3], dtype=torch.float64)
    vt = torch.tensor(
        [[1.0, -0.25, 0.5], [0.0, 0.75, -1.0]],
        dtype=torch.float64,
    )
    s_dense_np = np.array(
        [[0.0, 0.2, 0.0], [-0.1, 0.0, 0.3], [0.0, 0.0, 0.4]],
    )
    s_dense = torch.tensor(s_dense_np, dtype=torch.float64)
    indptr, indices, data = _dense_to_csr_arrays(s_dense_np)
    s_sparse = make_sparse_csr_tensor(indptr, indices, data, s_dense_np.shape)

    dense_list = [(0.4, u, s, vt, s_dense)]
    sparse_list = [(0.4, u, s, vt, s_sparse)]

    dense_terms = apply_cp_rho_V_rpca(rho_terms, [], dense_list)
    sparse_terms = apply_cp_rho_V_rpca_sparse(rho_terms, [], sparse_list)
    assert len(dense_terms) == len(sparse_terms)
    for dense_term, sparse_term in zip(dense_terms, sparse_terms):
        assert dense_term[0] == pytest.approx(sparse_term[0])
        for dense_vec, sparse_vec in zip(dense_term[1:], sparse_term[1:]):
            torch.testing.assert_close(
                dense_vec,
                sparse_vec,
                rtol=1e-12,
                atol=1e-12,
            )

    dx = 0.3
    dense_e = apply_cp_rho_E_rpca(rho_terms, [], dense_list, dx)
    sparse_e = apply_cp_rho_E_rpca_sparse(rho_terms, [], sparse_list, dx)
    torch.testing.assert_close(dense_e, sparse_e, rtol=1e-12, atol=1e-12)
