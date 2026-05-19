import numpy as np
import pytest

torch = pytest.importorskip("torch")

from src.approximation.cp_coulomb import (  # noqa: E402
    apply_cp_rho_E,
    apply_cp_rho_V,
    materialize_cp_potential,
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
