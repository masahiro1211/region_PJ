from .charge_potential import run_charge_potential_demo
from .regularized_coulomb import (
    build_regularized_coulomb_kernel_1d,
    build_regularized_coulomb_kernel_3d,
    build_regularized_coulomb_kernel_from_distances,
    flatten_xyz_grid,
    gaussian_density_1d,
    pairwise_distance_matrix,
    smoothstep_window,
    split_kernel_near_far,
)
from .separable_density import (
    SeparableDensityTerm,
    apply_exp_sum_to_separable_density,
    build_gaussian_kernel_1d,
    make_gaussian_density_terms,
    materialize_density_terms,
    outer3,
)
from .poisson_solver import (
    compute_monopole_bc,
    laplacian_matvec,
    rhs_with_bc,
    cg_solve,
    poisson_solve,
)

__all__ = [
    "run_charge_potential_demo",
    "build_regularized_coulomb_kernel_1d",
    "build_regularized_coulomb_kernel_3d",
    "build_regularized_coulomb_kernel_from_distances",
    "flatten_xyz_grid",
    "gaussian_density_1d",
    "pairwise_distance_matrix",
    "smoothstep_window",
    "split_kernel_near_far",
    "SeparableDensityTerm",
    "apply_exp_sum_to_separable_density",
    "build_gaussian_kernel_1d",
    "make_gaussian_density_terms",
    "materialize_density_terms",
    "outer3",
    "compute_monopole_bc",
    "laplacian_matvec",
    "rhs_with_bc",
    "cg_solve",
    "poisson_solve",
]
