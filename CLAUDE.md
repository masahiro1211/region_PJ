# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install (editable, with all dev dependencies)
pip install -e ".[dev]"

# Run tests
python -m pytest -q

# Lint
python -m pycodestyle src tests

# Run a single test
python -m pytest tests/test_torch_kernels.py::test_name -q
```

## Project Overview

Research project for fast evaluation of the 3D Hartree potential:

$$V_H(\mathbf{r}_1) = \int \frac{\rho(\mathbf{r}_2)}{|\mathbf{r}_1 - \mathbf{r}_2|}\, d\mathbf{r}_2$$

The core idea: approximate the Coulomb kernel `1/r ≈ Σ_k w_k exp(-α_k r²)`, exploit Gaussian separability to reduce `O(N⁶)` → `O(R N⁴)`, then compress the 1D kernel further with SVD / RPCA for `O(r N³)`. Primary accuracy metric is the Hartree energy relative error (target `O(10⁻³)`).

## Architecture

### Computation pipelines (all benchmarked in PyTorch for fair comparison)

Four paths in [src/approximation/torch_kernels.py](src/approximation/torch_kernels.py):

| Function | Complexity | Purpose |
|---|---|---|
| `apply_exp_sum_3d_full` | O(K N⁴) | Baseline |
| `apply_exp_sum_3d_lowrank_naive` | O(K N⁴) | Shows the bug: reconstructing K_r = UΣVᵀ first loses all savings |
| `apply_exp_sum_3d_lowrank` | O(K r N³) | Correct mode-n product: Vᵀ → Σ → U |
| `apply_exp_sum_3d_rpca` | O(K r N³) + O(nnz N²) | L (low-rank) + S (sparse/dense) |

The RPCA path splits kernel terms into `lowrank_only_list` (S ≈ 0) and `dense_list` (S ≠ 0), with dynamic dispatch to `torch.sparse.mm` (CSR) when nnz ≤ 0.1%, otherwise dense `tensordot`.

### Exponential-sum fitting pipeline

[src/approximation/exp_sum/](src/approximation/exp_sum/) — VARPRO:
- Outer loop: L-BFGS-B optimizes `log α_k` (nonlinear, R variables)
- Inner loop: NNLS solves for `w_k ≥ 0` analytically given `α_k`
- Fit grid is log-uniform; evaluation grid uses 2000 points

Heavy computations (VARPRO, RPCA) are cached by SHA256 key in `data/processed/cache/` via [src/utils/cache.py](src/utils/cache.py) `load_or_compute`.

### RPCA decomposition

[src/decomposition/rpca.py](src/decomposition/rpca.py):
- `rpca`: ALM (Augmented Lagrangian Method), uses full SVD per iteration
- `randomized_rpca`: same ALM but with randomized SVD; default `rank = N/4`

The diagonal singularity of `1/r` at `r=0` is handled by an analytic correction: the separable Gaussian approximation gives a finite value at the origin, so the difference `K_diag - Σ_k w_k` (where `K_diag ≈ 2.38/dx`) is added back as a diagonal correction.

### CP-format density path

[src/approximation/cp_coulomb.py](src/approximation/cp_coulomb.py) — when the charge density `ρ` itself has CP (CANDECOMP/PARAFAC) structure `ρ = Σ_m c_m fx_m ⊗ fy_m ⊗ fz_m`, all computations reduce to 1D matvec operations, avoiding explicit `N³` tensor construction for the energy (O(K M² N²) instead of O(K M N³)).

### Baselines

- FFT solver (periodic BC): [src/potential/charge_potential.py](src/potential/charge_potential.py)
- CG Poisson solver (monopole BC): [src/potential/poisson_solver.py](src/potential/poisson_solver.py)
- Tucker/HOSVD on density side: [src/decomposition/tucker.py](src/decomposition/tucker.py) `perform_tucker_rpca`

## Coding Conventions

- **Style**: PEP 8, 79-character line limit
- **Imports**: stdlib → external libraries → project-internal (one blank line between groups)
- **Docstrings**: All public functions get Japanese docstrings in NumPy/SciPy format (Parameters / Returns / Raises / Notes). Include array shapes.
- **Type hints**: All public function signatures; narrow `Union` types with `isinstance` before use
- **Return dicts**: Use `TypedDict` over plain `dict`
- **Numerical safety**: Use `np.divide(..., where=...)` for zero-division near singularities; note boundary-condition assumptions in comments
- **Optional deps**: Guard `import torch` with try/except; raise `ImportError` with install hint. Tests use `pytest.importorskip("torch")`.
- **Numerical tests**: `np.testing.assert_allclose` on small deterministic arrays. Heavy experiments go in notebooks or `tests/benchmarks/`, not unit tests.
- **Test naming**: Descriptive names that state the behaviour, e.g. `test_split_kernel_near_far_reconstructs_original_kernel`; one-line Japanese docstring per test stating preconditions / operation / expected result.

## Key Gotcha

The low-rank evaluation order matters critically: applying `K_r = U Σ Vᵀ` as a reconstructed N×N matrix costs `O(N⁴)` — the same as the full path. Savings only appear when mode-n product is applied in order `Vᵀ → Σ → U`, keeping the intermediate tensor dimension at `r` instead of `N`. `apply_exp_sum_3d_lowrank_naive` exists explicitly to demonstrate this failure mode in benchmarks.
