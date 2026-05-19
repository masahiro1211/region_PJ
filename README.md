# region_PJ

3 次元電荷密度 `rho(r)` から Hartree ポテンシャル

$$
V_H(\mathbf{r}_1) = \int \frac{\rho(\mathbf{r}_2)}{|\mathbf{r}_1 - \mathbf{r}_2|}\,d\mathbf{r}_2
$$

を高速に評価するための研究用コードです。中心にある考え方は、Coulomb カーネル `1/r` を Gaussian の指数和で近似し、Gaussian の次元分離性と 1D カーネルの低ランク構造を使って、3D 畳み込みの計算量を下げることです。

詳しい研究の流れ、実験結果、考察は [docs/research/flow.md](docs/research/flow.md) に分けています。この README は、プロジェクトの入口として必要な情報だけを置きます。

## Core Idea

Coulomb カーネルを

$$
\frac{1}{r} \approx \sum_{k=1}^{R} w_k e^{-\alpha_k r^2}
$$

で近似する。Gaussian は

$$
e^{-\alpha|\mathbf{r}_1-\mathbf{r}_2|^2}
= e^{-\alpha(x_1-x_2)^2}e^{-\alpha(y_1-y_2)^2}e^{-\alpha(z_1-z_2)^2}
$$

と分離できるので、3D 評価は 1D カーネル行列 `K` の軸方向適用に分解できる。

さらに 1D カーネルを

$$
K \approx U_r \Sigma_r V_r^T
$$

と圧縮し、`V_r^T -> Sigma_r -> U_r` の順で作用させることで、低ランクの計算量削減を保つ。`K_r` を先に再構成すると full rank と同じ扱いに戻るので、この順序が重要です。

密度 `rho` 自体が CP 形式

$$
\rho = \sum_m c_m\rho_m^x \otimes \rho_m^y \otimes \rho_m^z
$$

で与えられる場合は、ポテンシャルも CP 項のまま保持できます。

$$
V = dx^3 \sum_k w_k \sum_m c_m
(K_k\rho_m^x) \otimes (K_k\rho_m^y) \otimes (K_k\rho_m^z)
$$

この経路では、必要になるのは 1D matvec だけです。dense な `N^3` テンソルへの materialize は、可視化や dense API との比較が必要なときだけ行います。

## Current Focus

- VARPRO + NNLS による `1/r` の指数和近似
- Gaussian 分離による `O(N^6) -> O(R N^4)` の評価
- SVD による 1D カーネルの低ランク適用
- RPCA による `K = L + S` 分解と sparse/dense 動的ディスパッチ
- 対角補正による Hartree エネルギー精度の改善
- CP 形式密度に対する 1D matvec ベースのポテンシャル・エネルギー評価
- PyTorch 統一バックエンドでのタイミング比較

## Important Files

```text
src/
  approximation/
    exp_sum/                # 指数和近似、VARPRO、分離 Gaussian 評価
    torch_kernels.py        # PyTorch 版 full / low-rank / RPCA 評価
    cp_coulomb.py           # CP 密度向け PyTorch 評価
    low_rank.py             # SVD helper
  decomposition/
    rpca.py                 # RPCA / randomized RPCA
    tucker.py               # Tucker / HOSVD 系
  potential/
    separable_density.py    # CP 形式密度、CP 形式ポテンシャル
    gaussian_discretization.py
    charge_potential.py     # FFT 解法、解析 Gaussian ポテンシャル
    poisson_solver.py       # 実空間 Poisson CG 解法
  utils/
    grid.py
    metrics.py
    cache.py

notebooks/
  calc_potential_exp_expansion_approx.ipynb

scripts/
  run_exp_sum_fitting.py
  run_rpca_kernels.py
  run_rpca_error_sweep.py
  run_timing_benchmark.py

docs/
  research/flow.md          # 研究の流れと結果
  coding_standards.md
  testing.md
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Windows PowerShell の場合:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

PyTorch を含む notebook / benchmark 用の依存は `pyproject.toml` の optional dependencies も参照してください。

## Common Commands

指数和フィット:

```bash
python scripts/run_exp_sum_fitting.py --compute
```

1D カーネルの SVD/RPCA 保存:

```bash
python scripts/run_rpca_kernels.py --compute
```

RPCA 誤差 sweep:

```bash
python scripts/run_rpca_error_sweep.py
```

タイミング benchmark:

```bash
python scripts/run_timing_benchmark.py
```

テスト:

```bash
python -m pytest -q
```

## Minimal Examples

指数和フィット:

```python
from src.approximation.exp_sum.benchmark import BenchmarkRunner
from src.approximation.exp_sum.grid import LogUniformGrid
from src.approximation.exp_sum.varpro import VarproOptimizer

fit_grid = LogUniformGrid(r_min=1e-2, r_max=2 * 3**0.5 * 20, n_points=2000)
eval_grid = LogUniformGrid(r_min=1e-2, r_max=2 * 3**0.5 * 20, n_points=2000)

optimizer = VarproOptimizer(fit_grid, eval_grid, nonneg=True, max_iter=200_000)
runner = BenchmarkRunner(optimizer, ranks=range(1, 31))
fits = runner.run()
```

CP 形式密度への指数和適用:

```python
from src.potential.separable_density import (
    apply_exp_sum_to_separable_density_cp,
    make_gaussian_density_terms,
    materialize_potential_terms,
)

rho_terms = make_gaussian_density_terms(x_axis, alpha=1.0)
V_terms = apply_exp_sum_to_separable_density_cp(fit, x_axis, rho_terms)
V_dense = materialize_potential_terms(V_terms, dx)  # 必要な場合だけ
```

## Documentation

- [docs/research/flow.md](docs/research/flow.md): 研究の流れ、手法、知見、今後の課題
- [docs/testing.md](docs/testing.md): テスト規約
- [docs/coding_standards.md](docs/coding_standards.md): コーディング規約
- [References.md](References.md): 参考文献・リンク
- [homework.md](homework.md): 作業メモ

## Notes

- 精度評価では、ポテンシャル `V` の相対誤差だけでなく Hartree エネルギー `E_H` の相対誤差を重視します。
- `1/r` の特異点は指数和だけでは表現できないため、セル平均に基づく対角補正を入れています。
- タイミング比較では backend 差を避けるため、主要経路を PyTorch に寄せています。
