# region_PJ

3 次元電荷分布から生成される Hartree ポテンシャル

$$
V_H(\mathbf{r}_1) = \int \frac{\rho(\mathbf{r}_2)}{|\mathbf{r}_1 - \mathbf{r}_2|}\, d\mathbf{r}_2
$$

を、Coulomb カーネル $1/r$ の **指数和近似** と **低ランク／スパース分解** によって高速に評価することを目的とした研究用プロジェクト。

線形代数の基礎実装（SVD・Tucker 分解・RPCA）から、それらを物理問題へ適用するデモまでを一貫して扱う。

---

## 1. 背景と目的

### 1.1 背景

第一原理計算や密度汎関数理論 (DFT) において、電子密度 $\rho(\mathbf{r})$ から Hartree ポテンシャルを得る計算はもっとも重い処理のひとつである。各方向 $N$ 点の 3D グリッド上で素朴に積分すると **$O(N^6)$** の計算量となり、$N$ がやや大きくなるだけで現実的に解けなくなる。

代表的な高速化手法は次の 2 つに大別される。

1. **FFT を用いた周期境界での解法**（$O(N^3 \log N)$）
2. **多極子展開・カーネル分離による低ランク化**（$O(R\,N^4)$ など）

本プロジェクトでは、後者の路線で次元分離可能な Gaussian の重ね合わせとしてカーネルを近似し、さらに行列分解 (SVD / RPCA) で 1D カーネル自体を圧縮する。

### 1.2 目的

- Coulomb カーネル $1/r$ を **指数和**

$$
\frac{1}{r} \approx \sum_{k=1}^{R} w_k \, e^{-\alpha_k r^2}
$$

  で近似し、Hartree ポテンシャル評価を **$O(R \cdot N^4)$** に落とす。
- 1D Gaussian カーネル $K_{ij}=e^{-\alpha(x_i-x_j)^2}$ にさらに **SVD / Robust PCA** を施し、$O(r\,N^3)$ への削減を狙う。
- 評価指標として、ポテンシャル相対誤差だけでなく **Hartree エネルギー** $E = \tfrac{1}{2}\int\rho V\, d^3r$ の相対誤差を主指標に据え、目標値を **$O(10^{-3})$** とする。
- 計算量解析で予測される速度差を、**同一バックエンド（PyTorch）** での計測で確認する。

### 1.3 SVD と Tucker 分解の対応関係

| SVD | Tucker |
|---|---|
| 左特異ベクトル $U_r$ | 因子行列 $U_1, U_2, \ldots, U_N$ |
| 特異値 $\Sigma_r$ | コアテンソル $\mathcal{G}$ |
| 右特異ベクトル $V_r$ | 各モードの因子行列 |

---

## 2. 手法

### 2.1 ベースライン: FFT 解法と実空間 CG 解法

- **FFT 解法**: 周期境界条件のもとで $\hat V(\mathbf{k}) = 4\pi \hat \rho(\mathbf{k})/|\mathbf{k}|^2$ を評価（[src/potential/charge_potential.py](src/potential/charge_potential.py)）。
- **実空間 CG 解法**: モノポール境界条件 $V|_\partial = Q/r$ を与え、中央差分の Laplacian に対して共役勾配法でポアソン方程式を解く（[src/potential/poisson_solver.py](src/potential/poisson_solver.py)）。グリッドを細かくすると $O(dx^2)$ で解析解に近づくことを確認している（4/13 宿題）。

### 2.2 指数和近似（VARPRO）

指数和

$$
\frac{1}{r} \approx \sum_k w_k e^{-\alpha_k r^2}
$$

のフィッティングは、線形パラメータ $w_k$ と非線形パラメータ $\alpha_k$ を変数射影法 (VARPRO) で同時最適化する（[src/approximation/exp_sum.py](src/approximation/exp_sum.py)）。

- 外側ループ: `L-BFGS-B` で $\log\alpha_k$ を最適化（非線形、$R$ 変数）
- 内側ループ: 与えられた $\alpha_k$ に対して **NNLS** で $w_k\ge 0$ を解析的に解く
- グリッドは log-uniform、評価グリッドは 2000 点
- 結果は SHA256 ベースのキー付き **pkl キャッシュ** で再利用（[src/utils/cache.py](src/utils/cache.py)）

参考: Hackbusch (2019), Beylkin & Monzón (2005)。

### 2.3 次元分離による 3D 畳み込み

Gaussian は次元分離可能であり、

$$
e^{-\alpha|\mathbf{r}_1-\mathbf{r}_2|^2}
= e^{-\alpha(x_1-x_2)^2}\,e^{-\alpha(y_1-y_2)^2}\,e^{-\alpha(z_1-z_2)^2}
$$

3D 畳み込みは **1D Gaussian カーネル $K\in\mathbb{R}^{N\times N}$ を mode-n product で 3 軸に逐次適用**することに帰着する。素朴な評価の $O(N^6)$ から **$O(R\,N^4)$** に削減される（[src/approximation/exp_sum.py](src/approximation/exp_sum.py) `apply_separable_gaussian_3d`）。

### 2.4 1D カーネルの低ランク／スパース分解

1D Gaussian カーネル $K$ にさらに次の 2 系統を施し、計算量と精度を比較する。

- **SVD**: $K \approx U_r \Sigma_r V_r^\top$（[src/decomposition/svd.py](src/decomposition/svd.py)）
- **Robust PCA (RPCA)**: $K = L + S$（$L$ 低ランク、$S$ スパース）。ALM 版と randomized SVD 版を実装（[src/decomposition/rpca.py](src/decomposition/rpca.py)）

mode-n product を **正しい評価順** $U_r(\Sigma_r(V_r^\top \rho))$ で適用すると、各軸の計算量は **$O(r N^3)$**。一方、$U_r\Sigma_r V_r^\top$ を **先に組み立て直して** $K_r$ として作用させるとフルランク扱いに戻り $O(N^4)$ のまま、という典型的な落とし穴がある（5/2 宿題）。両者を区別して計測している。

### 2.5 Tucker (HOSVD) 系の経路

3D 密度 $\rho$ そのものに対する Tucker 分解（HOSVD）と、各 mode 展開に RPCA を施した HOSVD 改も実装（[src/decomposition/tucker.py](src/decomposition/tucker.py) `perform_tucker_rpca`）。これは「カーネル側」ではなく「密度側」を圧縮する経路で、ベースライン比較に用いる。

### 2.6 対角成分の補正

指数和近似では中心 $r=0$ で値が有限の $\sum_k w_k$ にとどまり、$1/r$ の特異性を表現できない。そこで一辺 $dx$ の立方体内で $1/r$ を解析積分した値

$$
K_{\text{diag}} = \frac{1}{dx^3} \iiint_{[-dx/2, dx/2]^3} \frac{1}{r}\,d^3r \approx \frac{2.38}{dx}
$$

を用いて、対角寄与を $K_{\text{diag}} - \sum_k w_k$ ぶんだけ後付けで補正する。

### 2.7 PyTorch への計算経路統一

NumPy / SciPy / `sparse-dot-mkl` 等を混在させると、関数ごとの BLAS / OpenMP 実装差で純粋な計算量比較が崩れる。これを避けるため、ベンチマーク部分（full / low-rank 順序ミス / low-rank 正解 / RPCA + sparse）を **すべて PyTorch テンソル演算 (`torch.tensordot`, `torch.sparse.mm`)** に統一した（最新コミット: `f3a5877 pytorchに統一`）。

---

## 3. 結果と考察

評価条件（代表値）: $N=151$、$L=20$、$\alpha=1.0$、単一 Gaussian 電荷。指数和ランク $R=1,\ldots,30$。

### 3.1 指数和近似の精度

`tests/`, ノートブック [notebooks/calc_potential_exp_expansion_approx.ipynb](notebooks/calc_potential_exp_expansion_approx.ipynb) より：

| $R$ | $L_2$ 相対誤差（$1/r$） | $L_\infty$ 相対誤差 |
|---:|---:|---:|
| 11 | 1.66e−1 | 9.10e−1 |
| 17 | 5.34e−3 | 5.34e−2 |
| 21 | 1.07e−3 | 3.68e−3 |
| 25 | 1.78e−4 | 1.12e−3 |
| 30 | 4.08e−5 | 5.00e−4 |

$R \gtrsim 20$ で関数近似自体は $10^{-3}$ を切る。

### 3.2 ポテンシャル / Hartree エネルギーの誤差

$V$ と $E=\tfrac{1}{2}\int\rho V\,d^3r$ の相対誤差を、対角補正の有無で比較した結果：

- **対角補正なし**: $R$ を増やしても $V$ 誤差が $10^{-1}$ 程度で頭打ち。$\sum_k w_k$ が $K_{\text{diag}}$ に届かないため。
- **対角補正あり**: $R=15$ 付近で $E$ 相対誤差は **$10^{-3}\sim10^{-4}$** に到達し、目標値 $O(10^{-3})$ を満たす。ただし $R=15$ 以降はプラトーに入る。

**プラトーの原因**: $E$ をリーマン和（`np.sum(rho * V) * dx**3`）で評価しているため、近似関数の改善ではなく **離散化誤差 $O(dx^2)$** が支配的になる。これは $N=151$, $L=20$ で $dx\approx 0.132$、対応する誤差スケールが $10^{-3}\sim10^{-4}$ となることと整合する。

### 3.3 1D カーネルの SVD / RPCA 圧縮

指数和 $R=11$ を固定し、1D カーネルの圧縮ランク $r$ をスイープ：

- **SVD のみ**: $r$ を上げるほど $V$, $E$ 誤差が単調に減少。
- **RPCA（閾値なし）**: $S$ をそのまま足すと SVD と同等以上だが、$S$ がほぼ密になり利得は小さい。
- **RPCA + 閾値**: $|S_{ij}|>\tau$ のみ残すことで、$\tau=10^{-2}$ 付近では非ゼロ率がかなり低くなり、精度を保ったまま計算量を抑えられる領域が見える。

### 3.4 計算時間（PyTorch 統一、$N=151$, $R=11$, $r=15$）

`%%timeit -n 10 -r 10` での実測（最新セル）：

| 経路 | 計算量（理論） | 実測 |
|---|---|---|
| full ($K$ をそのまま 3 軸に tensordot) | $O(N^4)$ | 137 ± 31 ms |
| low-rank（順序ミス: $K_r$ を再構成して適用） | $O(N^4)$ | 96.6 ± 1.3 ms |
| **low-rank（正しい順序: mode-n product）** | **$O(r N^3)$** | **61.6 ± 0.7 ms** |
| RPCA ($L$ は mode-n、$S$ は閾値後 sparse / dense) | (混在) | 123 ± 2.7 ms |

正しい mode-n product はフル評価の **約 2.2 倍速** となり、計算量解析（$O(rN^3)$ vs $O(N^4)$、$r/N\approx 0.1$）と一致する。一方、再構成 → tensordot の順序ミスは「低ランクなのに $O(N^4)$」になる典型を実測でも確認できた。

RPCA は現状ややオーバーヘッドが大きく、$L$ + sparse $S$ を別経路で足し合わせる構成のため「11 個のカーネル × dense+sparse 2 系統」が full より遅くなるケースがある。閾値 $\tau$ や sparse / dense の切替閾値（現状は非ゼロ率 0.1%）の調整余地が残っている。

### 3.5 知見の要点

- 目標精度 $E$ 相対誤差 $O(10^{-3})$ は **$R \ge 15$ + 対角補正** で達成可能。
- 精度の頭打ちは離散化誤差由来であり、$N$（あるいは $dx$）を細かくしないと $R$ を増やすだけでは改善しない。
- 計算量の理論差は **正しい評価順** で初めて顕在化する。プロファイリングは「同一バックエンド」が必須。

---

## 4. まとめと今後の展望

### 4.1 まとめ

1. Coulomb カーネル $1/r$ を VARPRO + NNLS で指数和近似し、$R$ vs 誤差曲線を取得した。
2. 次元分離 + mode-n product により、3D Hartree ポテンシャル計算を $O(N^6) \to O(R\,N^4) \to O(R\,r\,N^3)$ に段階的に削減した。
3. 対角補正により、近似による $r=0$ 付近の不一致を緩和し、Hartree エネルギーで目標精度 $O(10^{-3})$ を達成した。
4. PyTorch に経路を統一して計算時間を比較し、**理論計算量と実測の整合**を確認した。
5. 重い計算（VARPRO、RPCA）は SHA256 ベースの pkl キャッシュで再利用可能にした。

### 4.2 今後の展望

- **離散化誤差の打破**: グリッド細分化に対する $E$ 誤差の収束次数を再計測し、リーマン和を Simpson 則や trapezoidal で置き換えた場合の効果を見る。
- **短距離・長距離の分離**: 短距離は実空間（指数和 + 対角補正）、長距離は FFT で扱う Ewald 風スキームの導入（4/23 宿題の延長）。
- **RPCA 経路の最適化**: 閾値 $\tau$、sparse / dense 切替条件、$L$ のランクなどを系統的にスイープして実効最速点を探す。GPU バックエンドの効果検証も含む。
- **多中心・非対称な $\rho$ への拡張**: 現状は単一 Gaussian 中心の検証が中心。複数中心や DFT 由来の電子密度への適用に進む。
- **打ち切り次元の自動決定**: 特異値減衰やエネルギー保持比に基づくランク自動選択ルールを実装する。

---

## ディレクトリ構成

```text
region_PJ/
├── src/
│   ├── approximation/
│   │   ├── exp_sum.py              # VARPRO による 1/r ≈ Σ w_k exp(-α_k r²)
│   │   └── low_rank.py             # SVD / Tucker 統一インターフェース
│   ├── decomposition/
│   │   ├── svd.py                  # perform_svd
│   │   ├── tucker.py               # perform_tucker, perform_tucker_rpca
│   │   └── rpca.py                 # rpca, randomized_rpca
│   ├── potential/
│   │   ├── charge_potential.py     # FFT 解法、Tucker 評価ループ
│   │   └── poisson_solver.py       # 実空間 CG + monopole BC
│   └── utils/
│       ├── grid.py                 # 中心対称グリッド生成
│       ├── metrics.py              # relative_error, hartree_energy
│       ├── tensor_ops.py           # unfold, mode_n_product
│       └── cache.py                # pkl ベースの結果キャッシュ
├── notebooks/
│   ├── calc_exact_potential.ipynb
│   ├── calc_gaussian_potential_1D.ipynb
│   ├── calc_gaussian_potential_3D.ipynb
│   └── calc_potential_exp_expansion_approx.ipynb   # 主たる検証ノート
├── tests/
├── data/
│   └── processed/cache/            # VARPRO / RPCA 結果のキャッシュ
├── docs/coding_standards.md
├── References.md
├── homework.md                     # 進捗メモ（4/13, 4/23, 4/27, 5/1, 5/2）
├── requirements.txt
├── pyproject.toml
└── README.md
```

## セットアップ

### Windows (PowerShell)

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

### macOS / Linux

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## 使い方

### SVD / Tucker による低ランク近似

```python
import numpy as np
from src.approximation.low_rank import approximate

X_mat = np.random.randn(100, 80)
X_svd = approximate(X_mat, ranks=10, method="svd")

X_tensor = np.random.randn(20, 15, 10)
X_tucker = approximate(X_tensor, ranks=[5, 4, 3], method="tucker")
```

### 指数和フィッティング

```python
from src.approximation.exp_sum import LogUniformGrid, VarproOptimizer, BenchmarkRunner

fit_grid  = LogUniformGrid(r_min=1e-2, r_max=2*3**0.5*20, n_points=2000)
eval_grid = LogUniformGrid(r_min=1e-2, r_max=2*3**0.5*20, n_points=2000)

opt    = VarproOptimizer(fit_grid, eval_grid, nonneg=True, max_iter=200_000)
runner = BenchmarkRunner(opt, ranks=range(1, 31))
fits   = runner.run()
```

### 結果キャッシュ

```python
from src.utils.cache import load_or_compute

result = load_or_compute(
    namespace="exp_sum_fits",
    params={"R": 30, "N": 151, "L": 20.0},
    compute=lambda: runner.run(),
)
```

### 電荷ポテンシャルデモ

```python
from src.potential.charge_potential import run_charge_potential_demo

result = run_charge_potential_demo(N=32, ranks=[1, 2, 4, 8])
print(result["baseline_error"], result["ref_error"])
```

## テスト

```powershell
python -m pytest tests/
```

## 依存ライブラリ

| ライブラリ | 用途 |
|---|---|
| numpy | 数値計算・テンソル演算 |
| scipy | 特殊関数、`nnls`、`L-BFGS-B` |
| scikit-learn | randomized SVD |
| pytorch | 計算時間比較ベンチマークの統一バックエンド |
| sparse-dot-mkl | 旧経路でのスパース行列積（参考実装として保持） |
| pytest | テスト |

## メモ

1. ポテンシャル比較では定数シフト（ゲージ自由度）を補正して誤差を評価する。
2. 周期境界条件の影響を避けるため、内部マスク領域 ($R < L/3$ など) で誤差評価を行う。
3. 評価指標としては $V$ 単独の相対誤差より、Hartree エネルギー $E$ の相対誤差を優先する（5/1 メモ）。

## コーディング規約

[docs/coding_standards.md](docs/coding_standards.md) を参照。

## 参考文献・リンク

[References.md](References.md) を参照。Hackbusch のテンソル分解、Beylkin & Monzón の指数和近似、Robust PCA、ランダム化 SVD などへのリンクを掲載している。
