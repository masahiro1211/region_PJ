# 研究の流れ

この文書は、README から切り出した研究ログ寄りの説明を整理したものです。入口としての概要は [../../README.md](../../README.md) を参照してください。

## 1. 問題設定

3 次元電荷密度 `rho(r)` から Hartree ポテンシャル

$$
V_H(\mathbf{r}_1) = \int \frac{\rho(\mathbf{r}_2)}{|\mathbf{r}_1 - \mathbf{r}_2|}\,d\mathbf{r}_2
$$

を求める。3D グリッドで直接評価すると、各軸の点数を `N` として `O(N^6)` になる。FFT は周期境界では強いが、本研究では Coulomb カーネルを分離可能な形に近似し、実空間側で計算量を落とす経路を調べる。

## 2. ベースライン

### FFT 解法

[src/potential/charge_potential.py](../../src/potential/charge_potential.py) では、周期境界条件のもとで

$$
\hat V(\mathbf{k}) = \frac{4\pi \hat\rho(\mathbf{k})}{|\mathbf{k}|^2}
$$

を評価する FFT 経路を持つ。`k=0` はゲージ自由度として 0 に固定する。

### 実空間 Poisson 解法

[src/potential/poisson_solver.py](../../src/potential/poisson_solver.py) では、モノポール境界条件 `V|boundary = Q/r` を与え、中央差分 Laplacian に対して共役勾配法で解く。これは参照用の実空間解法として位置づけている。

## 3. Coulomb カーネルの指数和近似

Coulomb カーネルを

$$
\frac{1}{r} \approx \sum_{k=1}^{R} w_k e^{-\alpha_k r^2}
$$

で近似する。フィットは [src/approximation/exp_sum/varpro.py](../../src/approximation/exp_sum/varpro.py) の VARPRO 経路で行う。

- 非線形パラメータ: `log(alpha_k)` を `L-BFGS-B` で最適化
- 線形パラメータ: `w_k >= 0` を NNLS で解く
- グリッド: log-uniform
- 重いフィット結果は [src/utils/cache.py](../../src/utils/cache.py) のキャッシュで再利用

代表的な関数近似誤差では、`R >= 20` 付近で `1/r` の近似自体はおおむね `1e-3` を切る。

## 4. Gaussian 分離による 3D 評価

Gaussian は

$$
e^{-\alpha|\mathbf{r}_1-\mathbf{r}_2|^2}
= e^{-\alpha(x_1-x_2)^2}e^{-\alpha(y_1-y_2)^2}e^{-\alpha(z_1-z_2)^2}
$$

と分離できる。したがって 3D 畳み込みは 1D カーネル

$$
K_{ij}=e^{-\alpha(x_i-x_j)^2}
$$

を 3 軸に逐次適用する mode-n product になる。これにより `O(N^6)` から `O(R N^4)` へ落ちる。

主な実装は [src/approximation/exp_sum/separable.py](../../src/approximation/exp_sum/separable.py) と [src/potential/separable_density.py](../../src/potential/separable_density.py)。

## 5. CP 形式の密度

密度そのものが CP 形式

$$
\rho = \sum_m c_m\,\rho_m^x \otimes \rho_m^y \otimes \rho_m^z
$$

で与えられる場合、各指数和項のポテンシャルも

$$
V = dx^3 \sum_k w_k \sum_m c_m
(K_k\rho_m^x) \otimes (K_k\rho_m^y) \otimes (K_k\rho_m^z)
$$

として扱える。この経路では `K_k @ rho_m^x` のような 1D matvec だけで CP 形式の `V` を構成し、必要な場合だけ dense な `N^3` テンソルへ materialize する。

- NumPy 側: [src/potential/separable_density.py](../../src/potential/separable_density.py)
- PyTorch 側: [src/approximation/cp_coulomb.py](../../src/approximation/cp_coulomb.py)

エネルギー `E = 1/2 int rho V` は CP 項同士の 1D 内積の積で直接評価できるため、`V` の具現化を避けられる。

## 6. 1D カーネルの SVD 圧縮

1D カーネルを

$$
K \approx U_r \Sigma_r V_r^T
$$

と近似する。重要なのは、`K_r = U_r Sigma_r V_r^T` を先に再構成しないこと。正しい順序は各軸に対して

$$
V_r^T \rightarrow \Sigma_r \rightarrow U_r
$$

で作用させることで、中間軸の次元を `r` に保つ。この順序で初めて `O(r N^3)` の利得が出る。

[src/approximation/torch_kernels.py](../../src/approximation/torch_kernels.py) には、比較用に以下の経路がある。

- `apply_exp_sum_3d_full`: full 1D カーネルを 3 軸に適用
- `apply_exp_sum_3d_lowrank_naive`: 低ランク行列を再構成してから適用する失敗例
- `apply_exp_sum_3d_lowrank`: 正しい順序で低ランク因子を適用

## 7. RPCA による L + S 分解

Gaussian カーネルには、広域の滑らかな成分と対角近傍の局在ピークが混在する。そこで RPCA により

$$
K = L + S
$$

に分ける。

- `L`: 滑らかな低ランク成分
- `S`: 対角近傍の局在成分

`S` をしきい値 `tau` で丸め、十分に疎な場合だけ PyTorch の sparse CSR 経路を使う。現行実装では非ゼロ率 `0.1%` 以下を sparse 切替の目安にしている。

RPCA の実装は [src/decomposition/rpca.py](../../src/decomposition/rpca.py)。PyTorch 評価経路は [src/approximation/torch_kernels.py](../../src/approximation/torch_kernels.py)。

## 8. 対角補正

指数和近似では `r=0` の特異性を表せず、中心値は `sum_k w_k` にとどまる。そこで一辺 `dx` のセル内平均として

$$
K_{diag} \approx \frac{2.38}{dx}
$$

を使い、

$$
K_{diag} - \sum_k w_k
$$

を対角寄与として後付けする。Hartree エネルギー誤差はこの補正で大きく改善する。

## 9. 評価指標

主指標は Hartree エネルギー

$$
E_H = \frac{1}{2}\int \rho V\,d^3r
$$

の相対誤差。`V` の相対誤差だけでは、エネルギーとして重要な寄与との対応が弱い場合があるため、最終的な精度判断は `E_H` を優先する。

notebook の現在の構成では、指数和の離散化評価は連続系の解析エネルギー `analytic_gaussian_hartree_energy` との比較として整理している。

## 10. 代表的な知見

- `R >= 15` 程度の指数和近似と対角補正で、Hartree エネルギー相対誤差 `O(1e-3)` に到達する。
- `R` を増やしても、グリッド幅 `dx` に由来する離散化誤差で頭打ちになる。
- low-rank 評価は、因子を正しい順序で作用させた場合だけ計算量の利得が出る。
- RPCA は精度面では `L` の有効ランクを下げる効果が見える。一方で、Python ループや sparse/dense 分岐のオーバーヘッドにより、現状では常に高速とは限らない。
- ベンチマークは PyTorch に統一し、NumPy/SciPy/PyTorch のバックエンド差が混ざらないようにしている。

## 11. 主な実験入口

- [notebooks/calc_potential_exp_expansion_approx.ipynb](../../notebooks/calc_potential_exp_expansion_approx.ipynb): 主たる検証 notebook
- [scripts/run_exp_sum_fitting.py](../../scripts/run_exp_sum_fitting.py): 指数和フィット
- [scripts/run_rpca_kernels.py](../../scripts/run_rpca_kernels.py): 1D カーネルの SVD/RPCA 分解保存
- [scripts/run_rpca_error_sweep.py](../../scripts/run_rpca_error_sweep.py): SVD/RPCA 誤差 sweep
- [scripts/run_timing_benchmark.py](../../scripts/run_timing_benchmark.py): PyTorch 統一タイミング

## 12. 今後の課題

- 離散化誤差の収束次数をより体系的に測る。
- `tau`、`r`、sparse/dense 切替閾値を同時に最適化する。
- RPCA 経路の kernel loop をバッチ化し、Python 側のオーバーヘッドを減らす。
- 多中心・非対称密度や DFT 由来密度へ拡張する。
- CP 密度経路の入力生成と実データ接続を整える。
