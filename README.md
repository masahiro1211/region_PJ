# region_PJ

行列 SVD からテンソル Tucker 分解までを一貫して扱い、
低ランク近似と電荷ポテンシャル計算を検証する研究用プロジェクトです。

## 概要

このリポジトリでは、次の 2 系統を扱います。

1. 線形代数の基礎実装
2. 物理系デモ（ガウシアン電荷分布のポテンシャル計算）

SVD と Tucker の対応は以下です。

| SVD | Tucker |
|---|---|
| 左特異ベクトル行列 $U_r$ | 因子行列 $U_1, U_2, \ldots, U_N$ |
| 特異値行列 $\Sigma_r$ | コアテンソル $\mathcal{G}$ |
| 右特異ベクトル行列 $V_r$ | 各モードの因子行列 |

## 現在の機能

1. SVD による行列低ランク分解
2. HOSVD ベースの Tucker 分解と再構成
3. SVD/Tucker を統一した近似 API
4. FFT による電荷ポテンシャル計算
5. Tucker 圧縮時のポテンシャル誤差評価

## ディレクトリ構成

```text
region_PJ/
├── src/
│   ├── approximation/
│   │   └── low_rank.py            # SVD/Tucker 統一インターフェース
│   ├── decomposition/
│   │   ├── svd.py                 # perform_svd
│   │   └── tucker.py              # perform_tucker, reconstruct
│   ├── potential/
│   │   ├── __init__.py
│   │   └── charge_potential.py    # 電荷ポテンシャル評価ロジック
│   └── utils/
│       ├── metrics.py             # relative_error
│       └── tensor_ops.py          # unfold, mode_n_product
├── notebooks/
│   └── charge_potential_demo.py   # 実行用デモスクリプト
├── tests/
│   ├── test_svd.py
│   ├── test_tucker.py
│   ├── test_low_rank.py
│   └── test_charge_potential.py
├── data/
│   ├── raw/
│   └── processed/
├── results/
│   ├── figures/
│   └── logs/
├── requirements.txt
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

### SVD による低ランク近似

```python
import numpy as np
from src.decomposition.svd import perform_svd

X = np.random.randn(100, 80)
A, B = perform_svd(X, rank=10)
X_approx = A @ B
```

### Tucker 分解

```python
import numpy as np
from src.decomposition.tucker import perform_tucker, reconstruct

X = np.random.randn(20, 15, 10)
G, factors = perform_tucker(X, ranks=[5, 4, 3])
X_approx = reconstruct(G, factors)
```

### 統一インターフェース

```python
from src.approximation.low_rank import approximate

X_tucker = approximate(X, ranks=[5, 4, 3], method="tucker")
X_svd = approximate(X_matrix, ranks=10, method="svd")
```

### 電荷ポテンシャルデモ

```powershell
python notebooks/charge_potential_demo.py
```

モジュールとして利用する場合:

```python
from src.potential.charge_potential import run_charge_potential_demo

result = run_charge_potential_demo(N=32, ranks=[1, 2, 4, 8])
print(result["baseline_error"], result["ref_error"])
```

## テスト

```powershell
python -m pytest tests/
```

直近では 14 件のテストが通過しています。

## 依存ライブラリ

| ライブラリ | 用途 |
|---|---|
| numpy | 数値計算・テンソル演算 |
| scipy | 特殊関数、線形代数 |
| pytest | テスト |

## メモ

1. ポテンシャル比較では定数シフト（ゲージ自由度）を補正して誤差を評価します。
2. 境界近傍は周期境界条件の影響があるため、内部マスク領域で比較します。

## コーディング規約

コーディング規約は以下を参照してください。

1. [docs/coding_standards.md](docs/coding_standards.md)
