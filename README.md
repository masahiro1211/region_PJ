# region_PJ

SVD による行列分解からタッカーテンソル分解への拡張を目的とした研究プロジェクト。

## 概要

ガウス型電荷分布など空間的な密度場を低ランクテンソルで近似することを最終目標とする。  
まず行列（2次元）に対する SVD 分解を実装し、それをタッカー分解（HOSVD）へ一般化する。

### SVD との対応

| SVD | Tucker |
|---|---|
| 左特異ベクトル行列 $U_r$ | 因子行列 $U_1, U_2, \ldots, U_N$ |
| 特異値行列 $\Sigma_r$ | コアテンソル $\mathcal{G}$ |
| 右特異ベクトル行列 $V_r$ | 各モードの因子行列 |

## フォルダ構成

```
region_PJ/
├── src/
│   ├── decomposition/
│   │   ├── svd.py          # perform_svd: X ≈ A @ B
│   │   └── tucker.py       # perform_tucker: X ≈ G ×1 U1 ×2 U2 ...
│   ├── approximation/
│   │   └── low_rank.py     # SVD / Tucker の統一インターフェース
│   └── utils/
│       ├── tensor_ops.py   # unfold, mode_n_product
│       └── metrics.py      # relative_error
├── notebooks/              # 実験用 Jupyter ノートブック
├── data/
│   ├── raw/                # 生データ（Git 管理外）
│   └── processed/          # 前処理済みデータ（Git 管理外）
├── results/
│   ├── figures/            # プロット出力（Git 管理外）
│   └── logs/               # 実験ログ（Git 管理外）
├── tests/
│   ├── test_svd.py
│   └── test_tucker.py
├── requirements.txt
└── README.md
```

## セットアップ

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## 使い方

### SVD による低ランク近似

```python
import numpy as np
from src.decomposition.svd import perform_svd

X = np.random.randn(100, 80)
A, B = perform_svd(X, rank=10)
X_approx = A @ B  # X ≈ A @ B
```

### タッカー分解

```python
import numpy as np
from src.decomposition.tucker import perform_tucker, reconstruct

X = np.random.randn(20, 15, 10)   # 3次元テンソル
G, factors = perform_tucker(X, ranks=[5, 4, 3])
X_approx = reconstruct(G, factors)
```

### 統一インターフェース

```python
from src.approximation.low_rank import approximate

# Tucker（テンソル）
X_approx = approximate(X, ranks=[5, 4, 3], method="tucker")

# SVD（行列）
X_approx = approximate(X_matrix, ranks=10, method="svd")
```

## テスト

```bash
pytest tests/
```

## 依存ライブラリ

| ライブラリ | 用途 |
|---|---|
| numpy | 数値計算・線形代数 |
| scipy | SVD (`linalg.svd`) |
| pytest | テスト |
