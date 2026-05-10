# コーディング規約

このプロジェクトでの実装・レビュー基準をまとめる。

## 1. スタイル

1. Python の書式は PEP 8 に準拠する。
2. 行長は 79 文字以下を原則とする。
3. import はファイル先頭に置き、グループ順は標準ライブラリ、外部ライブラリ、プロジェクト内とする。
4. 文字コードは UTF-8 を前提とする。

## 2. 型ヒント

1. 公開関数には型ヒントを付ける。
2. Union が分岐する場合は isinstance 等で型を絞り込む。
3. API 返り値が辞書の場合は TypedDict を優先して定義する。

## 3. ドックストリング

1. すべての公開関数に日本語ドックストリングを付与する。
2. 形式は NumPy/SciPy 風の docstring を推奨する。

- 要約行
- 空行
- Parameters
- Returns
- Raises（必要な場合）
- Notes（必要な場合）

3. 何を返すかだけでなく、前提条件や例外条件を明記する。
4. 配列引数・返り値には可能な範囲で shape を書く。

例:

```python
def apply_separable_kernel_3d(
    alpha: float,
    x: np.ndarray,
    rho: np.ndarray,
) -> np.ndarray:
    """3D Gaussian カーネルを密度テンソルに作用させる。

    Parameters
    ----------
    alpha : float
        Gaussian の幅パラメータ。
    x : np.ndarray, shape (N,)
        1D グリッド点の座標配列。
    rho : np.ndarray, shape (N, N, N)
        電荷密度テンソル。

    Returns
    -------
    result : np.ndarray, shape (N, N, N)
        exp(-alpha |r1-r2|^2) カーネルで rho を畳み込んだ結果。
    """
```

## 4. 数値実装の注意点

1. 数値比較では必要に応じてマスクや定数シフト補正を使う。
2. ゼロ除算や特異点近傍は np.divide の where 引数などで安全に扱う。
3. 物理量比較時は境界条件の影響をコメントで明示する。

## 5. テスト

1. 変更時は pytest を実行し、既存テストを通す。
2. 新規モジュールには最低限のスキーマ確認テストを追加する。
3. 負例テストは型チェッカーとの整合を保つ。

## 6. 推奨チェックコマンド

```powershell
python -m pycodestyle src tests notebooks/charge_potential_demo.py
python -m pytest tests/
```
