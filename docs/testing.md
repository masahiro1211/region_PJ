# テスト規約

このプロジェクトでは `pytest` を使う。

テスト関数は、実装を開かなくても確認している振る舞いが分かるように
書く。以下の規約を使う。

- テスト名は振る舞いを説明する。例:
  `test_split_kernel_near_far_reconstructs_original_kernel`.
- 各テストには1行の日本語 docstring を付ける。
- docstring は「前提」「操作」「期待結果」が分かる形にする。
- 数値テストでは小さく決定的な配列を優先し、
  `np.testing.assert_allclose` を使う。
- notebook 規模の重い実験は unit test ではなく notebook または
  benchmark に置く。

推奨コマンド:

```bash
python -m pytest -q
python -m pycodestyle src tests
```
