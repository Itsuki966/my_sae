# SAE迎合性分析プロジェクト

LLMの迎合性（sycophancy）を分析し、SAE（Sparse Autoencoder）を使用して内部メカニズムを可視化するプロジェクトです。

## 🎯 目的

1. **迎合性の検出**: LLMが正しい回答を知っているにも関わらず、人間の疑問や疑念によって回答を変更する傾向を分析
2. **内部メカニズムの解明**: SAEを用いてLLMの内部構造を可視化し、迎合性の原因を特定
3. **改善された分析手法**: より確実な単一選択肢抽出と包括的な分析

## 📁 ファイル構成

### 🚀 メイン実行ファイル
```
├── sae_sycophancy_hybrid.py             # 統合版スクリプト（推奨）
├── sae_sycophancy_analysis_clean.ipynb  # メインNotebook（.pyからクラスをインポート）
```

### 📊 分析・学習用Notebook
```
├── sae_are_you_sure_analysis.ipynb      # Are You Sure分析詳細版
├── tutorial_2_0.ipynb                   # SAE学習・理解用チュートリアル
├── demo_sae_lens.ipynb                  # SAE Lensデモンストレーション
├── sample.ipynb                         # サンプルコード・実験用
├── sae_sycophancy.ipynb                 # 迎合性分析の初期版
```

### 📂 データ・設定
```
├── eval_dataset/
│   └── are_you_sure.jsonl              # 評価用データセット
├── pyproject.toml                       # Poetry依存関係管理
├── poetry.lock                          # 依存関係ロック
└── README.md                           # このファイル
```

## 🚀 クイックスタート

### 1. 環境セットアップ
```bash
# Poetry環境で依存関係をインストール
poetry install

# または pip で直接インストール
pip install torch sae-lens transformers plotly pandas numpy tqdm
```

### 2. 実行方法

#### 🎯 迎合性分析（メイン機能）

**方法A: 統合版スクリプト実行（推奨）**
```bash
# 最も簡単な実行方法
poetry run python sae_sycophancy_hybrid.py

# または直接実行
python sae_sycophancy_hybrid.py
```

**方法B: Jupyter Notebook**
```bash
# Jupyter起動
poetry run jupyter notebook

# sae_sycophancy_analysis_clean.ipynb を開いて実行
```

#### 📚 学習・デモ用Notebook

**SAE学習チュートリアル**
```bash
jupyter notebook tutorial_2_0.ipynb
```

**SAE Lensデモ**
```bash  
jupyter notebook demo_sae_lens.ipynb
```

**詳細分析**
```bash
jupyter notebook sae_are_you_sure_analysis.ipynb
```

## ⚙️ 設定カスタマイズ

### 主要な設定項目

実験設定は `sae_sycophancy_hybrid.py` の `ExperimentConfig` クラスで管理されています：

```python
@dataclass
class ExperimentConfig:
    # === モデル設定 ===
    model_name: str = "pythia-70m-deduped"           # 使用するLLMモデル
    sae_release: str = "pythia-70m-deduped-res-sm"   # SAEのリリース名
    sae_id: str = "blocks.5.hook_resid_post"         # 使用するSAEのID
    
    # === データ設定 ===
    sample_size: int = 50                            # 分析するサンプル数
    
    # === 生成設定 ===
    max_new_tokens: int = 8                          # 生成する最大トークン数
    temperature: float = 0.1                         # 生成の温度（低いほど決定的）
    
    # === 分析設定 ===
    top_k_features: int = 20                         # 分析する特徴の数
    show_details: bool = True                        # 詳細表示の有無
```

### 設定変更例

#### 1. サンプル数の調整
```python
# sae_sycophancy_hybrid.py内のconfig変数を変更
config = ExperimentConfig(
    sample_size=100,  # より多くのサンプルで分析
)
```

#### 2. モデルの変更
```python
config = ExperimentConfig(
    model_name="pythia-160m-deduped",  # より大きなモデルを使用
    sae_release="pythia-160m-deduped-res-sm",
    sae_id="blocks.7.hook_resid_post",
)
```

#### 3. 生成設定の調整
```python
config = ExperimentConfig(
    max_new_tokens=5,      # より短い回答を強制
    temperature=0.0,       # 完全に決定的
)
```

## 🔧 主な機能と改善点

### 1. 迎合性分析の核心機能
- **Are You Sure タスク**: LLMの回答を疑問視して再考を促す実験
- **単一選択肢抽出**: 改善されたパターンマッチングで確実な回答抽出
- **SAE特徴分析**: 内部表現の変化を可視化

### 2. 改善された分析手法
```python
class ImprovedAnswerExtractor:
    def extract(self, response: str) -> str:
        # 優先度付きパターンマッチング
        high_priority_patterns = [
            r'^([ABCDE])$',         # 完全に単一文字のみ
            r'^([ABCDE])[\.\)]',    # A. or A)
            r'answer.*?([ABCDE])',  # "answer is A"
        ]
        # ...
```

### 3. 包括的な分析・可視化
- 迎合性率、改善率、品質向上率の分析
- SAE特徴の一貫性分析
- 迎合性 vs 改善の比較分析

## 📊 出力される分析結果

### 1. 基本統計
- 最初の回答精度
- 最終回答精度
- 回答変更率
- 迎合性率

### 2. パターン分析
- 迎合性発生件数
- 改善発生件数
- 品質向上・劣化の件数

### 3. SAE特徴分析
- 迎合性に関連する特徴の特定
- 特徴活性化の変化量
- 一貫性スコア

### 4. 可視化
- 回答パターンの分布
- 特徴活性化の変化
- 迎合性 vs 改善の比較

## 🔍 各ファイルの詳細説明

### メイン実行ファイル

#### `sae_sycophancy_hybrid.py` ⭐推奨⭐
- **用途**: 迎合性分析のメイン実行ファイル
- **特徴**: Pythonスクリプトとしてもノートブックとしても実行可能
- **実行方法**: `python sae_sycophancy_hybrid.py`
- **含まれる機能**:
  - `ExperimentConfig`: 実験設定管理
  - `improved_extract_answer_letter()`: 改善された回答抽出
  - `run_sycophancy_analysis()`: メイン分析関数
  - `comprehensive_analyze_sycophancy_results()`: 包括的分析
  - `comprehensive_plot_sycophancy_results()`: 結果可視化

#### `sae_sycophancy_analysis_clean.ipynb`
- **用途**: Notebook形式での詳細分析
- **特徴**: `.py`からクラスと関数をインポートして使用
- **実行方法**: Jupyter Notebookで開いて実行
- **適用場面**: 
  - 対話的な分析を行いたい場合
  - 途中結果を確認しながら進めたい場合
  - 設定を細かく調整したい場合

### 学習・デモ用Notebook

#### `tutorial_2_0.ipynb`
- **用途**: SAE（Sparse Autoencoder）の学習と理解
- **対象**: SAEの基本概念を学びたい方
- **内容**: 
  - SAEの理論的背景
  - 実装例
  - 可視化手法

#### `demo_sae_lens.ipynb`
- **用途**: SAE Lensライブラリのデモンストレーション
- **対象**: SAE Lensの使い方を学びたい方
- **内容**:
  - ライブラリの基本的な使い方
  - 特徴抽出の例
  - 可視化機能

#### `sae_are_you_sure_analysis.ipynb`
- **用途**: Are You Sure タスクの詳細分析
- **対象**: 迎合性分析の詳細を理解したい方
- **内容**:
  - タスクの詳細な実装
  - 結果の深い分析
  - 問題ケースの調査

#### `sample.ipynb`
- **用途**: 実験・テスト用のサンプルコード
- **対象**: 新しいアイデアを試したい方
- **内容**: 
  - 様々なサンプルコード
  - 実験的な機能

#### `sae_sycophancy.ipynb`
- **用途**: 迎合性分析の初期版・参考用
- **対象**: 開発の歴史を確認したい方

## 🛠️ トラブルシューティング

### よくある問題と解決法

#### 1. UNKNOWN回答が多い場合
```python
config = ExperimentConfig(
    max_new_tokens=3,      # さらに短く
    temperature=0.0,       # 完全決定的
    debug_extraction=True  # デバッグ情報を表示
)
```

#### 2. メモリ不足の場合
```python
config = ExperimentConfig(
    sample_size=10,        # サンプル数を減らす
    top_k_features=10      # 分析する特徴数を減らす
)
```

#### 3. 処理が遅い場合
```python
config = ExperimentConfig(
    show_details=False,    # 詳細表示をOFF
    max_examples_shown=1   # 表示例を最小限に
)
```

#### 4. SAE Lensが見つからない場合
```bash
# インストール
pip install sae-lens

# または Poetry環境で
poetry add sae-lens
```

#### 5. データファイルが見つからない場合
```bash
# eval_dataset ディレクトリが存在することを確認
ls -la eval_dataset/

# データファイルが存在することを確認
ls -la eval_dataset/are_you_sure.jsonl
```

## 🎯 推奨ワークフロー

### 初心者向け
1. `tutorial_2_0.ipynb` でSAEの基本を学習
2. `demo_sae_lens.ipynb` でSAE Lensの使い方を習得
3. `sae_sycophancy_hybrid.py` で迎合性分析を実行

### 研究者向け
1. `sae_sycophancy_hybrid.py` で基本的な分析を実行
2. `sae_sycophancy_analysis_clean.ipynb` で詳細な対話的分析
3. `sae_are_you_sure_analysis.ipynb` で深い分析と検証

### 開発者向け
1. `sample.ipynb` で新機能をプロトタイプ
2. `sae_sycophancy_hybrid.py` に統合
3. テストと検証

## 📚 参考情報

- **SAE Lens**: [Sparse Autoencoder分析ライブラリ](https://github.com/jbloomAus/SAELens)
- **Pythia**: [実験に使用しているLLMシリーズ](https://github.com/EleutherAI/pythia)
- **Are You Sure データセット**: 迎合性分析用のベンチマークデータ
- **Transformer Lens**: [モデル分析用ライブラリ](https://github.com/neelnanda-io/TransformerLens)

## 🤝 貢献・カスタマイズ

改善提案やカスタマイズは、以下の観点で行ってください：

### 1. 分析精度の向上
- 単一選択肢抽出の精度向上
- 新しいパターンマッチング手法
- より効果的なプロンプト戦略

### 2. 可視化の改善
- より直感的なグラフ
- インタラクティブな可視化
- 詳細なSAE特徴分析

### 3. 実験設定の拡張
- 新しいモデルサポート
- 異なるデータセット対応
- カスタム分析メトリクス

### 4. パフォーマンス最適化
- メモリ使用量の削減
- 処理速度の向上
- バッチ処理の改善

---

**注意**: このプロジェクトは研究目的で作成されています。実際の本番環境での使用には十分なテストを行ってください。
