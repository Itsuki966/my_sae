# SAE迎合性分析プロジェクト

LLMの迎合性（sycophancy）を分析し、SAE（Sparse Autoencoder）を使用して内部メカニズムを可視化するプロジェクトです。

## 🎯 目的

1. **迎合性の検出**: LLMが正しい回答を知っているにも関わらず、人間の疑問や疑念によって回答を変更する傾向を分析
2. **内部メカニズムの解明**: SAEを用いてLLMの内部構造を可視化し、迎合性の原因を特定
3. **改善された分析手法**: より確実な単一選択肢抽出と包括的な分析

## 📁 整理後のファイル構成

### 🚀 メイン実行ファイル
```
├── sae_sycophancy_hybrid.py             # 統合版スクリプト（推奨）
├── sae_sycophancy_analysis_clean.ipynb  # メインNotebook（.pyからクラスをインポート）
```

### ⚙️ 設定・テストファイル
```
├── experiment_config.py                 # 実験設定管理（NEW!）
├── sae_test_light.py                    # 軽量テスト版（NEW!）
```

### 📊 学習・分析用Notebook
```
├── sae_are_you_sure_analysis.ipynb      # Are You Sure分析詳細版
├── tutorial_2_0.ipynb                   # SAE学習・理解用チュートリアル
```

### 📂 データ・プロジェクト管理
```
├── eval_dataset/
│   ├── are_you_sure.jsonl              # 評価用データセット
│   ├── answer.jsonl                     # 回答データ
│   └── feedback.jsonl                   # フィードバックデータ
├── pyproject.toml                       # Poetry依存関係管理
├── poetry.lock                          # 依存関係ロック
└── README.md                           # このファイル
```

## 🚀 使用方法

### 1. 初期設定
```bash
# 依存関係のインストール
poetry install

# 環境のアクティベート
poetry shell
```

### 2. 実行方法

#### 方法A: Pythonスクリプト直接実行（推奨）
```bash
# 統合版メインスクリプト（改善された分析機能付き）
python sae_sycophancy_hybrid.py

# 軽量テスト版（動作確認用）
python sae_test_light.py
```

#### 方法B: Jupyter Notebook実行
```bash
# JupyterLabを起動
jupyter lab

# または個別にnotebookを起動
jupyter notebook sae_sycophancy_analysis_clean.ipynb
```

### 3. 実験設定のカスタマイズ（NEW!）

新しい設定管理システム（`experiment_config.py`）を使用して、実験パラメータを簡単に変更できます：

```python
from experiment_config import ExperimentConfig, get_quick_test_config, get_full_analysis_config

# 事前定義された設定を使用
config = get_quick_test_config()  # 高速テスト用

# カスタム設定を作成
config = ExperimentConfig(
    n_samples=50,         # サンプル数
    batch_size=8,         # バッチサイズ  
    model_name="gpt-3.5-turbo",  # モデル名
    n_top_features=20,    # 表示する特徴量数
    min_confidence=0.8,   # 最小信頼度
    seed=42              # 再現性のためのシード
)
```

#### 利用可能な設定プリセット
- `get_quick_test_config()`: 高速テスト用（10サンプル）
- `get_full_analysis_config()`: 完全分析用（500サンプル）
- `get_debug_config()`: デバッグ用（詳細ログ有効）

### 4. 軽量テスト実行（NEW!）

プロジェクトの基本機能をテストするには：

```bash
python sae_test_light.py
```

このテストでは以下を確認します：
- データ読み込み機能
- 回答抽出システム
- プロンプト構築
- 設定管理システム

## 🔧 実験設定の詳細

### 主要パラメータ

| パラメータ名 | 説明 | デフォルト値 |
|-------------|------|-------------|
| `n_samples` | 分析するサンプル数 | 100 |
| `batch_size` | バッチサイズ | 4 |
| `model_name` | 使用するモデル | "gpt-3.5-turbo" |
| `sae_repo_id` | SAEリポジトリID | "jbloom/GPT2-Small-SAEs-Reformatted" |
| `layer_id` | 分析対象レイヤー | 6 |
| `n_top_features` | 表示する特徴量数 | 10 |
| `min_confidence` | 最小信頼度閾値 | 0.7 |
| `temperature` | 生成時の温度パラメータ | 0.1 |
| `max_tokens` | 最大トークン数 | 50 |

### 設定例

```python
# 高速プロトタイプ用
quick_config = ExperimentConfig(
    n_samples=10,
    batch_size=2,
    verbose=True
)

# 本格分析用
full_config = ExperimentConfig(
    n_samples=1000,
    batch_size=16,
    n_top_features=50,
    detailed_analysis=True
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
- **特徴**: 統合された分析機能とエラーハンドリング
- **実行方法**: `python sae_sycophancy_hybrid.py`
- **含まれる機能**:
  - `ImprovedAnswerExtractor`: 改善された回答抽出
  - 迎合性分析のメイン機能
  - 包括的な結果分析と可視化

#### `sae_sycophancy_analysis_clean.ipynb`
- **用途**: Notebook形式での詳細分析
- **特徴**: `.py`からクラスと関数をインポートして使用
- **実行方法**: Jupyter Notebookで開いて実行
- **適用場面**: 
  - 対話的な分析を行いたい場合
  - 途中結果を確認しながら進めたい場合
  - 設定を細かく調整したい場合

### 設定・テストファイル

#### `experiment_config.py` 🆕
- **用途**: 実験設定の統合管理
- **特徴**: マジックナンバーを排除し、設定を集約
- **含まれる機能**:
  - `ExperimentConfig`: 設定データクラス
  - プリセット設定（クイックテスト、フル分析、デバッグ）
  - バリデーション機能

#### `sae_test_light.py` 🆕
- **用途**: 軽量テスト版（動作確認用）
- **特徴**: 基本機能のみ実装、高速実行
- **実行方法**: `python sae_test_light.py`
- **テスト内容**:
  - データ読み込みテスト
  - 回答抽出テスト
  - プロンプト構築テスト
  - 設定管理テスト

### 学習・分析用Notebook

#### `tutorial_2_0.ipynb`
- **用途**: SAE（Sparse Autoencoder）の学習と理解
- **対象**: SAEの基本概念を学びたい方
- **内容**: 
  - SAEの理論的背景
  - 実装例
  - 可視化手法

#### `sae_are_you_sure_analysis.ipynb`
- **用途**: Are You Sure タスクの詳細分析
- **対象**: 迎合性分析の詳細を理解したい方
- **内容**:
  - タスクの詳細な実装
  - 結果の深い分析
  - 問題ケースの調査

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
1. `sae_test_light.py` で基本機能の動作確認
2. `tutorial_2_0.ipynb` でSAEの基本を学習
3. `sae_sycophancy_hybrid.py` で迎合性分析を実行

### 研究者向け
1. `experiment_config.py` で実験設定をカスタマイズ
2. `sae_sycophancy_hybrid.py` で基本的な分析を実行
3. `sae_sycophancy_analysis_clean.ipynb` で詳細な対話的分析
4. `sae_are_you_sure_analysis.ipynb` で深い分析と検証

### 開発者向け
1. `sae_test_light.py` で機能テスト
2. `experiment_config.py` で新しい設定を追加
3. `sae_sycophancy_hybrid.py` に新機能を統合

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
