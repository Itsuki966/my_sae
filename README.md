# SAE迎合性分析 - 改善版

LLMの迎合性（sycophancy）を分析し、SAE（Sparse Autoencoder）を使用して内部メカニズムを可視化するプロジェクトです。

## 🎯 目的

1. **迎合性の検出**: LLMが正しい回答を知っているにも関わらず、人間の疑問や疑念によって回答を変更する傾向を分析
2. **内部メカニズムの解明**: SAEを用いてLLMの内部構造を可視化し、迎合性の原因を特定
3. **改善された分析手法**: より確実な単一選択肢抽出と包括的な分析

## 📁 ファイル構成

```
├── sae_sycophancy_analysis_clean.ipynb  # メインのノートブック（改善版）
├── sae_sycophancy_improved.py           # スタンドアロン実行スクリプト  
├── sae_sycophancy_hybrid.py             # ハイブリッド版（.py と .ipynb 両対応）
├── quick_sycophancy_test.py             # クイックテスト用スクリプト
├── sae_test_light.py                    # 軽量テスト（依存関係最小限）
├── setup_environment.py                 # 環境セットアップスクリプト
├── eval_dataset/
│   └── are_you_sure.jsonl              # 評価用データセット
└── README.md                           # このファイル
```

## 🚀 クイックスタート

### 1. 環境セットアップ
```bash
# 環境セットアップスクリプトを実行（推奨）
python setup_environment.py

# または手動でPoetry依存関係をインストール
poetry install
```

### 2. 軽量テスト
```bash
# 基本機能をテスト
python sae_test_light.py
```

### 3. 実行方法（3つの選択肢）

#### オプション A: ハイブリッド版（推奨）
```bash
# Python スクリプトとして実行
poetry run python sae_sycophancy_hybrid.py

# または Jupyter Notebook として実行
poetry run jupyter notebook
# ↳ sae_sycophancy_hybrid.py を .ipynb として開く
```

#### オプション B: スタンドアロン版  
```bash
poetry run python sae_sycophancy_improved.py
```

#### オプション C: 従来のNotebook版
```bash
poetry run jupyter notebook sae_sycophancy_analysis_clean.ipynb
```

## ⚙️ 設定カスタマイズ

### 主要な設定項目

実験設定は各ファイルの `ExperimentConfig` クラスで一元管理されています：

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

### よく変更する設定

#### 1. サンプル数の調整
```python
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
    repetition_penalty=1.2 # 繰り返しをより強く抑制
)
```

## 🔧 主な改善点

### 1. 単一選択肢抽出の改善
- **問題**: LLMが全ての選択肢を出力する
- **解決**: 改良されたプロンプト戦略と回答抽出パターン

```python
class ImprovedAnswerExtractor:
    def extract(self, response: str) -> str:
        # 優先度付きパターンマッチング
        high_priority_patterns = [
            r'^([ABCDE])$',  # 完全に単一文字のみ
            r'^([ABCDE])[\.\)]',  # A. or A)
            r'answer.*?([ABCDE])',  # "answer is A"
        ]
        # ...
```

### 2. 実験設定の一元管理
- **問題**: マジックナンバーが散在
- **解決**: `ExperimentConfig` クラスで設定を一元管理

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

## 🔍 使用例

### ノートブック版での設定変更
```python
# セル1: 設定の変更
config = ExperimentConfig(
    sample_size=30,        # テスト用に少なめ
    show_details=True,     # 詳細表示ON
    max_new_tokens=5,      # 短い回答を強制
    temperature=0.0        # 完全決定的
)
```

### スタンドアロン版での設定変更
```python
# sae_sycophancy_improved.py内で直接変更
config = ExperimentConfig(
    sample_size=100,
    show_details=False,    # 大量データ処理時は詳細表示OFF
    model_name="pythia-160m-deduped"
)
```

## 📈 期待される改善効果

1. **抽出成功率の向上**: UNKNOWN回答の大幅減少
2. **迎合性検出精度の向上**: より正確な迎合性パターンの特定
3. **分析の包括性**: 多角的な分析による深い洞察
4. **設定の柔軟性**: 様々な実験条件での簡単なテスト

## 🛠️ トラブルシューティング

### よくある問題と解決法

#### 1. UNKNOWN回答が多い
```python
config = ExperimentConfig(
    max_new_tokens=3,      # さらに短く
    temperature=0.0,       # 完全決定的
    debug_extraction=True  # デバッグ情報を表示
)
```

#### 2. メモリ不足
```python
config = ExperimentConfig(
    sample_size=10,        # サンプル数を減らす
    top_k_features=10      # 分析する特徴数を減らす
)
```

#### 3. 処理が遅い
```python
config = ExperimentConfig(
    show_details=False,    # 詳細表示をOFF
    max_examples_shown=1   # 表示例を最小限に
)
```

## 📚 参考情報

- **SAE Lens**: Sparse Autoencoder分析ライブラリ
- **Pythia**: 実験に使用しているLLMシリーズ
- **Are You Sure データセット**: 迎合性分析用のベンチマークデータ

## 🤝 貢献

改善提案やバグ報告は、以下の観点でお願いします：
1. 単一選択肢抽出の精度向上
2. 新しいSAE特徴分析手法
3. 可視化の改善
4. 実験設定の拡張

---

**注意**: このプロジェクトは研究目的で作成されています。実際の本番環境での使用には十分なテストを行ってください。
