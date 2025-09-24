# LLM迎合性分析プロジェクト

LLMの迎合性（sycophancy）を分析し、SAE（Sparse Autoencoder）を使用して内部メカニズムを可視化するプロジェクトです。

## 概要

このプロジェクトは、LLMが「本当にその答えで合ってる？」という挑戦的な質問に対してどの程度回答を変更するかを分析し、その際の内部表現（SAE特徴）を可視化します。

## 主要ファイル

- `sycophancy_analyzer.py`: メイン分析スクリプト
- `config.py`: 実験設定管理ファイル
- `eval_dataset/are_you_sure.jsonl`: 評価用データセット

## 環境要件

- Python 3.10+
- Poetry または pip を使用した依存関係管理

### ローカル環境（Mac/開発用）

```bash
# Poetry使用（推奨）
poetry install

# pip使用
pip install -r requirements.txt
```

### サーバー環境（Linux/CUDA環境）

```bash
# 基本依存関係のインストール
poetry install

# flash-attnを含むサーバー専用依存関係のインストール
poetry install --with server

# または個別インストール
poetry add flash-attn --group server
```

**注意**: flash-attnはLinux+CUDA環境でのみ動作します。Mac環境では自動的にスキップされます。

## メインスクリプトの使用方法

### 基本実行

```bash
# 軽量テスト（GPT-2、サンプル数5）
python sycophancy_analyzer.py --mode test

# 環境自動選択実行
python sycophancy_analyzer.py --mode auto

# Llama3テスト実行
python sycophancy_analyzer.py --mode llama3-test

# Gemma-2Bテスト実行
python sycophancy_analyzer.py --mode gemma-2b-test
```

### 主要オプション

- `--mode`: 実行モード
  - `test`: GPT-2軽量テスト（サンプル数5）
  - `production`: GPT-2本番実行
  - `llama3-test`: Llama3テスト（サンプル数5）
  - `llama3-prod`: Llama3本番実行（サンプル数1000）
  - `llama3-memory`: Llama3メモリ効率化実行
  - `gemma-2b-test`: Gemma-2Bテスト（サンプル数5）
  - `gemma-2b-prod`: Gemma-2B本番実行（サンプル数1000）
  - `auto`: 環境に応じて自動選択

- `--sample-size`: サンプル数を上書き
- `--verbose`: 詳細ログ表示
- `--debug`: プロンプト・応答の表示
- `--memory-limit`: 最大メモリ使用量（GB）
- `--use-fp16`: float16精度の強制使用

## 実験設定の変更方法

## 実験設定の変更方法

設定は `config.py` ファイルで管理されています。主要な設定項目：

### ModelConfig（モデル設定）
```python
@dataclass
class ModelConfig:
    name: str = "gpt2"                      # 使用モデル
    sae_release: str = "gpt2-small-res-jb"  # SAEリリース名
    sae_id: str = "blocks.5.hook_resid_pre" # SAE ID
    device: str = "auto"                    # デバイス設定
    use_fp16: bool = True                   # float16使用（メモリ節約）
    max_memory_gb: float = None             # 最大メモリ使用量
```

### DataConfig（データ設定）
```python
@dataclass
class DataConfig:
    sample_size: int = 50    # 分析サンプル数
    random_seed: int = 42    # 再現性用シード
```

### GenerationConfig（生成設定）
```python
@dataclass
class GenerationConfig:
    max_new_tokens: int = 5  # 生成トークン数
    temperature: float = 0   # 生成温度（0=決定的）
    top_p: float = 0.9       # top-pサンプリング
```

### プリセット設定

- `TEST_CONFIG`: GPT-2軽量テスト（サンプル数5）
- `LLAMA3_TEST_CONFIG`: Llama3テスト（サンプル数5）
- `GEMMA2B_TEST_CONFIG`: Gemma-2Bテスト（サンプル数5）
- `SERVER_LARGE_CONFIG`: サーバー用大規模実験（サンプル数1000）
- `MAC_CONFIG`: Mac環境用軽量設定

## 主要なマジックナンバー

### SAE設定
- `blocks.5.hook_resid_pre`: GPT-2のブロック5の残差接続前
- `blocks.21.hook_resid_pre`: Llama3のブロック21の残差接続前
- `blocks.12.hook_resid_post`: Gemma-2Bのブロック12の残差接続後
- `top_k_features=20`: 分析する上位特徴数
- `activation_threshold=0.1`: SAE活性化閾値

### 生成設定
- `max_new_tokens=5`: 選択肢（A-E）回答に十分なトークン数
- `temperature=0`: 決定的な生成（再現性確保）
- `top_p=0.9`: 上位90%の確率質量からサンプリング

### メモリ最適化
- `use_fp16=True`: メモリ使用量を約半分に削減
- `low_cpu_mem_usage=True`: モデル読み込み時のCPU使用量削減
- `device_map="auto"`: 自動デバイス配置

## 実行結果

### 出力ファイル
- `results/analysis_results_[model]_[samples].json`: 詳細分析結果
- `plots/overview.html`: 概要ダッシュボード
- `plots/accuracy_comparison.html`: 精度比較グラフ

### 分析指標
- **迎合率**: 挑戦的質問で回答を変更した割合
- **初回正答率**: 最初の回答の正確性
- **挑戦後正答率**: 挑戦後の回答の正確性