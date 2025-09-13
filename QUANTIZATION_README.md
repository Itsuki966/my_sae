# 量子化機能テスト

## 概要

このプロジェクトに bitsandbytes による 4bit/8bit 量子化機能を実装しました。これにより、メモリ使用量を大幅に削減してより大きなLLMモデル（Llama3など）を使用できるようになります。

## 新機能

### 1. 量子化設定（config.py）

新しい量子化オプションが追加されました：

```python
# 4bit量子化設定
QUANTIZED_4BIT_TEST_CONFIG = ExperimentConfig(
    model=ModelConfig(
        name="meta-llama/Llama-3.2-3B",
        use_quantization=True,
        quantization_config="4bit",
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype="float16",
        max_memory_gb=8.0,
        # ...
    )
)

# 8bit量子化設定
QUANTIZED_8BIT_TEST_CONFIG = ExperimentConfig(
    model=ModelConfig(
        use_quantization=True,
        quantization_config="8bit",
        load_in_8bit=True,
        # ...
    )
)
```

### 2. SycophancyAnalyzer の量子化対応

`sycophancy_analyzer.py` に量子化対応の新しいメソッドが追加されました：

- `create_quantization_config()`: BitsAndBytesConfig の作成
- `setup_models_with_quantization()`: 量子化を使用したモデル読み込み
- 自動フォールバック機能（量子化 → accelerate → 標準）

## テストファイル

### 1. test_quantization_simple.py

基本的な依存関係と量子化機能をチェックする軽量テスト：

```bash
python3 test_quantization_simple.py
```

### 2. test_quantization.py

実際にLlama3モデルを量子化して読み込み、迎合性分析を実行するフルテスト：

```bash
python3 test_quantization.py
```

### 3. run_quantization_test.sh

必要な依存関係の自動インストールとテスト実行を行うスクリプト：

```bash
bash run_quantization_test.sh
```

## インストールと実行

### 1. 必要な依存関係

```bash
pip install bitsandbytes accelerate psutil
```

### 2. 簡単テスト実行

```bash
# 依存関係の自動インストールとテスト実行
bash run_quantization_test.sh

# または手動実行
python3 test_quantization_simple.py  # 基本テスト
python3 test_quantization.py         # フルテスト
```

## 量子化の効果

### メモリ使用量の削減

- **4bit量子化**: 約75%のメモリ削減
- **8bit量子化**: 約50%のメモリ削減

### 例：Llama-3.2-3B モデル

| 設定 | 推定メモリ使用量 | 説明 |
|------|------------------|------|
| Float32 | ~12GB | 標準精度 |
| Float16 | ~6GB | 半精度 |
| 8bit量子化 | ~3GB | 50%削減 |
| 4bit量子化 | ~1.5GB | 75%削減 |

## 設定の選び方

### 環境別推奨設定

1. **Mac (Apple Silicon)**
   - `QUANTIZED_4BIT_TEST_CONFIG` または `QUANTIZED_8BIT_TEST_CONFIG`
   - メモリ制限: 8GB以下

2. **GPU環境（8GB以下）**
   - `QUANTIZED_4BIT_TEST_CONFIG`
   - より積極的なメモリ節約

3. **GPU環境（16GB以上）**
   - `QUANTIZED_8BIT_TEST_CONFIG`
   - 精度とメモリのバランス

### カスタム設定例

```python
from config import ExperimentConfig, ModelConfig

# カスタム量子化設定
custom_config = ExperimentConfig(
    model=ModelConfig(
        name="meta-llama/Llama-3.2-3B",
        use_quantization=True,
        quantization_config="4bit",
        load_in_4bit=True,
        max_memory_gb=6.0,  # 使用環境に応じて調整
    ),
    data=DataConfig(sample_size=10),  # テスト用に少なく
)
```

## トラブルシューティング

### 1. bitsandbytes インストールエラー

```bash
# CUDA環境の場合
pip install bitsandbytes --extra-index-url https://download.pytorch.org/whl/cu118

# CPU/MPS環境の場合
pip install bitsandbytes
```

### 2. メモリ不足エラー

- `max_memory_gb` を小さく設定
- `sample_size` を減らす
- より積極的な量子化設定（4bit）を使用

### 3. モデル読み込みエラー

アナライザーは自動フォールバック機能を持っています：
1. 量子化読み込み試行
2. accelerate読み込み試行
3. 標準読み込み（フォールバック）

## パフォーマンス注意点

### 精度への影響

- **4bit量子化**: 若干の精度低下（通常1-3%）
- **8bit量子化**: 最小限の精度低下（通常1%未満）

### 実行速度

- 量子化により初期読み込み時間は短縮
- 推論速度は環境により変動
- メモリ効率性の向上により安定性が向上

## 使用例

```python
from config import QUANTIZED_4BIT_TEST_CONFIG
from sycophancy_analyzer import SycophancyAnalyzer

# 量子化アナライザーの作成
analyzer = SycophancyAnalyzer(QUANTIZED_4BIT_TEST_CONFIG)

# 迎合性分析の実行
results = analyzer.analyze_sycophancy()

# 結果表示
print(f"迎合率: {results['summary']['sycophancy_rate']:.2%}")
```

この量子化機能により、限られたメモリ環境でもより大きなモデルを使用した迎合性分析が可能になります。
