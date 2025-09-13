#!/usr/bin/env python3
"""
量子化機能の簡単な動作確認スクリプト

このスクリプトは量子化機能の基本的な動作を確認します。
"""

import sys
import time
import torch
from config import QUANTIZED_4BIT_TEST_CONFIG

def check_dependencies():
    """必要な依存関係をチェック"""
    print("🔍 依存関係チェック中...")
    
    # PyTorch
    print(f"✅ PyTorch: {torch.__version__}")
    
    # CUDA/MPS
    if torch.cuda.is_available():
        print(f"✅ CUDA: 利用可能 ({torch.cuda.get_device_name()})")
    elif torch.backends.mps.is_available():
        print(f"✅ MPS: 利用可能 (Apple Silicon)")
    else:
        print(f"⚠️ GPU: 利用不可（CPUのみ）")
    
    # bitsandbytes
    try:
        import bitsandbytes as bnb
        print(f"✅ bitsandbytes: {bnb.__version__}")
        return True
    except ImportError:
        print("❌ bitsandbytes: インストールされていません")
        print("   インストールコマンド: pip install bitsandbytes")
        return False
    
    # transformers
    try:
        import transformers
        print(f"✅ transformers: {transformers.__version__}")
    except ImportError:
        print("❌ transformers: インストールされていません")
        return False
    
    # SAE Lens
    try:
        import sae_lens
        print(f"✅ sae_lens: 利用可能")
    except ImportError:
        print("❌ sae_lens: インストールされていません")
        return False
    
    return True

def test_basic_quantization():
    """基本的な量子化機能をテスト"""
    print("\n🧪 基本的な量子化テスト")
    
    try:
        from transformers import BitsAndBytesConfig
        
        # 4bit設定
        config_4bit = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16
        )
        print("✅ 4bit量子化設定作成成功")
        
        # 8bit設定
        config_8bit = BitsAndBytesConfig(
            load_in_8bit=True,
        )
        print("✅ 8bit量子化設定作成成功")
        
        return True
        
    except Exception as e:
        print(f"❌ 量子化設定作成エラー: {e}")
        return False

def test_analyzer_import():
    """SycophancyAnalyzerのインポートテスト"""
    print("\n🔍 アナライザーインポートテスト")
    
    try:
        from sycophancy_analyzer import SycophancyAnalyzer
        print("✅ SycophancyAnalyzer インポート成功")
        return True
    except Exception as e:
        print(f"❌ SycophancyAnalyzer インポートエラー: {e}")
        return False

def test_quantized_config():
    """量子化設定のテスト"""
    print("\n🔧 量子化設定テスト")
    
    try:
        config = QUANTIZED_4BIT_TEST_CONFIG
        print(f"✅ 設定読み込み成功")
        print(f"   モデル: {config.model.name}")
        print(f"   量子化: {config.model.quantization_config}")
        print(f"   4bit: {config.model.load_in_4bit}")
        print(f"   サンプル数: {config.data.sample_size}")
        return True
    except Exception as e:
        print(f"❌ 設定テストエラー: {e}")
        return False

def main():
    """メイン実行関数"""
    print("🚀 量子化機能 動作確認開始")
    print("=" * 50)
    
    # 依存関係チェック
    if not check_dependencies():
        print("\n❌ 必要な依存関係が不足しています")
        return False
    
    # 基本的な量子化テスト
    if not test_basic_quantization():
        print("\n❌ 基本的な量子化テストが失敗しました")
        return False
    
    # アナライザーインポートテスト
    if not test_analyzer_import():
        print("\n❌ アナライザーのインポートが失敗しました")
        return False
    
    # 量子化設定テスト
    if not test_quantized_config():
        print("\n❌ 量子化設定のテストが失敗しました")
        return False
    
    print("\n" + "=" * 50)
    print("🎉 すべてのテストが成功しました！")
    print("\n次のステップ:")
    print("   python test_quantization.py  # 完全な量子化テスト実行")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
