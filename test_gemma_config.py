#!/usr/bin/env python3
"""
Gemma-2B設定のテストスクリプト

新しく追加したgemma-2bの設定が正しく動作するかを確認します。
"""

import sys
import os

def test_gemma_config():
    """Gemma-2B設定のテスト"""
    print("🧪 Gemma-2B設定テストを開始...")
    
    try:
        # config.pyから新しい設定をインポート
        from config import GEMMA2B_TEST_CONFIG, GEMMA2B_PROD_CONFIG
        
        print("✅ Gemma-2B設定のインポートに成功")
        
        # テスト設定の確認
        print("\n📋 GEMMA2B_TEST_CONFIG:")
        print(f"  モデル名: {GEMMA2B_TEST_CONFIG.model.name}")
        print(f"  SAE リリース: {GEMMA2B_TEST_CONFIG.model.sae_release}")
        print(f"  SAE ID: {GEMMA2B_TEST_CONFIG.model.sae_id}")
        print(f"  サンプルサイズ: {GEMMA2B_TEST_CONFIG.data.sample_size}")
        print(f"  デバイス: {GEMMA2B_TEST_CONFIG.model.device}")
        print(f"  詳細出力: {GEMMA2B_TEST_CONFIG.debug.verbose}")
        
        # 本番設定の確認
        print("\n🚀 GEMMA2B_PROD_CONFIG:")
        print(f"  モデル名: {GEMMA2B_PROD_CONFIG.model.name}")
        print(f"  SAE リリース: {GEMMA2B_PROD_CONFIG.model.sae_release}")
        print(f"  SAE ID: {GEMMA2B_PROD_CONFIG.model.sae_id}")
        print(f"  サンプルサイズ: {GEMMA2B_PROD_CONFIG.data.sample_size}")
        print(f"  デバイス: {GEMMA2B_PROD_CONFIG.model.device}")
        print(f"  accelerate使用: {GEMMA2B_PROD_CONFIG.model.use_accelerate}")
        print(f"  float16使用: {GEMMA2B_PROD_CONFIG.model.use_fp16}")
        
        # 設定値の検証
        assert GEMMA2B_TEST_CONFIG.model.name == "gemma-2b-it"
        assert GEMMA2B_TEST_CONFIG.model.sae_release == "gemma-2b-it-res-jb"
        assert GEMMA2B_TEST_CONFIG.model.sae_id == "blocks.12.hook_resid_post"
        assert GEMMA2B_TEST_CONFIG.data.sample_size == 5
        
        assert GEMMA2B_PROD_CONFIG.model.name == "gemma-2b-it"
        assert GEMMA2B_PROD_CONFIG.model.sae_release == "gemma-2b-it-res-jb"
        assert GEMMA2B_PROD_CONFIG.model.sae_id == "blocks.12.hook_resid_post"
        assert GEMMA2B_PROD_CONFIG.data.sample_size == 1000
        
        print("\n✅ 設定値の検証に成功")
        
    except ImportError as e:
        print(f"❌ インポートエラー: {e}")
        return False
    except AssertionError as e:
        print(f"❌ 設定値検証エラー: {e}")
        return False
    except Exception as e:
        print(f"❌ 予期しないエラー: {e}")
        return False
    
    return True

def test_sycophancy_analyzer_import():
    """sycophancy_analyzer.pyでのインポートテスト"""
    print("\n🔧 sycophancy_analyzer.pyでのインポートテスト...")
    
    try:
        # sycophancy_analyzer.pyから設定をインポート
        sys.path.insert(0, os.path.dirname(__file__))
        
        # メイン関数の部分だけインポートしてテスト
        from sycophancy_analyzer import get_config_from_mode
        
        print("✅ get_config_from_mode関数のインポートに成功")
        
        # モック引数を作成
        class MockArgs:
            sample_size = None
            memory_limit = None
            use_fp16 = False
            disable_accelerate = False
            verbose = False
            debug = False
        
        mock_args = MockArgs()
        
        # テスト設定の取得
        test_config = get_config_from_mode('gemma-2b-test', mock_args)
        print(f"✅ gemma-2b-test設定の取得に成功: {test_config.model.name}")
        
        # 本番設定の取得
        prod_config = get_config_from_mode('gemma-2b-prod', mock_args)
        print(f"✅ gemma-2b-prod設定の取得に成功: {prod_config.model.name}")
        
        # 設定値の検証
        assert test_config.model.name == "gemma-2b-it"
        assert prod_config.model.name == "gemma-2b-it"
        assert test_config.data.sample_size == 5
        assert prod_config.data.sample_size == 1000
        
        print("✅ sycophancy_analyzer.pyでの設定取得に成功")
        
    except Exception as e:
        print(f"❌ sycophancy_analyzer.pyテストエラー: {e}")
        return False
    
    return True

def main():
    """メインテスト関数"""
    print("🚀 Gemma-2B設定の包括的テストを開始")
    print("=" * 60)
    
    success = True
    
    # 基本設定テスト
    if not test_gemma_config():
        success = False
    
    # sycophancy_analyzer.pyでのテスト
    if not test_sycophancy_analyzer_import():
        success = False
    
    print("\n" + "=" * 60)
    if success:
        print("🎉 すべてのテストが成功しました！")
        print("\n📋 使用可能なコマンド:")
        print("  python sycophancy_analyzer.py --mode gemma-2b-test")
        print("  python sycophancy_analyzer.py --mode gemma-2b-prod")
        print("  python sycophancy_analyzer.py --mode gemma-2b-test --sample-size 10")
        print("  python sycophancy_analyzer.py --mode gemma-2b-prod --verbose")
    else:
        print("❌ 一部のテストが失敗しました")
        sys.exit(1)

if __name__ == "__main__":
    main()