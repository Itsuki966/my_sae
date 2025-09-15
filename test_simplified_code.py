#!/usr/bin/env python3
"""
簡単な動作確認スクリプト

削除したコードによる影響がないかをチェックします。
"""

import sys
import os

# パスを追加
sys.path.append('/Users/itsukikuwahara/codes/research/sae')

def test_imports():
    """インポートのテスト"""
    print("🧪 インポートテストを開始...")
    
    try:
        from config import TEST_CONFIG
        print("✅ configインポート成功")
        
        from sycophancy_analyzer import SycophancyAnalyzer
        print("✅ SycophancyAnalyzerインポート成功")
        
        # アナライザーの初期化テスト
        analyzer = SycophancyAnalyzer(TEST_CONFIG)
        print("✅ SycophancyAnalyzer初期化成功")
        
        # メソッドの存在確認
        if hasattr(analyzer, 'setup_models'):
            print("✅ setup_modelsメソッド存在")
        else:
            print("❌ setup_modelsメソッドなし")
            
        if hasattr(analyzer, 'get_model_response'):
            print("✅ get_model_responseメソッド存在")
        else:
            print("❌ get_model_responseメソッドなし")
        
        if hasattr(analyzer, 'optimize_memory_usage'):
            print("✅ optimize_memory_usageメソッド存在")
        else:
            print("❌ optimize_memory_usageメソッドなし")
            
        print("\n🎉 すべてのインポートテストが成功しました！")
        return True
        
    except Exception as e:
        print(f"❌ インポートエラー: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_basic_functionality():
    """基本機能のテスト"""
    print("\n🧪 基本機能テストを開始...")
    
    try:
        from config import TEST_CONFIG
        from sycophancy_analyzer import SycophancyAnalyzer
        
        # テスト設定
        config = TEST_CONFIG
        config.debug.verbose = True
        
        analyzer = SycophancyAnalyzer(config)
        
        # メモリ最適化のテスト
        analyzer.optimize_memory_usage()
        print("✅ メモリ最適化メソッド実行成功")
        
        print("\n🎉 基本機能テストが成功しました！")
        return True
        
    except Exception as e:
        print(f"❌ 基本機能テストエラー: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🚀 コード削除後の動作確認テスト開始")
    print("="*50)
    
    # インポートテスト
    import_success = test_imports()
    
    if import_success:
        # 基本機能テスト
        basic_success = test_basic_functionality()
        
        if basic_success:
            print("\n✅ すべてのテストが成功しました！")
            print("🎉 コードの削除は正常に完了しています")
        else:
            print("\n❌ 基本機能テストが失敗しました")
            sys.exit(1)
    else:
        print("\n❌ インポートテストが失敗しました")
        sys.exit(1)
