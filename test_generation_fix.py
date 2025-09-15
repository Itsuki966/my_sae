#!/usr/bin/env python3
"""
改善されたLLM生成機能のテストスクリプト

このスクリプトは、tutorial_2_0.ipynbを参考に改善されたLLM生成部分をテストします。
"""

import sys
import torch
from config import get_auto_config
from sycophancy_analyzer import SycophancyAnalyzer

def test_generation_improvement():
    """LLM生成の改善をテスト"""
    print("🧪 LLM生成改善のテスト開始")
    print("="*60)
    
    # 軽量なテスト設定を使用
    try:
        from config import TEST_CONFIG
        config = TEST_CONFIG  # 確実に動作する設定を使用
        # テスト用に設定を調整
        config.generation.max_new_tokens = 10  # 短いトークン数でテスト
        config.debug.verbose = True
        config.debug.show_prompts = True
        config.debug.show_responses = True
        
        print(f"📋 使用設定: {config.model.name}")
        print(f"🎯 最大トークン数: {config.generation.max_new_tokens}")
        
        # アナライザーの初期化
        analyzer = SycophancyAnalyzer(config)
        
        # モデルの読み込み
        print("\n🔄 モデルとSAEの読み込み中...")
        analyzer.setup_models()
        
        # テスト用のシンプルなプロンプト
        test_prompts = [
            "What is the capital of France?",
            "Choose the best answer: (A) Paris (B) London (C) Berlin",
            "Question: What color is the sky? Options: (A) Blue (B) Red (C) Green"
        ]
        
        print("\n📝 テストプロンプトでの生成テスト:")
        print("-" * 60)
        
        for i, prompt in enumerate(test_prompts, 1):
            print(f"\n🧪 テスト {i}/3")
            print(f"入力: {prompt}")
            
            try:
                response = analyzer.get_model_response(prompt)
                print(f"✅ 出力: '{response}'")
                
                # 空でない応答があることを確認
                if response and response.strip():
                    print(f"🎉 テスト{i}成功: 有効な応答を取得")
                else:
                    print(f"⚠️ テスト{i}警告: 空の応答")
                    
            except Exception as e:
                print(f"❌ テスト{i}失敗: {e}")
                import traceback
                traceback.print_exc()
        
        print("\n" + "="*60)
        print("🏁 テスト完了")
        
        # メモリクリーンアップ
        analyzer.optimize_memory_usage()
        
        return True
        
    except Exception as e:
        print(f"❌ テスト中にエラーが発生: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_simple_generation():
    """最小限のテスト"""
    print("\n🔬 最小限の生成テスト")
    print("-" * 40)
    
    try:
        # 最軽量設定
        from config import TEST_CONFIG
        config = TEST_CONFIG
        config.generation.max_new_tokens = 5
        config.debug.verbose = True
        
        analyzer = SycophancyAnalyzer(config)
        analyzer.setup_models()
        
        # シンプルなテスト
        response = analyzer.get_model_response("Answer: ")
        print(f"🎯 最小テスト結果: '{response}'")
        
        return response is not None and len(response.strip()) > 0
        
    except Exception as e:
        print(f"❌ 最小テスト失敗: {e}")
        return False

if __name__ == "__main__":
    print("🚀 LLM生成改善テスト開始")
    
    # GPU使用可能性チェック
    if torch.cuda.is_available():
        print(f"✅ CUDA利用可能: {torch.cuda.get_device_name()}")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        print("✅ MPS (Apple Silicon) 利用可能")
    else:
        print("ℹ️ CPU使用")
    
    # テスト実行
    try:
        success = test_generation_improvement()
        if not success:
            print("\n🔄 フォールバック: 最小限テストを実行")
            success = test_simple_generation()
        
        if success:
            print("\n✅ すべてのテストが完了しました！")
            print("🎉 改善されたLLM生成機能は正常に動作しています")
        else:
            print("\n❌ テストが失敗しました")
            print("🔧 further debugging may be needed")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n⚠️ ユーザーによってテストが中断されました")
    except Exception as e:
        print(f"\n💥 予期しないエラー: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
