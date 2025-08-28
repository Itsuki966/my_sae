#!/usr/bin/env python3
"""
Llama3の回答生成修正をクイックテストするスクリプト
"""

def test_llama3_generation():
    """Llama3の回答生成テスト"""
    print("🦙 Llama3修正版クイックテスト")
    print("="*50)
    
    try:
        from sycophancy_analyzer import SycophancyAnalyzer
        from config import LLAMA3_TEST_CONFIG
        
        print("🔧 分析器初期化中...")
        analyzer = SycophancyAnalyzer(LLAMA3_TEST_CONFIG)
        
        print("🔄 モデル読み込み中...")
        analyzer.setup_models()
        
        print("\n📝 シンプルなテストプロンプト:")
        simple_prompt = "What is 2 + 2? Answer: A) 3, B) 4, C) 5. Just respond with the letter:"
        
        print(f"プロンプト: {simple_prompt}")
        print("="*60)
        
        response = analyzer.get_model_response(simple_prompt)
        
        print("="*60)
        print(f"✅ 応答: '{response}'")
        print(f"📏 応答長: {len(response)}文字")
        
        if response and len(response.strip()) > 0:
            print("🎉 回答生成成功!")
            
            # 回答抽出テスト
            extracted = analyzer.extract_answer_letter(response)
            if extracted:
                print(f"✅ 回答抽出成功: {extracted}")
                if extracted == 'B':
                    print("🎯 正解! (2+2=4)")
                else:
                    print(f"⚠️ 期待した回答と異なります (期待: B, 実際: {extracted})")
            else:
                print("⚠️ 回答抽出失敗")
            
            return True
        else:
            print("❌ 応答が空です")
            return False
            
    except Exception as e:
        print(f"❌ テストエラー: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_llama3_generation()
