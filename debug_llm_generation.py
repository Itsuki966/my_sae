#!/usr/bin/env python3
"""
LLM生成問題のデバッグ用テストスクリプト

このスクリプトでは、LLMの生成が空になる原因を特定するための
段階的なテストを実行します。
"""

import torch
import os
import sys
from typing import Optional

# 環境変数設定
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

try:
    from sae_lens import SAE, HookedSAETransformer
    SAE_AVAILABLE = True
except ImportError:
    print("❌ SAE Lensが利用できません")
    SAE_AVAILABLE = False
    sys.exit(1)

from config import TEST_CONFIG, LLAMA3_TEST_CONFIG

class LLMGenerationDebugger:
    """LLM生成問題のデバッグクラス"""
    
    def __init__(self, config):
        self.config = config
        self.model = None
        self.tokenizer = None
        self.device = config.model.device
        
    def test_basic_model_loading(self):
        """基本的なモデル読み込みテスト"""
        print("=" * 50)
        print("🔍 基本的なモデル読み込みテスト")
        print("=" * 50)
        
        try:
            print(f"📱 デバイス: {self.device}")
            print(f"🤖 モデル: {self.config.model.name}")
            
            # モデル読み込み
            print("🔄 モデル読み込み中...")
            self.model = HookedSAETransformer.from_pretrained(
                self.config.model.name,
                device=self.device
            )
            print("✅ モデル読み込み成功")
            
            # Tokenizerの取得
            self.tokenizer = self.model.tokenizer
            print("✅ Tokenizer取得成功")
            
            # 基本情報の表示
            print(f"🔧 語彙サイズ: {self.tokenizer.vocab_size}")
            print(f"🔧 EOSトークンID: {self.tokenizer.eos_token_id}")
            print(f"🔧 パッドトークンID: {getattr(self.tokenizer, 'pad_token_id', 'None')}")
            
            return True
            
        except Exception as e:
            print(f"❌ モデル読み込みエラー: {e}")
            return False
    
    def test_simple_tokenization(self):
        """シンプルなトークン化テスト"""
        print("\n" + "=" * 50)
        print("🔍 シンプルなトークン化テスト")
        print("=" * 50)
        
        test_text = "Hello, world!"
        print(f"📝 テストテキスト: '{test_text}'")
        
        try:
            # トークン化
            tokens = self.tokenizer.encode(test_text, return_tensors="pt")
            print(f"🔢 トークン: {tokens}")
            print(f"🔢 トークン数: {tokens.shape[1]}")
            
            # デコード
            decoded = self.tokenizer.decode(tokens[0], skip_special_tokens=True)
            print(f"📝 デコード結果: '{decoded}'")
            
            # デバイス移動
            tokens_on_device = tokens.to(self.device)
            print(f"📱 デバイス移動: {tokens.device} → {tokens_on_device.device}")
            
            return True
            
        except Exception as e:
            print(f"❌ トークン化エラー: {e}")
            return False
    
    def test_simple_generation(self):
        """シンプルな生成テスト"""
        print("\n" + "=" * 50)
        print("🔍 シンプルな生成テスト")
        print("=" * 50)
        
        test_prompt = "The capital of France is"
        print(f"📝 テストプロンプト: '{test_prompt}'")
        
        try:
            # トークン化
            inputs = self.tokenizer.encode(test_prompt, return_tensors="pt").to(self.device)
            original_length = inputs.shape[1]
            print(f"🔢 入力トークン数: {original_length}")
            
            # シンプルな生成（最小パラメータ）
            with torch.no_grad():
                try:
                    outputs = self.model.generate(
                        inputs,
                        max_new_tokens=5,  # 最小限のトークン数
                        temperature=1.0,   # 標準的な温度
                        do_sample=False,   # グリーディ生成
                        pad_token_id=self.tokenizer.eos_token_id
                    )
                    
                    # 生成された部分を取得
                    generated_part = outputs[0][original_length:]
                    response = self.tokenizer.decode(generated_part, skip_special_tokens=True)
                    
                    print(f"✅ 生成成功: '{response}'")
                    return True
                    
                except Exception as gen_error:
                    print(f"⚠️ generate()メソッドエラー: {gen_error}")
                    return self.test_manual_generation(inputs, original_length)
                    
        except Exception as e:
            print(f"❌ 生成テストエラー: {e}")
            return False
    
    def test_manual_generation(self, inputs: torch.Tensor, original_length: int):
        """手動生成テスト（フォールバック）"""
        print("🔄 手動生成テストに切り替え")
        
        try:
            generated_tokens = inputs.clone()
            
            with torch.no_grad():
                for step in range(3):  # 3トークンのみ生成
                    # フォワードパス
                    logits = self.model(generated_tokens)
                    next_token_logits = logits[0, -1, :]
                    
                    # グリーディ選択
                    next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                    
                    # トークン追加
                    generated_tokens = torch.cat([generated_tokens, next_token.unsqueeze(0)], dim=1)
                    
                    # 中間結果表示
                    current_text = self.tokenizer.decode(
                        generated_tokens[0, original_length:], 
                        skip_special_tokens=True
                    )
                    print(f"ステップ {step + 1}: '{current_text}'")
                    
                    # EOSチェック
                    if next_token.item() == self.tokenizer.eos_token_id:
                        print("🛑 EOSトークンで終了")
                        break
            
            # 最終結果
            final_text = self.tokenizer.decode(
                generated_tokens[0, original_length:], 
                skip_special_tokens=True
            )
            print(f"✅ 手動生成成功: '{final_text}'")
            return True
            
        except Exception as e:
            print(f"❌ 手動生成エラー: {e}")
            return False
    
    def test_choice_question_generation(self):
        """選択肢質問での生成テスト"""
        print("\n" + "=" * 50)
        print("🔍 選択肢質問での生成テスト")
        print("=" * 50)
        
        # シンプルな選択肢質問
        prompt = """Question: What is 2 + 2?

Options:
(A) 3
(B) 4
(C) 5

Select the best answer. Respond with only the letter (A-C).
"""
        
        print(f"📝 テストプロンプト:")
        print(prompt)
        print("-" * 30)
        
        try:
            # トークン化
            inputs = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
            original_length = inputs.shape[1]
            
            # 複数の生成設定でテスト
            test_configs = [
                {"max_new_tokens": 1, "temperature": 0.0, "do_sample": False},
                {"max_new_tokens": 3, "temperature": 0.0, "do_sample": False},
                {"max_new_tokens": 5, "temperature": 0.1, "do_sample": True},
                {"max_new_tokens": 10, "temperature": 0.5, "do_sample": True},
            ]
            
            for i, config in enumerate(test_configs):
                print(f"\n🧪 テスト設定 {i+1}: {config}")
                
                try:
                    with torch.no_grad():
                        outputs = self.model.generate(
                            inputs,
                            **config,
                            pad_token_id=self.tokenizer.eos_token_id
                        )
                        
                        generated_part = outputs[0][original_length:]
                        response = self.tokenizer.decode(generated_part, skip_special_tokens=True)
                        
                        print(f"  生成結果: '{response}'")
                        print(f"  長さ: {len(response)} 文字")
                        
                        if response.strip():
                            print("  ✅ 非空の応答")
                        else:
                            print("  ❌ 空の応答")
                            
                except Exception as e:
                    print(f"  ❌ エラー: {e}")
            
            return True
            
        except Exception as e:
            print(f"❌ 選択肢質問テストエラー: {e}")
            return False
    
    def run_all_tests(self):
        """全テストの実行"""
        print("🚀 LLM生成デバッグテストを開始")
        
        tests = [
            ("モデル読み込み", self.test_basic_model_loading),
            ("トークン化", self.test_simple_tokenization),
            ("シンプル生成", self.test_simple_generation),
            ("選択肢質問生成", self.test_choice_question_generation),
        ]
        
        results = {}
        
        for test_name, test_func in tests:
            if test_name in ["トークン化", "シンプル生成", "選択肢質問生成"] and not hasattr(self, 'model'):
                print(f"⏭️ {test_name}をスキップ（モデル読み込み失敗のため）")
                results[test_name] = False
                continue
                
            try:
                success = test_func()
                results[test_name] = success
            except Exception as e:
                print(f"❌ {test_name}で予期しないエラー: {e}")
                results[test_name] = False
        
        # 結果サマリー
        print("\n" + "=" * 50)
        print("📊 テスト結果サマリー")
        print("=" * 50)
        
        for test_name, success in results.items():
            status = "✅ 成功" if success else "❌ 失敗"
            print(f"{test_name}: {status}")
        
        # 推奨事項
        print("\n🔧 推奨される修正:")
        if not results.get("モデル読み込み", False):
            print("- モデル読み込み処理の修正が必要")
        if not results.get("シンプル生成", False):
            print("- 生成パラメータの調整が必要")
            print("- デバイス設定の確認が必要")
        if results.get("シンプル生成", False) and not results.get("選択肢質問生成", False):
            print("- プロンプト形式の最適化が必要")

def main():
    """メイン関数"""
    print("🔬 LLM生成問題デバッグスクリプト")
    
    # GPT-2のテスト
    print("\n🤖 GPT-2モデルのテスト")
    gpt2_debugger = LLMGenerationDebugger(TEST_CONFIG)
    gpt2_debugger.run_all_tests()
    
    # Llama3のテスト（オプション）
    print("\n🦙 Llama3モデルのテスト")
    try:
        llama_debugger = LLMGenerationDebugger(LLAMA3_TEST_CONFIG)
        llama_debugger.run_all_tests()
    except Exception as e:
        print(f"⚠️ Llama3テストをスキップ: {e}")

if __name__ == "__main__":
    main()
