#!/usr/bin/env python3
"""
修正されたコードの動作確認用テストスクリプト

このスクリプトは以下の機能をテストします：
1. 基本的なインポートと設定
2. LLMとの対話（プロンプトと応答の表示）
3. 実際のデータセットを使った迎合性分析

新機能：
- デバッグ設定による詳細な出力制御
- プロンプトと応答の自動表示
- SAE活性化の詳細情報表示
- TEST_CONFIG による詳細デバッグモード
"""

import sys
import os

def test_imports():
    """インポートテスト"""
    print("📦 インポートテスト開始...")
    
    try:
        from config import ExperimentConfig, DEFAULT_CONFIG, LIGHTWEIGHT_CONFIG, TEST_CONFIG
        print("✅ config.py のインポート成功")
        
        from sycophancy_analyzer import SycophancyAnalyzer
        print("✅ sycophancy_analyzer.py のインポート成功")
        
        return True
    except Exception as e:
        print(f"❌ インポートエラー: {e}")
        return False

def test_config():
    """設定テスト"""
    print("\n⚙️ 設定テスト開始...")
    
    try:
        from config import DEFAULT_CONFIG, LIGHTWEIGHT_CONFIG, TEST_CONFIG
        
        # 設定の基本チェック
        print(f"✅ デフォルト設定 - モデル: {DEFAULT_CONFIG.model.name}")
        print(f"✅ デフォルト設定 - SAE: {DEFAULT_CONFIG.model.sae_id}")
        print(f"✅ デフォルト設定 - サンプルサイズ: {DEFAULT_CONFIG.data.sample_size}")
        print(f"✅ デフォルト設定 - デバイス: {DEFAULT_CONFIG.model.device}")
        
        print(f"✅ 軽量設定 - サンプルサイズ: {LIGHTWEIGHT_CONFIG.data.sample_size}")
        print(f"✅ 軽量設定 - デバッグ出力: {LIGHTWEIGHT_CONFIG.debug.show_prompts}")
        
        print(f"✅ テスト設定 - サンプルサイズ: {TEST_CONFIG.data.sample_size}")
        print(f"✅ テスト設定 - 詳細デバッグ: {TEST_CONFIG.debug.show_activations}")
        
        return True
    except Exception as e:
        print(f"❌ 設定テストエラー: {e}")
        return False

def test_analyzer_init():
    """分析器初期化テスト"""
    print("\n🔬 分析器初期化テスト開始...")
    
    try:
        from sycophancy_analyzer import SycophancyAnalyzer
        from config import LIGHTWEIGHT_CONFIG
        
        # 分析器の初期化
        analyzer = SycophancyAnalyzer(LIGHTWEIGHT_CONFIG)
        
        print("✅ SycophancyAnalyzer 初期化成功")
        print(f"✅ 設定されたモデル: {analyzer.config.model.name}")
        print(f"✅ 設定されたデバイス: {analyzer.device}")
        
        return True
    except Exception as e:
        print(f"❌ 分析器初期化エラー: {e}")
        return False

def test_dataset_loading():
    """データセット読み込みテスト"""
    print("\n📂 データセット読み込みテスト開始...")
    
    try:
        from sycophancy_analyzer import SycophancyAnalyzer
        from config import LIGHTWEIGHT_CONFIG
        
        analyzer = SycophancyAnalyzer(LIGHTWEIGHT_CONFIG)
        
        # データセットファイルの存在確認
        dataset_path = analyzer.config.data.dataset_path
        if not os.path.exists(dataset_path):
            print(f"⚠️ データセットファイルが見つかりません: {dataset_path}")
            print("📝 eval_dataset/are_you_sure.jsonl を配置してください")
            return False
        
        # データセット読み込みテスト
        dataset = analyzer.load_dataset()
        print(f"✅ データセット読み込み成功: {len(dataset)}件")
        
        # データ構造確認
        if dataset:
            sample = dataset[0]
            print(f"✅ サンプルデータ構造確認: {list(sample.keys())}")
            if 'base' in sample:
                print(f"✅ base キー内容: {list(sample['base'].keys())}")
        
        return True
    except Exception as e:
        print(f"❌ データセット読み込みエラー: {e}")
        return False

def test_llm_interaction():
    """LLM対話テスト（プロンプトと応答の確認）"""
    print("\n🤖 LLM対話テスト開始...")
    
    try:
        from sycophancy_analyzer import SycophancyAnalyzer
        from config import TEST_CONFIG
        
        # テスト用設定を使用（詳細なデバッグ出力有効）
        analyzer = SycophancyAnalyzer(TEST_CONFIG)
        
        # モデルセットアップ（時間がかかる可能性あり）
        print("⚠️ モデルセットアップ中... (時間がかかる場合があります)")
        analyzer.setup_models()
        
        # テスト用のサンプルプロンプト
        test_prompt = """Question: What is the capital of France?

A) London
B) Berlin  
C) Paris
D) Madrid

Please choose the best answer (A, B, C, or D):"""
        
        print("\n" + "="*60)
        print("🧪 テスト用プロンプトと応答の確認")
        print("="*60)
        
        # LLMからの応答取得（デバッグ出力が自動で表示される）
        response = analyzer.get_model_response(test_prompt)
        
        # 応答の分析
        extracted_answer = analyzer.extract_answer_letter(response)
        print(f"\n📊 抽出された回答: {extracted_answer}")
        
        # テスト成功の条件チェック
        if response and len(response.strip()) > 0:
            print("\n✅ LLM対話テスト成功")
            print(f"✅ 応答長: {len(response)}文字")
            if extracted_answer:
                print(f"✅ 回答抽出成功: {extracted_answer}")
            else:
                print("⚠️ 回答抽出は失敗しましたが、応答は取得できました")
            return True
        else:
            print("\n❌ LLMから応答を取得できませんでした")
            return False
            
    except Exception as e:
        print(f"❌ LLM対話テストエラー: {e}")
        print("💡 ヒント: モデルのダウンロードには時間がかかります")
        return False

def test_sycophancy_analysis_sample():
    """迎合性分析サンプルテスト（実際のデータを使用）"""
    print("\n🎯 迎合性分析サンプルテスト開始...")
    
    try:
        from sycophancy_analyzer import SycophancyAnalyzer
        from config import TEST_CONFIG
        
        # テスト用設定を使用（詳細なデバッグ出力有効）
        analyzer = SycophancyAnalyzer(TEST_CONFIG)
        
        # データセット確認
        if not os.path.exists(analyzer.config.data.dataset_path):
            print(f"⚠️ データセットファイルが見つかりません: {analyzer.config.data.dataset_path}")
            return False
        
        dataset = analyzer.load_dataset()
        if not dataset:
            print("❌ データセットが空です")
            return False
        
        print(f"✅ データセット読み込み成功: {len(dataset)}件")
        
        # モデルセットアップ
        print("⚠️ モデルセットアップ中... (初回は時間がかかります)")
        analyzer.setup_models()
        
        # 1つのサンプルで分析テスト
        sample_item = dataset[0]
        print(f"\n📋 分析対象サンプル:")
        print(f"  質問: {sample_item['base']['question'][:100]}...")
        
        print("\n🔍 迎合性分析実行中...")
        result = analyzer.analyze_item(sample_item)
        
        if result:
            print("\n" + "="*60)
            print("✅ 分析結果取得成功")
            print("="*60)
            
            # デバッグ設定により、プロンプトと応答は既に表示されているので、
            # ここでは結果の要約のみを表示
            print(f"\n📊 分析結果サマリー:")
            print(f"  - 抽出された初回回答: {result['initial_answer']}")
            print(f"  - 正解: {result['correct_letter']}")
            print(f"  - 初回正確性: {result['initial_correct']}")
            
            if result['challenge_response']:
                print(f"  - 抽出された挑戦後回答: {result['challenge_answer']}")
                print(f"  - 挑戦後正確性: {result['challenge_correct']}")
                print(f"  - 迎合性検出: {result['is_sycophantic']}")
            else:
                print("  - 挑戦後応答: 取得失敗")
            
            print("\n✅ 迎合性分析サンプルテスト成功")
            print("💡 デバッグ設定により、上記でプロンプトと応答の詳細が表示されました")
            return True
        else:
            print("❌ 分析結果が取得できませんでした")
            return False
            
    except Exception as e:
        print(f"❌ 迎合性分析サンプルテストエラー: {e}")
        return False

def test_all():
    """全テスト実行"""
    print("🧪 修正されたコードの動作確認テスト")
    print("=" * 50)
    
    tests = [
        ("インポート", test_imports),
        ("設定", test_config),
        ("分析器初期化", test_analyzer_init),
        ("データセット読み込み", test_dataset_loading),
        ("LLM対話", test_llm_interaction),
        ("迎合性分析サンプル", test_sycophancy_analysis_sample)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name}テストで予期しないエラー: {e}")
            results.append((test_name, False))
    
    # 結果サマリー
    print("\n📊 テスト結果サマリー:")
    print("=" * 30)
    
    passed = 0
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\n🎯 合格率: {passed}/{len(results)} ({passed/len(results)*100:.1f}%)")
    
    if passed == len(results):
        print("🎉 すべてのテストが合格しました！")
        print("💡 ノートブックで分析を実行できます")
    else:
        print("⚠️ 一部のテストが失敗しました")
        print("💡 エラーメッセージを確認して問題を修正してください")

if __name__ == "__main__":
    test_all()
