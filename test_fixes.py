#!/usr/bin/env python3
"""
修正されたコードの動作確認用テストスクリプト
"""

import sys
import os

def test_imports():
    """インポートテスト"""
    print("📦 インポートテスト開始...")
    
    try:
        from config import ExperimentConfig, DEFAULT_CONFIG, LIGHTWEIGHT_CONFIG
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
        from config import DEFAULT_CONFIG, LIGHTWEIGHT_CONFIG
        
        # 設定の基本チェック
        print(f"✅ デフォルト設定 - モデル: {DEFAULT_CONFIG.model.name}")
        print(f"✅ デフォルト設定 - SAE: {DEFAULT_CONFIG.model.sae_id}")
        print(f"✅ デフォルト設定 - サンプルサイズ: {DEFAULT_CONFIG.data.sample_size}")
        print(f"✅ デフォルト設定 - デバイス: {DEFAULT_CONFIG.model.device}")
        
        print(f"✅ 軽量設定 - サンプルサイズ: {LIGHTWEIGHT_CONFIG.data.sample_size}")
        
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

def test_all():
    """全テスト実行"""
    print("🧪 修正されたコードの動作確認テスト")
    print("=" * 50)
    
    tests = [
        ("インポート", test_imports),
        ("設定", test_config),
        ("分析器初期化", test_analyzer_init),
        ("データセット読み込み", test_dataset_loading)
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
