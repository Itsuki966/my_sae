#!/usr/bin/env python3
"""
新規作成ファイルの動作確認スクリプト

このスクリプトは新しく作成されたファイルが正常に動作するかを簡単にテストします。
"""

import sys
import os
import traceback

def test_config_import():
    """設定モジュールのインポートテスト"""
    print("🔧 設定モジュールのテスト...")
    try:
        from config import ExperimentConfig, DEFAULT_CONFIG, LIGHTWEIGHT_CONFIG
        print("✅ config.py のインポート成功")
        
        # 設定の基本動作テスト
        config = DEFAULT_CONFIG
        print(f"  モデル名: {config.model.name}")
        print(f"  サンプルサイズ: {config.data.sample_size}")
        print(f"  デバイス: {config.model.device}")
        
        return True
    except Exception as e:
        print(f"❌ config.py のインポート失敗: {e}")
        return False

def test_analyzer_import():
    """分析器クラスのインポートテスト"""
    print("\\n🔬 分析器クラスのテスト...")
    try:
        from sycophancy_analyzer import SycophancyAnalyzer
        print("✅ sycophancy_analyzer.py のインポート成功")
        
        # 分析器の初期化テスト（モデル読み込みは除く）
        from config import LIGHTWEIGHT_CONFIG
        analyzer = SycophancyAnalyzer(LIGHTWEIGHT_CONFIG)
        print(f"  分析器初期化成功: {analyzer.config.model.name}")
        
        return True
    except Exception as e:
        print(f"❌ sycophancy_analyzer.py のインポート失敗: {e}")
        return False

def test_dataset_access():
    """データセットファイルのアクセステスト"""
    print("\\n📂 データセットアクセステスト...")
    try:
        dataset_path = "eval_dataset/are_you_sure.jsonl"
        if os.path.exists(dataset_path):
            print(f"✅ データセットファイル存在確認: {dataset_path}")
            
            # ファイルサイズ確認
            file_size = os.path.getsize(dataset_path)
            print(f"  ファイルサイズ: {file_size:,} bytes")
            
            # 最初の数行を読み込み
            import json
            with open(dataset_path, 'r', encoding='utf-8') as f:
                line_count = 0
                for line in f:
                    line_count += 1
                    if line_count <= 3:
                        data = json.loads(line.strip())
                        print(f"  サンプル {line_count}: {data['base']['question'][:50]}...")
                    if line_count > 10:  # 最初の10行だけカウント
                        break
            
            print(f"  データセット行数（確認分）: {min(line_count, 10)}+")
            return True
        else:
            print(f"❌ データセットファイルが見つかりません: {dataset_path}")
            return False
            
    except Exception as e:
        print(f"❌ データセットアクセスエラー: {e}")
        return False

def test_dependencies():
    """依存関係のテスト"""
    print("\\n📦 依存関係のテスト...")
    
    dependencies = [
        ('torch', 'PyTorch'),
        ('pandas', 'Pandas'),
        ('numpy', 'NumPy'),
        ('plotly', 'Plotly'),
        ('tqdm', 'tqdm'),
    ]
    
    results = []
    for module_name, display_name in dependencies:
        try:
            __import__(module_name)
            print(f"✅ {display_name} 利用可能")
            results.append(True)
        except ImportError:
            print(f"❌ {display_name} がインストールされていません")
            results.append(False)
    
    # SAE Lensの特別チェック
    try:
        from sae_lens import SAE, HookedSAETransformer
        print("✅ SAE Lens 利用可能")
        results.append(True)
    except ImportError:
        print("⚠️ SAE Lens がインストールされていません（pip install sae-lens）")
        results.append(False)
    
    return all(results)

def test_notebook_file():
    """ノートブックファイルの存在確認"""
    print("\\n📓 ノートブックファイルのテスト...")
    
    notebook_path = "sycophancy_analysis_improved.ipynb"
    if os.path.exists(notebook_path):
        print(f"✅ 改善版ノートブック存在確認: {notebook_path}")
        
        # ファイルサイズ確認
        file_size = os.path.getsize(notebook_path)
        print(f"  ファイルサイズ: {file_size:,} bytes")
        
        return True
    else:
        print(f"❌ ノートブックファイルが見つかりません: {notebook_path}")
        return False

def main():
    """メインテスト関数"""
    print("🧪 新規作成ファイルの動作確認テスト")
    print("=" * 50)
    print(f"🐍 Python バージョン: {sys.version}")
    print(f"📂 現在のディレクトリ: {os.getcwd()}")
    
    # テスト実行
    tests = [
        ("設定モジュール", test_config_import),
        ("分析器クラス", test_analyzer_import),
        ("データセットアクセス", test_dataset_access),
        ("依存関係", test_dependencies),
        ("ノートブックファイル", test_notebook_file),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name}テストでエラー: {e}")
            traceback.print_exc()
            results.append((test_name, False))
    
    # 結果サマリー
    print("\\n" + "=" * 50)
    print("📊 テスト結果サマリー:")
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\\n📈 成功率: {passed}/{total} ({passed/total:.1%})")
    
    if passed == total:
        print("\\n🎉 すべてのテストが成功しました！")
        print("🚀 sycophancy_analyzer.py または sycophancy_analysis_improved.ipynb を実行してください")
    else:
        print("\\n⚠️ いくつかのテストが失敗しました")
        print("🔧 上記のエラーメッセージを確認して必要な修正を行ってください")
        
        # 修正提案
        if not any(result for name, result in results if name == "依存関係"):
            print("\\n📦 依存関係インストール推奨コマンド:")
            print("pip install torch pandas numpy plotly tqdm sae-lens")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
