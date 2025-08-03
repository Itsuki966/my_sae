#!/usr/bin/env python3
"""
SAE迎合性分析 - 環境セットアップスクリプト
"""

import os
import sys
from pathlib import Path

def check_python_version():
    """Python バージョンチェック"""
    print("🐍 Python環境チェック")
    print(f"   バージョン: {sys.version}")
    
    if sys.version_info < (3, 8):
        print("❌ Python 3.8以上が必要です")
        return False
    else:
        print("✅ Python バージョンOK")
        return True

def check_poetry():
    """Poetry の確認"""
    print("\n📦 Poetry環境チェック")
    
    if os.system("poetry --version") == 0:
        print("✅ Poetry がインストールされています")
        return True
    else:
        print("❌ Poetry がインストールされていません")
        print("   インストール: curl -sSL https://install.python-poetry.org | python3 -")
        return False

def check_project_files():
    """プロジェクトファイルの確認"""
    print("\n📁 プロジェクトファイルチェック")
    
    required_files = [
        "pyproject.toml",
        "eval_dataset/are_you_sure.jsonl",
        "sae_sycophancy_hybrid.py"
    ]
    
    missing_files = []
    for file_path in required_files:
        if os.path.exists(file_path):
            print(f"✅ {file_path}")
        else:
            print(f"❌ {file_path} - 不足")
            missing_files.append(file_path)
    
    return len(missing_files) == 0

def install_dependencies():
    """依存関係のインストール"""
    print("\n📦 依存関係インストール")
    
    print("Poetry依存関係をインストール中...")
    if os.system("poetry install") == 0:
        print("✅ Poetry依存関係インストール完了")
        return True
    else:
        print("❌ Poetry依存関係インストールに失敗")
        return False

def run_tests():
    """テストの実行"""
    print("\n🧪 軽量テスト実行")
    
    if os.system("python sae_test_light.py") == 0:
        print("✅ 軽量テスト完了")
        return True
    else:
        print("❌ 軽量テストに失敗")
        return False

def main():
    """メイン処理"""
    print("🚀 SAE迎合性分析 - 環境セットアップ")
    print("=" * 50)
    
    checks = [
        ("Python バージョン", check_python_version),
        ("Poetry", check_poetry),
        ("プロジェクトファイル", check_project_files),
        ("依存関係インストール", install_dependencies),
        ("軽量テスト", run_tests)
    ]
    
    failed_checks = []
    
    for check_name, check_func in checks:
        try:
            if not check_func():
                failed_checks.append(check_name)
        except Exception as e:
            print(f"❌ {check_name}: エラー - {e}")
            failed_checks.append(check_name)
    
    print(f"\n📋 セットアップ結果")
    print("-" * 30)
    
    if not failed_checks:
        print("🎉 すべてのチェックに合格！")
        print("\n🚀 実行可能なコマンド:")
        print("   poetry run python sae_sycophancy_hybrid.py")
        print("   poetry run jupyter notebook sae_sycophancy_analysis_clean.ipynb")
    else:
        print(f"⚠️ 失敗したチェック: {', '.join(failed_checks)}")
        print("\n🔧 修正が必要です:")
        
        if "Poetry" in failed_checks:
            print("   - Poetry をインストールしてください")
        if "プロジェクトファイル" in failed_checks:
            print("   - 不足しているファイルを確認してください")
        if "依存関係インストール" in failed_checks:
            print("   - poetry install を再実行してください")

if __name__ == "__main__":
    main()
