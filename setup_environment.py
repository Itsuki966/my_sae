#!/usr/bin/env python3
"""
環境自動設定スクリプト

このスクリプトは実行環境を自動検出し、最適な依存関係をインストールします。
Poetry依存関係グループを活用してサーバー・ローカル環境の使い分けを行います。
"""

import os
import sys
import platform
import subprocess
import shutil
from pathlib import Path

def run_command(command, description):
    """コマンドを実行し、結果を返す"""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ {description} 成功")
            return True, result.stdout
        else:
            print(f"❌ {description} 失敗: {result.stderr}")
            return False, result.stderr
    except Exception as e:
        print(f"❌ {description} エラー: {e}")
        return False, str(e)

def detect_environment():
    """実行環境を検出"""
    system = platform.system()
    
    # GPU利用可能性チェック
    gpu_available = False
    try:
        import torch
        gpu_available = torch.cuda.is_available()
    except ImportError:
        pass
    
    # MPS（Mac）利用可能性チェック
    mps_available = False
    if system == "Darwin":
        try:
            import torch
            mps_available = torch.backends.mps.is_available()
        except ImportError:
            pass
    
    # SSH接続かどうかをチェック（サーバー環境の指標）
    ssh_connection = os.environ.get('SSH_CONNECTION') is not None
    
    # Jupyter環境の必要性を判定
    jupyter_needed = not ssh_connection and system == "Darwin"
    
    return {
        'system': system,
        'gpu_available': gpu_available,
        'mps_available': mps_available,
        'ssh_connection': ssh_connection,
        'jupyter_needed': jupyter_needed
    }

def check_poetry():
    """Poetryがインストールされているかチェック"""
    return shutil.which('poetry') is not None

def setup_poetry_environment(env_info):
    """Poetry環境をセットアップ"""
    print(f"🖥️ 検出された環境: {env_info['system']}")
    print(f"🔧 GPU利用可能: {env_info['gpu_available']}")
    print(f"🍎 MPS利用可能: {env_info['mps_available']}")
    print(f"🌐 SSH接続: {env_info['ssh_connection']}")
    print(f"📓 Jupyter必要: {env_info['jupyter_needed']}")
    
    if not check_poetry():
        print("❌ Poetryがインストールされていません")
        print("📦 Poetryのインストール方法:")
        print("   curl -sSL https://install.python-poetry.org | python3 -")
        return False
    
    print("\\n🎯 環境に最適化された依存関係をインストールします...")
    
    # サーバー環境（Jupyter不要）
    if env_info['ssh_connection'] or not env_info['jupyter_needed']:
        print("🖥️ サーバー環境モード: 本番用依存関係のみインストール")
        success, output = run_command("poetry install --no-root", "本番用依存関係のインストール")
        
        if success:
            print("\\n✅ サーバー環境セットアップ完了！")
            print("\\n🚀 実行方法:")
            print("   python sycophancy_analyzer.py")
        
    # ローカル環境（Jupyter含む）
    else:
        print("💻 ローカル環境モード: 全依存関係をインストール")
        success, output = run_command("poetry install", "全依存関係のインストール")
        
        if success:
            print("\\n✅ ローカル環境セットアップ完了！")
            print("\\n🚀 実行方法:")
            print("   Python: python sycophancy_analyzer.py")
            print("   Jupyter: jupyter notebook sycophancy_analysis_improved.ipynb")
    
    return success

def create_run_scripts(env_info):
    """実行用スクリプトを作成"""
    print("\\n📝 実行スクリプトを作成中...")
    
    # テスト実行スクリプト
    test_script = '''#!/bin/bash
# 環境テストスクリプト
echo "🧪 環境テスト開始..."
python test_new_files.py
'''
    
    with open("run_test.sh", "w") as f:
        f.write(test_script)
    
    os.chmod("run_test.sh", 0o755)
    print("✅ run_test.sh 作成完了")
    
    # Jupyter起動スクリプト（ローカル環境のみ）
    if env_info['jupyter_needed']:
        jupyter_script = '''#!/bin/bash
# Jupyter Notebook起動スクリプト
echo "📓 Jupyter Notebook起動中..."
poetry run jupyter notebook sycophancy_analysis_improved.ipynb
'''
        
        with open("start_jupyter.sh", "w") as f:
            f.write(jupyter_script)
        
        os.chmod("start_jupyter.sh", 0o755)
        print("✅ start_jupyter.sh 作成完了")

def validate_installation():
    """インストールを検証"""
    print("\\n🔍 インストール検証中...")
    
    # Pythonの基本インポートテスト
    test_imports = [
        ('torch', 'PyTorch'),
        ('pandas', 'pandas'),
        ('numpy', 'numpy'),
        ('plotly', 'plotly'),
        ('tqdm', 'tqdm'),
    ]
    
    success_count = 0
    for module, name in test_imports:
        try:
            __import__(module)
            print(f"✅ {name} インポート成功")
            success_count += 1
        except ImportError:
            print(f"❌ {name} インポート失敗")
    
    # SAE Lensの特別チェック
    try:
        from sae_lens import SAE
        print("✅ SAE Lens インポート成功")
        success_count += 1
    except ImportError:
        print("⚠️ SAE Lens インポート失敗（pip install sae-lens が必要）")
    
    total_tests = len(test_imports) + 1
    print(f"\\n📊 検証結果: {success_count}/{total_tests} 成功 ({success_count/total_tests:.1%})")
    
    return success_count == total_tests

def main():
    """メイン実行関数"""
    print("🔬 SAE迎合性分析プロジェクト - 環境自動セットアップ")
    print("=" * 60)
    print(f"🐍 Python バージョン: {sys.version}")
    print(f"💻 OS: {platform.system()} {platform.release()}")
    print(f"📂 作業ディレクトリ: {os.getcwd()}")
    
    # 環境検出
    env_info = detect_environment()
    
    # Poetry環境セットアップ
    if not setup_poetry_environment(env_info):
        print("\\n❌ セットアップに失敗しました")
        sys.exit(1)
    
    # 実行スクリプト作成
    create_run_scripts(env_info)
    
    # 検証
    if validate_installation():
        print("\\n🎉 環境セットアップが完全に完了しました！")
        print("\\n📋 次のステップ:")
        print("   1. ./run_test.sh で環境テスト")
        
        if env_info['jupyter_needed']:
            print("   2. ./start_jupyter.sh でJupyter起動")
            print("   3. python sycophancy_analyzer.py で分析実行")
        else:
            print("   2. python sycophancy_analyzer.py で分析実行")
            
    else:
        print("\\n⚠️ 一部の依存関係に問題があります")
        print("🔧 手動で以下をインストールしてください:")
        print("   poetry add sae-lens")
    
    # 環境設定ファイル生成
    env_config = f'''# 環境設定ファイル（自動生成）
# 生成日時: {platform.uname()}

DETECTED_SYSTEM="{env_info['system']}"
GPU_AVAILABLE={env_info['gpu_available']}
MPS_AVAILABLE={env_info['mps_available']}
SSH_CONNECTION={env_info['ssh_connection']}
JUPYTER_NEEDED={env_info['jupyter_needed']}

# 推奨設定
{"RECOMMENDED_MODE=server" if env_info['ssh_connection'] else "RECOMMENDED_MODE=local"}
'''
    
    with open("env_config.py", "w") as f:
        f.write(env_config)
    
    print("\\n✅ env_config.py 作成完了")

if __name__ == "__main__":
    main()
