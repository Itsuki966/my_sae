#!/usr/bin/env python3
"""
サーバー環境専用セットアップスクリプト
.pyファイルのみを実行するサーバー環境に最適化された依存関係のインストール
"""
import subprocess
import sys
import os
from pathlib import Path

def run_command(cmd: str, check: bool = True) -> subprocess.CompletedProcess:
    """コマンドを実行し、結果を返す"""
    print(f"実行中: {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if check and result.returncode != 0:
        print(f"エラー: {result.stderr}")
        sys.exit(1)
    return result

def check_python_version():
    """Python バージョンをチェック"""
    version = sys.version_info
    if version.major != 3 or version.minor < 10:
        print(f"Python 3.10以上が必要です。現在のバージョン: {version.major}.{version.minor}")
        sys.exit(1)
    print(f"Python バージョン確認: {version.major}.{version.minor}.{version.micro}")

def install_core_dependencies():
    """サーバー環境用のコア依存関係のみをインストール"""
    print("サーバー環境用依存関係をインストール中...")
    
    # コア依存関係のみをインストール（Jupyter関連は除外）
    core_packages = [
        "torch>=2.0.0",
        "transformers>=4.35.0,<5.0.0",
        "accelerate>=0.24.0", 
        "safetensors>=0.4.0",
        "sae-lens>=3.0.0",
        "einops>=0.7.0",
        "jaxtyping>=0.2.0",
        "pandas>=2.0.0,<3.0.0",
        "numpy>=1.24.0,<2.0.0",
        "tqdm>=4.65.0,<5.0.0",
        "requests>=2.31.0",
        "datasets>=2.14.0"
    ]
    
    # Poetryが利用可能かチェック
    try:
        run_command("poetry --version", check=False)
        use_poetry = True
        print("Poetryを使用してインストールします")
    except:
        use_poetry = False
        print("Poetryが見つかりません。pipを使用してインストールします")
    
    if use_poetry:
        # Poetryを使用してサーバー用最小構成をインストール
        run_command("poetry install --only=main --extras=server")
    else:
        # pipで直接インストール
        for package in core_packages:
            run_command(f"pip install '{package}'")

def create_server_config():
    """サーバー環境用設定ファイルを作成"""
    server_config = """# サーバー環境設定
import os
os.environ['TOKENIZERS_PARALLELISM'] = 'false'  # サーバーでの並列処理警告を抑制
os.environ['HF_DATASETS_OFFLINE'] = '0'  # データセットのオンライン確認を有効化

# 自動設定を使用
from config import get_auto_config
config = get_auto_config()
"""
    
    with open("server_config.py", "w", encoding="utf-8") as f:
        f.write(server_config)
    print("server_config.py を作成しました")

def create_run_script():
    """サーバー実行用スクリプトを作成"""
    run_script = """#!/bin/bash
# サーバー環境での実行スクリプト

echo "SAE Sycophancy Analysis - Server Mode"
echo "Python version: $(python --version)"
echo "Environment: SERVER"

# 環境変数設定
export TOKENIZERS_PARALLELISM=false
export HF_DATASETS_OFFLINE=0

# メイン分析スクリプトを実行
python sycophancy_analyzer.py "$@"
"""
    
    with open("run_server.sh", "w") as f:
        f.write(run_script)
    os.chmod("run_server.sh", 0o755)
    print("run_server.sh を作成しました")

def verify_installation():
    """インストールの検証"""
    print("インストールの検証中...")
    
    test_imports = [
        "torch",
        "transformers", 
        "sae_lens",
        "pandas",
        "numpy"
    ]
    
    for module in test_imports:
        try:
            __import__(module)
            print(f"✓ {module}")
        except ImportError as e:
            print(f"✗ {module}: {e}")
            return False
    
    print("すべてのコア依存関係が正常にインストールされました")
    return True

def main():
    """メイン処理"""
    print("=== SAE Server Environment Setup ===")
    print("サーバー環境用の最小依存関係セットアップを開始します")
    print("注意: Jupyter Notebook関連のパッケージはインストールされません\n")
    
    # Python バージョンチェック
    check_python_version()
    
    # 依存関係インストール
    install_core_dependencies()
    
    # 設定ファイル作成
    create_server_config()
    
    # 実行スクリプト作成
    create_run_script()
    
    # インストール検証
    if verify_installation():
        print("\n=== セットアップ完了 ===")
        print("サーバー環境の準備が完了しました")
        print("実行方法:")
        print("  ./run_server.sh")
        print("  または")
        print("  python sycophancy_analyzer.py")
    else:
        print("\n=== セットアップ失敗 ===")
        print("依存関係のインストールに問題があります")
        sys.exit(1)

if __name__ == "__main__":
    main()
