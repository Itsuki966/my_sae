#!/usr/bin/env python
"""
メモリ管理のテストスクリプト
Llama3での動作確認とメモリ不足対策の検証
"""

import torch
import psutil
import os
from config import ExperimentConfig, LLAMA3_MEMORY_OPTIMIZED_CONFIG, TEST_CONFIG
from sycophancy_analyzer import SycophancyAnalyzer

def check_system_memory():
    """システムメモリ情報を確認"""
    virtual_memory = psutil.virtual_memory()
    print("📊 システムメモリ情報:")
    print(f"   - 総メモリ: {virtual_memory.total / (1024**3):.2f} GB")
    print(f"   - 利用可能: {virtual_memory.available / (1024**3):.2f} GB")
    print(f"   - 使用率: {virtual_memory.percent:.1f}%")
    
    if torch.cuda.is_available():
        print("\n🎯 GPU情報:")
        for i in range(torch.cuda.device_count()):
            total_memory = torch.cuda.get_device_properties(i).total_memory
            allocated = torch.cuda.memory_allocated(i)
            cached = torch.cuda.memory_reserved(i)
            print(f"   GPU {i}: 総メモリ {total_memory/(1024**3):.2f}GB, "
                  f"使用中 {allocated/(1024**3):.2f}GB, "
                  f"キャッシュ {cached/(1024**3):.2f}GB")
    else:
        print("\n⚠️ CUDA使用不可（CPUモード）")

def test_gpt2_memory_usage():
    """gpt2でのメモリ使用量テスト"""
    print("\n🔍 GPT2メモリ使用量テスト")
    check_system_memory()
    
    try:
        config = ExperimentConfig()
        config.model.name = "gpt2"
        config.model.sae_release = "gpt2-small-res-jb"
        config.model.sae_id = "blocks.8.hook_resid_pre"
        
        analyzer = SycophancyAnalyzer(config)
        print("\n🔄 GPT2モデル読み込み中...")
        
        success = analyzer.setup_models()
        if success:
            print("✅ GPT2読み込み成功")
            print(f"🔧 メモリ使用状況:")
            memory_info = analyzer.get_model_memory_footprint()
            for key, value in memory_info.items():
                if isinstance(value, (int, float)):
                    print(f"   - {key}: {value:.2f}")
                else:
                    print(f"   - {key}: {value}")
        else:
            print("❌ GPT2読み込み失敗")
            
        # クリーンアップ
        del analyzer
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
    except Exception as e:
        print(f"❌ GPT2テストエラー: {e}")

def test_llama3_memory_requirements():
    """Llama3のメモリ要件を推定"""
    print("\n🦙 Llama3メモリ要件推定")
    
    # システムメモリチェック
    virtual_memory = psutil.virtual_memory()
    available_gb = virtual_memory.available / (1024**3)
    
    print(f"📊 利用可能メモリ: {available_gb:.2f} GB")
    
    # Llama3-1Bの推定メモリ要件
    llama3_1b_estimated = 4.0  # GB (モデル + SAE + 実行時メモリ)
    llama3_3b_estimated = 12.0  # GB
    
    print(f"📈 Llama3-1B推定要件: ~{llama3_1b_estimated} GB")
    print(f"📈 Llama3-3B推定要件: ~{llama3_3b_estimated} GB")
    
    if available_gb < llama3_1b_estimated:
        print("⚠️ メモリ不足の可能性があります（Llama3-1B）")
        print("💡 推奨対策:")
        print("   - ハイブリッド配置（CPU+GPU）の使用")
        print("   - float16精度の使用")
        print("   - バッチサイズの削減")
        print("   - 他のプロセスの終了")
    else:
        print("✅ Llama3-1Bは動作可能と思われます")
    
    if available_gb < llama3_3b_estimated:
        print("⚠️ メモリ不足の可能性があります（Llama3-3B）")
    else:
        print("✅ Llama3-3Bも動作可能と思われます")

def test_memory_management_functions():
    """メモリ管理関数のテスト"""
    print("\n🔧 メモリ管理関数テスト")
    
    try:
        config = ExperimentConfig()
        analyzer = SycophancyAnalyzer(config)
        
        # メモリ使用量取得テスト
        print("📊 get_model_memory_footprint()テスト:")
        memory_info = analyzer.get_model_memory_footprint()
        for key, value in memory_info.items():
            print(f"   - {key}: {value}")
        
        # メモリ最適化テスト
        print("\n🔄 optimize_memory_usage()テスト:")
        analyzer.optimize_memory_usage()
        print("✅ メモリ最適化完了")
        
        # SAEデバイス取得テスト  
        print("\n🎯 get_current_sae_device()テスト:")
        device = analyzer.get_current_sae_device()
        print(f"   現在のSAEデバイス: {device}")
        
    except Exception as e:
        print(f"❌ メモリ管理関数テストエラー: {e}")

if __name__ == "__main__":
    print("🧪 メモリ管理テスト開始")
    print("=" * 50)
    
    check_system_memory()
    test_gpt2_memory_usage()
    test_llama3_memory_requirements()
    test_memory_management_functions()
    
    print("\n🏁 テスト完了")
