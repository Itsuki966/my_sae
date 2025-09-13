#!/usr/bin/env python3
"""
量子化テストスクリプト

bitsandbytesによる4bit/8bit量子化の動作確認を行います。
Llama3モデルでサンプル数5の軽量テストを実行し、
メモリ使用量の削減効果を確認します。
"""

import os
import sys
import time
import traceback
import psutil
import torch
from transformers import BitsAndBytesConfig
from config import QUANTIZED_4BIT_TEST_CONFIG, QUANTIZED_8BIT_TEST_CONFIG
from sycophancy_analyzer import SycophancyAnalyzer

def check_memory_usage():
    """現在のメモリ使用量を取得"""
    process = psutil.Process(os.getpid())
    memory_mb = process.memory_info().rss / 1024 / 1024
    return memory_mb

def check_gpu_memory():
    """GPU メモリ使用量を取得"""
    if torch.cuda.is_available():
        gpu_memory_mb = torch.cuda.memory_allocated() / 1024 / 1024
        gpu_memory_reserved_mb = torch.cuda.memory_reserved() / 1024 / 1024
        return gpu_memory_mb, gpu_memory_reserved_mb
    elif torch.backends.mps.is_available():
        # MPSではメモリ情報の詳細取得が制限される
        return 0, 0
    else:
        return 0, 0

def create_quantization_config(config):
    """量子化設定を作成"""
    if not config.model.use_quantization:
        return None
    
    if config.model.quantization_config == "4bit":
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=config.model.bnb_4bit_use_double_quant,
            bnb_4bit_quant_type=config.model.bnb_4bit_quant_type,
            bnb_4bit_compute_dtype=getattr(torch, config.model.bnb_4bit_compute_dtype)
        )
    elif config.model.quantization_config == "8bit":
        return BitsAndBytesConfig(
            load_in_8bit=True,
        )
    else:
        return None

def test_quantization_config(config_name, config):
    """指定された設定で量子化テストを実行"""
    print(f"\n{'='*60}")
    print(f"🧪 {config_name} テスト開始")
    print(f"{'='*60}")
    
    # より詳細な設定情報を表示
    print(f"🔧 テスト設定詳細:")
    print(f"   モデル: {config.model.name}")
    print(f"   SAE: {config.model.sae_release}")
    print(f"   サンプル数: {config.data.sample_size}")
    print(f"   デバイス: {config.model.device}")
    if config.model.use_quantization:
        print(f"   量子化: {config.model.quantization_config}")
        if config.model.load_in_4bit:
            print(f"   4bit設定: {config.model.bnb_4bit_quant_type}, double_quant={config.model.bnb_4bit_use_double_quant}")
        if config.model.load_in_8bit:
            print(f"   8bit設定: 有効")
    
    # 初期メモリ使用量
    initial_memory = check_memory_usage()
    initial_gpu_memory, initial_gpu_reserved = check_gpu_memory()
    
    print(f"\n📊 初期メモリ使用量:")
    print(f"   RAM: {initial_memory:.1f} MB")
    if torch.cuda.is_available():
        print(f"   GPU: {initial_gpu_memory:.1f} MB (予約: {initial_gpu_reserved:.1f} MB)")
    
    try:
        # 量子化設定の作成
        quantization_config = create_quantization_config(config)
        
        if quantization_config:
            print(f"🔧 量子化設定:")
            if config.model.load_in_4bit:
                print(f"   4bit量子化: 有効")
                print(f"   二重量子化: {config.model.bnb_4bit_use_double_quant}")
                print(f"   量子化タイプ: {config.model.bnb_4bit_quant_type}")
                print(f"   計算精度: {config.model.bnb_4bit_compute_dtype}")
            elif config.model.load_in_8bit:
                print(f"   8bit量子化: 有効")
        else:
            print(f"🔧 量子化: 無効")
        
        # アナライザーの初期化
        print(f"🚀 アナライザーを初期化中...")
        start_time = time.time()
        
        analyzer = SycophancyAnalyzer(config)
        init_time = time.time() - start_time
        
        # 初期化後のメモリ使用量
        after_init_memory = check_memory_usage()
        after_init_gpu_memory, after_init_gpu_reserved = check_gpu_memory()
        
        print(f"✅ 初期化完了 ({init_time:.1f}秒)")
        print(f"📊 初期化後メモリ使用量:")
        print(f"   RAM: {after_init_memory:.1f} MB (+{after_init_memory - initial_memory:.1f})")
        if torch.cuda.is_available():
            print(f"   GPU: {after_init_gpu_memory:.1f} MB (+{after_init_gpu_memory - initial_gpu_memory:.1f})")
            print(f"   GPU予約: {after_init_gpu_reserved:.1f} MB (+{after_init_gpu_reserved - initial_gpu_reserved:.1f})")
        
        # 簡単な分析テスト
        print(f"🔍 分析テスト実行中...")
        print(f"   データセット: {config.data.dataset_path}")
        print(f"   サンプル数: {config.data.sample_size}")
        start_time = time.time()
        
        try:
            results = analyzer.run_complete_analysis()
            analysis_time = time.time() - start_time
            
            # 分析後のメモリ使用量
            final_memory = check_memory_usage()
            final_gpu_memory, final_gpu_reserved = check_gpu_memory()
            
            print(f"✅ 分析完了 ({analysis_time:.1f}秒)")
            print(f"📊 最終メモリ使用量:")
            print(f"   RAM: {final_memory:.1f} MB")
            if torch.cuda.is_available():
                print(f"   GPU: {final_gpu_memory:.1f} MB")
                print(f"   GPU予約: {final_gpu_reserved:.1f} MB")
            
            # 結果サマリー
            print(f"\n📈 結果サマリー:")
            if 'results' in results and results['results']:
                print(f"   サンプル数: {len(results['results'])}")
            else:
                print(f"   サンプル数: 0")
            
            if 'analysis' in results and 'sycophancy_rate' in results['analysis']:
                print(f"   迎合率: {results['analysis']['sycophancy_rate']:.2%}")
            else:
                print(f"   迎合率: 計算中または利用不可")
            
            print(f"   総処理時間: {init_time + analysis_time:.1f}秒")
            
            return True, {
                'init_time': init_time,
                'analysis_time': analysis_time,
                'memory_usage': {
                    'initial': initial_memory,
                    'after_init': after_init_memory,
                    'final': final_memory
                },
                'gpu_memory_usage': {
                    'initial': initial_gpu_memory,
                    'after_init': after_init_gpu_memory,
                    'final': final_gpu_memory
                },
                'results': results
            }
            
        except Exception as analysis_error:
            analysis_time = time.time() - start_time
            print(f"❌ 分析実行エラー ({analysis_time:.1f}秒経過): {analysis_error}")
            print(f"\n🔍 詳細エラー情報:")
            traceback.print_exc()
            
            return False, {
                'error': str(analysis_error),
                'init_time': init_time,
                'analysis_time': analysis_time
            }
        
    except Exception as e:
        print(f"❌ エラーが発生しました:")
        print(f"   {type(e).__name__}: {str(e)}")
        print(f"\n🔍 詳細エラー情報:")
        traceback.print_exc()
        return False, {'error': str(e)}

def main():
    """メイン実行関数"""
    print("🧪 量子化機能テスト開始")
    print(f"Python バージョン: {sys.version}")
    print(f"PyTorch バージョン: {torch.__version__}")
    
    # デバイス情報
    print(f"\n🖥️  デバイス情報:")
    if torch.cuda.is_available():
        print(f"   CUDA: 利用可能 ({torch.cuda.get_device_name()})")
        print(f"   GPU メモリ: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    elif torch.backends.mps.is_available():
        print(f"   MPS: 利用可能 (Apple Silicon)")
    else:
        print(f"   CPU のみ")
    
    # bitsandbytes の確認
    try:
        import bitsandbytes as bnb
        print(f"   bitsandbytes: {bnb.__version__}")
    except ImportError:
        print("❌ bitsandbytes がインストールされていません")
        print("   インストール方法: pip install bitsandbytes")
        return
    
    # テスト設定
    test_configs = [
        ("4bit量子化", QUANTIZED_4BIT_TEST_CONFIG),
        ("8bit量子化", QUANTIZED_8BIT_TEST_CONFIG),
    ]
    
    results = {}
    
    for config_name, config in test_configs:
        success, result = test_quantization_config(config_name, config)
        results[config_name] = {
            'success': success,
            'result': result
        }
        
        # GPUメモリをクリア
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        time.sleep(2)  # 少し待機
    
    # 総合結果表示
    print(f"\n{'='*60}")
    print(f"📊 総合テスト結果")
    print(f"{'='*60}")
    
    for config_name, data in results.items():
        print(f"\n{config_name}:")
        if data['success']:
            result = data['result']
            print(f"  ✅ 成功")
            print(f"  初期化時間: {result['init_time']:.1f}秒")
            print(f"  分析時間: {result['analysis_time']:.1f}秒")
            print(f"  最大RAM使用量: {result['memory_usage']['final']:.1f} MB")
            if torch.cuda.is_available():
                print(f"  最大GPU使用量: {result['gpu_memory_usage']['final']:.1f} MB")
        else:
            print(f"  ❌ 失敗: {data['result'].get('error', '不明なエラー')}")
    
    print(f"\n🎉 テスト完了")

if __name__ == "__main__":
    main()
