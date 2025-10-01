"""
メモリ最適化設定ファイル
CUDA 9.1環境でflash-attnが使用できない場合の代替最適化手法
"""
import torch
import os
from typing import Dict, Any, Optional

class MemoryOptimizer:
    """メモリ使用量を最適化するクラス"""
    
    def __init__(self, cuda_version: str = "9.1"):
        self.cuda_version = cuda_version
        self.flash_attn_available = self._check_flash_attn()
        
    def _check_flash_attn(self) -> bool:
        """flash-attnが利用可能かチェック"""
        try:
            import flash_attn
            return True
        except ImportError:
            return False
    
    def get_memory_config(self, model_name: str = "default") -> Dict[str, Any]:
        """環境に応じたメモリ最適化設定を返す"""
        config = {
            # 基本設定
            "use_fp16": True,
            "use_gradient_checkpointing": True,
            "low_cpu_mem_usage": True,
            "device_map": "auto",
            
            # CUDA 9.1用追加最適化
            "torch_dtype": torch.float16,
            "attn_implementation": "eager",  # flash_attnの代わり
            "max_memory_per_gpu": "6GB",  # GPU使用量制限
            
            # PyTorch最適化設定
            "torch_compile": False,  # CUDA 9.1では無効
            "use_cache": True,
            "pad_token_id": None,
        }
        
        # Gemma-2B特化の最適化
        if "gemma" in model_name.lower():
            config.update({
                "max_memory_per_gpu": "4GB",  # Gemma-2Bはより少ないメモリで動作
                "memory_fraction": 0.7,       # メモリ使用率をより制限
                "offload_to_cpu": True,       # 積極的にCPUオフロード
                "use_sequential_cpu_offload": True,  # シーケンシャルオフロード
                "enable_cpu_offload": True,   # CPU オフロード有効化
            })
            print("🎯 Gemma-2B specific optimizations applied")
        elif "llama" in model_name.lower():
            config.update({
                "max_memory_per_gpu": "8GB",  # Llamaは少し多めのメモリ
                "memory_fraction": 0.8,
            })
            print("🦙 Llama specific optimizations applied")
        
        if self.flash_attn_available:
            config["attn_implementation"] = "flash_attention_2"
        else:
            print(f"⚠️  flash-attn not available (CUDA {self.cuda_version}). Using eager attention.")
            
        return config
    
    def optimize_model_loading(self, model_config: Dict[str, Any]) -> Dict[str, Any]:
        """モデル読み込み時の最適化設定"""
        # モデル名を取得してモデル特化の最適化を適用
        model_name = model_config.get("name", "default")
        memory_config = self.get_memory_config(model_name)
        
        # 既存設定に最適化設定をマージ
        optimized_config = {**model_config, **memory_config}
        
        return optimized_config
    
    def set_memory_fraction(self, fraction: float = 0.8):
        """GPU メモリ使用量を制限"""
        if torch.cuda.is_available():
            torch.cuda.set_per_process_memory_fraction(fraction)
            print(f"🔧 GPU memory fraction set to {fraction}")
    
    def enable_memory_efficient_attention(self):
        """メモリ効率的なAttentionを有効化"""
        # PyTorch 2.0以降のScaled Dot Product Attentionを使用
        if hasattr(torch.nn.functional, 'scaled_dot_product_attention'):
            print("✅ Using PyTorch native scaled_dot_product_attention")
            return True
        return False
    
    def clear_cache(self):
        """GPU キャッシュをクリア"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print("🧹 GPU cache cleared")
    
    def optimize_for_gemma2b(self) -> Dict[str, Any]:
        """Gemma-2B特化のメモリ最適化設定"""
        config = self.get_memory_config("gemma-2b-it")
        
        # Gemma-2B追加最適化
        gemma_specific = {
            # モデルサイズに適した設定
            "max_memory": {"0": "4GB"},  # GPU 0に4GB制限
            "low_cpu_mem_usage": True,
            "use_safetensors": True,
            
            # 生成最適化
            "pad_token_id": 0,  # Gemmaのpad token
            "eos_token_id": 1,  # Gemmaのeos token
            
            # バッチ処理最適化
            "batch_size": 1,    # メモリ効率重視でバッチサイズ1
            "gradient_accumulation_steps": 4,  # グラデーション蓄積
        }
        
        config.update(gemma_specific)
        return config


def get_cuda_optimized_config() -> Dict[str, Any]:
    """CUDA環境に最適化された設定を返す"""
    optimizer = MemoryOptimizer()
    
    base_config = {
        "use_fp16": True,
        "max_memory_gb": 6,
        "device": "cuda" if torch.cuda.is_available() else "cpu"
    }
    
    return optimizer.optimize_model_loading(base_config)


def get_gemma2b_optimized_config() -> Dict[str, Any]:
    """Gemma-2B特化の最適化設定を返す"""
    optimizer = MemoryOptimizer()
    
    base_config = {
        "name": "gemma-2b-it",
        "use_fp16": True,
        "max_memory_gb": 4,  # Gemma-2B用に制限
        "device": "cuda" if torch.cuda.is_available() else "cpu"
    }
    
    return optimizer.optimize_for_gemma2b()


# 環境情報の表示
def print_memory_info():
    """メモリ情報を表示"""
    print("🔍 Memory Optimization Info:")
    print(f"  - CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  - GPU Count: {torch.cuda.device_count()}")
        print(f"  - Current GPU: {torch.cuda.get_device_name(0)}")
        print(f"  - CUDA Version: {torch.version.cuda}")
        
        # GPU メモリ情報
        total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        allocated_memory = torch.cuda.memory_allocated(0) / 1024**3
        cached_memory = torch.cuda.memory_reserved(0) / 1024**3
        
        print(f"  - Total GPU Memory: {total_memory:.2f} GB")
        print(f"  - Allocated Memory: {allocated_memory:.2f} GB")
        print(f"  - Cached Memory: {cached_memory:.2f} GB")
    
    optimizer = MemoryOptimizer()
    print(f"  - Flash Attention: {'✅ Available' if optimizer.flash_attn_available else '❌ Not Available'}")


if __name__ == "__main__":
    print_memory_info()
    config = get_cuda_optimized_config()
    print(f"\n🚀 Optimized Config: {config}")