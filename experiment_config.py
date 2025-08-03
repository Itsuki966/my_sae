#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SAE迎合性分析 - 実験設定管理
実験に関わるマジックナンバーを一元管理し、設定変更を容易にする
"""

import torch
from dataclasses import dataclass
from typing import Optional

@dataclass
class ExperimentConfig:
    """実験設定を管理するデータクラス"""
    
    # ===== 🤖 モデル設定 =====
    model_name: str = "pythia-70m-deduped"
    sae_release: str = "pythia-70m-deduped-res-sm"
    sae_id: str = "blocks.5.hook_resid_post"
    
    # ===== 📊 データ設定 =====
    dataset_path: str = "eval_dataset/are_you_sure.jsonl"
    sample_size: int = 50
    
    # ===== 🎛️ 生成設定 =====
    max_new_tokens: int = 8
    temperature: float = 0.1
    do_sample: bool = True
    repetition_penalty: float = 1.0
    top_p: float = 1.0
    top_k: int = 50
    
    # ===== 🔍 分析設定 =====
    top_k_features: int = 20
    activation_threshold: float = 0.1
    
    # ===== 📝 表示設定 =====
    show_details: bool = True
    detail_samples: int = 3
    max_examples_shown: int = 3
    
    # ===== 🐛 デバッグ設定 =====
    debug_extraction: bool = False
    debug_activations: bool = False
    verbose_logging: bool = False
    
    # ===== 💻 システム設定 =====
    device: Optional[str] = None
    random_seed: int = 42
    
    def __post_init__(self):
        """初期化後の処理"""
        if self.device is None:
            if torch.backends.mps.is_available():
                self.device = "mps"
            elif torch.cuda.is_available():
                self.device = "cuda"
            else:
                self.device = "cpu"

# ===== 📋 プリセット設定 =====

def get_quick_test_config() -> ExperimentConfig:
    """クイックテスト用の設定"""
    return ExperimentConfig(
        sample_size=10,
        max_new_tokens=5,
        temperature=0.0,
        show_details=True,
        detail_samples=5
    )

def get_full_analysis_config() -> ExperimentConfig:
    """完全分析用の設定"""
    return ExperimentConfig(
        sample_size=100,
        max_new_tokens=8,
        temperature=0.1,
        show_details=False,
        detail_samples=0
    )

def get_debug_config() -> ExperimentConfig:
    """デバッグ用の設定"""
    return ExperimentConfig(
        sample_size=5,
        max_new_tokens=10,
        temperature=0.0,
        show_details=True,
        detail_samples=5,
        debug_extraction=True,
        debug_activations=True,
        verbose_logging=True
    )

def get_larger_model_config() -> ExperimentConfig:
    """より大きなモデル用の設定"""
    return ExperimentConfig(
        model_name="pythia-160m-deduped",
        sae_release="pythia-160m-deduped-res-sm",
        sae_id="blocks.7.hook_resid_post",
        sample_size=50,
        max_new_tokens=8,
        temperature=0.1
    )

def get_deterministic_config() -> ExperimentConfig:
    """完全決定的な設定"""
    return ExperimentConfig(
        sample_size=50,
        max_new_tokens=5,
        temperature=0.0,
        do_sample=False,
        top_p=1.0,
        top_k=1
    )

# ===== 🎯 設定検証 =====

def validate_config(config: ExperimentConfig) -> bool:
    """設定の妥当性を検証"""
    errors = []
    
    # 基本チェック
    if config.sample_size <= 0:
        errors.append("sample_size は正の整数である必要があります")
    
    if config.max_new_tokens <= 0:
        errors.append("max_new_tokens は正の整数である必要があります")
    
    if not (0.0 <= config.temperature <= 2.0):
        errors.append("temperature は 0.0-2.0 の範囲である必要があります")
    
    if not (0.0 <= config.top_p <= 1.0):
        errors.append("top_p は 0.0-1.0 の範囲である必要があります")
    
    if config.top_k_features <= 0:
        errors.append("top_k_features は正の整数である必要があります")
    
    # ファイル存在チェック
    import os
    if not os.path.exists(config.dataset_path):
        errors.append(f"データセットファイルが見つかりません: {config.dataset_path}")
    
    # エラー出力
    if errors:
        print("❌ 設定エラー:")
        for error in errors:
            print(f"   - {error}")
        return False
    
    print("✅ 設定検証完了")
    return True

def print_config_summary(config: ExperimentConfig):
    """設定内容のサマリーを表示"""
    print("📋 実験設定サマリー")
    print("=" * 40)
    
    print(f"🤖 モデル: {config.model_name}")
    print(f"📊 サンプル数: {config.sample_size}")
    print(f"💻 デバイス: {config.device}")
    print(f"🎛️ 最大トークン: {config.max_new_tokens}")
    print(f"🌡️ 温度: {config.temperature}")
    print(f"🔍 詳細表示: {config.show_details}")
    
    if config.debug_extraction or config.debug_activations:
        print(f"🐛 デバッグモード: ON")
    
    print("=" * 40)

# ===== 📚 使用例 =====

if __name__ == "__main__":
    print("🎯 SAE迎合性分析 - 設定管理デモ")
    print()
    
    # デフォルト設定
    print("1. デフォルト設定:")
    default_config = ExperimentConfig()
    print_config_summary(default_config)
    print()
    
    # クイックテスト設定
    print("2. クイックテスト設定:")
    quick_config = get_quick_test_config()
    print_config_summary(quick_config)
    print()
    
    # 設定検証
    print("3. 設定検証:")
    validate_config(default_config)
    print()
    
    print("🚀 利用方法:")
    print("   from experiment_config import ExperimentConfig, get_quick_test_config")
    print("   config = get_quick_test_config()")
    print("   # または")
    print("   config = ExperimentConfig(sample_size=100, temperature=0.0)")
