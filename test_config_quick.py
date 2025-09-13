#!/usr/bin/env python3
"""
Quick test to verify config changes
"""

from config import QUANTIZED_4BIT_TEST_CONFIG

print("🔍 設定テスト開始")
config = QUANTIZED_4BIT_TEST_CONFIG
print(f"✅ モデル: {config.model.name}")
print(f"✅ 量子化: {config.model.quantization_config}")
print(f"✅ デバイス: {config.model.device}")
print("🎉 設定テスト完了")
