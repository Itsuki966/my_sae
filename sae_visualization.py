import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Optional, Tuple, Any
from matplotlib.colors import LogNorm
import japanize_matplotlib
from transformers import PreTrainedTokenizer

from sae_model import SparseAutoencoder

class SAEVisualizer:
    """SAEの特徴を可視化するためのクラス"""
    
    def __init__(self, sae_model: SparseAutoencoder, tokenizer: PreTrainedTokenizer):
        """
        Args:
            sae_model: 可視化するSAEモデル
            tokenizer: 使用するトークナイザー
        """
        self.sae_model = sae_model
        self.tokenizer = tokenizer
        self.device = next(sae_model.parameters()).device
        
    def analyze_token_features(self, activations_dict: Dict[str, torch.Tensor]) -> Tuple[List[Dict], List[torch.Tensor]]:
        """
        各トークンのSAE特徴を分析する
        
        Args:
            activations_dict: テキストごとの活性化辞書
            
        Returns:
            Tuple: トークン情報のリストと、SAE特徴のリスト
        """
        token_info_list = []
        all_sae_features_list = []
        global_token_idx = 0

        for original_text, token_activations in activations_dict.items():
            # 活性ベクトルをデバイスに転送
            token_activations = token_activations.to(self.device)
            
            # 学習済みのSAEモデルを使用して、トークンの活性化をエンコード
            with torch.no_grad():
                sae_model_pre_relu = self.sae_model.encoder(token_activations)
                sae_features_for_text = self.sae_model.relu(sae_model_pre_relu)

            all_sae_features_list.append(sae_features_for_text.cpu())

            # トークンを取得
            inputs = self.tokenizer(original_text, return_tensors="pt", padding="max_length", 
                                   truncation=True, max_length=128)
            
            # パディングトークンを除外するためにAttention Maskを使用
            attention_mask = inputs["attention_mask"].squeeze(0)
            input_ids_squeeze = inputs["input_ids"].squeeze(0)

            # トークンIDを取得
            actual_tokens_ids_for_text = input_ids_squeeze[attention_mask == 1]
            actual_tokens_str_list = self.tokenizer.convert_ids_to_tokens(actual_tokens_ids_for_text)

            if len(actual_tokens_str_list) != sae_features_for_text.shape[0]:
                print(f"Warning: Mismatch in token count for text: {original_text}")
                continue
            
            for token_idx_in_text in range(sae_features_for_text.shape[0]):
                token_info_list.append({
                    "original_text": original_text,
                    "token_idx_in_text": token_idx_in_text,
                    "token_str": actual_tokens_str_list[token_idx_in_text],
                    "global_token_idx": global_token_idx,
                })
                
                global_token_idx += 1
                
        return token_info_list, all_sae_features_list
        
    def show_top_activating_tokens(self, token_info_list: List[Dict], 
                                 all_sae_features_list: List[torch.Tensor],
                                 num_features: int = 10, 
                                 tokens_per_feature: int = 5):
        """
        各SAE特徴を最も活性化させるトークンを表示
        
        Args:
            token_info_list: トークン情報のリスト
            all_sae_features_list: SAE特徴のリスト
            num_features: 表示する特徴数
            tokens_per_feature: 各特徴ごとに表示するトークン数
        """
        if not all_sae_features_list:
            print("No SAE features found for the token.")
            return

        concatenated_sae_features = torch.cat(all_sae_features_list, dim=0)
        sae_total_features = concatenated_sae_features.shape[0]
        
        num_sae_features_to_analyze = min(num_features, sae_total_features)
        num_top_tokens_per_feature = tokens_per_feature

        for feature_idx_to_analyze in range(num_sae_features_to_analyze):
            feature_column_activation = concatenated_sae_features[:, feature_idx_to_analyze]
            
            actual_k = min(num_top_tokens_per_feature, len(feature_column_activation))
            if actual_k == 0:
                continue
            
            top_k_values, top_k_global_indices = torch.topk(feature_column_activation, k=actual_k)
            
            print(f"\n--- SAE Feature {feature_idx_to_analyze} を最も強く活性化するトークン")
            
            if top_k_values.numel() == 0:
                print("No top tokens found for this feature.")
                continue
            
            for rank, (activation_value, global_token_idx_item) in enumerate(zip(top_k_values, top_k_global_indices)):
                global_idx = global_token_idx_item.item()
                if global_idx < len(token_info_list):
                    token_info = token_info_list[global_idx]
                    
                    text_snippet = token_info["original_text"]
                    
                    # 文脈表示のために、元のテキストを再度トークナイズ(表示用)
                    inputs_ctx = self.tokenizer(text_snippet,
                                            return_tensors="pt",
                                            truncation=True,
                                            max_length=128,
                                            padding="max_length",
                                            return_attention_mask=True)
                    ids_ctx = inputs_ctx["input_ids"].squeeze()[inputs_ctx["attention_mask"].squeeze() == 1]
                    tokens_ctx = self.tokenizer.convert_ids_to_tokens(ids_ctx)
                    
                    tok_idx_in_ctx = token_info["token_idx_in_text"]
                    
                    context_window_size = 3
                    start_idx = max(0, tok_idx_in_ctx - context_window_size)
                    end_idx = min(len(tokens_ctx), tok_idx_in_ctx + context_window_size + 1)
                    
                    context_display_parts = []
                    for i in range(start_idx, end_idx):
                        if i == tok_idx_in_ctx:
                            context_display_parts.append(f"**{tokens_ctx[i]}**")
                        else:
                            context_display_parts.append(tokens_ctx[i])
                    context_str = " ".join(context_display_parts)

                    print(f"  順位 {rank + 1}: 活性化値 = {activation_value.item():.4f}")
                    print(f"    トークン: '{token_info['token_str']}' (テキスト内の実トークンindex: {tok_idx_in_ctx})")
                    print(f"    文脈: {context_str}")
                    text_preview = (text_snippet[:70] + '...') if len(text_snippet) > 70 else text_snippet
                    print(f"    元テキスト (一部): \"{text_preview}\"")
                else:
                    print(f"  順位 {rank + 1}: エラー - グローバルインデックス {global_idx} が範囲外です。")

    def visualize_token_feature_activations(self, token_info_list: List[Dict], 
                                         all_sae_features_list: List[torch.Tensor]):
        """
        トークンごとのSAE特徴活性化を可視化
        
        Args:
            token_info_list: トークン情報のリスト
            all_sae_features_list: SAE特徴のリスト
        """
        if not all_sae_features_list or not token_info_list:
            print("活性化データまたはトークン情報が存在しません。")
            return
            
        # 全トークンの特徴活性化値を取得
        concatenated_features = torch.cat(all_sae_features_list, dim=0).cpu().numpy()
        
        # 表示するトークン数と特徴数を制限
        max_tokens_to_display = min(20, concatenated_features.shape[0])
        max_features_to_display = min(50, concatenated_features.shape[1])
        
        # 表示するデータの準備
        selected_tokens = token_info_list[:max_tokens_to_display]
        token_labels = [f"{info['token_str']} ({info['token_idx_in_text']})" for info in selected_tokens]
        
        # 特徴活性化データのサブセットを取得
        activations_subset = concatenated_features[:max_tokens_to_display, :max_features_to_display]
        
        # 1. ヒートマップによる可視化
        plt.figure(figsize=(15, 10))
        
        # ヒートマップの作成（対数スケールを使用して低い活性化値も見やすくする）
        sns.heatmap(
            activations_subset,
            xticklabels=[f"F{i}" for i in range(max_features_to_display)],
            yticklabels=token_labels,
            cmap="viridis",
            norm=LogNorm(vmin=max(0.001, activations_subset.min())),
            cbar_kws={"label": "活性化値（対数スケール）"}
        )
        
        plt.title("トークンごとのSAE特徴活性化ヒートマップ")
        plt.xlabel("SAE特徴")
        plt.ylabel("トークン")
        plt.tight_layout()
        plt.show()
        
        # 2. 各トークンの上位活性化特徴を棒グラフで表示
        self._plot_top_features_per_token(concatenated_features, selected_tokens)
        
        # 3. 特徴ごとの活性化分布を表示
        self._plot_top_tokens_per_feature(concatenated_features, token_info_list)
    
    def _plot_top_features_per_token(self, concatenated_features: np.ndarray, 
                                   selected_tokens: List[Dict], 
                                   num_tokens: int = 5, 
                                   top_features: int = 10):
        """各トークンの上位活性化特徴を棒グラフで表示"""
        num_tokens_for_bar = min(num_tokens, len(selected_tokens))
        top_features_per_token = top_features
        
        fig, axes = plt.subplots(num_tokens_for_bar, 1, figsize=(12, 3*num_tokens_for_bar))
        if num_tokens_for_bar == 1:
            axes = [axes]
        
        for i, ax in enumerate(axes):
            if i >= len(selected_tokens):
                break
                
            # このトークンの活性化値
            token_activations = concatenated_features[i]
            # 上位特徴を取得
            top_indices = np.argsort(token_activations)[::-1][:top_features_per_token]
            top_values = token_activations[top_indices]
            
            # 棒グラフの作成
            bars = ax.bar(
                [f"F{idx}" for idx in top_indices],
                top_values
            )
            
            # トークンの情報を表示
            token_info = selected_tokens[i]
            ax.set_title(f"トークン: '{token_info['token_str']}' (idx: {token_info['token_idx_in_text']})")
            ax.set_ylabel("活性化値")
            
            # 各バーの上に値を表示
            for bar, val in zip(bars, top_values):
                ax.text(
                    bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 0.01,
                    f"{val:.3f}",
                    ha='center',
                    va='bottom',
                    fontsize=8,
                    rotation=45
                )
        
        plt.tight_layout()
        plt.show()
    
    def _plot_top_tokens_per_feature(self, concatenated_features: np.ndarray, 
                                   token_info_list: List[Dict], 
                                   num_features: int = 6, 
                                   top_tokens: int = 10):
        """特徴ごとの上位活性化トークンを表示"""
        num_features_to_analyze = min(num_features, concatenated_features.shape[1])
        feature_indices = list(range(num_features_to_analyze))
        
        fig, axes = plt.subplots(len(feature_indices), 1, figsize=(12, 3*len(feature_indices)))
        if len(feature_indices) == 1:
            axes = [axes]
            
        for i, feature_idx in enumerate(feature_indices):
            feature_activations = concatenated_features[:, feature_idx]
            
            # 活性化値でソートして上位のトークンを取得
            sorted_indices = np.argsort(feature_activations)[::-1]
            top_n = min(top_tokens, len(sorted_indices))
            
            token_strs = [token_info_list[idx]['token_str'] for idx in sorted_indices[:top_n]]
            sorted_activations = feature_activations[sorted_indices[:top_n]]
            
            # 棒グラフの作成
            bars = axes[i].bar(token_strs, sorted_activations)
            axes[i].set_title(f"SAE特徴 F{feature_idx} の上位活性化トークン")
            axes[i].set_ylabel("活性化値")
            axes[i].set_xticklabels(token_strs, rotation=45, ha='right')
            
            # 各バーの上に値を表示
            for bar, val in zip(bars, sorted_activations):
                axes[i].text(
                    bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 0.01,
                    f"{val:.3f}",
                    ha='center',
                    va='bottom',
                    fontsize=8
                )
        
        plt.tight_layout()
        plt.show()