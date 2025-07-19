import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import torch
from typing import List, Dict
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def analyze_conversation_with_sae(text: str, model, sae):
    """会話テキストをSAEで分析

        Args:
            text (str): 分析する会話テキスト
            model: 使用する言語モデル
            sae: SAEモデル
            
        Returns:
            dict: トークン、特徴活性、再構成結果、元の活性
    """
    
    # トークナイズ
    tokens = model.to_tokens(text)

    # モデルの順伝搬
    with torch.no_grad():
        _, cache = model.run_with_cache(tokens)

        # SAEで特徴活性化を取得
        feature_acts = sae.encode(cache[sae.cfg.hook_name])
        reconstructed = sae.decode(feature_acts)
        
        return{
            "tokens": tokens,
            "feature_acts": feature_acts,
            "reconstructed": reconstructed,
            "original_acts": cache[sae.cfg.hook_name]
        }
        
def visualize_feature_activations(feature_acts, tokens, model, top_k=20):
    """特徴活性の可視化：トークンごとの上位特徴量をヒートマップで表示
         Args:
                feature_acts (torch.Tensor): 特徴活性化のテンソル
                tokens (torch.Tensor): トークンのテンソル
                model: 使用する言語モデル
                top_k (int): 表示する上位特徴量の数
                
          Returns:
                None: ヒートマップを表示
    """
    
    # トークンを文字列に変換
    token_strs = [model.to_string(token) for token in tokens[0]]
    top_features = torch.topk(feature_acts[0], k=top_k, dim=-1)
    
    # 各位置での最も活性化した特徴量を取得
    heatmap_data = top_features.values.cpu().numpy()
    
    # ヒートマップデータを準備
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15,10))
    
    # 上位特徴量のヒートマップ
    im1 = ax1.imshow(heatmap_data.T, aspect='auto', cmap='viridis')
    ax1.set_xlabel('Token Position')
    ax1.set_ylabel(f'Top {top_k} Features')
    ax1.set_title('Top Feature Activations Across Tokens')
    ax1.set_xticks(range(len(token_strs)))
    ax1.set_xticklabels(token_strs, rotation=45, ha='right')
    ax1.set_yticks(range(top_k), labels=[f'Rank {i+1}' for i in range(top_k)])
    plt.colorbar(im1, ax=ax1, label='Activation Strength')
    
    # 特徴量の分布
    all_acts = feature_acts[0].cpu().numpy().flatten()
    ax2.hist(all_acts[all_acts > 0], bins=50, alpha=0.7, color='skyblue')
    ax2.set_xlabel('Activation Value')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Distribution of Non-zero Feature Activations')
    ax2.set_yscale('log')
    
    plt.tight_layout()
    plt.show()
    
def plot_sparsity_analysis(feature_acts):
    """スパース性の分析：アクティブな特徴量の分布とスパース性の測定"""
    
    # 各位置でのアクティブな特徴量の数
    active_features_per_token = (feature_acts[0] > 0).sum(dim=-1).cpu().numpy()
    
    # 各特徴量がアクティブになった回数
    feature_activation_counts = (feature_acts[0] > 0).sum(dim=0).cpu().numpy()
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # アクティブ特徴量数の分布
    axes[0, 0].bar(range(len(active_features_per_token)), active_features_per_token)
    axes[0, 0].set_xlabel('Token Position')
    axes[0, 0].set_ylabel('Number of Active Features')
    axes[0, 0].set_title('Active Features per Token')
    
    # スパース性の分布
    sparsity_per_token = 1 - (active_features_per_token / feature_acts.shape[-1])
    axes[0, 1].hist(sparsity_per_token, bins=20, alpha=0.7, color='orange')
    axes[0, 1].set_xlabel('Sparsity (1 - active_ratio)')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title('Sparsity Distribution')
    
    # 特徴量活性化頻度
    axes[1, 0].hist(feature_activation_counts, bins=50, alpha=0.7, color='green')
    axes[1, 0].set_xlabel('Number of Activations')
    axes[1, 0].set_ylabel('Number of Features')
    axes[1, 0].set_title('Feature Activation Frequency')
    axes[1, 0].set_yscale('log')
    
    # 累積分布
    sorted_counts = np.sort(feature_activation_counts)[::-1]
    axes[1, 1].plot(range(len(sorted_counts)), sorted_counts)
    axes[1, 1].set_xlabel('Feature Rank')
    axes[1, 1].set_ylabel('Activation Count')
    axes[1, 1].set_title('Feature Activation Frequency (Ranked)')
    axes[1, 1].set_yscale('log')
    
    plt.tight_layout()
    plt.show()
    
def interactive_feature_explorer(results, model):
    """インタラクティブな特徴量探索"""
    tokens = results['tokens']
    feature_acts = results['feature_acts']
    
    token_strs = [model.to_string(token) for token in tokens[0]]
    
    # 各トークンでの上位特徴量
    top_features_per_token = torch.topk(feature_acts[0], k=10, dim=-1)
    
    # データフレーム作成
    data = []
    for pos, token_str in enumerate(token_strs):
        for rank in range(10):
            feature_id = top_features_per_token.indices[pos, rank].item()
            activation = top_features_per_token.values[pos, rank].item()
            data.append({
                'position': pos,
                'token': token_str,
                'feature_id': feature_id,
                'activation': activation,
                'rank': rank + 1
            })
    
    df = pd.DataFrame(data)
    
    # プロットリーを使ったインタラクティブな可視化
    fig = px.scatter(df, 
                     x='position', 
                     y='feature_id', 
                     size='activation',
                     color='activation',
                     hover_data=['token', 'rank'],
                     title='Interactive Feature Activation Map',
                     labels={'position': 'Token Position', 
                            'feature_id': 'Feature ID'})
    
    fig.update_layout(height=600)
    fig.show()

def reconstruction_quality_analysis(results):
    """再構成品質の分析"""
    original = results['original_acts']
    reconstructed = results['reconstructed']
    
    # MSE計算
    mse = torch.mean((original - reconstructed) ** 2, dim=-1)
    
    # コサイン類似度計算
    cos_sim = torch.nn.functional.cosine_similarity(
        original.flatten(0, 1), 
        reconstructed.flatten(0, 1), 
        dim=-1
    ).reshape(original.shape[:-1])
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # MSE分布
    axes[0].hist(mse.flatten().cpu().numpy(), bins=30, alpha=0.7, color='red')
    axes[0].set_xlabel('Mean Squared Error')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('Reconstruction MSE Distribution')
    
    # コサイン類似度分布
    axes[1].hist(cos_sim.flatten().cpu().numpy(), bins=30, alpha=0.7, color='blue')
    axes[1].set_xlabel('Cosine Similarity')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title('Reconstruction Cosine Similarity')
    
    # 位置ごとの再構成品質
    axes[2].plot(cos_sim[0].cpu().numpy(), 'o-', label='Cosine Similarity')
    axes[2].set_xlabel('Token Position')
    axes[2].set_ylabel('Cosine Similarity')
    axes[2].set_title('Reconstruction Quality by Position')
    axes[2].legend()
    
    plt.tight_layout()
    plt.show()
    
    print(f"Average MSE: {mse.mean().item():.6f}")
    print(f"Average Cosine Similarity: {cos_sim.mean().item():.4f}")