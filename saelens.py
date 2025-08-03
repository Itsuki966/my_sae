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
    """会話テキストをSAEで分析（改善版）

        Args:
            text (str): 分析する会話テキスト
            model: 使用する言語モデル
            sae: SAEモデル
            
        Returns:
            dict: トークン、特徴活性、再構成結果、元の活性
    """
    
    try:
        # 入力検証
        if not text or not text.strip():
            raise ValueError("テキストが空です")
        
        # テキストの長さを制限（メモリ効率のため）
        if len(text) > 2000:
            text = text[:2000]
            print("警告: テキストが長すぎるため、最初の2000文字のみを使用します")
        
        # トークナイズ
        tokens = model.to_tokens(text)
        
        # トークン数を制限（計算効率のため）
        if tokens.shape[1] > 512:
            tokens = tokens[:, :512]
            print("警告: トークン数が多すぎるため、最初の512トークンのみを使用します")

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
            
    except Exception as e:
        print(f"SAE分析中にエラーが発生しました: {e}")
        return None
        
def visualize_feature_activations(feature_acts, tokens, model, top_k=20):
    """特徴活性の可視化：トークンごとの上位特徴量をヒートマップで表示（改善版）
         Args:
                feature_acts (torch.Tensor): 特徴活性化のテンソル
                tokens (torch.Tensor): トークンのテンソル
                model: 使用する言語モデル
                top_k (int): 表示する上位特徴量の数
                
          Returns:
                None: ヒートマップを表示
    """
    
    try:
        # 入力検証
        if feature_acts is None or tokens is None:
            print("エラー: 入力データが無効です")
            return
        
        if len(feature_acts.shape) < 3 or feature_acts.shape[0] == 0:
            print("エラー: 特徴活性化データの形状が無効です")
            return
        
        # top_kを動的に調整
        max_features = feature_acts.shape[-1]
        top_k = min(top_k, max_features)
        
        # トークンを文字列に変換（安全な方法）
        try:
            token_strs = model.to_str_tokens(tokens[0])
        except:
            # フォールバック: トークンIDを文字列として使用
            token_strs = [str(token.item()) for token in tokens[0]]
        
        # トークン数を制限（可視化のため）
        max_tokens = min(50, len(token_strs))
        token_strs = token_strs[:max_tokens]
        feature_acts_subset = feature_acts[0, :max_tokens, :]
        
        top_features = torch.topk(feature_acts_subset, k=top_k, dim=-1)
        
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
        positive_acts = all_acts[all_acts > 0]
        
        if len(positive_acts) > 0:
            ax2.hist(positive_acts, bins=50, alpha=0.7, color='skyblue')
            ax2.set_xlabel('Activation Value')
            ax2.set_ylabel('Frequency')
            ax2.set_title('Distribution of Non-zero Feature Activations')
            ax2.set_yscale('log')
        else:
            ax2.text(0.5, 0.5, 'No positive activations found', 
                    ha='center', va='center', transform=ax2.transAxes)
            ax2.set_title('Distribution of Non-zero Feature Activations')
        
        plt.tight_layout()
        plt.show()
        
        print(f"可視化完了: {len(token_strs)}トークン, 上位{top_k}特徴量")
        
    except Exception as e:
        print(f"可視化中にエラーが発生しました: {e}")
        import traceback
        traceback.print_exc()
    
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
    """インタラクティブな特徴量探索（改善版）"""
    
    try:
        # 入力検証
        if results is None or 'tokens' not in results or 'feature_acts' not in results:
            print("エラー: 結果データが無効です")
            return
        
        tokens = results['tokens']
        feature_acts = results['feature_acts']
        
        if tokens.shape[0] == 0 or feature_acts.shape[0] == 0:
            print("エラー: データが空です")
            return
        
        # トークンを安全に文字列に変換
        try:
            token_strs = model.to_str_tokens(tokens[0])
        except:
            token_strs = [f"Token_{i}" for i in range(tokens.shape[1])]
        
        # データサイズを制限（パフォーマンスのため）
        max_tokens = min(30, len(token_strs))
        max_features = min(10, feature_acts.shape[-1])
        
        token_strs = token_strs[:max_tokens]
        feature_acts_subset = feature_acts[0, :max_tokens, :]
        
        # 各トークンでの上位特徴量
        top_features_per_token = torch.topk(feature_acts_subset, k=max_features, dim=-1)
        
        # データフレーム作成
        data = []
        for pos, token_str in enumerate(token_strs):
            for rank in range(max_features):
                if rank < top_features_per_token.indices.shape[1]:
                    feature_id = top_features_per_token.indices[pos, rank].item()
                    activation = top_features_per_token.values[pos, rank].item()
                    data.append({
                        'position': pos,
                        'token': token_str,
                        'feature_id': feature_id,
                        'activation': activation,
                        'rank': rank + 1
                    })
        
        if not data:
            print("エラー: プロット用のデータが生成されませんでした")
            return
        
        df = pd.DataFrame(data)
        
        # プロットリーを使ったインタラクティブな可視化
        fig = px.scatter(df, 
                         x='position', 
                         y='feature_id', 
                         size='activation',
                         color='activation',
                         hover_data=['token', 'rank'],
                         title=f'Interactive Feature Activation Map ({max_tokens} tokens, top {max_features} features)',
                         labels={'position': 'Token Position', 
                                'feature_id': 'Feature ID'})
        
        fig.update_layout(height=600)
        fig.show()
        
        print(f"インタラクティブ可視化完了: {len(token_strs)}トークン, 上位{max_features}特徴量")
        
    except Exception as e:
        print(f"インタラクティブ探索中にエラーが発生しました: {e}")
        import traceback
        traceback.print_exc()

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

def batch_analyze_texts(texts: List[str], model, sae, batch_size: int = 8):
    """
    複数のテキストを効率的にバッチ処理でSAE分析する
    
    Args:
        texts (List[str]): 分析するテキストのリスト
        model: 使用する言語モデル
        sae: SAEモデル
        batch_size (int): バッチサイズ
        
    Returns:
        List[dict]: 各テキストの分析結果のリスト
    """
    results = []
    
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        print(f"バッチ {i//batch_size + 1}/{(len(texts) + batch_size - 1)//batch_size} 処理中...")
        
        for text in batch_texts:
            result = analyze_conversation_with_sae(text, model, sae)
            if result is not None:
                results.append(result)
        
        # メモリをクリア
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    return results

def validate_sae_inputs(feature_acts, tokens, model):
    """
    SAE入力データの検証を行う
    
    Args:
        feature_acts: 特徴活性化データ
        tokens: トークンデータ
        model: モデル
        
    Returns:
        bool: 検証結果
    """
    try:
        if feature_acts is None or tokens is None:
            print("エラー: 入力データがNoneです")
            return False
        
        if not isinstance(feature_acts, torch.Tensor) or not isinstance(tokens, torch.Tensor):
            print("エラー: 入力データがTensorではありません")
            return False
        
        if len(feature_acts.shape) < 2 or len(tokens.shape) < 2:
            print("エラー: データの次元が不正です")
            return False
        
        if feature_acts.shape[0] == 0 or tokens.shape[0] == 0:
            print("エラー: データが空です")
            return False
        
        return True
        
    except Exception as e:
        print(f"検証中にエラー: {e}")
        return False

def optimize_memory_usage():
    """
    メモリ使用量を最適化する
    """
    import gc
    
    # ガベージコレクションを実行
    gc.collect()
    
    # GPU/MPSメモリをクリア
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif torch.backends.mps.is_available():
        torch.mps.empty_cache()