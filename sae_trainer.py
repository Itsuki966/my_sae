import torch
import torch.optim as optim
from transformers import AutoModel, AutoTokenizer
from typing import List, Tuple, Dict, Optional, Any, Union
import os
import matplotlib.pyplot as plt
import numpy as np
import math

from sae_model import SparseAutoencoder
from activation_utils import get_llm_activations_residual_stream

# 定数
DEFAULT_SEQ_LEN = 128   # デフォルトのシーケンス長
SAE_FEATURE_RATIO = 1.1   # この値を大きくすると解釈しやすい表現が得られるが、学習コストが上がる
LEARNING_RATE = 1e-3    # SAEを学習する際の学習率

def extract_specific_layer_activations(
    llm_model_name: str,
    texts: List[str],
    target_layer_idx: int,
    num_samples: int,
    max_length: int = DEFAULT_SEQ_LEN
) -> Tuple[torch.Tensor, dict]:
    """
    特定の層のLLMの活性化を抽出する
    
    Args:
        llm_model_name: 使用するLLMのモデル名
        texts: 処理するテキストのリスト
        target_layer_idx: 抽出する層のインデックス
        num_samples: 使用するサンプル数
        max_length: 最大シーケンス長
        
    Returns:
        Tuple[torch.Tensor, Dict]: 活性化テンソルと、テキストごとの活性化辞書
    """
    
    tokenizer = AutoTokenizer.from_pretrained(llm_model_name)
    llm_model = AutoModel.from_pretrained(llm_model_name)
    training_texts = [texts[i % len(texts)] for i in range(num_samples)]
    activations, activation_dict = get_llm_activations_residual_stream(
        llm_model, tokenizer, training_texts, layer_index=target_layer_idx, max_length=max_length
    )
    return activations, activation_dict

def extract_all_layer_activations(
    llm_model_name: str,
    texts: List[str],
    num_samples: int,
    max_length: int = DEFAULT_SEQ_LEN
) -> Dict[int, torch.Tensor]:
    """
    LLMの全層の活性化を抽出する
    
    Args:
        llm_model_name: 使用するLLMのモデル名
        texts: 処理するテキストのリスト
        num_samples: 使用するサンプル数
        max_length: 最大シーケンス長
        
    Returns:
        Dict[int, torch.Tensor]: 層インデックスをキー、活性化テンソルを値とする辞書
    """
    
    tokenizer = AutoTokenizer.from_pretrained(llm_model_name)
    llm_model = AutoModel.from_pretrained(llm_model_name)
    num_layers = llm_model.config.num_hidden_layers
    all_layer_activations = {}
    
    print("")
    print(f"Extracting activations from {num_layers} layers of the model: {llm_model_name}")
    training_texts = [texts[i % len(texts)] for i in range(num_samples)]

    for layer_idx in range(num_layers):
        print(f"Extracting activations from layer {layer_idx + 1}/{num_layers}...")
        activations, _ = get_llm_activations_residual_stream(
            llm_model, tokenizer, training_texts, layer_index=layer_idx, max_length=max_length
        )
        all_layer_activations[layer_idx] = activations
        print(f"Layer {layer_idx + 1} activations shape: {activations.shape}")
    
    print("All layer activations extraction complete.")
    return all_layer_activations

def create_data_loader(activations: torch.Tensor, batch_size: int) -> torch.utils.data.DataLoader:
    """
    活性化テンソルからDataLoaderを作成
    
    Args:
        activations: 活性化テンソル
        batch_size: バッチサイズ
        
    Returns:
        torch.utils.data.DataLoader: 訓練用DataLoader
    """
    
    dataset = torch.utils.data.TensorDataset(activations)
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

def train_sparse_autoencoder(
    activations: torch.Tensor,
    data_loader: torch.utils.data.DataLoader,
    num_epochs: int = 200,
    sae_l1_coeff: float = 1e-4,
    llm_model_name: str = "",
    layer_idx: int = -1,
    save_dir: Optional[str] = None,
    skip_plot: bool = True
) -> Tuple[SparseAutoencoder, list, list, list, int, int]:
    """
    特定の層のLLMの活性化からSAEを学習する
    
    Args:
        activations: 活性化テンソル
        data_loader: 訓練用DataLoader
        num_epochs: 訓練エポック数
        sae_l1_coeff: L1正則化係数
        llm_model_name: LLMモデル名（記録用）
        layer_idx: 層インデックス（記録用）
        save_dir: モデル保存ディレクトリ（指定があれば保存）
        skip_plot: プロット表示をスキップするかどうか
        
    Returns:
        Tuple: 
            SparseAutoencoderのインスタンス
            学習損失、再構成損失、スパース性損失のリスト
            SAEの特徴次元、入力次元
    """
    device = torch.device("mps" if torch.mps.is_available() else "cpu")
    print(f"Using device: {device}")


    if activations.shape[0] == 0:
        print("No activations were extracted. Exiting.")
        return None, [], [], [], 0, 0

    input_dim = activations.shape[1]
    sae_feature_dim = int(input_dim * SAE_FEATURE_RATIO)
    print("----------LLM・SAEの情報----------")
    print(f"LLMモデル名: {llm_model_name}")
    print(f"対象層インデックス: {layer_idx + 1}")
    print(f"LLMの活性化ベクトルの次元数: {input_dim}, SAEの特徴次元: {sae_feature_dim}")
    print("----------------------------------")
    sae_model = SparseAutoencoder(input_dim, sae_feature_dim, l1_coeff=sae_l1_coeff).to(device)
    optimizer = optim.Adam(sae_model.parameters(), lr=LEARNING_RATE)

    print("")
    print(f"Starting SAE training for layer {layer_idx + 1} for {num_epochs} epochs...")

    training_losses, reconstruction_losses, sparsity_losses = [], [], []

    for epoch in range(num_epochs):
        sae_model.train()
        epoch_total_loss, epoch_recon_loss, epoch_sparse_loss = 0, 0, 0
        num_batches = 0

        for batch_data in data_loader:
            activations_batch = batch_data[0].to(device)
            optimizer.zero_grad()
            reconstructed_batch, features_batch = sae_model(activations_batch)
            loss, recon_loss, sparse_loss = sae_model.compute_loss(
                activations_batch, reconstructed_batch, features_batch
            )
            loss.backward()
            optimizer.step()

            epoch_total_loss += loss.item()
            epoch_recon_loss += recon_loss.item()
            epoch_sparse_loss += sparse_loss.item()
            num_batches += 1

        avg_total_loss = epoch_total_loss / num_batches
        avg_recon_loss = epoch_recon_loss / num_batches
        avg_sparse_loss = epoch_sparse_loss / num_batches

        training_losses.append(avg_total_loss)
        reconstruction_losses.append(avg_recon_loss)
        sparsity_losses.append(avg_sparse_loss)

        if (epoch + 1) % 20 == 0 or epoch == 0 or epoch == num_epochs - 1:
            print(f"Epoch {epoch + 1}/{num_epochs}, Total Loss: {avg_total_loss:.4f}, Recon Loss: {avg_recon_loss:.4f}, Sparse Loss: {avg_sparse_loss:.4f}")
            with torch.no_grad():
                _, features_eval = sae_model(activations[:min(100, activations.shape[0])].to(device))
                active_features = (features_eval > 0).float().sum(dim=0)
                print(f"Active features: {active_features.sum().item()}/{sae_feature_dim} ({active_features.sum().item()/sae_feature_dim*100:.2f}%)")

    print(f"Training complete for layer {layer_idx + 1}.")
    
    # 学習曲線のプロット（skip_plotがFalseの場合のみ表示）
    if not skip_plot:
        plot_training_curves(training_losses, reconstruction_losses, sparsity_losses)
    
    # モデルの保存
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"sae_model_layer_{layer_idx}.pt")
        sae_model.save_model(save_path)
        print(f"Model saved to {save_path}")
        
        # 損失のプロットも保存
        if not skip_plot:
            plot_training_curves(training_losses, reconstruction_losses, sparsity_losses, 
                                save_path=os.path.join(save_dir, f"sae_training_curves_layer_{layer_idx}.png"))
    
    return sae_model, training_losses, reconstruction_losses, sparsity_losses, sae_feature_dim, input_dim

def plot_training_curves(training_losses: List[float], 
                       reconstruction_losses: List[float], 
                       sparsity_losses: List[float],
                       save_path: Optional[str] = None):
    """
    訓練曲線をプロットする
    
    Args:
        training_losses: 訓練損失のリスト
        reconstruction_losses: 再構成損失のリスト
        sparsity_losses: スパース性損失のリスト
        save_path: 保存パス（指定があれば保存）
    """
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(training_losses, label='Total Loss')
    plt.plot(reconstruction_losses, label='Reconstruction Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training & Reconstruction Loss')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(sparsity_losses, label='Sparsity Loss', color='green')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Sparsity Loss')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"Training curves saved to {save_path}")
    
    plt.show()

def plot_all_layers_training_curves(all_losses: Dict[int, Tuple[List[float], List[float], List[float]]], 
                                  save_path: Optional[str] = None):
    """
    全ての層の訓練曲線を複数のサブプロットとして1つの図にまとめる
    
    Args:
        all_losses: 層インデックスをキー、(訓練損失, 再構成損失, スパース性損失)のタプルを値とする辞書
        save_path: 保存パス（指定があれば保存）
    """
    num_layers = len(all_losses)
    
    # レイアウトを決定（行数、列数を自動計算）
    cols = min(4, num_layers)  # 最大4列まで
    rows = math.ceil(num_layers / cols)
    
    # サブプロットの大きさを調整
    fig_width = 5 * cols
    fig_height = 3 * rows
    
    fig, axes = plt.subplots(rows, cols, figsize=(fig_width, fig_height))
    if rows == 1 and cols == 1:
        axes = np.array([axes])  # 1x1の場合、axes をリストに変換
    axes = axes.flatten()  # 2次元配列を1次元に変換
    
    # 各層ごとにサブプロットを作成
    for i, (layer_idx, (total_loss, recon_loss, sparse_loss)) in enumerate(sorted(all_losses.items())):
        ax = axes[i]
        epochs = range(1, len(total_loss) + 1)
        
        # 損失の可視化
        line1, = ax.plot(epochs, total_loss, 'b-', label='Total Loss')
        line2, = ax.plot(epochs, recon_loss, 'g--', label='Recon Loss')
        line3, = ax.plot(epochs, sparse_loss, 'r-.', label='Sparse Loss')
        
        ax.set_title(f'Layer {layer_idx + 1}')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # 対数スケールの方が見やすい場合がある
        if max(total_loss) / min(total_loss) > 100:  # 損失の範囲が広い場合
            ax.set_yscale('log')
        
        # 凡例の設定
        if i == 0:  # 最初のサブプロットにだけ凡例を表示
            ax.legend()
    
    # 未使用のサブプロットを非表示にする
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])
    
    plt.tight_layout()
    fig.suptitle('Training Curves for All Layers', fontsize=16)
    plt.subplots_adjust(top=0.92)  # タイトル用のスペースを確保
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Combined training curves saved to {save_path}")
    
    plt.show()

def train_all_layer_saes(
    llm_model_name: str,
    texts: List[str],
    num_samples: int,
    batch_size: int = 64,
    num_epochs: int =200,
    sae_l1_coeff: float = 1e-4,
    save_dir: Optional[str] = None,
    plot_individual_layers: bool = False
) -> Dict[int, SparseAutoencoder]:
    """
    全ての層に対してSAEを学習する
    
    Args:
        llm_model_name: 使用するLLMのモデル名
        texts: 処理するテキストのリスト
        num_samples: 使用するサンプル数
        batch_size: バッチサイズ
        num_epochs: 訓練エポック数
        sae_l1_coeff: L1正則化係数
        save_dir: モデル保存ディレクトリ
        plot_individual_layers: 各層の学習曲線を個別に表示するかどうか
        
    Returns:
        Dict[int, SparseAutoencoder]: 層インデックスをキー、SAEモデルを値とする辞書
    """
    
    # 全ての層から活性を取得
    all_layer_activations = extract_all_layer_activations(
        llm_model_name, texts, num_samples
    )
    
    # 各層のSAEを格納する辞書
    layer_saes = {}
    # 各層の損失を格納する辞書
    all_layer_losses = {}
    
    # 各層ごとにSAEを学習
    for layer_idx, activations in all_layer_activations.items():
        print(f"\nTraining SAE for layer {layer_idx + 1}...")
        dataloader = create_data_loader(activations, batch_size)
        
        # 個別の層のプロットを無効化（後でまとめてプロットするため）
        if not plot_individual_layers:
            # 元のplot_training_curves関数を呼び出さないように修正
            sae_model, training_losses, reconstruction_losses, sparsity_losses, _, _ = train_sparse_autoencoder(
                activations, dataloader, num_epochs=num_epochs, sae_l1_coeff=sae_l1_coeff,
                llm_model_name=llm_model_name, layer_idx=layer_idx, save_dir=save_dir,
                skip_plot=True  # プロットをスキップするフラグを追加
            )
        else:
            # 従来通り個別のプロットも表示
            sae_model, training_losses, reconstruction_losses, sparsity_losses, _, _ = train_sparse_autoencoder(
                activations, dataloader, num_epochs=num_epochs, sae_l1_coeff=sae_l1_coeff,
                llm_model_name=llm_model_name, layer_idx=layer_idx, save_dir=save_dir
            )
        
        layer_saes[layer_idx] = sae_model
        all_layer_losses[layer_idx] = (training_losses, reconstruction_losses, sparsity_losses)
    
    # 全ての層の結果をまとめてプロット
    print("\nAll layers training complete. Plotting combined results...")
    combined_save_path = None
    if save_dir:
        combined_save_path = os.path.join(save_dir, "all_layers_training_curves.png")
    
    plot_all_layers_training_curves(all_layer_losses, save_path=combined_save_path)
                
    return layer_saes