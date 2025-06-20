import torch
import torch.optim as optim
from transformers import AutoModel, AutoTokenizer
from typing import List, Tuple, Dict

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
    """特定の層のLLMの活性化を抽出する"""
    
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
    """LLMの全層の活性化を抽出する"""
    
    tokenizer = AutoTokenizer.from_pretrained(llm_model_name)
    llm_model = AutoModel.from_pretrained(llm_model_name)
    num_layers = llm_model.config.num_hidden_layers
    all_layer_activations = {}
    
    print("")
    print(f"Extracting activations from {num_layers} layers of the model: {llm_model_name}")

    for layer_idx in range(num_layers):
        print(f"Extracting activations from layer {layer_idx + 1}/{num_layers}...")
        activations, _ = get_llm_activations_residual_stream(
            llm_model, tokenizer, texts, layer_index=layer_idx, max_length=max_length
        )
        all_layer_activations[layer_idx] = activations
        print(f"Layer {layer_idx + 1} activations shape: {activations.shape}")
    
    print("All layer activations extraction complete.")
    return all_layer_activations

def create_data_loader(activations: torch.Tensor, batch_size: int) -> torch.utils.data.DataLoader:
    """活性化テンソルからDataLoaderを作成"""
    
    dataset = torch.utils.data.TensorDataset(activations)
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

def train_sparse_autoencoder(
    activations: torch.Tensor,
    data_loader: torch.utils.data.DataLoader,
    num_epochs: int = 200,
    sae_l1_coeff: float = 1e-4,
    llm_model_name: str = "",
    layer_idx: int = -1
) -> Tuple[SparseAutoencoder, list, list, list, int, int]:
    """
    特定の層のLLMの活性化からSAEを学習する
    -SparseAutoencoderのインスタンス
    -学習損失、再構成損失、スパース性損失のリスト
    -SAEの特徴次元、入力次元
    """
    device = torch.device("mps" if torch.mps.is_available() else "cpu")
    print(f"Using device: {device}")


    if activations.shape[0] == 0:
        print("No activations were extracted. Exiting.")
        return

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

        print(f"Epoch {epoch + 1}/{num_epochs}, Total Loss: {avg_total_loss:.4f}, Recon Loss: {avg_recon_loss:.4f}, Sparse Loss: {avg_sparse_loss:.4f}")
        if (epoch + 1) % 20 == 0:
            with torch.no_grad():
                _, features_eval = sae_model(activations.to(device))
                print(f"Evaluated features shape: {features_eval.shape}")

    print(f"Training complete for layer {layer_idx + 1}.")
    return sae_model, training_losses, reconstruction_losses, sparsity_losses, sae_feature_dim, input_dim

def train_all_layer_saes(
    llm_model_name: str,
    texts: List[str],
    num_samples: int,
    batch_size: int = 64,
    num_epochs: int =200,
    sae_l1_coeff: float = 1e-4
) -> Dict[int, SparseAutoencoder]:
    """全ての層に対してSAEを学習する"""
    
    # 全ての層から活性を取得
    all_layer_activations = extract_all_layer_activations(
        llm_model_name, texts, num_samples
    )
    
    # 各層のSAEを格納する辞書
    layer_saes = {}
    
    # 各層ごとにSAEを学習
    for layer_idx, activations in all_layer_activations.items():
        print("")
        print(f"Training SAE for layer {layer_idx + 1}...")
        
        data_loader = create_data_loader(activations, batch_size)
        sae_model, training_loss, recon_loss, sparse_loss, feature_dim, input_dim = train_sparse_autoencoder(
            activations, data_loader,
            num_epochs=num_epochs,
            sae_l1_coeff=sae_l1_coeff,
            llm_model_name=llm_model_name,
            layer_idx=layer_idx
        )
        layer_saes[layer_idx] = sae_model
                
    return layer_saes