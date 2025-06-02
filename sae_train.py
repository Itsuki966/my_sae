import torch
import torch.optim as optim
from transformers import AutoModel, AutoTokenizer
import matplotlib.pyplot as plt
import numpy as np

from sae_model import SparseAutoencoder
from activation_utils import get_llm_activations

if __name__ == "__main__":
    device = torch.device("mps" if torch.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    llm_model_name = "bert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(llm_model_name)
    llm_model = AutoModel.from_pretrained(llm_model_name).to(device)
    sample_texts = "This is a sample text for testing the Sparse Autoencoder."

    target_layer_idx = 5
    print(f"Extracting activations from layer {target_layer_idx} of {llm_model_name}.")

    num_samples_for_training  = 100

    num_samples_for_training = 500
    training_texts = [sample_texts[i % len(sample_texts)] for i in range(num_samples_for_training)]
    
    llm_activations = get_llm_activations(llm_model, tokenizer, training_texts, 
                                          layer_index=target_layer_idx)
    input_dim = llm_activations.shape[1] # LLMの活性化ベクトルの次元
    print(f"LLM activations shape: {llm_activations.shape}") # (num_samples, hidden_dim)

    # 4. SAEのセットアップ
    # feature_dim は通常 input_dim より大きく設定し、過完備な表現を目指す
    # 解釈可能性のため、input_dim の 2倍, 4倍, 8倍などが試される
    sae_feature_dim = input_dim * 4 
    sae_l1_coeff = 1e-4  # スパース性の度合いを調整する係数
    
    sae_model = SparseAutoencoder(input_dim, sae_feature_dim, l1_coeff=sae_l1_coeff).to(device)
    optimizer = optim.Adam(sae_model.parameters(), lr=1e-3)

    # 5. 訓練ループ
    num_epochs = 200  # 実際の訓練ではより多くのエポック数が必要
    batch_size = 64   # VRAMに応じて調整
    
    dataset = torch.utils.data.TensorDataset(llm_activations)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    print(f"Starting SAE training for {num_epochs} epochs...")
    training_losses, reconstruction_losses, sparsity_losses = [], [], []

    for epoch in range(num_epochs):
        epoch_total_loss, epoch_recon_loss, epoch_sparse_loss = 0, 0, 0
        num_batches = 0
        for batch_data in dataloader:
            activations_batch = batch_data[0].to(device)

            optimizer.zero_grad()
            reconstructed_batch, features_batch = sae_model(activations_batch)
            loss, recon_loss, sparse_loss = sae_model.compute_loss(activations_batch, reconstructed_batch, features_batch)
            
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

        if (epoch + 1) % 20 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Total Loss: {avg_total_loss:.6f}, Recon Loss: {avg_recon_loss:.6f}, Sparsity Loss: {avg_sparse_loss:.6f}")
            with torch.no_grad():
                _, features_eval = sae_model(llm_activations.to(device))
                avg_l0_norm = (features_eval > 1e-6).float().sum(dim=1).mean().item() 
                print(f"  Average L0 norm of features: {avg_l0_norm:.2f} / {sae_feature_dim}")

    print("Training complete.")

    # --- 6. 可視化 ---
    print("Visualizing results...")

    # a) 訓練損失のプロット
    plt.figure(figsize=(18, 5))
    plt.subplot(1, 3, 1)
    plt.plot(training_losses); plt.title('Total Loss'); plt.xlabel('Epoch'); plt.ylabel('Loss')
    plt.subplot(1, 3, 2)
    plt.plot(reconstruction_losses); plt.title('Reconstruction Loss'); plt.xlabel('Epoch'); plt.ylabel('Loss')
    plt.subplot(1, 3, 3)
    plt.plot(sparsity_losses); plt.title('Sparsity Loss (scaled)'); plt.xlabel('Epoch'); plt.ylabel('Loss')
    plt.tight_layout(); plt.show()

    # b) エンコーダの重みの可視化
    # sae_model.encoder.weight は (feature_dim, input_dim) の形状
    encoder_weights = sae_model.encoder.weight.data.cpu().numpy()
    
    plt.figure(figsize=(10, 8))
    num_features_to_show = min(32, sae_feature_dim) 
    num_input_dims_to_show = min(100, input_dim) 
    
    plt.imshow(encoder_weights[:num_features_to_show, :num_input_dims_to_show], aspect='auto', cmap='viridis')
    plt.colorbar(label='Weight Value')
    plt.title(f'Encoder Weights (First {num_features_to_show} Features vs First {num_input_dims_to_show} Input Dims)')
    plt.xlabel('Input LLM Activation Dimension Index')
    plt.ylabel('SAE Feature Index')
    plt.show()

    # c) 特徴量の活性化のスパース性ヒストグラム
    sae_model.eval()
    with torch.no_grad():
        sample_activations_for_hist = llm_activations[:min(1000, len(llm_activations))].to(device) 
        _, features_hist = sae_model(sample_activations_for_hist)
        features_hist = features_hist.cpu().numpy()

    if sae_feature_dim > 0:
        feature_idx_to_plot = np.random.randint(0, sae_feature_dim) # ランダムな特徴を選択
        plt.figure(figsize=(8, 5))
        plt.hist(features_hist[:, feature_idx_to_plot], bins=50, log=True)
        plt.title(f'Activation Distribution for SAE Feature {feature_idx_to_plot}')
        plt.xlabel('Activation Value'); plt.ylabel('Frequency (log scale)')
        plt.show()

    l0_norms_per_sample = (features_hist > 1e-6).sum(axis=1)
    plt.figure(figsize=(8, 5))
    plt.hist(l0_norms_per_sample, bins=min(50, sae_feature_dim))
    plt.title('Distribution of L0 Norms (Active Features per Sample)')
    plt.xlabel('Number of Active Features'); plt.ylabel('Frequency')
    plt.show()
    print(f"Average number of active features per sample (L0 norm) on test subset: {np.mean(l0_norms_per_sample):.2f} out of {sae_feature_dim}")

    # d) 再構成品質の例
    if len(llm_activations) > 0:
        original_sample = llm_activations[0].unsqueeze(0).to(device)
        with torch.no_grad():
            reconstructed_sample, _ = sae_model(original_sample)
        
        original_sample_np = original_sample.cpu().squeeze().numpy()
        reconstructed_sample_np = reconstructed_sample.cpu().squeeze().numpy()

        plt.figure(figsize=(15, 5))
        plt.subplot(1, 2, 1)
        plt.plot(original_sample_np, label='Original', alpha=0.7)
        plt.plot(reconstructed_sample_np, label='Reconstructed', linestyle='--', alpha=0.7)
        plt.title('Original vs. Reconstructed Activations (Sample 0)'); plt.xlabel('Dimension'); plt.ylabel('Activation')
        plt.legend(); plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 2, 2)
        error = original_sample_np - reconstructed_sample_np
        plt.plot(error, label='Reconstruction Error', color='red', alpha=0.7)
        plt.title('Reconstruction Error (Sample 0)'); plt.xlabel('Dimension'); plt.ylabel('Error')
        plt.legend(); plt.grid(True, alpha=0.3)
        
        plt.tight_layout(); plt.show()
        mse = np.mean((original_sample_np - reconstructed_sample_np)**2)
        print(f"MSE for sample 0 reconstruction: {mse:.6f}")

    print("\nFurther steps for interpretation:")
    print("- Analyze which input texts maximally activate specific SAE features.")
    print("- Correlate SAE feature activations with specific linguistic properties or task performance.")
    print("- If decoder weights are tied or constrained (e.g. non-negative), their interpretation can also be insightful.")    