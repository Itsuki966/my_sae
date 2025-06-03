import torch
import torch.optim as optim
from transformers import AutoModel, AutoTokenizer
import matplotlib.pyplot as plt
import numpy as np

from sae_model import SparseAutoencoder
from activation_utils import get_llm_activations_residual_stream

if __name__ == "__main__":
    device = torch.device("mps" if torch.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    llm_model_name = "distilgpt2"
    tokenizer = AutoTokenizer.from_pretrained(llm_model_name)
    llm_model = AutoModel.from_pretrained(llm_model_name).to(device)

    sample_texts = [
        "The quick brown fox jumps over the lazy dog.",
        "Large language models are powerful tools for various NLP tasks.",
        "Sparse autoencoders can help us understand the internal representations of LLMs.",
        "PyTorch is a widely used deep learning framework.",
        "Interpreting neural networks is a key challenge in modern AI research."
    ]
    
    if llm_model_name == "distilgpt2":
        target_layer_idx = 2
    else:
        target_layer_idx = 5
    print(f"Extracting activations from layer {target_layer_idx} of {llm_model_name}.")

    num_samples_for_training = 500
    training_texts = [sample_texts[i % len(sample_texts)] for i in range(num_samples_for_training)]
    
    llm_activations = get_llm_activations_residual_stream(llm_model, 
                                                          tokenizer, 
                                                          training_texts, 
                                                          layer_index=target_layer_idx, 
                                                          max_length=128) # シーケンス長
    
    if llm_activations.shape[0] == 0:
        print("No activations were extracted. Exiting.")
        exit()

    input_dim = llm_activations.shape[1] # LLMの活性化ベクトルの次元 (GPTのhidden_size)
    print(f"LLM activations shape: {llm_activations.shape}") # (total_actual_tokens, hidden_dim)
    
    sae_feature_dim = input_dim * 4  # 過完備な表現を目指す
    sae_l1_coeff = 1e-4  # スパース性の度合いを調整する係数
    sae_model = SparseAutoencoder(input_dim, sae_feature_dim, l1_coeff=sae_l1_coeff).to(device)
    optimizer = optim.Adam(sae_model.parameters(), lr=1e-3)

    num_epochs = 200
    batch_size = 256

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
    
    plt.figure(figsize=(18, 5))
    plt.subplot(1, 3, 1); plt.plot(training_losses); plt.title('Total Loss')
    plt.subplot(1, 3, 2); plt.plot(reconstruction_losses); plt.title('Reconstruction Loss')
    plt.subplot(1, 3, 3); plt.plot(sparsity_losses); plt.title('Sparsity Loss')
    plt.tight_layout(); plt.show()

    encoder_weights = sae_model.encoder.weight.data.cpu().numpy()
    plt.figure(figsize=(10, 8))
    num_features_to_show = min(32, sae_feature_dim) 
    num_input_dims_to_show = min(100, input_dim) 
    plt.imshow(encoder_weights[:num_features_to_show, :num_input_dims_to_show], aspect='auto', cmap='viridis')
    plt.colorbar(label='Weight Value'); plt.title(f'Encoder Weights (First {num_features_to_show} Feats vs First {num_input_dims_to_show} In Dims)')
    plt.xlabel('Input LLM Activation Dimension Index'); plt.ylabel('SAE Feature Index'); plt.show()    