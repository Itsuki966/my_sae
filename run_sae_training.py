from sae_trainer import extract_activations, create_data_loader, train_sparse_autoencoder

if __name__ == "__main__":
    
    # extract_activationsの変数設定
    llm_model_name = "bert-base-uncased"    # 使用するLLMモデル名    
    texts = ["This is a sample text for testing the Sparse Autoencoder."] * 500  # 使用するテキスト
    target_layer_idx = 5    # 抽出するLLMの層インデックス
    num_samples_for_training = 500  # 訓練に使用するサンプル数

    # create_data_loaderの変数設定
    batch_size = 256  # バッチサイズ

    # train_sparse_autoencoderの変数設定
    num_epochs = 200  # 訓練エポック数
    sae_l1_coeff = 1e-4 # スパース性の度合いを調整する係数

    activations = extract_activations(
        llm_model_name,
        texts,
        target_layer_idx,
        num_samples_for_training
    )
    
    if activations.shape[0] == 0:
        print("No activations were extracted. Exiting.")
    else:
        data_loader = create_data_loader(activations, batch_size)
        sae_model, training_losses, reconstruction_losses, sparsity_losses = train_sparse_autoencoder(
            activations,
            data_loader,
            num_epochs=num_epochs,
            sae_l1_coeff=sae_l1_coeff
        )
        
    
    