import torch
import torch.nn as nn
import torch.optim as optim
from transformers import AutoModel, AutoTokenizer
import matplotlib.pyplot as plt
import numpy as np
from typing import Tuple, Dict, Any

# SAEのモデルを定義
class SparseAutoencoder(nn.Module):
    def __init__(self, input_dim, feature_dim, l1_coeff=1e-5):
        super(SparseAutoencoder, self).__init__()
        self.input_dim = input_dim
        self.feature_dim = feature_dim
        self.l1_coeff = l1_coeff

        self.encoder = nn.Linear(input_dim, feature_dim)
        self.relu = nn.ReLU()
        self.decoder = nn.Linear(feature_dim, input_dim)
        
    def forward(self, x):
        encoded_pre_relu = self.encoder(x)
        features = self.relu(encoded_pre_relu)
        reconstructed = self.decoder(features)
        return reconstructed, features

    def compute_loss(self, x_original, x_reconstructed, features):
        reconstruction_loss = nn.functional.mse_loss(x_reconstructed, x_original)
        sparsity_loss = torch.mean(torch.abs(features))
        total_loss = reconstruction_loss + self.l1_coeff * sparsity_loss
        return total_loss, reconstruction_loss, sparsity_loss
        
    def get_features(self, x):
        """入力からSAE特徴を抽出するメソッド"""
        with torch.no_grad():
            encoded_pre_relu = self.encoder(x)
            features = self.relu(encoded_pre_relu)
        return features
        
    def save_model(self, path: str):
        """モデルを保存するヘルパーメソッド"""
        state = {
            'input_dim': self.input_dim,
            'feature_dim': self.feature_dim,
            'l1_coeff': self.l1_coeff,
            'state_dict': self.state_dict()
        }
        torch.save(state, path)
        
    @classmethod
    def load_model(cls, path: str, device=None):
        """保存されたモデルをロードするクラスメソッド"""
        if device is None:
            device = torch.device("mps" if torch.mps.is_available() else "cpu")
        state = torch.load(path, map_location=device)
        model = cls(
            input_dim=state['input_dim'],
            feature_dim=state['feature_dim'],
            l1_coeff=state['l1_coeff']
        )
        model.load_state_dict(state['state_dict'])
        model.to(device)
        return model

