import torch
import torch.nn as nn
import torch.optim as optim
from transformers import AutoModel, AutoTokenizer
import matplotlib.pyplot as plt
import numpy as np

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
    
