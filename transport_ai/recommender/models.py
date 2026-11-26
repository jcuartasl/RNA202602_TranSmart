"""Dos torres pequeñas con embeddings ID y contenido; no usa Popularity del CSV.

La torre de usuario usa UserID conocido + Preferences. La torre de destino
usa DestinationID conocido + Type. Los índices 0 representan desconocidos.
Se pueden puntuar destinos y usuarios inéditos mediante las variables de
contenido, aunque su precisión depende de los datos de entrenamiento.
"""
from __future__ import annotations

from math import sqrt

import torch
from torch import nn


class TwoTowerRecommender(nn.Module):
    def __init__(self, n_users: int, n_items: int, n_types: int = 5,
                 embedding_dim: int = 8, hidden_dim: int = 32,
                 latent_dim: int = 16, dropout: float = 0.1):
        super().__init__()
        self.user_embedding = nn.Embedding(n_users + 1, embedding_dim, padding_idx=0)
        self.item_embedding = nn.Embedding(n_items + 1, embedding_dim, padding_idx=0)
        self.user_tower = nn.Sequential(
            nn.Linear(embedding_dim + n_types, hidden_dim), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(hidden_dim, latent_dim)
        )
        self.item_tower = nn.Sequential(
            nn.Linear(embedding_dim + n_types, hidden_dim), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(hidden_dim, latent_dim)
        )
        self.bias = nn.Parameter(torch.zeros(1))
        self.latent_dim = latent_dim

    def forward(self, user_indices, item_indices, user_preferences, item_types):
        user_repr = self.user_tower(torch.cat((self.user_embedding(user_indices), user_preferences), dim=-1))
        item_repr = self.item_tower(torch.cat((self.item_embedding(item_indices), item_types), dim=-1))
        return (user_repr * item_repr).sum(dim=-1) / sqrt(self.latent_dim) + self.bias

    def count_parameters(self) -> int:
        return sum(param.numel() for param in self.parameters() if param.requires_grad)
