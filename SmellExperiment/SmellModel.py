"""Encoder and learnable prototypes from the centralized SMELL experiment."""

import torch
from torch import nn
from torch.nn import functional as F


class TwoNN(nn.Module):
    def __init__(self, embedding_size=64):
        super().__init__()
        self.base = nn.Sequential(
            nn.Linear(784, 200),
            nn.ReLU(inplace=True),
            nn.Linear(200, 200),
            nn.ReLU(inplace=True),
        )
        self.embedding = nn.Sequential(nn.Linear(200, embedding_size), nn.ReLU())

    def forward(self, images):
        return self.embedding(self.base(torch.flatten(images, start_dim=1)))


class PrototypicalCenterLoss(nn.Module):
    def __init__(self, num_classes=10, embedding_size=64, margin=1.0):
        super().__init__()
        self.prototypes = nn.Parameter(torch.randn(num_classes, embedding_size))
        self.margin = margin

    def forward(self, embeddings, labels):
        intra = (embeddings - self.prototypes[labels]).square().sum(dim=1).mean()
        distances = torch.cdist(self.prototypes, self.prototypes)
        diagonal = torch.eye(len(self.prototypes), device=embeddings.device) * 1e12
        inter = F.relu(self.margin - distances - diagonal).mean()
        return intra + inter


class PairPrototypeLoss(nn.Module):
    def __init__(self, embedding_size=64, similar_prototypes=2, dissimilar_prototypes=2):
        super().__init__()
        self.similar_prototypes = nn.Parameter(torch.cat([
            torch.zeros(1, embedding_size),
            torch.randn(similar_prototypes - 1, embedding_size),
        ]))
        self.dissimilar_prototypes = nn.Parameter(
            torch.randn(dissimilar_prototypes, embedding_size) + 2.0
        )

    def forward(self, first, second, labels):
        differences = torch.abs(first - second).unsqueeze(1)
        similar_distances = (differences - self.similar_prototypes).square().sum(dim=2)
        dissimilar_distances = (differences - self.dissimilar_prototypes).square().sum(dim=2)
        # Log-space form of the original 1 / (1 + distance**2) probabilities.
        similar_scores = -torch.log1p(similar_distances)
        dissimilar_scores = -torch.log1p(dissimilar_distances)
        denominator = torch.logsumexp(torch.cat([similar_scores, dissimilar_scores], dim=1), dim=1)
        log_similar = torch.logsumexp(similar_scores, dim=1) - denominator
        log_dissimilar = torch.logsumexp(dissimilar_scores, dim=1) - denominator
        return -torch.where(labels.bool(), log_similar, log_dissimilar).mean()


class SmellModel(nn.Module):
    def __init__(self, embedding_size=64, num_classes=10):
        super().__init__()
        self.encoder = TwoNN(embedding_size)
        self.center_loss = PrototypicalCenterLoss(num_classes, embedding_size)
        self.pair_loss = PairPrototypeLoss(embedding_size)

    def forward(self, images):
        return self.encoder(images)
