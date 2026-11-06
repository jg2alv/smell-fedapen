"""Two-hidden-layer MLP for Fashion-MNIST or CIFAR-10 images."""

import torch
from torch import nn
from torch.nn import functional as F


class PoCModel_2HiddenLayers(nn.Module):
    """Image classifier: input_size -> 256 -> 128 -> num_classes."""

    def __init__(self, num_classes: int = 10, input_size: int = 784) -> None:
        super().__init__()
        self.hidden1 = nn.Linear(input_size, 256)
        self.hidden2 = nn.Linear(256, 128)
        self.output_layer = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.hidden1(x))
        x = F.relu(self.hidden2(x))
        return self.output_layer(x)
