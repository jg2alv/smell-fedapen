"""CNN architectures shared by the baseline and FedAPEN experiments."""

import torch
from torch import nn
from torch.nn import functional as F


class FashionMNISTCNN(nn.Module):
    """Two 3x3 convolutions with six channels and stride two (Appendix A.2.1)."""

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=3, stride=2)
        self.conv2 = nn.Conv2d(6, 6, kernel_size=3, stride=2)
        self.fc = nn.Linear(6 * 6 * 6, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = torch.flatten(x, start_dim=1)
        return self.fc(x)


class CIFARCNN(nn.Module):
    """The homogeneous CNN2_BN used in the supplied implementation."""

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(3, 64, 3), nn.BatchNorm2d(64), nn.ReLU())
        self.conv2 = nn.Sequential(nn.Conv2d(64, 128, 3), nn.BatchNorm2d(128), nn.ReLU())
        self.conv3 = nn.Sequential(nn.Conv2d(128, 256, 3), nn.BatchNorm2d(256), nn.ReLU())
        self.maxpool = nn.MaxPool2d(2, 2)
        self.fc = nn.Linear(256 * 4 * 4, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.maxpool(self.conv1(x))
        x = self.maxpool(self.conv2(x))
        x = self.conv3(x)
        return self.fc(torch.flatten(x, start_dim=1))


def create_model(dataset: str, initialization: str = "kaiming") -> nn.Module:
    model = {"fashion-mnist": FashionMNISTCNN, "cifar10": CIFARCNN}[dataset]()
    if initialization == "none":
        return model
    initializers = {"kaiming": nn.init.kaiming_normal_, "xavier": nn.init.xavier_normal_}
    initializer = initializers[initialization]
    for module in model.modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            initializer(module.weight)
    return model
