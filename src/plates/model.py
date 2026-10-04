import torch.nn as nn
from torchvision import models


def get_model(num_classes: int = 2, freeze_backbone: bool = True) -> nn.Module:
    """Return ResNet-18 with a replaced classification head."""
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False

    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model
