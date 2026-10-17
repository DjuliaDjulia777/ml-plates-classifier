"""ResNet-18 model for binary plate classification."""

import torch.nn as nn
from torchvision import models


def get_model(
    num_classes: int = 2,
    freeze_backbone: bool = True,
    pretrained: bool = True,
) -> nn.Module:
    """Return ResNet-18 with a replaced classification head.

    Args:
        num_classes: Number of output classes.
        freeze_backbone: If True, only the new head is trainable.
        pretrained: If True, load ImageNet weights (requires internet on
            the first call). Use False for inference from a checkpoint
            and for tests.

    Returns:
        The model.
    """
    weights = models.ResNet18_Weights.DEFAULT if pretrained else None
    model = models.resnet18(weights=weights)

    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False

    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def set_train_mode(model: nn.Module, freeze_backbone: bool) -> None:
    """Switch the model to training mode.

    With a frozen backbone the BatchNorm layers are kept in eval mode,
    otherwise their running statistics would still change during training.

    Args:
        model: Model to switch.
        freeze_backbone: Whether the backbone is frozen.
    """
    model.train()
    if freeze_backbone:
        for module in model.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.eval()
