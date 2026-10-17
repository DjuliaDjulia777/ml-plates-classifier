import torch
import torch.nn as nn

from plates.model import get_model, set_train_mode


def test_model_output_shape():
    model = get_model(num_classes=2, pretrained=False)
    model.eval()
    with torch.no_grad():
        output = model(torch.randn(1, 3, 224, 224))
    assert output.shape == (1, 2)


def test_model_freeze_backbone():
    model = get_model(num_classes=2, freeze_backbone=True, pretrained=False)
    for name, param in model.named_parameters():
        if name.startswith("fc."):
            assert param.requires_grad, f"{name} should be trainable"
        else:
            assert not param.requires_grad, f"{name} should be frozen"


def test_model_unfreeze_backbone():
    model = get_model(num_classes=2, freeze_backbone=False, pretrained=False)
    assert all(p.requires_grad for p in model.parameters())


def test_model_custom_num_classes():
    model = get_model(num_classes=5, pretrained=False)
    assert model.fc.out_features == 5


def test_set_train_mode_keeps_batchnorm_frozen():
    model = get_model(freeze_backbone=True, pretrained=False)
    set_train_mode(model, freeze_backbone=True)
    assert model.training
    batch_norms = [m for m in model.modules() if isinstance(m, nn.BatchNorm2d)]
    assert batch_norms and all(not m.training for m in batch_norms)


def test_set_train_mode_unfrozen_trains_batchnorm():
    model = get_model(freeze_backbone=False, pretrained=False)
    set_train_mode(model, freeze_backbone=False)
    batch_norms = [m for m in model.modules() if isinstance(m, nn.BatchNorm2d)]
    assert all(m.training for m in batch_norms)
