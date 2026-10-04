import torch

from plates.model import get_model


def test_model_output_shape():
    model = get_model(num_classes=2)
    model.eval()
    dummy_input = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        output = model(dummy_input)
    assert output.shape == (1, 2)


def test_model_freeze_backbone():
    model = get_model(num_classes=2, freeze_backbone=True)
    for name, param in model.named_parameters():
        if not name.startswith("fc."):
            assert not param.requires_grad, f"{name} should be frozen"


def test_model_unfreeze_backbone():
    model = get_model(num_classes=2, freeze_backbone=False)
    trainable = [p for p in model.parameters() if p.requires_grad]
    assert len(trainable) > 2


def test_model_custom_num_classes():
    model = get_model(num_classes=5)
    assert model.fc.out_features == 5
