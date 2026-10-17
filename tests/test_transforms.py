import torch
from PIL import Image

from plates.transforms import get_train_transform, get_val_transform


def test_train_transform_output_shape():
    tensor = get_train_transform(image_size=224)(Image.new("RGB", (300, 400)))
    assert tensor.shape == (3, 224, 224)
    assert tensor.dtype == torch.float32


def test_val_transform_output_shape():
    tensor = get_val_transform(image_size=128)(Image.new("RGB", (300, 400)))
    assert tensor.shape == (3, 128, 128)
    assert tensor.dtype == torch.float32


def test_val_transform_is_deterministic():
    transform = get_val_transform(image_size=64)
    image = Image.new("RGB", (100, 100), color="gray")
    assert torch.equal(transform(image), transform(image))


def test_train_transform_changes_image():
    torch.manual_seed(0)
    transform = get_train_transform(image_size=64)
    image = Image.new("RGB", (100, 100), color="white")
    outputs = [transform(image) for _ in range(5)]
    assert not all(torch.equal(outputs[0], t) for t in outputs[1:])
