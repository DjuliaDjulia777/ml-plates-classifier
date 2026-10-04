import torch
from PIL import Image

from plates.transforms import get_train_transform, get_val_transform


def test_train_transform_output_shape():
    transform = get_train_transform(image_size=224)
    image = Image.new("RGB", (300, 400), color="white")
    tensor = transform(image)
    assert tensor.shape == (3, 224, 224)
    assert tensor.dtype == torch.float32


def test_val_transform_output_shape():
    transform = get_val_transform(image_size=128)
    image = Image.new("RGB", (300, 400), color="white")
    tensor = transform(image)
    assert tensor.shape == (3, 128, 128)
    assert tensor.dtype == torch.float32


def test_val_transform_is_deterministic():
    transform = get_val_transform(image_size=64)
    image = Image.new("RGB", (100, 100), color="gray")
    t1 = transform(image)
    t2 = transform(image)
    assert torch.equal(t1, t2)


def test_train_transform_changes_image():
    transform = get_train_transform(image_size=64)
    image = Image.new("RGB", (100, 100), color="white")
    outputs = [transform(image) for _ in range(5)]
    same = all(torch.equal(outputs[0], t) for t in outputs[1:])
    assert not same
