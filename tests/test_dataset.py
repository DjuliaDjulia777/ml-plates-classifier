import pytest
from PIL import Image

from plates.dataset import PlatesDataset, UnlabeledImageDataset
from plates.transforms import get_val_transform


def test_plates_dataset_len(fake_dataset):
    assert len(PlatesDataset(str(fake_dataset))) == 10


def test_plates_dataset_labels(fake_dataset):
    ds = PlatesDataset(str(fake_dataset))
    for path, label in ds.samples:
        expected = 0 if "cleaned" in path else 1
        assert label == expected


def test_plates_dataset_item(fake_dataset):
    img, label = PlatesDataset(str(fake_dataset))[0]
    assert img.size == (32, 32)
    assert label == 0  # sorted order: cleaned comes first


def test_plates_dataset_applies_transform(fake_dataset):
    ds = PlatesDataset(str(fake_dataset), transform=get_val_transform(16))
    img, _ = ds[0]
    assert tuple(img.shape) == (3, 16, 16)


def test_plates_dataset_is_sorted(fake_dataset):
    paths = [p for p, _ in PlatesDataset(str(fake_dataset)).samples]
    cleaned = [p for p in paths if "cleaned" in p]
    assert cleaned == sorted(cleaned)


def test_plates_dataset_ignores_non_images(fake_dataset):
    (fake_dataset / "cleaned" / "notes.txt").write_text("not an image")
    assert len(PlatesDataset(str(fake_dataset))) == 10


def test_plates_dataset_missing_class_folder(tmp_path):
    with pytest.raises(FileNotFoundError):
        PlatesDataset(str(tmp_path))


def test_unlabeled_dataset_len_and_ids(tmp_path):
    for name in ["b", "a", "c"]:
        Image.new("RGB", (32, 32)).save(tmp_path / f"{name}.jpg")
    ds = UnlabeledImageDataset(str(tmp_path))
    assert len(ds) == 3
    assert [ds[i][1] for i in range(3)] == ["a", "b", "c"]
