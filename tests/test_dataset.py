import pytest
from PIL import Image

from plates.dataset import PlatesDataset, TestDataset


@pytest.fixture
def fake_dataset(tmp_path):
    """Create a small fake dataset with cleaned/ and dirty/ folders."""
    for cls in ["cleaned", "dirty"]:
        d = tmp_path / cls
        d.mkdir()
        for i in range(3):
            Image.new("RGB", (32, 32), color="white").save(d / f"{cls}_{i}.jpg")
    return tmp_path


def test_plates_dataset_len(fake_dataset):
    ds = PlatesDataset(str(fake_dataset))
    assert len(ds) == 6


def test_plates_dataset_item(fake_dataset):
    ds = PlatesDataset(str(fake_dataset))
    img, label = ds[0]
    assert img.size == (32, 32)
    assert label in {0, 1}


def test_test_dataset_len(tmp_path):
    for i in range(4):
        Image.new("RGB", (32, 32)).save(tmp_path / f"img_{i}.jpg")
    ds = TestDataset(str(tmp_path))
    assert len(ds) == 4
