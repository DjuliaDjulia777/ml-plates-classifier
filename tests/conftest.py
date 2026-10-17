import pytest
from PIL import Image


@pytest.fixture
def fake_dataset(tmp_path):
    """Create a labeled dataset with 5 images in cleaned/ and dirty/."""
    root = tmp_path / "train"
    for cls, color in [("cleaned", "white"), ("dirty", "brown")]:
        folder = root / cls
        folder.mkdir(parents=True)
        for i in range(5):
            Image.new("RGB", (32, 32), color=color).save(folder / f"{cls}_{i}.jpg")
    return root
