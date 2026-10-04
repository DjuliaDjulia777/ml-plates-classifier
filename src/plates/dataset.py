import os
from typing import Callable, List, Optional, Tuple

from PIL import Image
from torch.utils.data import Dataset


class PlatesDataset(Dataset):
    """Dataset for train/val with folder structure root_dir/cleaned, root_dir/dirty."""

    CLASSES = ["cleaned", "dirty"]

    def __init__(self, root_dir: str, transform: Optional[Callable] = None) -> None:
        self.root_dir = root_dir
        self.transform = transform
        self.class_to_idx = {cls: i for i, cls in enumerate(self.CLASSES)}
        self.samples: List[Tuple[str, int]] = []

        for cls in self.CLASSES:
            cls_dir = os.path.join(root_dir, cls)
            if not os.path.isdir(cls_dir):
                continue
            for fname in os.listdir(cls_dir):
                if fname.lower().endswith((".png", ".jpg", ".jpeg")):
                    self.samples.append(
                        (os.path.join(cls_dir, fname), self.class_to_idx[cls])
                    )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        image = Image.open(path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, label


class TestDataset(Dataset):
    """Dataset for test with a flat folder of images without labels."""

    def __init__(self, root_dir: str, transform: Optional[Callable] = None) -> None:
        self.root_dir = root_dir
        self.transform = transform
        self.images: List[str] = [
            os.path.join(root_dir, fname)
            for fname in os.listdir(root_dir)
            if fname.lower().endswith((".png", ".jpg", ".jpeg"))
        ]

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int):
        path = self.images[idx]
        image = Image.open(path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        img_id = os.path.splitext(os.path.basename(path))[0]
        return image, img_id
