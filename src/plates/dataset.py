"""Datasets for the plates classifier."""

import os
from typing import Any, Callable, List, Optional, Tuple

from PIL import Image
from torch.utils.data import Dataset

IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg")


def list_images(folder: str) -> List[str]:
    """Return sorted paths of all images in a folder.

    Sorting makes the order independent of the file system, so the
    train/val split and the prediction order are reproducible.

    Args:
        folder: Path to a directory with images.

    Returns:
        Sorted list of image file paths.

    Raises:
        FileNotFoundError: If the directory does not exist.
    """
    if not os.path.isdir(folder):
        raise FileNotFoundError(f"Directory not found: {folder}")
    return sorted(
        os.path.join(folder, name)
        for name in os.listdir(folder)
        if name.lower().endswith(IMAGE_EXTENSIONS)
    )


def load_rgb(path: str) -> Image.Image:
    """Open an image and convert it to RGB, closing the file afterwards.

    Args:
        path: Path to the image file.

    Returns:
        RGB image.
    """
    with Image.open(path) as img:
        return img.convert("RGB")


class PlatesDataset(Dataset):
    """Labeled dataset with folders root_dir/cleaned and root_dir/dirty."""

    CLASSES = ["cleaned", "dirty"]

    def __init__(self, root_dir: str, transform: Optional[Callable] = None) -> None:
        """Index all images of both classes.

        Args:
            root_dir: Directory containing one sub-folder per class.
            transform: Optional transform applied to each image.

        Raises:
            FileNotFoundError: If a class folder is missing.
            ValueError: If no images were found.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.class_to_idx = {cls: i for i, cls in enumerate(self.CLASSES)}
        self.samples: List[Tuple[str, int]] = []

        for cls in self.CLASSES:
            for path in list_images(os.path.join(root_dir, cls)):
                self.samples.append((path, self.class_to_idx[cls]))

        if not self.samples:
            raise ValueError(f"No images found in {root_dir}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[Any, int]:
        path, label = self.samples[idx]
        image = load_rgb(path)
        if self.transform is not None:
            image = self.transform(image)
        return image, label


class UnlabeledImageDataset(Dataset):
    """Flat folder of images without labels, used for inference."""

    def __init__(self, root_dir: str, transform: Optional[Callable] = None) -> None:
        """Index all images of the folder.

        Args:
            root_dir: Directory with images.
            transform: Optional transform applied to each image.

        Raises:
            FileNotFoundError: If the directory does not exist.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.images: List[str] = list_images(root_dir)

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> Tuple[Any, str]:
        path = self.images[idx]
        image = load_rgb(path)
        if self.transform is not None:
            image = self.transform(image)
        img_id = os.path.splitext(os.path.basename(path))[0]
        return image, img_id
