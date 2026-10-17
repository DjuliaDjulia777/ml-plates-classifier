"""Shared helpers: config loading, seeding and visualization."""

import random
from typing import Any, Dict, Optional

import numpy as np
import torch
import yaml

from plates.transforms import IMAGENET_MEAN, IMAGENET_STD


def load_config(path: str) -> Dict[str, Any]:
    """Load a YAML config file.

    Args:
        path: Path to the YAML file.

    Returns:
        Parsed config as a nested dictionary.
    """
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def set_seed(seed: int = 42) -> None:
    """Fix random seeds for reproducibility.

    Args:
        seed: Seed value for random, numpy and torch.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def imshow(inp: torch.Tensor, title: Optional[str] = None) -> None:
    """Show a tensor normalized with ImageNet statistics.

    Args:
        inp: Image tensor of shape (3, H, W).
        title: Optional plot title.
    """
    import matplotlib.pyplot as plt  # Imported lazily: only needed for plots.

    image = inp.detach().cpu().numpy().transpose((1, 2, 0))
    image = np.array(IMAGENET_STD) * image + np.array(IMAGENET_MEAN)
    image = np.clip(image, 0, 1)
    plt.imshow(image)
    if title is not None:
        plt.title(title)
    plt.axis("off")
