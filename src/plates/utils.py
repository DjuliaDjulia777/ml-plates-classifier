import random
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import torch

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD = np.array([0.229, 0.224, 0.225])


def set_seed(seed: int = 42) -> None:
    """Fix random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def imshow(inp: torch.Tensor, title: Optional[str] = None) -> None:
    """Show a normalized ImageNet tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    inp = IMAGENET_STD * inp + IMAGENET_MEAN
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.axis("off")
