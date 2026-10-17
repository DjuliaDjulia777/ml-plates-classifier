"""Training script: python -m plates.train --config configs/config.yaml."""

import argparse
import json
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from plates.dataset import PlatesDataset
from plates.metrics import classification_metrics
from plates.model import get_model, set_train_mode
from plates.transforms import get_train_transform, get_val_transform
from plates.utils import load_config, set_seed

POSITIVE_CLASS = 1  # Index of "dirty": the class we want to detect.


def split_indices(
    total: int, val_split: float, seed: int
) -> Tuple[List[int], List[int]]:
    """Split sample indices into disjoint train and validation parts.

    Args:
        total: Number of samples.
        val_split: Fraction of samples used for validation.
        seed: Seed of the random generator.

    Returns:
        Tuple (train_indices, val_indices).

    Raises:
        ValueError: If either part would be empty.
    """
    val_size = int(val_split * total)
    train_size = total - val_size
    if val_size == 0 or train_size == 0:
        raise ValueError(f"Cannot split {total} samples with val_split={val_split}")
    generator = torch.Generator().manual_seed(seed)
    permutation = torch.randperm(total, generator=generator).tolist()
    return permutation[:train_size], permutation[train_size:]


def build_dataloaders(cfg: Dict[str, Any]) -> Tuple[DataLoader, DataLoader]:
    """Build train and validation dataloaders from one shared split.

    Args:
        cfg: Parsed config.

    Returns:
        Tuple (train_loader, val_loader).
    """
    data_cfg, train_cfg = cfg["data"], cfg["training"]
    size = data_cfg["image_size"]

    train_full = PlatesDataset(data_cfg["train_dir"], get_train_transform(size))
    val_full = PlatesDataset(data_cfg["train_dir"], get_val_transform(size))
    train_idx, val_idx = split_indices(
        len(train_full), data_cfg["val_split"], train_cfg["seed"]
    )

    train_loader = DataLoader(
        Subset(train_full, train_idx),
        batch_size=train_cfg["batch_size"],
        shuffle=True,
        num_workers=train_cfg["num_workers"],
    )
    val_loader = DataLoader(
        Subset(val_full, val_idx),
        batch_size=train_cfg["batch_size"],
        shuffle=False,
        num_workers=train_cfg["num_workers"],
    )
    return train_loader, val_loader


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: Optional[optim.Optimizer] = None,
    freeze_backbone: bool = False,
) -> Dict[str, float]:
    """Run one training epoch (if optimizer is given) or one validation pass.

    Args:
        model: Model to run.
        loader: Dataloader with (inputs, labels) batches.
        criterion: Loss function.
        device: Device for computation.
        optimizer: Optimizer. If None, the model is only evaluated.
        freeze_backbone: Whether the backbone is frozen (affects BatchNorm).

    Returns:
        Dictionary with loss, accuracy, precision, recall and f1.
    """
    training = optimizer is not None
    if training:
        set_train_mode(model, freeze_backbone)
    else:
        model.eval()

    total_loss = 0.0
    all_preds: List[int] = []
    all_labels: List[int] = []
    desc = "Train" if training else "Val"

    with torch.set_grad_enabled(training):
        for inputs, labels in tqdm(loader, desc=desc, leave=False):
            inputs, labels = inputs.to(device), labels.to(device)
            if optimizer is not None:
                optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            if optimizer is not None:
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * inputs.size(0)
            all_preds.extend(outputs.argmax(dim=1).cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    metrics = classification_metrics(all_preds, all_labels, POSITIVE_CLASS)
    metrics["loss"] = total_loss / len(all_labels)
    return metrics


def main(config_path: str) -> Dict[str, Any]:
    """Train the model and save the best checkpoint and its metrics.

    The best epoch is selected by validation F1 for the "dirty" class.

    Args:
        config_path: Path to the YAML config.

    Returns:
        Validation metrics of the best epoch.
    """
    cfg = load_config(config_path)
    set_seed(cfg["training"]["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train_loader, val_loader = build_dataloaders(cfg)

    freeze = cfg["model"]["freeze_backbone"]
    model = get_model(
        num_classes=cfg["model"]["num_classes"],
        freeze_backbone=freeze,
        pretrained=cfg["model"]["pretrained"],
    ).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=cfg["training"]["learning_rate"],
    )

    models_dir = cfg["paths"]["models_dir"]
    os.makedirs(models_dir, exist_ok=True)
    ckpt_path = os.path.join(models_dir, "best_model.pth")
    metrics_path = os.path.join(models_dir, "metrics.json")

    num_epochs = cfg["training"]["num_epochs"]
    best_f1 = -1.0
    best_metrics: Dict[str, Any] = {}

    for epoch in range(1, num_epochs + 1):
        train_m = run_epoch(model, train_loader, criterion, device, optimizer, freeze)
        val_m = run_epoch(model, val_loader, criterion, device)

        print(
            f"Epoch {epoch:02d}/{num_epochs} | "
            f"train_loss={train_m['loss']:.4f} train_acc={train_m['accuracy']:.4f} | "
            f"val_loss={val_m['loss']:.4f} val_acc={val_m['accuracy']:.4f} "
            f"val_f1={val_m['f1']:.4f}"
        )

        if val_m["f1"] > best_f1:
            best_f1 = val_m["f1"]
            best_metrics = {"epoch": epoch, **val_m}
            torch.save(model.state_dict(), ckpt_path)
            print(f"  -> saved best model (val_f1={best_f1:.4f})")

    report = {
        "selection_metric": "val_f1",
        "positive_class": PlatesDataset.CLASSES[POSITIVE_CLASS],
        "best": best_metrics,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "config": cfg,
    }
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"Saved metrics to {metrics_path}")
    return best_metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the plates classifier.")
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()
    main(args.config)
