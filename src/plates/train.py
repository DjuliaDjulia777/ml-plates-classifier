import argparse
import os

import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from torch.utils.data import DataLoader, Subset, random_split
from tqdm import tqdm

from plates.dataset import PlatesDataset
from plates.model import get_model
from plates.transforms import get_train_transform, get_val_transform
from plates.utils import set_seed


def load_config(path: str) -> dict:
    """Load YAML config."""
    with open(path, "r") as f:
        return yaml.safe_load(f)


def build_dataloaders(cfg: dict):
    """Build train and val dataloaders using the same split indices."""
    total = len(PlatesDataset(cfg["data"]["train_dir"]))
    val_size = int(cfg["data"]["val_split"] * total)
    train_size = total - val_size

    generator = torch.Generator().manual_seed(cfg["training"]["seed"])
    train_subset, val_subset = random_split(
        range(total), [train_size, val_size], generator=generator
    )

    train_dataset = Subset(
        PlatesDataset(cfg["data"]["train_dir"], transform=get_train_transform()),
        train_subset.indices,
    )
    val_dataset = Subset(
        PlatesDataset(cfg["data"]["train_dir"], transform=get_val_transform()),
        val_subset.indices,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg["training"]["batch_size"],
        shuffle=True,
        num_workers=cfg["training"]["num_workers"],
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg["training"]["batch_size"],
        shuffle=False,
        num_workers=cfg["training"]["num_workers"],
    )
    return train_loader, val_loader


def train_one_epoch(model, loader, criterion, optimizer, device):
    """Run one training epoch and return loss and accuracy."""
    model.train()
    running_loss, running_correct = 0.0, 0
    for inputs, labels in tqdm(loader, desc="Train", leave=False):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        _, preds = torch.max(outputs, 1)
        running_loss += loss.item() * inputs.size(0)
        running_correct += (preds == labels).sum().item()

    n = len(loader.dataset)
    return running_loss / n, running_correct / n


@torch.no_grad()
def validate(model, loader, criterion, device):
    """Run validation and return loss and accuracy."""
    model.eval()
    running_loss, running_correct = 0.0, 0
    for inputs, labels in tqdm(loader, desc="Val", leave=False):
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        _, preds = torch.max(outputs, 1)
        running_loss += loss.item() * inputs.size(0)
        running_correct += (preds == labels).sum().item()

    n = len(loader.dataset)
    return running_loss / n, running_correct / n


def main(config_path: str) -> None:
    """Train the model from the config."""
    cfg = load_config(config_path)
    set_seed(cfg["training"]["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train_loader, val_loader = build_dataloaders(cfg)

    model = get_model(num_classes=cfg["model"]["num_classes"]).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg["training"]["learning_rate"],
    )

    best_val_acc = 0.0
    os.makedirs(cfg["paths"]["models_dir"], exist_ok=True)
    ckpt_path = os.path.join(cfg["paths"]["models_dir"], "best_model.pth")

    for epoch in range(cfg["training"]["num_epochs"]):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        print(
            f"Epoch {epoch + 1:02d}/{cfg['training']['num_epochs']} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), ckpt_path)
            print(f"  -> saved best model (val_acc={best_val_acc:.4f})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()
    main(args.config)
