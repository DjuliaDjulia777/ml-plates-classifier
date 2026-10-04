import argparse

import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from plates.dataset import PlatesDataset, TestDataset
from plates.model import get_model
from plates.transforms import get_val_transform


def load_config(path: str) -> dict:
    """Load YAML config."""
    with open(path, "r") as f:
        return yaml.safe_load(f)


@torch.no_grad()
def predict(model, loader, device, class_names):
    """Run inference and return a DataFrame with ids and labels."""
    model.eval()
    ids, preds = [], []
    for inputs, batch_ids in tqdm(loader, desc="Predict"):
        inputs = inputs.to(device)
        outputs = model(inputs)
        _, pred = torch.max(outputs, 1)
        ids.extend(batch_ids)
        preds.extend([class_names[p] for p in pred.cpu().numpy()])
    return pd.DataFrame({"id": ids, "label": preds})


def main(config_path: str, ckpt_path: str, output_path: str) -> None:
    """Load checkpoint and save predictions to CSV."""
    cfg = load_config(config_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    test_dataset = TestDataset(cfg["data"]["test_dir"], transform=get_val_transform())
    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg["training"]["batch_size"],
        shuffle=False,
        num_workers=cfg["training"]["num_workers"],
    )

    model = get_model(num_classes=cfg["model"]["num_classes"]).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))

    df = predict(model, test_loader, device, PlatesDataset.CLASSES)
    df.to_csv(output_path, index=False)
    print(f"Saved {len(df)} predictions to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--ckpt", default="models/best_model.pth")
    parser.add_argument("--output", default="submission.csv")
    args = parser.parse_args()
    main(args.config, args.ckpt, args.output)
