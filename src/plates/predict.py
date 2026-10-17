"""Inference script: python -m plates.predict --ckpt models/best_model.pth."""

import argparse
from typing import Any, Dict, List, Optional

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from plates.dataset import PlatesDataset, UnlabeledImageDataset
from plates.model import get_model
from plates.transforms import get_val_transform
from plates.utils import load_config


def decide(label: str, confidence: float, threshold: float) -> str:
    """Map a prediction to an action of the dishwashing quality-control process.

    Args:
        label: Predicted class name ("cleaned" or "dirty").
        confidence: Probability of the predicted class.
        threshold: Minimal confidence to trust a "cleaned" prediction.

    Returns:
        "serve" for a confidently clean plate, "manual_review" for a
        clean plate with low confidence, "rewash" for a dirty plate.
    """
    if label == "cleaned":
        return "serve" if confidence >= threshold else "manual_review"
    return "rewash"


@torch.no_grad()
def predict(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    class_names: List[str],
    threshold: float,
) -> pd.DataFrame:
    """Run inference and collect labels, confidences and decisions.

    Args:
        model: Trained model.
        loader: Dataloader with (inputs, ids) batches.
        device: Device for computation.
        class_names: Class names indexed by class id.
        threshold: Confidence threshold for the "serve" decision.

    Returns:
        DataFrame with columns id, label, confidence and decision.
    """
    model.eval()
    ids: List[str] = []
    labels: List[str] = []
    confidences: List[float] = []
    for inputs, batch_ids in tqdm(loader, desc="Predict"):
        probs = torch.softmax(model(inputs.to(device)), dim=1)
        conf, pred = torch.max(probs, dim=1)
        ids.extend(batch_ids)
        labels.extend(class_names[i] for i in pred.cpu().tolist())
        confidences.extend(conf.cpu().tolist())
    decisions = [decide(lb, c, threshold) for lb, c in zip(labels, confidences)]
    return pd.DataFrame(
        {"id": ids, "label": labels, "confidence": confidences, "decision": decisions}
    )


def main(
    config_path: str,
    ckpt_path: str,
    output_path: str,
    details_path: Optional[str] = None,
) -> pd.DataFrame:
    """Load a checkpoint and save predictions to CSV.

    Args:
        config_path: Path to the YAML config.
        ckpt_path: Path to the model checkpoint.
        output_path: Where to save the id,label CSV.
        details_path: Optional path for the CSV with confidence and decision.

    Returns:
        The full predictions table.
    """
    cfg: Dict[str, Any] = load_config(config_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = UnlabeledImageDataset(
        cfg["data"]["test_dir"], get_val_transform(cfg["data"]["image_size"])
    )
    loader = DataLoader(
        dataset,
        batch_size=cfg["training"]["batch_size"],
        shuffle=False,
        num_workers=cfg["training"]["num_workers"],
    )

    model = get_model(num_classes=cfg["model"]["num_classes"], pretrained=False)
    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
    model.to(device)

    df = predict(
        model,
        loader,
        device,
        PlatesDataset.CLASSES,
        cfg["inference"]["confidence_threshold"],
    )
    df[["id", "label"]].to_csv(output_path, index=False)
    print(f"Saved {len(df)} predictions to {output_path}")
    if details_path:
        df.to_csv(details_path, index=False)
        print(f"Saved detailed predictions to {details_path}")
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict plate cleanliness.")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--ckpt", default="models/best_model.pth")
    parser.add_argument("--output", default="submission.csv")
    parser.add_argument(
        "--details",
        default=None,
        help="Optional CSV with confidence and decision columns.",
    )
    args = parser.parse_args()
    main(args.config, args.ckpt, args.output, args.details)
