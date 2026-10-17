import pandas as pd
import yaml
from PIL import Image

from plates import predict, train
from plates.predict import decide


def test_decide_follows_process_rules():
    assert decide("cleaned", 0.99, 0.95) == "serve"
    assert decide("cleaned", 0.80, 0.95) == "manual_review"
    assert decide("dirty", 0.99, 0.95) == "rewash"
    assert decide("dirty", 0.40, 0.95) == "rewash"


def test_train_then_predict_smoke(tmp_path, fake_dataset):
    test_dir = tmp_path / "test"
    test_dir.mkdir()
    for i in range(3):
        Image.new("RGB", (32, 32)).save(test_dir / f"img_{i}.jpg")

    cfg = {
        "data": {
            "train_dir": str(fake_dataset),
            "test_dir": str(test_dir),
            "val_split": 0.2,
            "image_size": 64,
        },
        "model": {"num_classes": 2, "freeze_backbone": True, "pretrained": False},
        "training": {
            "batch_size": 4,
            "num_epochs": 1,
            "learning_rate": 0.001,
            "num_workers": 0,
            "seed": 0,
        },
        "inference": {"confidence_threshold": 0.5},
        "paths": {"models_dir": str(tmp_path / "models")},
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")

    best = train.main(str(config_path))
    assert 0.0 <= best["f1"] <= 1.0
    assert (tmp_path / "models" / "best_model.pth").exists()
    assert (tmp_path / "models" / "metrics.json").exists()

    output = tmp_path / "submission.csv"
    details = tmp_path / "details.csv"
    predict.main(
        str(config_path),
        str(tmp_path / "models" / "best_model.pth"),
        str(output),
        str(details),
    )
    assert list(pd.read_csv(output).columns) == ["id", "label"]
    detailed = pd.read_csv(details)
    assert list(detailed.columns) == ["id", "label", "confidence", "decision"]
    assert len(detailed) == 3
