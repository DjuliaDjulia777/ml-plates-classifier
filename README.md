# ML Plates Classifier

![CI](https://github.com/DjuliaDjulia777/ml-plates-classifier/actions/workflows/ci.yml/badge.svg)
![Python](https://img.shields.io/badge/python-3.11%20%7C%203.12-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.14-ee4c2c)
![Poetry](https://img.shields.io/badge/managed%20by-poetry-60a5fa)
![Code style](https://img.shields.io/badge/code%20style-black-000000)
![License](https://img.shields.io/badge/license-MIT-green)

Binary image classification of clean and dirty plates using transfer learning (ResNet-18).

## Task

Given photos of plates, predict whether each plate is `cleaned` or `dirty`.

## Documentation

- [model_card.md](model_card.md) — Model Card (Google template): purpose, data, metrics, limitations.
- [Model_BPMN.md](Model_BPMN.md) — place of the model in the dishwashing quality-control process.
- [Model_retrain_BPMN.md](Model_retrain_BPMN.md) — automatic retraining process, from trigger to deployment.

## Project structure

- `src/plates/` — source code
  * `dataset.py` — PyTorch Dataset classes (`PlatesDataset`, `UnlabeledImageDataset`)
  * `transforms.py` — image transforms and augmentations
  * `model.py` — ResNet-18 model, frozen-BatchNorm helper
  * `metrics.py` — accuracy, precision, recall, F1
  * `train.py` — training loop, saves best checkpoint and `metrics.json`
  * `predict.py` — inference: label, confidence, decision
  * `utils.py` — config loading, seed, imshow
- `configs/config.yaml` — hyperparameters and inference threshold
- `tests/` — pytest tests
- `notebooks/` — original notebook
- `Makefile` — shortcuts (Linux, macOS, Git Bash)
- `data/raw/` — dataset (not committed)
- `models/` — trained weights and metrics (not committed)

## Requirements

- Python 3.11 or 3.12
- Poetry

## Install

```
poetry install
```

## Data

Put the PlatesV2 dataset into `data/raw/plates/` so that the structure looks like:

```
data/raw/plates/
  train/
    cleaned/
    dirty/
  test/
```

Run all commands from the repository root: paths in `configs/config.yaml` are relative.

## Train

```
poetry run python -m plates.train --config configs/config.yaml
```

The best checkpoint (by validation F1 for the `dirty` class) is saved to `models/best_model.pth`, its metrics to `models/metrics.json`.

## Predict

```
poetry run python -m plates.predict --config configs/config.yaml --ckpt models/best_model.pth --output submission.csv --details predictions_detailed.csv
```

`submission.csv` contains `id,label`. The optional `--details` file also contains `confidence` and `decision` (`serve`, `manual_review`, `rewash`), see [Model_BPMN.md](Model_BPMN.md). The confidence threshold is set in `configs/config.yaml`.

## Results

Validation (8 images, best epoch 3): accuracy 0.625, precision 0.800, recall 0.667, F1 0.727 for class `dirty`. The dataset is very small, so the numbers are noisy: this is an educational prototype. Details and limitations: [model_card.md](model_card.md).

## Tests

```
poetry run pytest
```

28 tests: datasets, transforms, model, metrics, train/val split, and a train-then-predict smoke test.

## Code quality

```
poetry run pre-commit run --all-files
```

Runs trailing-whitespace, end-of-file-fixer, check-yaml, check-added-large-files, and black, isort, flake8, mypy (tool versions come from `poetry.lock`). CI (GitHub Actions) runs the same checks and the tests on Python 3.11 and 3.12.

## Virtual environment

The project uses Poetry. The virtual environment itself is not committed to Git (see `.gitignore`). Reproducibility is provided by `pyproject.toml` and `poetry.lock`. Anyone can recreate the exact environment with `poetry install`.

## License

MIT, see [LICENSE](LICENSE).
