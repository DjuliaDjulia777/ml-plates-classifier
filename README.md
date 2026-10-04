# ML Plates Classifier

Binary image classification of clean and dirty plates using transfer learning (ResNet-18).

## Task

Given photos of plates, predict whether each plate is `cleaned` or `dirty`.

## Project structure

- `src/plates/` — source code
  - `dataset.py` — PyTorch Dataset classes
  - `transforms.py` — image transforms and augmentations
  - `model.py` — ResNet-18 model
  - `train.py` — training loop
  - `predict.py` — inference and submission.csv
  - `utils.py` — helpers (seed, imshow)
- `configs/config.yaml` — hyperparameters
- `tests/` — pytest tests
- `data/raw/` — dataset (not committed)
- `models/` — trained weights (not committed)

## Requirements

- Python 3.11+
- Poetry

## Install

    poetry install

## Data

Put the PlatesV2 dataset into `data/raw/plates/` so that the structure looks like:

    data/raw/plates/
      train/
        cleaned/
        dirty/
      test/

## Train

    poetry run python -m plates.train --config configs/config.yaml

The best checkpoint is saved to `models/best_model.pth`.

## Predict

    poetry run python -m plates.predict --config configs/config.yaml --ckpt models/best_model.pth --output submission.csv

## Tests

    poetry run pytest

## Virtual environment

The project uses Poetry. The virtual environment itself is not committed to Git (see .gitignore). Reproducibility is provided by pyproject.toml and poetry.lock. Anyone can recreate the exact environment with `poetry install`.
