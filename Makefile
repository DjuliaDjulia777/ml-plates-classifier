.PHONY: install train predict test lint format

install:
	poetry install

train:
	poetry run python -m plates.train --config configs/config.yaml

predict:
	poetry run python -m plates.predict --config configs/config.yaml --ckpt models/best_model.pth --output submission.csv

test:
	poetry run pytest

lint:
	poetry run flake8 src tests
	poetry run mypy src

format:
	poetry run black src tests
	poetry run isort src tests
