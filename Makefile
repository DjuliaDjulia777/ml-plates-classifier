.PHONY: install train predict test lint format check pre-commit clean

install:
	poetry install

train:
	poetry run python -m plates.train --config configs/config.yaml

predict:
	poetry run python -m plates.predict --config configs/config.yaml --ckpt models/best_model.pth --output submission.csv --details predictions_detailed.csv

test:
	poetry run pytest

lint:
	poetry run black --check src tests
	poetry run isort --check-only src tests
	poetry run flake8 src tests
	poetry run mypy src

format:
	poetry run black src tests
	poetry run isort src tests

check: lint test

pre-commit:
	poetry run pre-commit run --all-files

clean:
	rm -rf .pytest_cache .mypy_cache
