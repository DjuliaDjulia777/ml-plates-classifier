import pytest

from plates.metrics import classification_metrics


def test_metrics_known_example():
    m = classification_metrics([1, 1, 0, 0], [1, 0, 0, 1], positive_class=1)
    assert m == {"accuracy": 0.5, "precision": 0.5, "recall": 0.5, "f1": 0.5}


def test_metrics_perfect():
    m = classification_metrics([0, 1, 1, 0], [0, 1, 1, 0])
    assert m["accuracy"] == m["f1"] == 1.0


def test_metrics_no_positive_predictions():
    m = classification_metrics([0, 0], [1, 0])
    assert m["precision"] == 0.0 and m["f1"] == 0.0


def test_metrics_rejects_bad_input():
    with pytest.raises(ValueError):
        classification_metrics([], [])
    with pytest.raises(ValueError):
        classification_metrics([1], [1, 0])
