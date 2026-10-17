"""Classification metrics without extra dependencies."""

from typing import Dict, Sequence


def classification_metrics(
    preds: Sequence[int],
    labels: Sequence[int],
    positive_class: int = 1,
) -> Dict[str, float]:
    """Compute accuracy, precision, recall and F1 for a binary task.

    Args:
        preds: Predicted class indices.
        labels: True class indices.
        positive_class: Index of the class treated as positive.

    Returns:
        Dictionary with keys accuracy, precision, recall and f1.
        Metrics with an empty denominator are 0.0.

    Raises:
        ValueError: If the inputs are empty or have different lengths.
    """
    if len(preds) != len(labels) or len(labels) == 0:
        raise ValueError("preds and labels must be non-empty and equally long")

    tp = fp = fn = correct = 0
    for pred, label in zip(preds, labels):
        correct += int(pred == label)
        if pred == positive_class and label == positive_class:
            tp += 1
        elif pred == positive_class:
            fp += 1
        elif label == positive_class:
            fn += 1

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "accuracy": correct / len(labels),
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }
