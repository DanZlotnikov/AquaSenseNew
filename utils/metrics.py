"""Evaluation metrics for the fish prediction model.

Two metric groups:
    presence_metrics  — binary classification (fish present / absent)
    length_metrics    — regression on fish-present samples only

Both functions operate on plain numpy arrays so they can be used independently
of the training framework.
"""

import numpy as np
from sklearn.metrics import precision_recall_fscore_support


def presence_metrics(
    logits: np.ndarray,
    targets: np.ndarray,
    threshold: float = 0.5,
) -> dict:
    """
    Compute classification metrics for the presence head.

    Args:
        logits:    Raw model logits (before sigmoid), shape (N,).
        targets:   Ground-truth binary labels {0, 1}, shape (N,).
        threshold: Probability threshold for declaring fish present.

    Returns:
        Dict with keys: accuracy, precision, recall, f1.
    """
    probs   = 1.0 / (1.0 + np.exp(-logits))        # sigmoid
    preds   = (probs >= threshold).astype(int)
    targets = targets.astype(int)

    accuracy = float((preds == targets).mean())
    precision, recall, f1, _ = precision_recall_fscore_support(
        targets, preds, average="binary", zero_division=0
    )
    return {
        "accuracy":  accuracy,
        "precision": float(precision),
        "recall":    float(recall),
        "f1":        float(f1),
    }


def length_metrics(pred_lengths: np.ndarray, true_lengths: np.ndarray) -> dict:
    """
    Compute regression metrics for the length head.

    Both arrays must already be filtered to fish-present samples
    (true_lengths > 0 is guaranteed after masking, so MAPE is safe).

    Args:
        pred_lengths: Predicted fish lengths in cm, shape (M,).
        true_lengths: Ground-truth fish lengths in cm, shape (M,).

    Returns:
        Dict with keys: mae (cm), rmse (cm), mape (%).
    """
    errors = pred_lengths - true_lengths
    mae    = float(np.abs(errors).mean())
    rmse   = float(np.sqrt((errors ** 2).mean()))
    mape   = float((np.abs(errors) / true_lengths).mean() * 100.0)
    return {"mae": mae, "rmse": rmse, "mape": mape}


def compute_all_metrics(
    logit_presence: np.ndarray,
    pred_length: np.ndarray,
    y_presence: np.ndarray,
    y_length: np.ndarray,
    threshold: float = 0.5,
) -> dict:
    """
    Compute presence and length metrics over a full dataset split.

    Length metrics are computed only for ground-truth fish-present samples.
    If no fish-present samples exist, length metrics are returned as 0.0.

    Args:
        logit_presence : Presence logits, shape (N,).
        pred_length    : Predicted lengths in cm, shape (N,).
        y_presence     : Ground-truth presence labels {0, 1}, shape (N,).
        y_length       : Ground-truth lengths in cm (0 when absent), shape (N,).
        threshold      : Presence classification threshold.

    Returns:
        Combined dict of presence and length metrics.
    """
    pres_m = presence_metrics(logit_presence, y_presence, threshold)

    mask = y_presence.astype(bool)
    if mask.sum() > 0:
        len_m = length_metrics(pred_length[mask], y_length[mask])
    else:
        len_m = {"mae": 0.0, "rmse": 0.0, "mape": 0.0}

    return {**pres_m, **len_m}
