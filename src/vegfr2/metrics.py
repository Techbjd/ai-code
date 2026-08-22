"""Classification metrics for virtual screening evaluation."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from sklearn.metrics import matthews_corrcoef, roc_auc_score


def classification_metrics(
    y_true: Sequence[int], y_prob: Sequence[float], threshold: float = 0.5
) -> dict:
    """Compute ACC/SEN/SPE/MCC/AUC plus a confusion-matrix dict."""
    y_true_arr = np.asarray(list(y_true), dtype=np.int64)
    y_prob_arr = np.asarray(list(y_prob), dtype=np.float64)
    y_pred = (y_prob_arr >= threshold).astype(np.int64)
    tp = int(((y_pred == 1) & (y_true_arr == 1)).sum())
    tn = int(((y_pred == 0) & (y_true_arr == 0)).sum())
    fp = int(((y_pred == 1) & (y_true_arr == 0)).sum())
    fn = int(((y_pred == 0) & (y_true_arr == 1)).sum())
    n = tp + tn + fp + fn
    acc = float((tp + tn) / n) if n > 0 else 0.0
    sen = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    spe = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    try:
        mcc = float(matthews_corrcoef(y_true_arr, y_pred))
    except Exception:
        mcc = 0.0
    auc: float | None
    auc = (
        float(roc_auc_score(y_true_arr, y_prob_arr))
        if len(set(y_true_arr.tolist())) >= 2
        else None
    )
    return {
        'acc': acc,
        'sen': sen,
        'spe': spe,
        'mcc': mcc,
        'auc': auc,
        'confusion_matrix': {'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn},
    }
