"""Tests for vegfr2.metrics module."""

from __future__ import annotations

import math
import pytest

from vegfr2.metrics import classification_metrics


def test_metrics_perfect_predictions():
    y_true = [1, 1, 0, 0]
    y_prob = [0.95, 0.85, 0.10, 0.05]
    m = classification_metrics(y_true, y_prob, threshold=0.5)

    assert m["acc"] == 1.0
    assert m["sen"] == 1.0
    assert m["spe"] == 1.0
    assert m["mcc"] == 1.0
    assert m["auc"] == 1.0
    assert m["confusion_matrix"] == {"tp": 2, "tn": 2, "fp": 0, "fn": 0}


def test_metrics_hand_computed_case():
    # 4 active, 3 inactive = 7 total
    # tp=3, fn=1 (active=4)
    # tn=2, fp=1 (inactive=3)
    y_true = [1, 1, 1, 1, 0, 0, 0]
    y_prob = [0.9, 0.8, 0.7, 0.2, 0.6, 0.1, 0.3]
    m = classification_metrics(y_true, y_prob, threshold=0.5)

    assert m["confusion_matrix"]["tp"] == 3
    assert m["confusion_matrix"]["fn"] == 1
    assert m["confusion_matrix"]["tn"] == 2
    assert m["confusion_matrix"]["fp"] == 1

    assert m["acc"] == pytest.approx(5 / 7, abs=1e-3)
    assert m["sen"] == pytest.approx(3 / 4, abs=1e-3)
    assert m["spe"] == pytest.approx(2 / 3, abs=1e-3)
    # MCC = (3*2 - 1*1) / sqrt((3+1)(3+1)(2+1)(2+1)) = 5 / sqrt(16 * 9) = 5 / 12 = 0.41666...
    assert m["mcc"] == pytest.approx(5 / 12, abs=1e-3)


def test_metrics_single_class_auc_none():
    """When only one class is present in y_true, AUC cannot be computed and must be None."""
    y_true = [1, 1, 1]
    y_prob = [0.9, 0.8, 0.7]
    m = classification_metrics(y_true, y_prob)
    assert m["auc"] is None
    assert m["acc"] == 1.0
