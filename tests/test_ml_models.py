"""Tests for vegfr2.ml_models (RF, SVM, XGBoost)."""

from __future__ import annotations

import numpy as np
import pytest

from vegfr2.ml_models import load_ml_model, predict_ml_model, save_ml_model, train_ml_model


@pytest.fixture
def synthetic_data():
    rng = np.random.RandomState(42)
    X = rng.randn(60, 64)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    return X[:40], y[:40], X[40:], y[40:]


@pytest.mark.parametrize("name", ["rf", "svm", "xgb"])
def test_ml_train_predict_range(name, synthetic_data):
    X_train, y_train, X_test, y_test = synthetic_data
    est = train_ml_model(name, X_train, y_train, seed=42)
    probs = predict_ml_model(est, X_test)

    assert probs.shape == (20,)
    assert (probs >= 0.0).all() and (probs <= 1.0).all()


@pytest.mark.parametrize("name", ["rf", "svm", "xgb"])
def test_ml_save_load_roundtrip(name, synthetic_data, tmp_path):
    X_train, y_train, X_test, _ = synthetic_data
    est = train_ml_model(name, X_train, y_train, seed=42)
    orig_probs = predict_ml_model(est, X_test)

    model_path = tmp_path / f"{name}.pkl"
    save_ml_model(est, model_path)
    loaded = load_ml_model(model_path)
    new_probs = predict_ml_model(loaded, X_test)

    assert np.allclose(orig_probs, new_probs)


def test_unknown_ml_raises():
    with pytest.raises(ValueError):
        train_ml_model("unknown", np.zeros((10, 5)), np.zeros(10))
