"""Classical ML models (RF, SVM, XGBoost) for fingerprint-based VEGFR2 activity prediction."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier


def train_ml_model(
    name: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray | None = None,
    y_val: np.ndarray | None = None,
    seed: int = 42,
) -> Any:
    name = name.lower()
    if name == "rf":
        est = RandomForestClassifier(n_estimators=300, random_state=seed, n_jobs=-1)
    elif name == "svm":
        est = SVC(probability=True, kernel="rbf", C=10, gamma="scale", random_state=seed)
    elif name == "xgb":
        est = XGBClassifier(
            n_estimators=400,
            max_depth=6,
            learning_rate=0.1,
            tree_method="hist",
            eval_metric="logloss",
            n_jobs=-1,
            random_state=seed,
        )
    else:
        raise ValueError(f"Unknown ML model: {name}. Available: rf, svm, xgb")
    est.fit(X_train, y_train)
    return est


def predict_ml_model(estimator: Any, X: np.ndarray) -> np.ndarray:
    return estimator.predict_proba(X)[:, 1]


def save_ml_model(estimator: Any, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump(estimator, f)
    return path


def load_ml_model(path: str | Path) -> Any:
    with Path(path).open("rb") as f:
        return pickle.load(f)