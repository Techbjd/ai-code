"""Hyperparameter optimization using Optuna (lazy import)."""

from __future__ import annotations

from typing import Any, Callable


def optimize_gnn(objective: Callable[[Any], float], n_trials: int = 20) -> tuple[dict, float]:
    try:
        import optuna
    except ImportError as e:
        raise ImportError("optuna is required for --hpo: pip install optuna") from e

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials)
    return study.best_params, study.best_value