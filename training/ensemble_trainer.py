"""Optimize ensemble weights/thresholds from OOF predictions."""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import json
import numpy as np
import pandas as pd
import optuna
from sklearn.metrics import f1_score


@dataclass
class EnsembleConfig:
    model_order: List[str]
    n_trials: int = 200  # optuna trials per action
    n_jobs: int = 1
    save_path: Optional[Path] = None  # if set, dump JSON with weights/thresholds


def _optimize_action(oof_pred_probs: Dict[str, np.ndarray], y_action: np.ndarray, *, n_trials: int, n_jobs: int):
    """Mirror notebook optuna search: weights in [-1,1], normalized; threshold in [0,1]."""

    def objective(trial: optuna.Trial) -> float:
        weights = np.array([trial.suggest_float(model, -1.0, 1.0) for model in oof_pred_probs.keys()], dtype=np.float64)
        if np.allclose(weights.sum(), 0):
            weights = np.ones_like(weights)
        weights = weights / weights.sum()
        pred_probs = sum(w * oof_pred_probs[m] for w, m in zip(weights, oof_pred_probs.keys()))
        threshold = trial.suggest_float("threshold", 0.0, 1.0)
        return f1_score(y_action, pred_probs >= threshold, zero_division=0)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, n_jobs=n_jobs)

    best_weights = np.array([study.best_params[model] for model in oof_pred_probs.keys()], dtype=np.float64)
    if np.allclose(best_weights.sum(), 0):
        best_weights = np.ones_like(best_weights)
    best_weights = best_weights / best_weights.sum()
    return {"threshold": float(study.best_params["threshold"]), "weight": best_weights.tolist()}


def train_ensemble_weights(
    oof_preds: Dict[str, pd.DataFrame],
    labels: pd.DataFrame,
    *,
    cfg: EnsembleConfig,
) -> Tuple[Dict[str, float], Dict[str, List[float]]]:
    """
    Tune per-action ensemble weights/thresholds from OOF probs of multiple base models.
    oof_preds: dict model_name -> DataFrame of framewise probs (index aligned to labels)
    labels: DataFrame of true labels (NaNs allowed to mask)
    """
    actions = list(labels.columns)
    thresholds: Dict[str, float] = {}
    weights: Dict[str, List[float]] = {}

    model_names = [m for m in cfg.model_order if m in oof_preds]
    if not model_names:
        raise ValueError("No matching models found in oof_preds for provided model_order.")

    for action in actions:
        mask = labels[action].notna().values
        if mask.sum() == 0:
            continue
        y_action = labels[action][mask].astype(int).values
        oof_prob_dict = {m: oof_preds[m][action].values[mask].astype(np.float64) for m in model_names if action in oof_preds[m].columns}
        if len(oof_prob_dict) == 0:
            continue
        try:
            best = _optimize_action(oof_prob_dict, y_action, n_trials=cfg.n_trials, n_jobs=cfg.n_jobs)
            thresholds[action] = best["threshold"]
            weights[action] = best["weight"]
        except Exception:
            thresholds[action] = 0.5
            weights[action] = [1.0 / len(oof_prob_dict)] * len(oof_prob_dict)

    if cfg.save_path:
        cfg.save_path.parent.mkdir(parents=True, exist_ok=True)
        with cfg.save_path.open("w", encoding="utf-8") as f:
            json.dump({"thresholds": thresholds, "weights": weights, "model_order": model_names}, f, indent=2)

    return thresholds, weights
