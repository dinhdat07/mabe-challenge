"""Cross-validation and threshold tuning for XGBoost models."""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple
import gc
import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedGroupKFold
from koolbox import Trainer


@dataclass
class XGBConfig:
    model: object
    n_splits: int = 3
    save_dir: Path = Path("models")


def compute_scale_pos_weight(y: np.ndarray) -> float:
    n_pos = max(1, int((y == 1).sum()))
    n_neg = max(1, int((y == 0).sum()))
    return float(n_neg / n_pos)


def tune_threshold(oof_pred: np.ndarray, y_true: np.ndarray, step: float = 0.01) -> float:
    best_thr = 0.5
    best_f1 = -1
    for thr in np.arange(0, 1 + step, step):
        f1 = f1_score(y_true, (oof_pred >= thr).astype(int), zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_thr = thr
    return float(best_thr)


def cross_validate_classifier(
    X: pd.DataFrame,
    label: pd.DataFrame,
    meta: pd.DataFrame,
    *,
    cfg: XGBConfig,
) -> Tuple[List[pd.DataFrame], List[tuple], Dict[str, float]]:
    """Lightweight extraction of the notebook CV loop (per-action binary training)."""
    oof = pd.DataFrame(index=meta.index)
    f1_list: List[tuple] = []
    submission_list: List[pd.DataFrame] = []
    thresholds: Dict[str, float] = {}

    cv = StratifiedGroupKFold(cfg.n_splits)

    for action in label.columns:
        action_mask = ~label[action].isna().values
        y_action = label[action][action_mask].values.astype(int)
        X_action = X[action_mask]
        groups_action = meta.video_id[action_mask]

        if len(np.unique(groups_action)) < cfg.n_splits:
            continue
        if (y_action == 0).all():
            oof_action = np.zeros(len(y_action), dtype=float)
        else:
            estimator = clone(cfg.model)
            estimator.set_params(scale_pos_weight=compute_scale_pos_weight(y_action))
            trainer = Trainer(
                estimator=estimator,
                cv=cv,
                cv_args={"groups": groups_action},
                metric=f1_score,
                task="binary",
                verbose=False,
                save=True,
                save_path=str(cfg.save_dir / action),
                use_early_stopping=True,
            )
            trainer.fit(
                X_action,
                y_action,
                fit_args={"early_stopping_rounds": 50, "eval_metric": "aucpr", "verbose": False},
            )
            oof_action = trainer.oof_preds
            thresholds[action] = tune_threshold(oof_action, y_action)
            f1_list.append((action, f1_score(y_action, (oof_action >= thresholds[action]).astype(int), zero_division=0)))
            del trainer
            gc.collect()

        oof_column = np.zeros(len(label), dtype=float)
        oof_column[action_mask] = oof_action
        oof[action] = oof_column

    submission_list.append(oof)
    return submission_list, f1_list, thresholds
