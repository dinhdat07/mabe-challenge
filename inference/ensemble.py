"""Inference for weighted ensemble of multiple XGB models."""

from pathlib import Path
from typing import Dict, List
import glob
import gc
import joblib
import numpy as np
import pandas as pd

from mabe.data.generator import generate_mouse_data
from mabe.features.single import transform_single
from mabe.features.pair import transform_pair
from mabe.features.utils import fps_from_meta
from mabe.postprocess import predict_multiclass


def _load_trainer(model_path: Path) -> object | None:
    matches = glob.glob(str(model_path / "*trainer*.pkl"))
    if len(matches) == 1:
        return joblib.load(matches[0])
    return None


def submit(
    subset: pd.DataFrame,
    *,
    fps_lookup: dict,
    body_parts: List[str],
    mode: str,
    section: int,
    thresholds: Dict[str, float],
    weights: Dict[str, List[float]],
    model_roots: Dict[str, Path],
    train_tracking_dir: Path,
    test_tracking_dir: Path,
) -> List[pd.DataFrame]:
    submission_list: List[pd.DataFrame] = []
    sample_gen = generate_mouse_data(
        subset,
        mode=mode,
        is_train=False,
        train_tracking_dir=train_tracking_dir,
        test_tracking_dir=test_tracking_dir,
    )

    for _, track_df, meta_df, actions in sample_gen:
        fps = fps_from_meta(meta_df, fps_lookup, default_fps=30.0)
        X = transform_single(track_df, body_parts, fps) if mode == "single" else transform_pair(track_df, body_parts, fps)

        pred = pd.DataFrame(index=X.index)
        for action in actions:
            action_weights = weights.get(action, [])
            model_preds = []
            for model_name, weight in zip(model_roots.keys(), action_weights or []):
                model_path = model_roots[model_name] / str(section) / action
                trainer = _load_trainer(model_path)
                if trainer is None:
                    model_preds.append(np.zeros(X.shape[0]))
                else:
                    model_preds.append(trainer.predict(X))
            if model_preds:
                pred[action] = sum(w * p for w, p in zip(action_weights, model_preds))
            else:
                pred[action] = 0.0

        if pred.shape[1] > 0:
            submission = predict_multiclass(pred, meta_df, thresholds)
            submission_list.append(submission)

        del track_df
        gc.collect()

    return submission_list
