"""Inference and section loop for XGB models."""

from pathlib import Path
from typing import Dict, List, Optional
import glob
import gc
import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator

from mabe.data.generator import generate_mouse_data
from mabe.features.single import transform_single
from mabe.features.pair import transform_pair
from mabe.features.utils import fps_from_meta
from mabe.postprocess import predict_multiclass


def _load_trainer(model_dir: Path, section: int, action: str) -> Optional[BaseEstimator]:
    """Load a saved koolbox Trainer (or estimator) pickle for a given section/action."""
    pattern = model_dir / str(section) / action / "*trainer*.pkl"
    matches = glob.glob(str(pattern))
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
    model_dir: Path,
    train_tracking_dir: Path,
    test_tracking_dir: Path,
) -> List[pd.DataFrame]:
    """Run inference for one section and one mode (single/pair) using saved XGB trainers."""
    submission_list: List[pd.DataFrame] = []
    sample_gen = generate_mouse_data(
        subset,
        mode=mode,
        is_train=False,
        train_tracking_dir=train_tracking_dir,
        test_tracking_dir=test_tracking_dir,
    )

    for _, track_df, meta_df, actions in sample_gen:
        video_id = meta_df["video_id"].iloc[0]
        agent_id = meta_df["agent_id"].iloc[0]
        target_id = meta_df["target_id"].iloc[0]
        fps = fps_from_meta(meta_df, fps_lookup, default_fps=30.0)
        X = transform_single(track_df, body_parts, fps) if mode == "single" else transform_pair(track_df, body_parts, fps)

        pred = pd.DataFrame(index=X.index)
        for action in actions:
            trainer = _load_trainer(model_dir, section, action)
            if trainer is None:
                pred[action] = 0.0
                continue
            pred[action] = trainer.predict(X)

        if pred.shape[1] > 0:
            submission = predict_multiclass(pred, meta_df, thresholds)
            submission_list.append(submission)

        del track_df
        gc.collect()

    return submission_list


def process_mode(
    *,
    mode: str,
    subset: pd.DataFrame,
    body_parts: List[str],
    fps_lookup: dict,
    section: int,
    thresholds: Dict[str, Dict[str, float]],
    model_dir: Path,
    train_tracking_dir: Path,
    test_tracking_dir: Path,
    f1_list: Optional[List] = None,
    submission_list: Optional[List[pd.DataFrame]] = None,
) -> None:
    """Wrapper for a single mode; thresholds is nested dict {mode: {section: {action: thr}}}."""
    submission_list = submission_list if submission_list is not None else []
    section_thresholds = thresholds.get(mode, {}).get(str(section), {})
    temp = submit(
        subset=subset,
        fps_lookup=fps_lookup,
        body_parts=body_parts,
        mode=mode,
        section=section,
        thresholds=section_thresholds,
        model_dir=model_dir,
        train_tracking_dir=train_tracking_dir,
        test_tracking_dir=test_tracking_dir,
    )
    submission_list.extend(temp)
