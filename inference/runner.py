"""Top-level orchestration for section-wise inference across model types."""

from pathlib import Path
import json
import pandas as pd
from typing import Dict, List

from mabe.inference import xgb as xgb_infer
from mabe.inference import cnn as cnn_infer
from mabe.inference import tcn as tcn_infer
from mabe.inference import ensemble as ens_infer
from mabe.postprocess import clean_and_fill_submission


def _iter_body_parts(train_df: pd.DataFrame) -> List[str]:
    return list(train_df.body_parts_tracked.unique())


def run_xgb_loop(
    *,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    mode: str,
    thresholds: Dict[str, Dict[str, Dict[str, float]]],
    model_dir: Path,
    train_tracking_dir: Path,
    test_tracking_dir: Path,
) -> pd.DataFrame:
    submission_list = []
    body_parts_list = _iter_body_parts(train_df)

    for section, body_parts_tracked_str in enumerate(body_parts_list):
        try:
            body_parts = json.loads(body_parts_tracked_str)
        except Exception:
            body_parts = []

        subset = train_df if mode == "validate" else test_df
        subset = subset[subset.body_parts_tracked == body_parts_tracked_str]
        fps_lookup = (
            subset[["video_id", "frames_per_second"]]
            .drop_duplicates("video_id")
            .set_index("video_id")["frames_per_second"]
            .to_dict()
        )

        xgb_infer.process_mode(
            mode=mode,
            subset=subset,
            body_parts=body_parts,
            fps_lookup=fps_lookup,
            section=section,
            thresholds=thresholds,
            model_dir=model_dir,
            train_tracking_dir=train_tracking_dir,
            test_tracking_dir=test_tracking_dir,
            submission_list=submission_list,
        )

    combined = pd.concat(submission_list) if submission_list else pd.DataFrame()
    meta_df = train_df if mode == "validate" else test_df
    return clean_and_fill_submission(combined, meta_df, is_train=(mode == "validate"))


def run_cnn_loop(
    *,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    mode: str,
    n_folds: int,
    window_size: int,
    model_root: Path,
    train_tracking_dir: Path,
    test_tracking_dir: Path,
) -> pd.DataFrame:
    submission_list = []
    body_parts_list = _iter_body_parts(train_df)

    for section, body_parts_tracked_str in enumerate(body_parts_list):
        try:
            body_parts = json.loads(body_parts_tracked_str)
        except Exception:
            body_parts = []

        subset = train_df if mode == "validate" else test_df
        subset = subset[subset.body_parts_tracked == body_parts_tracked_str]
        fps_lookup = (
            subset[["video_id", "frames_per_second"]]
            .drop_duplicates("video_id")
            .set_index("video_id")["frames_per_second"]
            .to_dict()
        )

        section_dir = model_root / f"sec{section}_{mode}"
        submission_list.extend(
            cnn_infer.submit(
                subset=subset,
                fps_lookup=fps_lookup,
                body_parts=body_parts,
                mode=mode,
                section_dir=section_dir,
                n_folds=n_folds,
                window_size=window_size,
                train_tracking_dir=train_tracking_dir,
                test_tracking_dir=test_tracking_dir,
            )
        )

    combined = pd.concat(submission_list) if submission_list else pd.DataFrame()
    meta_df = train_df if mode == "validate" else test_df
    return clean_and_fill_submission(combined, meta_df, is_train=(mode == "validate"))

def run_tcn_loop(
    *,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    mode: str,
    n_folds: int,
    window_size: int,
    model_root: Path,
    train_tracking_dir: Path,
    test_tracking_dir: Path,
) -> pd.DataFrame:
    submission_list = []
    body_parts_list = _iter_body_parts(train_df)

    for section, body_parts_tracked_str in enumerate(body_parts_list):
        try:
            body_parts = json.loads(body_parts_tracked_str)
        except Exception:
            body_parts = []

        subset = train_df if mode == "validate" else test_df
        subset = subset[subset.body_parts_tracked == body_parts_tracked_str]
        fps_lookup = (
            subset[["video_id", "frames_per_second"]]
            .drop_duplicates("video_id")
            .set_index("video_id")["frames_per_second"]
            .to_dict()
        )

        section_dir = model_root / f"sec{section}_{mode}"
        submission_list.extend(
            tcn_infer.submit(
                subset=subset,
                fps_lookup=fps_lookup,
                body_parts=body_parts,
                mode=mode,
                section_dir=section_dir,
                n_folds=n_folds,
                window_size=window_size,
                train_tracking_dir=train_tracking_dir,
                test_tracking_dir=test_tracking_dir,
            )
        )

    combined = pd.concat(submission_list) if submission_list else pd.DataFrame()
    meta_df = train_df if mode == "validate" else test_df
    return clean_and_fill_submission(combined, meta_df, is_train=(mode == "validate"))


def run_ensemble_loop(
    *,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    mode: str,
    thresholds: Dict[str, Dict[str, float]],
    weights: Dict[str, Dict[str, List[float]]],
    model_roots: Dict[str, Path],
    train_tracking_dir: Path,
    test_tracking_dir: Path,
) -> pd.DataFrame:
    submission_list = []
    body_parts_list = _iter_body_parts(train_df)

    for section, body_parts_tracked_str in enumerate(body_parts_list):
        try:
            body_parts = json.loads(body_parts_tracked_str)
        except Exception:
            body_parts = []

        subset = train_df if mode == "validate" else test_df
        subset = subset[subset.body_parts_tracked == body_parts_tracked_str]
        fps_lookup = (
            subset[["video_id", "frames_per_second"]]
            .drop_duplicates("video_id")
            .set_index("video_id")["frames_per_second"]
            .to_dict()
        )

        section_thresholds = thresholds.get(mode, {}).get(str(section), {})
        section_weights = weights.get(mode, {}).get(str(section), {})

        submission_list.extend(
            ens_infer.submit(
                subset=subset,
                fps_lookup=fps_lookup,
                body_parts=body_parts,
                mode=mode,
                section=section,
                thresholds=section_thresholds,
                weights=section_weights,
                model_roots=model_roots,
                train_tracking_dir=train_tracking_dir,
                test_tracking_dir=test_tracking_dir,
            )
        )

    combined = pd.concat(submission_list) if submission_list else pd.DataFrame()
    meta_df = train_df if mode == "validate" else test_df
    return clean_and_fill_submission(combined, meta_df, is_train=(mode == "validate"))
