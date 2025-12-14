"""Prediction smoothing and submission helpers."""

import numpy as np
import pandas as pd


def clean_and_fill_submission(submission: pd.DataFrame, meta_df: pd.DataFrame, is_train: bool = True) -> pd.DataFrame:
    """Ensure each agent/target/video has at least one row; fill missing with dummy no-action segment."""
    if submission.empty:
        return submission

    if is_train:
        expected = meta_df[["video_id", "agent_id", "target_id"]].drop_duplicates()
    else:
        expected = meta_df[["video_id", "agent_id", "target_id"]].drop_duplicates()

    merged = expected.merge(
        submission,
        on=["video_id", "agent_id", "target_id"],
        how="left",
        indicator=True,
        suffixes=("", "_pred"),
    )
    missing_rows = merged["_merge"] == "left_only"
    dummy_rows = merged.loc[missing_rows, ["video_id", "agent_id", "target_id"]]

    if not dummy_rows.empty:
        dummy_rows["action"] = "rear"
        dummy_rows["start_frame"] = 0
        dummy_rows["stop_frame"] = 1
        submission = pd.concat([submission, dummy_rows], ignore_index=True)
    return submission


def predict_multiclass(pred: pd.DataFrame, meta: pd.DataFrame, thresholds: dict) -> pd.DataFrame:
    """Convert per-frame probabilities to event segments with smoothing."""
    pred_smoothed = pred.rolling(window=5, min_periods=1, center=True).mean()
    threshold_array = np.array([thresholds.get(col, 0.27) for col in pred.columns])
    margins = pred_smoothed.values - threshold_array[None, :]
    ama = np.argmax(margins, axis=1)
    max_margin = margins[np.arange(len(ama)), ama]
    ama = np.where(max_margin >= 0.0, ama, -1)
    ama = pd.Series(ama, index=meta.video_frame)

    changes_mask = (ama != ama.shift(1)).values
    ama_changes = ama[changes_mask]
    meta_changes = meta[changes_mask]

    mask = ama_changes.values >= 0
    mask[-1] = False
    submission_part = pd.DataFrame(
        {
            "video_id": meta_changes["video_id"][mask].values,
            "agent_id": meta_changes["agent_id"][mask].values,
            "target_id": meta_changes["target_id"][mask].values,
            "action": pred.columns[ama_changes[mask].values],
            "start_frame": ama_changes.index[mask],
            "stop_frame": ama_changes.index[1:][mask[:-1]],
        }
    )

    stop_video_id = meta_changes["video_id"][1:][mask[:-1]].values
    stop_agent_id = meta_changes["agent_id"][1:][mask[:-1]].values
    stop_target_id = meta_changes["target_id"][1:][mask[:-1]].values

    for i in range(len(submission_part)):
        video_id = submission_part.video_id.iloc[i]
        agent_id = submission_part.agent_id.iloc[i]
        target_id = submission_part.target_id.iloc[i]
        if i < len(stop_video_id):
            if stop_video_id[i] != video_id or stop_agent_id[i] != agent_id or stop_target_id[i] != target_id:
                new_stop_frame = meta.query("(video_id == @video_id)").video_frame.max() + 1
                submission_part.iat[i, submission_part.columns.get_loc("stop_frame")] = new_stop_frame
        else:
            new_stop_frame = meta.query("(video_id == @video_id)").video_frame.max() + 1
            submission_part.iat[i, submission_part.columns.get_loc("stop_frame")] = new_stop_frame

    duration = submission_part.stop_frame - submission_part.start_frame
    submission_part = submission_part[duration >= 3].reset_index(drop=True)
    return submission_part
