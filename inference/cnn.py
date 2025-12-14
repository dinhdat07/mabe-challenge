"""Inference loop for the 1D CNN ensemble."""

from pathlib import Path
from typing import Dict, List
import gc
import json
import numpy as np
import pandas as pd
import torch
import joblib

from mabe.data.generator import generate_mouse_data
from mabe.features.single import transform_single
from mabe.features.pair import transform_pair
from mabe.features.utils import fps_from_meta
from mabe.training.cnn_trainer import MouseResNet1D


def load_resnet_resources(base_dir: Path, n_folds: int, device: str = "cuda"):
    pack = {"scalers": [], "models": [], "thresholds": {}, "feature_cols": [], "action_names": []}

    feat_col_path = base_dir / "feature_cols.json"
    with feat_col_path.open("r", encoding="utf-8") as f:
        pack["feature_cols"] = json.load(f)

    thresholds = {}
    counts = {}
    for fold in range(n_folds):
        scaler_path = base_dir / f"scaler_fold{fold}.pkl"
        model_path = base_dir / f"model_fold{fold}.pth"
        thr_path = base_dir / f"thresholds_fold{fold}.json"
        pack["scalers"].append(joblib.load(scaler_path))

        state = torch.load(model_path, map_location=device)
        n_feat = state["stem.0.weight"].shape[1]
        last_key = [k for k in state.keys() if "weight" in k][-1]
        n_class = state[last_key].shape[0]

        model = MouseResNet1D(n_feat=n_feat, n_class=n_class).to(device)
        model.load_state_dict(state)
        model.eval()
        pack["models"].append(model)

        if thr_path.exists():
            with thr_path.open("r", encoding="utf-8") as f:
                thrs = json.load(f)
            for k, v in thrs.items():
                thresholds[k] = thresholds.get(k, 0.0) + v
                counts[k] = counts.get(k, 0) + 1

    if thresholds:
        for k in thresholds:
            thresholds[k] /= counts[k]
    else:
        thresholds = {}

    pack["thresholds"] = thresholds
    pack["action_names"] = list(thresholds.keys())
    return pack


def predict_multiclass(pred: np.ndarray, meta: pd.DataFrame, thresholds: np.ndarray, action_names: List[str]) -> pd.DataFrame:
    pred_smoothed = pd.DataFrame(pred).rolling(window=5, min_periods=1, center=True).mean().values
    ama = np.argmax(pred_smoothed, axis=1)
    max_proba = pred_smoothed[np.arange(len(ama)), ama]
    action_thresholds = thresholds[ama]
    ama = np.where(max_proba >= action_thresholds, ama, -1)
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
            "action": [action_names[i] for i in ama_changes[mask].values],
            "start_frame": ama_changes.index[mask],
            "stop_frame": ama_changes.index[1:][mask[:-1]],
        }
    )
    duration = submission_part.stop_frame - submission_part.start_frame
    return submission_part[duration >= 3].reset_index(drop=True)


def submit(
    subset: pd.DataFrame,
    *,
    fps_lookup: dict,
    body_parts: List[str],
    mode: str,
    section_dir: Path,
    n_folds: int,
    window_size: int,
    train_tracking_dir: Path,
    test_tracking_dir: Path,
) -> List[pd.DataFrame]:
    submission_list: List[pd.DataFrame] = []
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pack = load_resnet_resources(section_dir, n_folds=n_folds, device=device)
    action_names = pack["action_names"]
    thr_array = np.array([pack["thresholds"].get(a, 0.5) for a in action_names])

    sample_gen = generate_mouse_data(
        subset,
        mode=mode,
        is_train=False,
        train_tracking_dir=train_tracking_dir,
        test_tracking_dir=test_tracking_dir,
    )

    for _, track_df, meta_df, actions in sample_gen:
        fps = fps_from_meta(meta_df, fps_lookup, default_fps=30.0)
        X_df = transform_single(track_df, body_parts, fps) if mode == "single" else transform_pair(track_df, body_parts, fps)
        X_df = X_df.reindex(columns=pack["feature_cols"], fill_value=0)
        X = X_df.values.astype(np.float32)

        n_frames = len(X)
        full_preds = np.zeros((n_frames, len(action_names)), dtype=np.float32)
        if n_frames < window_size:
            continue

        n_windows = max(1, n_frames - window_size + 1)
        accum_probs = 0
        for scaler, model in zip(pack["scalers"], pack["models"]):
            X_scaled = scaler.transform(np.nan_to_num(X))
            windows = np.array([X_scaled[t : t + window_size] for t in range(n_windows)])
            inp = torch.tensor(windows, dtype=torch.float32).transpose(1, 2).to(device)
            with torch.no_grad():
                probs = torch.sigmoid(model(inp)).cpu().numpy()
                accum_probs += probs

        avg_probs = accum_probs / float(len(pack["models"]))
        offset = window_size // 2
        full_preds[offset : offset + len(avg_probs)] = avg_probs
        full_preds[:offset] = avg_probs[0]
        full_preds[offset + len(avg_probs) :] = avg_probs[-1]

        submission = predict_multiclass(full_preds, meta_df, thr_array, action_names)
        submission_list.append(submission)
        del track_df
        gc.collect()
    return submission_list
