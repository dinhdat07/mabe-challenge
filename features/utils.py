"""Shared scaling helpers for FPS-aware windows."""

import pandas as pd


def scaled_window(n_frames_30fps: int, fps: float, min_frac: float = 0.2, min_abs: int = 1) -> tuple[int, int]:
    ws = max(1, int(round(n_frames_30fps * float(fps) / 30.0)))
    min_periods = max(min_abs, int(round(ws * min_frac)))
    return ws, min_periods


def fps_from_meta(meta_df: pd.DataFrame, fallback_lookup: dict, default_fps: float = 30.0) -> float:
    if "frames_per_second" in meta_df.columns and pd.notnull(meta_df["frames_per_second"]).any():
        return float(meta_df["frames_per_second"].iloc[0])
    vid = meta_df["video_id"].iloc[0]
    return float(fallback_lookup.get(vid, default_fps))


def scale(n_frames_at_30fps: int, fps: float, ref: float = 30.0) -> int:
    return max(1, int(round(n_frames_at_30fps * float(fps) / ref)))


def scale_signed(n_frames_at_30fps: int, fps: float, ref: float = 30.0) -> int:
    if n_frames_at_30fps == 0:
        return 0
    sign = 1 if n_frames_at_30fps > 0 else -1
    mag = max(1, int(round(abs(n_frames_at_30fps) * float(fps) / ref)))
    return sign * mag
