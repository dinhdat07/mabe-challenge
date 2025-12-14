"""Single-mouse feature engineering."""

import numpy as np
import pandas as pd
from .utils import scaled_window, scale


def add_curvature_features(X: pd.DataFrame, center_x: pd.Series, center_y: pd.Series, fps: float) -> pd.DataFrame:
    vx = center_x.diff()
    vy = center_y.diff()
    ax = vx.diff()
    ay = vy.diff()

    cross = vx * ay - vy * ax
    speed = np.sqrt(vx**2 + vy**2)
    curvature = np.abs(cross) / (speed**3 + 1e-6)

    for base_w in [25, 50, 75]:
        ws, mp = scaled_window(base_w, fps, min_frac=0.2)
        X[f"curv_mean_{base_w}"] = curvature.rolling(ws, min_periods=mp).mean()

    angle = np.arctan2(vy, vx)
    angle_change = np.abs(angle.diff())
    ws, mp = scaled_window(30, fps, min_frac=0.2)
    X["turn_rate_30"] = angle_change.rolling(ws, min_periods=mp).sum()
    return X


def add_multiscale_features(X: pd.DataFrame, center_x: pd.Series, center_y: pd.Series, fps: float) -> pd.DataFrame:
    speed = np.sqrt(center_x.diff() ** 2 + center_y.diff() ** 2) * float(fps)
    scales = [20, 40, 60, 80]
    for base_w in scales:
        ws, mp = scaled_window(base_w, fps, min_frac=0.25)
        if len(speed) >= ws:
            X[f"sp_m{base_w}"] = speed.rolling(ws, min_periods=mp).mean()
            X[f"sp_s{base_w}"] = speed.rolling(ws, min_periods=mp).std()
    if all(f"sp_m{s}" in X.columns for s in (scales[0], scales[-1])):
        X["sp_ratio"] = X[f"sp_m{scales[0]}"] / (X[f"sp_m{scales[-1]}"] + 1e-6)
    return X


def add_state_features(X: pd.DataFrame, center_x: pd.Series, center_y: pd.Series, fps: float) -> pd.DataFrame:
    speed = np.sqrt(center_x.diff() ** 2 + center_y.diff() ** 2) * float(fps)
    ws_ma, mp_ma = scaled_window(15, fps, min_frac=1 / 3)
    speed_ma = speed.rolling(ws_ma, min_periods=mp_ma).mean()
    try:
        bins = [-np.inf, 0.5 * fps, 2.0 * fps, 5.0 * fps, np.inf]
        speed_states = pd.cut(speed_ma, bins=bins, labels=[0, 1, 2, 3]).astype(float)
        for base_w in [20, 40, 60, 80]:
            ws, mp = scaled_window(base_w, fps, min_frac=0.2)
            if len(speed_states) < ws:
                continue
            for state in [0, 1, 2, 3]:
                X[f"s{state}_{base_w}"] = (
                    (speed_states == state).astype(float).rolling(ws, min_periods=mp).mean()
                )
            state_changes = (speed_states != speed_states.shift(1)).astype(float)
            X[f"trans_{base_w}"] = state_changes.rolling(ws, min_periods=mp).sum()
    except Exception:
        pass
    return X


def add_longrange_features(X: pd.DataFrame, center_x: pd.Series, center_y: pd.Series, fps: float) -> pd.DataFrame:
    for base_w in [30, 60, 120]:
        ws, mp = scaled_window(base_w, fps, min_frac=1 / 6, min_abs=5)
        if len(center_x) >= ws:
            X[f"x_ml{base_w}"] = center_x.rolling(ws, min_periods=mp).mean()
            X[f"y_ml{base_w}"] = center_y.rolling(ws, min_periods=mp).mean()
    for span in [30, 60, 120]:
        s, _ = scaled_window(span, fps, min_frac=0.0)
        X[f"x_e{span}"] = center_x.ewm(span=s, min_periods=1).mean()
        X[f"y_e{span}"] = center_y.ewm(span=s, min_periods=1).mean()
    speed = np.sqrt(center_x.diff() ** 2 + center_y.diff() ** 2) * float(fps)
    for base_w in [30, 60, 120]:
        ws, mp = scaled_window(base_w, fps, min_frac=1 / 6, min_abs=5)
        if len(speed) >= ws:
            X[f"sp_pct{base_w}"] = speed.rolling(ws, min_periods=mp).rank(pct=True)
    return X


def add_cumulative_distance_single(
    X: pd.DataFrame, cx: pd.Series, cy: pd.Series, fps: float, horizon_frames_base: int = 180, colname: str = "path_cum180"
) -> pd.DataFrame:
    L = max(1, scale(horizon_frames_base, fps))
    step = np.hypot(cx.diff(), cy.diff())
    path = step.rolling(2 * L + 1, min_periods=max(5, L // 6), center=True).sum()
    X[colname] = path.fillna(0.0).astype(np.float32)
    return X


def add_groom_microfeatures(X: pd.DataFrame, df: pd.DataFrame, fps: float) -> pd.DataFrame:
    if ("ear_left" not in df.columns) or ("ear_right" not in df.columns):
        return X
    lag = scale(2, fps)
    le = df["ear_left"]
    re = df["ear_right"]
    le_lag = le.shift(lag)
    re_lag = re.shift(lag)

    le_ch = np.sqrt((le["x"] - le_lag["x"]) ** 2 + (le["y"] - le_lag["y"]) ** 2)
    re_ch = np.sqrt((re["x"] - re_lag["x"]) ** 2 + (re["y"] - re_lag["y"]) ** 2)
    X["ear_jitter"] = (le_ch + re_ch) / 2.0

    min_periods = max(1, lag // 2)
    for base_w in [10, 20, 40]:
        w = scale(base_w, fps)
        roll = dict(window=w, min_periods=min_periods, center=True)
        X[f"ear_jitter_m{base_w}"] = X["ear_jitter"].rolling(**roll).mean()
        X[f"ear_jitter_s{base_w}"] = X["ear_jitter"].rolling(**roll).std()
    return X


def add_speed_asymmetry_future_past_single(X: pd.DataFrame, df: pd.DataFrame, fps: float) -> pd.DataFrame:
    if "nose" not in df.columns or "tail_base" not in df.columns:
        return X
    lag = scale(2, fps)
    nose = df["nose"]
    tail = df["tail_base"]
    nose_lag = nose.shift(lag)
    tail_lag = tail.shift(lag)
    nose_ch = np.sqrt((nose["x"] - nose_lag["x"]) ** 2 + (nose["y"] - nose_lag["y"]) ** 2)
    tail_ch = np.sqrt((tail["x"] - tail_lag["x"]) ** 2 + (tail["y"] - tail_lag["y"]) ** 2)
    X["head_speed"] = nose_ch
    X["rear_speed"] = tail_ch
    X["asym"] = X["head_speed"] - X["rear_speed"]

    min_periods = max(1, lag // 2)
    for base_w in [10, 20, 40]:
        w = scale(base_w, fps)
        roll = dict(window=w, min_periods=min_periods, center=True)
        X[f"asym_m{base_w}"] = X["asym"].rolling(**roll).mean()
        X[f"asym_s{base_w}"] = X["asym"].rolling(**roll).std()
    return X


def _speed(cx: pd.Series, cy: pd.Series, fps: float) -> pd.Series:
    return np.hypot(cx.diff(), cy.diff()) * float(fps)


def _roll_future_mean(s: pd.Series, w: int, min_p: int = 1) -> pd.Series:
    return s.rolling(w, min_periods=min_p).mean().shift(-w)


def _roll_future_var(s: pd.Series, w: int, min_p: int = 2) -> pd.Series:
    return s.rolling(w, min_periods=min_p).var().shift(-w)


def add_gauss_shift_speed_future_past_single(X: pd.DataFrame, df: pd.DataFrame, fps: float) -> pd.DataFrame:
    if "body_center" not in df.columns:
        return X
    cx = df["body_center"]["x"]
    cy = df["body_center"]["y"]
    speed = _speed(cx, cy, fps)
    max_win = scale(30, fps)
    min_p = max(1, max_win // 3)
    for base_w in [10, 20, 30]:
        w = scale(base_w, fps)
        X[f"spd_fut_mean_{base_w}"] = _roll_future_mean(speed, w, min_p=min_p)
        X[f"spd_fut_var_{base_w}"] = _roll_future_var(speed, w, min_p=min_p)
        X[f"spd_past_mean_{base_w}"] = speed.rolling(w, min_periods=min_p).mean()
        X[f"spd_past_var_{base_w}"] = speed.rolling(w, min_periods=min_p).var()
    return X


def add_single_extra_features(X: pd.DataFrame, single_mouse: pd.DataFrame, available_parts: list[str], fps: float) -> pd.DataFrame:
    if "body_center" in available_parts:
        bc = single_mouse["body_center"]
        X = add_curvature_features(X, bc["x"], bc["y"], fps)
        X = add_multiscale_features(X, bc["x"], bc["y"], fps)
        X = add_state_features(X, bc["x"], bc["y"], fps)
        X = add_longrange_features(X, bc["x"], bc["y"], fps)
        X = add_cumulative_distance_single(X, bc["x"], bc["y"], fps)
        X = add_speed_asymmetry_future_past_single(X, single_mouse, fps)
        X = add_gauss_shift_speed_future_past_single(X, single_mouse, fps)
    X = add_groom_microfeatures(X, single_mouse, fps)
    return X


def transform_single(single_mouse: pd.DataFrame, body_parts_tracked: list[str], fps: float) -> pd.DataFrame:
    available_parts = list(single_mouse.columns.get_level_values(0))
    features = {}

    for part in body_parts_tracked:
        if part not in available_parts:
            continue
        df = single_mouse[part]
        for coord in ["x", "y"]:
            features[f"{part}_{coord}"] = df[coord]
            features[f"{part}_{coord}_d1"] = df[coord].diff()
            features[f"{part}_{coord}_d2"] = df[coord].diff().diff()
    X = pd.DataFrame(features)
    X = add_single_extra_features(X, single_mouse, available_parts, fps)
    return X.astype(np.float32, copy=False)
