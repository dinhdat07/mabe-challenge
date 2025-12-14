"""Pair/interaction feature engineering."""

import itertools
import numpy as np
import pandas as pd
from .utils import scale, scale_signed


def add_interaction_features(X: pd.DataFrame, mouse_pair: pd.DataFrame, avail_A, avail_B, fps: float) -> pd.DataFrame:
    if "body_center" in avail_A and "body_center" in avail_B:
        Ax = mouse_pair["A"]["body_center"]["x"]
        Ay = mouse_pair["A"]["body_center"]["y"]
        Bx = mouse_pair["B"]["body_center"]["x"]
        By = mouse_pair["B"]["body_center"]["y"]
        lag = scale(15, fps)
        cd2 = np.square(mouse_pair["A"]["body_center"] - mouse_pair["B"]["body_center"]).sum(axis=1, skipna=False)
        roll_c = dict(window=lag, min_periods=1, center=True)
        X["cd2_mean"] = cd2.rolling(**roll_c).mean()
        X["cd2_std"] = cd2.rolling(**roll_c).std()
        vel_cos = ((Ax.diff() * Bx.diff()) + (Ay.diff() * By.diff())) / (
            np.sqrt(Ax.diff() ** 2 + Ay.diff() ** 2) * np.sqrt(Bx.diff() ** 2 + By.diff() ** 2) + 1e-6
        )
        for off in [-30, -15, 0, 15, 30]:
            o = scale_signed(off, fps)
            X[f"va_{off}"] = vel_cos.shift(-o)
    return X


def add_egocentric_interaction_features(X: pd.DataFrame, mouse_pair: pd.DataFrame, avail_A, avail_B, fps: float) -> pd.DataFrame:
    if "nose" in avail_A and "nose" in avail_B and "tail_base" in avail_A and "tail_base" in avail_B:
        dir_A = mouse_pair["A"]["nose"] - mouse_pair["A"]["tail_base"]
        dir_B = mouse_pair["B"]["nose"] - mouse_pair["B"]["tail_base"]
        dot = dir_A["x"] * dir_B["x"] + dir_A["y"] * dir_B["y"]
        nA = np.sqrt(dir_A["x"] ** 2 + dir_A["y"] ** 2)
        nB = np.sqrt(dir_B["x"] ** 2 + dir_B["y"] ** 2)
        X["rel_ori"] = dot / (nA * nB + 1e-6)
    if "nose" in avail_A and "nose" in avail_B:
        nn = np.sqrt(
            (mouse_pair["A"]["nose"]["x"] - mouse_pair["B"]["nose"]["x"]) ** 2
            + (mouse_pair["A"]["nose"]["y"] - mouse_pair["B"]["nose"]["y"]) ** 2
        )
        for lag in [10, 20, 40]:
            l = scale(lag, fps)
            X[f"nn_lg{lag}"] = nn.shift(l)
            X[f"nn_ch{lag}"] = nn - nn.shift(l)
            is_close = (nn < 10.0).astype(float)
            X[f"cl_ps{lag}"] = is_close.rolling(l, min_periods=1).mean()
    return X


def add_asymmetry_features(X: pd.DataFrame, mouse_pair: pd.DataFrame, avail_A, avail_B, fps: float) -> pd.DataFrame:
    if "nose" in avail_A and "nose" in avail_B and "tail_base" in avail_A and "tail_base" in avail_B:
        noseA = mouse_pair["A"]["nose"]
        tailA = mouse_pair["A"]["tail_base"]
        noseB = mouse_pair["B"]["nose"]
        tailB = mouse_pair["B"]["tail_base"]
        if "body_center" in avail_A and "body_center" in avail_B:
            bodyA = mouse_pair["A"]["body_center"]
            bodyB = mouse_pair["B"]["body_center"]
            vAB = bodyB - bodyA
            vA = noseA - tailA
            vB = noseB - tailB
            crossA = vAB["x"] * vA["y"] - vAB["y"] * vA["x"]
            crossB = vAB["x"] * vB["y"] - vAB["y"] * vB["x"]
            X["asym_turn"] = crossA - crossB
    return X


def transform_pair(mouse_pair: pd.DataFrame, body_parts_tracked: list[str], fps: float) -> pd.DataFrame:
    avail_A = mouse_pair["A"].columns.get_level_values(0)
    avail_B = mouse_pair["B"].columns.get_level_values(0)

    features = {}
    for p1, p2 in itertools.product(body_parts_tracked, repeat=2):
        if p1 in avail_A and p2 in avail_B:
            diff = mouse_pair["A"][p1] - mouse_pair["B"][p2]
            dist2 = np.square(diff).sum(axis=1, skipna=False)
            features[f"12+{p1}+{p2}"] = dist2

    X = pd.DataFrame(features)
    X = add_egocentric_interaction_features(X, mouse_pair, avail_A, avail_B, fps)
    X = add_asymmetry_features(X, mouse_pair, avail_A, avail_B, fps)
    X = add_interaction_features(X, mouse_pair, avail_A, avail_B, fps)
    return X.astype(np.float32, copy=False)
