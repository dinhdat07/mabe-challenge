"""Training runner: build features per section/mode and train CNN/TCN/XGB."""
from pathlib import Path
from typing import List, Tuple
import json
import gc
import pandas as pd
from mabe.data.generator import generate_mouse_data, DROP_BODY_PARTS
from mabe.features.single import transform_single
from mabe.features.pair import transform_pair
from mabe.features.utils import fps_from_meta
from mabe.training import CNNConfig, TCNConfig, XGBConfig
from mabe.training import train_evaluate_cnn, train_evaluate_tcn, cross_validate_classifier
from mabe.training import EnsembleConfig, train_ensemble_weights
from mabe.config import Paths

def _iter_sections(train_df: pd.DataFrame) -> List[str]:
    return list(train_df.body_parts_tracked.unique())

def _build_feat_label(subset: pd.DataFrame, *, mode: str, body_parts: List[str], fps_lookup: dict, train_tracking_dir: Path, is_train: bool = True):
    data_list, label_list, meta_list = [], [], []
    sample_gen = generate_mouse_data(subset, mode=mode, is_train=is_train, train_tracking_dir=train_tracking_dir, test_tracking_dir=train_tracking_dir)
    for switch, data, meta, label in sample_gen:
        if switch != mode:
            continue
        fps = fps_from_meta(meta, fps_lookup, default_fps=30.0)
        X = transform_single(data, body_parts, fps) if mode == "single" else transform_pair(data, body_parts, fps)
        data_list.append(X.astype("float32"))
        meta_list.append(meta.reset_index(drop=True))
        if is_train:
            label_list.append(label.reset_index(drop=True))
    if not data_list:
        return None
    X_all = pd.concat(data_list, ignore_index=True)
    meta_all = pd.concat(meta_list, ignore_index=True)
    if is_train:
        y_all = pd.concat(label_list, ignore_index=True)
        return X_all, y_all, meta_all
    return X_all, None, meta_all

def train_section(train_df: pd.DataFrame, *, section: int, body_parts_tracked_str: str, paths: Paths, cnn_cfg: CNNConfig | None = None, tcn_cfg: TCNConfig | None = None, xgb_cfg: XGBConfig | None = None, modes: Tuple[str, ...] = ("single", "pair")):
    body_parts = json.loads(body_parts_tracked_str)
    if len(body_parts) > 5:
        body_parts = [b for b in body_parts if b not in DROP_BODY_PARTS]
    subset = train_df[train_df.body_parts_tracked == body_parts_tracked_str]
    fps_lookup = subset[["video_id", "frames_per_second"]].drop_duplicates("video_id").set_index("video_id")["frames_per_second"].to_dict()
    thresholds_xgb = {}
    oof_store: dict = {}
    for mode in modes:
        built = _build_feat_label(subset, mode=mode, body_parts=body_parts, fps_lookup=fps_lookup, train_tracking_dir=paths.train_tracking_dir, is_train=True)
        if built is None:
            continue
        X, y, meta = built
        if cnn_cfg:
            cfg = CNNConfig(**cnn_cfg.__dict__)
            cfg.model_dir = Path(cnn_cfg.model_dir) / f"sec{section}_{mode}"
            train_evaluate_cnn([X], [y], [meta], cfg=cfg)
        if tcn_cfg:
            cfg = TCNConfig(**tcn_cfg.__dict__)
            cfg.model_dir = Path(tcn_cfg.model_dir) / f"sec{section}_{mode}"
            train_evaluate_tcn([X], [y], cfg=cfg)
        if xgb_cfg:
            cfg = XGBConfig(**xgb_cfg.__dict__)
            cfg.save_dir = Path(xgb_cfg.save_dir) / f"sec{section}_{mode}"
            _, _, thr = cross_validate_classifier(X, y, meta, cfg=cfg)
            thresholds_xgb.setdefault(mode, {})[str(section)] = thr
        oof_store.setdefault(mode, {})[str(section)] = {"X": X, "y": y, "meta": meta}
        gc.collect()
    return thresholds_xgb, oof_store

def train_all(train_df: pd.DataFrame, *, paths: Paths, cnn_cfg: CNNConfig | None = None, tcn_cfg: TCNConfig | None = None, xgb_cfg: XGBConfig | None = None, modes: Tuple[str, ...] = ("single", "pair")):
    sections = _iter_sections(train_df)
    thresholds_all = {"single": {}, "pair": {}}
    oof_all = {}
    for sec_idx, bp_str in enumerate(sections):
        thr, oof = train_section(train_df, section=sec_idx, body_parts_tracked_str=bp_str, paths=paths, cnn_cfg=cnn_cfg, tcn_cfg=tcn_cfg, xgb_cfg=xgb_cfg, modes=modes)
        for mode, d in thr.items():
            thresholds_all.setdefault(mode, {}).update(d)
        oof_all[f"section{sec_idx}"] = oof
    return thresholds_all, oof_all
