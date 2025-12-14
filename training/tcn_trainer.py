"""Training loop for causal Temporal Convolutional Network (TCN) baseline."""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple
import json
import gc

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedGroupKFold

from mabe.training.cnn_trainer import FocalBCEWithLogitsLoss, tune_thresholds_per_class, save_feature_columns


class MABeLazyDataset(Dataset):
    """Sliding window dataset; center frame label for each window."""

    def __init__(self, feat_list, scaler: StandardScaler, label_list=None, window_size: int = 30):
        self.feat_list = feat_list
        self.scaler = scaler
        self.label_list = label_list
        self.window_size = window_size
        self.index_map = []
        for vid, arr in enumerate(self.feat_list):
            L = len(arr)
            if L >= window_size:
                self.index_map.extend([(vid, t) for t in range(L - window_size + 1)])

    def __len__(self) -> int:
        return len(self.index_map)

    def __getitem__(self, idx):
        v_idx, t = self.index_map[idx]
        window = self.feat_list[v_idx][t : t + self.window_size]
        if self.scaler is not None:
            window = self.scaler.transform(window)
        x = torch.tensor(window, dtype=torch.float32).transpose(0, 1)  # (feat, seq)
        if self.label_list is None:
            return x
        center_frame = t + self.window_size // 2
        y = torch.tensor(self.label_list[v_idx][center_frame], dtype=torch.float32)
        return x, y


class Chomp1d(nn.Module):
    def __init__(self, chomp: int):
        super().__init__()
        self.chomp = chomp

    def forward(self, x):
        return x[..., :-self.chomp] if self.chomp > 0 else x


class TemporalBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, k: int = 3, d: int = 1, dropout: float = 0.1):
        super().__init__()
        pad = (k - 1) * d
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, k, padding=pad, dilation=d),
            Chomp1d(pad),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(out_ch, out_ch, k, padding=pad, dilation=d),
            Chomp1d(pad),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.down = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x):
        return nn.functional.relu(self.net(x) + self.down(x))


class MouseTCN(nn.Module):
    """Causal TCN mirroring the notebook architecture."""

    def __init__(self, n_feat: int, n_class: int, channels: Tuple[int, ...] = (64, 64, 128, 128), k: int = 3, dropout: float = 0.1):
        super().__init__()
        blocks = []
        in_c = n_feat
        for i, out_c in enumerate(channels):
            blocks.append(TemporalBlock(in_c, out_c, k=k, d=2**i, dropout=dropout))
            in_c = out_c
        self.tcn = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(in_c, n_class)

    def forward(self, x):
        x = self.tcn(x)
        x = self.pool(x).squeeze(-1)
        return self.fc(x)


@dataclass
class TCNConfig:
    model_dir: Path = Path("models/tcn")
    n_splits: int = 3
    window_size: int = 30
    patience: int = 5
    lr: float = 1e-3
    weight_decay: float = 1e-4
    dropout: float = 0.1


def _compute_video_labels(label_np: List[np.ndarray]) -> np.ndarray:
    video_labels = []
    for arr in label_np:
        counts = arr.sum(axis=0)
        video_labels.append(np.argmax(counts) if counts.sum() > 0 else -1)
    return np.array(video_labels)


def train_evaluate_tcn(feat_list: List[pd.DataFrame], label_list: List[pd.DataFrame], *, cfg: TCNConfig):
    """Train causal TCN with StratifiedGroupKFold; mirrors notebook logic."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg.model_dir.mkdir(parents=True, exist_ok=True)

    ref_cols = sorted({c for df in feat_list for c in df.columns})
    save_feature_columns(ref_cols, cfg.model_dir / "feature_cols.json")

    feat_np = []
    for df in feat_list:
        arr = df.reindex(columns=ref_cols, fill_value=0).values.astype(np.float32)
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        feat_np.append(arr)

    all_actions = set()
    for df in label_list:
        all_actions.update(df.columns)
    ref_actions = sorted(all_actions)

    label_np = [df.reindex(columns=ref_actions, fill_value=0).values.astype(np.float32) for df in label_list]
    video_indices = np.arange(len(feat_np))
    video_labels = _compute_video_labels(label_np)
    video_groups = video_indices.copy()

    sgkf = StratifiedGroupKFold(n_splits=cfg.n_splits)
    fold_scores = []

    for fold, (train_idx, val_idx) in enumerate(sgkf.split(video_indices, video_labels, video_groups)):
        train_feats = [feat_np[i] for i in train_idx]
        val_feats = [feat_np[i] for i in val_idx]
        train_lbls = [label_np[i] for i in train_idx]
        val_lbls = [label_np[i] for i in val_idx]

        scaler = StandardScaler()
        for arr in train_feats:
            scaler.partial_fit(arr)

        ds_train = MABeLazyDataset(train_feats, scaler, train_lbls, window_size=cfg.window_size)
        ds_val = MABeLazyDataset(val_feats, scaler, val_lbls, window_size=cfg.window_size)
        loader_train = DataLoader(ds_train, batch_size=256, shuffle=True, num_workers=0)
        loader_val = DataLoader(ds_val, batch_size=512, shuffle=False, num_workers=0)

        n_feat = len(ref_cols)
        n_class = len(ref_actions)
        model = MouseTCN(n_feat=n_feat, n_class=n_class, dropout=cfg.dropout).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

        pos_counts = np.zeros(n_class)
        total_samples = 0
        for arr in train_lbls:
            if len(arr) >= cfg.window_size:
                valid_lbls = arr[cfg.window_size // 2 : -(cfg.window_size // 2)]
                pos_counts += valid_lbls.sum(axis=0)
                total_samples += len(valid_lbls)
        raw_weights = (total_samples - pos_counts) / (pos_counts + 1e-6)
        raw_weights = np.clip(raw_weights, 1.0, 100.0)
        criterion = FocalBCEWithLogitsLoss(pos_weight=torch.tensor(raw_weights, dtype=torch.float32).to(device), gamma=1.0)

        best_f1 = 0.0
        no_improve = 0
        for epoch in range(50):
            model.train()
            for xb, yb in loader_train:
                xb, yb = xb.to(device), yb.to(device)
                optimizer.zero_grad()
                loss = criterion(model(xb), yb)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            model.eval()
            val_probs, val_true = [], []
            with torch.no_grad():
                for xb, yb in loader_val:
                    val_probs.append(torch.sigmoid(model(xb.to(device))).cpu())
                    val_true.append(yb)
            vp = torch.cat(val_probs).numpy()
            vt = torch.cat(val_true).numpy()
            mean_f1 = f1_score(vt, (vp > 0.5).astype(int), average="macro", zero_division=0)
            if mean_f1 > best_f1 + 1e-4:
                best_f1 = mean_f1
                no_improve = 0
                torch.save(model.state_dict(), cfg.model_dir / f"model_fold{fold}.pth")
                joblib.dump(scaler, cfg.model_dir / f"scaler_fold{fold}.pkl")
                best_thrs, _ = tune_thresholds_per_class(vt, vp, ref_actions)
                with open(cfg.model_dir / f"thresholds_fold{fold}.json", "w", encoding="utf-8") as f:
                    json.dump(best_thrs, f)
            else:
                no_improve += 1
                if no_improve >= cfg.patience:
                    break

        fold_scores.append(best_f1)

        del model, scaler, ds_train, ds_val, loader_train, loader_val
        gc.collect()

    return fold_scores
