"""Training loop for the 1D CNN (ResNet) model."""

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


class FocalBCEWithLogitsLoss(nn.Module):
    def __init__(self, pos_weight=None, gamma: float = 2.0, reduction: str = "mean"):
        super().__init__()
        self.pos_weight = pos_weight
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        bce = nn.functional.binary_cross_entropy_with_logits(logits, targets, pos_weight=self.pos_weight, reduction="none")
        probas = torch.sigmoid(logits)
        pt = torch.where(targets == 1, probas, 1 - probas)
        loss = (1 - pt) ** self.gamma * bce
        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


class MABeLazyDataset(Dataset):
    """Sliding window dataset over precomputed features."""

    def __init__(self, feat_list, scaler: StandardScaler, label_list=None, window_size: int = 30):
        self.feat_list = feat_list
        self.scaler = scaler
        self.label_list = label_list
        self.window_size = window_size
        self.index_map = []
        for vid, arr in enumerate(self.feat_list):
            for t in range(max(1, len(arr) - window_size + 1)):
                self.index_map.append((vid, t))

    def __len__(self) -> int:
        return len(self.index_map)

    def __getitem__(self, idx):
        v_idx, t = self.index_map[idx]
        window = self.feat_list[v_idx][t : t + self.window_size]
        if self.scaler is not None:
            window = self.scaler.transform(window)
        X = torch.tensor(window, dtype=torch.float32).transpose(0, 1)
        if self.label_list is None:
            return X
        center_frame = t + self.window_size // 2
        y = torch.tensor(self.label_list[v_idx][center_frame], dtype=torch.float32)
        return X, y


class ResidualBlock1D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, dilation: int = 1, dropout: float = 0.2):
        super().__init__()
        padding = (kernel_size - 1) * dilation // 2
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding, dilation=dilation)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, padding=padding, dilation=dilation)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.shortcut = nn.Conv1d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        residual = self.shortcut(x)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.dropout(x)
        x = self.bn2(self.conv2(x))
        x += residual
        return self.relu(x)


class MouseResNet1D(nn.Module):
    def __init__(self, n_feat: int, n_class: int, base_filters: int = 64):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv1d(n_feat, base_filters, kernel_size=7, padding=3), nn.BatchNorm1d(base_filters), nn.ReLU())
        self.layer1 = ResidualBlock1D(base_filters, base_filters, dilation=1)
        self.layer2 = ResidualBlock1D(base_filters, base_filters * 2, dilation=2)
        self.layer3 = ResidualBlock1D(base_filters * 2, base_filters * 4, dilation=4)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(base_filters * 4, n_class)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.global_pool(x).squeeze(-1)
        return self.fc(x)


@dataclass
class CNNConfig:
    model_dir: Path = Path("models/1d-cnn")
    n_splits: int = 3
    window_size: int = 30
    base_filters: int = 64
    patience: int = 5
    lr: float = 1e-3
    weight_decay: float = 1e-4


def tune_thresholds_per_class(y_true: np.ndarray, y_pred_prob: np.ndarray, class_names: list[str]) -> tuple[dict, dict]:
    best_thrs = {}
    best_f1s = {}
    thresholds = np.linspace(0.01, 0.99, 99)
    for idx, name in enumerate(class_names):
        p = y_pred_prob[:, idx]
        y_t = y_true[:, idx]
        pred_matrix = p[:, None] >= thresholds[None, :]
        tp = (pred_matrix & (y_t[:, None] == 1)).sum(axis=0)
        fp = (pred_matrix & (y_t[:, None] == 0)).sum(axis=0)
        fn = ((~pred_matrix) & (y_t[:, None] == 1)).sum(axis=0)
        f1_scores = 2 * tp / (2 * tp + fp + fn + 1e-7)
        best_idx = int(np.argmax(f1_scores))
        best_f1s[name] = float(f1_scores[best_idx])
        best_thrs[name] = float(thresholds[best_idx])
    return best_thrs, best_f1s


def save_feature_columns(ref_cols: list[str], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(ref_cols, f)


def train_evaluate_cnn(feat_list: List[pd.DataFrame], label_list: List[pd.DataFrame], meta_list: List[pd.DataFrame], *, cfg: CNNConfig):
    """Minimal CNN trainer extracted from the notebook."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg.model_dir.mkdir(parents=True, exist_ok=True)

    ref_cols = sorted({c for df in feat_list for c in df.columns})
    save_feature_columns(ref_cols, cfg.model_dir / "feature_cols.json")

    feat_np = []
    for df in feat_list:
        arr = df.reindex(columns=ref_cols, fill_value=0).values.astype(np.float32)
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        feat_np.append(arr)
    label_np = []
    all_actions = set()
    for df in label_list:
        all_actions.update(df.columns)
    ref_actions = sorted(all_actions)
    for df in label_list:
        label_np.append(df.reindex(columns=ref_actions, fill_value=0).values.astype(np.float32))

    video_indices = np.arange(len(feat_np))
    video_labels = []
    for arr in label_np:
        counts = arr.sum(axis=0)
        video_labels.append(np.argmax(counts) if counts.sum() > 0 else -1)

    fold_scores = []
    for fold in range(cfg.n_splits):
        # Placeholder split: simple hold-out per fold id.
        val_mask = video_indices % cfg.n_splits == fold
        train_idx = video_indices[~val_mask]
        val_idx = video_indices[val_mask]

        train_feats = [feat_np[i] for i in train_idx]
        val_feats = [feat_np[i] for i in val_idx]
        train_lbls = [label_np[i] for i in train_idx]
        val_lbls = [label_np[i] for i in val_idx]

        scaler = StandardScaler()
        for arr in train_feats:
            scaler.partial_fit(arr)

        ds_train = MABeLazyDataset(train_feats, scaler, train_lbls, window_size=cfg.window_size)
        ds_val = MABeLazyDataset(val_feats, scaler, val_lbls, window_size=cfg.window_size)
        loader_train = DataLoader(ds_train, batch_size=1024, shuffle=True, num_workers=0)
        loader_val = DataLoader(ds_val, batch_size=2048, shuffle=False, num_workers=0)

        model = MouseResNet1D(n_feat=len(ref_cols), n_class=len(ref_actions), base_filters=cfg.base_filters).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        pos_counts = np.zeros(len(ref_actions))
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
        for _ in range(cfg.patience):
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
            fold_scores.append(f1_score(vt, (vp > 0.5).astype(int), average="macro", zero_division=0))
            best_f1 = max(best_f1, fold_scores[-1])

        torch.save(model.state_dict(), cfg.model_dir / f"model_fold{fold}.pth")
        joblib.dump(scaler, cfg.model_dir / f"scaler_fold{fold}.pkl")
        best_thrs, _ = tune_thresholds_per_class(vt, vp, ref_actions)
        with open(cfg.model_dir / f"thresholds_fold{fold}.json", "w", encoding="utf-8") as f:
            json.dump(best_thrs, f)

        del model, scaler, ds_train, ds_val, loader_train, loader_val
        gc.collect()

    return fold_scores
