"""Configuration helpers for training and inference."""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import yaml


@dataclass
class Paths:
    train_csv: Path
    test_csv: Path
    train_annotation_dir: Path
    train_tracking_dir: Path
    test_tracking_dir: Path
    model_dir: Path


@dataclass
class TrainingConfig:
    mode: str = "validate"  # or "submit"
    n_splits: int = 3
    random_state: int = 2024


def load_config(path: Path) -> tuple[Paths, TrainingConfig]:
    """Load YAML config and return typed paths/config."""
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    paths = Paths(
        train_csv=Path(cfg["data"]["train_csv"]),
        test_csv=Path(cfg["data"]["test_csv"]),
        train_annotation_dir=Path(cfg["data"]["train_annotation_dir"]),
        train_tracking_dir=Path(cfg["data"]["train_tracking_dir"]),
        test_tracking_dir=Path(cfg["data"]["test_tracking_dir"]),
        model_dir=Path(cfg["output"]["model_dir"]),
    )
    train_cfg = TrainingConfig(
        mode=cfg.get("mode", "validate"),
        n_splits=cfg.get("n_splits", 3),
        random_state=cfg.get("random_state", 2024),
    )
    return paths, train_cfg
