"""Label dataframe construction."""

from pathlib import Path
from typing import Iterable, Tuple, List
import json
import pandas as pd
from tqdm import tqdm


def create_solution_df(dataset: pd.DataFrame, annotation_root: Path) -> Tuple[pd.DataFrame, List[str]]:
    """Load all annotation parquet files and concatenate into a ground-truth dataframe."""
    solution = []
    missing_file: List[str] = []
    for _, row in tqdm(dataset.iterrows(), total=len(dataset)):
        lab_id = row["lab_id"]
        if isinstance(lab_id, str) and lab_id.startswith("MABe22"):
            continue

        video_id = row["video_id"]
        path = annotation_root / lab_id / f"{video_id}.parquet"
        try:
            anno = pd.read_parquet(path)
        except FileNotFoundError:
            missing_file.append(str(path))
            continue

        anno["lab_id"] = lab_id
        anno["video_id"] = video_id
        anno["behaviors_labeled"] = row["behaviors_labeled"]

        anno["target_id"] = (
            anno["target_id"].where(anno["target_id"] == anno["agent_id"], anno["target_id"].apply(lambda s: f"mouse{s}"))
        )
        anno["target_id"] = anno["target_id"].replace(anno["agent_id"], "self")
        anno["agent_id"] = anno["agent_id"].apply(lambda s: f"mouse{s}")

        solution.append(anno)

    if not solution:
        return pd.DataFrame(), missing_file
    return pd.concat(solution), missing_file
