"""Iterators that yield per-mouse tracking slices and labels."""

from pathlib import Path
from typing import Generator, Iterable, Optional, Tuple, Union
import gc
import itertools
import json
import numpy as np
import pandas as pd

DROP_BODY_PARTS = [
    "headpiece_bottombackleft",
    "headpiece_bottombackright",
    "headpiece_bottomfrontleft",
    "headpiece_bottomfrontright",
    "headpiece_topbackleft",
    "headpiece_topbackright",
    "headpiece_topfrontleft",
    "headpiece_topfrontright",
    "spine_1",
    "spine_2",
    "tail_middle_1",
    "tail_middle_2",
    "tail_midpoint",
]


def generate_mouse_data(
    dataset: pd.DataFrame,
    *,
    train_tracking_dir: Path,
    test_tracking_dir: Path,
    mode: Optional[str] = None,
    is_train: bool = True,
    drop_body_parts: Iterable[str] = DROP_BODY_PARTS,
) -> Generator[Tuple[str, pd.DataFrame, pd.DataFrame, Union[pd.DataFrame, np.ndarray]], None, None]:
    """Yield per-sample tracking data and labels/features for single or pair modes."""
    data_dir = train_tracking_dir if is_train else test_tracking_dir

    for _, row in dataset.iterrows():
        lab_id = row.lab_id
        if isinstance(lab_id, str) and lab_id.startswith("MABe22"):
            continue
        if not isinstance(row.behaviors_labeled, str):
            continue

        video_id = row.video_id
        tracking_path = data_dir / lab_id / f"{video_id}.parquet"
        vid = pd.read_parquet(tracking_path)

        if len(np.unique(vid.bodypart)) > 5:
            vid = vid.query("~bodypart.isin(@drop_body_parts)")

        pvid = vid.pivot(index="video_frame", columns=["mouse_id", "bodypart"], values=["x", "y"])
        del vid
        gc.collect()

        pvid = pvid.reorder_levels([1, 2, 0], axis=1).T.sort_index().T
        pvid = pvid / row.pix_per_cm_approx

        raw_behaviors = json.loads(row.behaviors_labeled)
        cleaned = {b.replace("'", "") for b in raw_behaviors}
        behaviors_split = [b.split(",") for b in sorted(cleaned)]
        vid_beh = pd.DataFrame(behaviors_split, columns=["agent", "target", "action"])

        if is_train:
            try:
                anno_path = str(tracking_path).replace("train_tracking", "train_annotation")
                anno = pd.read_parquet(anno_path)
            except FileNotFoundError:
                continue

        if mode is None or mode == "single":
            vid_beh_single = vid_beh.query("target == 'self'")
            for agent_str in np.unique(vid_beh_single.agent):
                try:
                    mouse_id = int(agent_str[-1])
                    agent_actions = np.unique(vid_beh_single.query("agent == @agent_str").action)
                    single_mouse = pvid.loc[:, mouse_id]
                    meta = pd.DataFrame(
                        {"video_id": video_id, "agent_id": agent_str, "target_id": "self", "video_frame": single_mouse.index}
                    )
                    if is_train:
                        single_label = pd.DataFrame(0.0, columns=agent_actions, index=single_mouse.index)
                        anno_single = anno.query("(agent_id == @mouse_id) & (target_id == @mouse_id)")
                        for _, anno_row in anno_single.iterrows():
                            single_label.loc[anno_row["start_frame"] : anno_row["stop_frame"], anno_row["action"]] = 1.0
                        yield "single", single_mouse, meta, single_label
                    else:
                        yield "single", single_mouse, meta, agent_actions
                except KeyError:
                    continue

        if mode is None or mode == "pair":
            vid_behaviors_pair = vid_beh.query("target != 'self'")
            if len(vid_behaviors_pair) == 0:
                continue

            mouse_ids = np.unique(pvid.columns.get_level_values("mouse_id"))
            for agent_id, target_id in itertools.permutations(mouse_ids, 2):
                agent_str = f"mouse{agent_id}"
                target_str = f"mouse{target_id}"
                pair_actions = np.unique(
                    vid_behaviors_pair.query("(agent == @agent_str) & (target == @target_str)").action
                )

                mouse_pair = pd.concat([pvid[agent_id], pvid[target_id]], axis=1, keys=["A", "B"])
                meta = pd.DataFrame(
                    {"video_id": video_id, "agent_id": agent_str, "target_id": target_str, "video_frame": mouse_pair.index}
                )

                if is_train:
                    pair_label = pd.DataFrame(0.0, columns=pair_actions, index=mouse_pair.index)
                    anno_pair = anno.query("(agent_id == @agent_id) & (target_id == @target_id)")
                    for _, anno_row in anno_pair.iterrows():
                        pair_label.loc[anno_row["start_frame"] : anno_row["stop_frame"], anno_row["action"]] = 1.0
                    yield "pair", mouse_pair, meta, pair_label
                else:
                    yield "pair", mouse_pair, meta, pair_actions
