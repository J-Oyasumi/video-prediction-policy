"""
Prepare json file for `evaluate_robocasa.py`.

This script builds a list of episodes for a **single RoboCasa task** with
distributions "train" / "val" / "target" and saves it as json.

Expected output format (see `evaluate_robocasa.py`):

[
    {
        "task_name": "CloseBlenderLid",
        "dataset_dir": "data_raw/v1.0/pretrain/atomic/CloseBlenderLid/20250822/lerobot",
        "episode_id": 18,
            "distribution": "train",
    },
    ...
]
"""

import argparse
import json
import os
import random
from pathlib import Path
from typing import List, Dict, Any


def _load_episode_meta(ann_path: Path) -> Dict[str, Any]:
    with ann_path.open("r") as f:
        return json.load(f)


def _infer_dataset_dir(ann_path: Path, repo_root: Path) -> str:
    """
    Infer dataset_dir from a single annotation file.

    We take the first video_path, join with annotation dir, then strip the
    trailing "videos/.../episode_xxx.mp4" part and keep up to "lerobot".
    """
    label = _load_episode_meta(ann_path)
    videos = label.get("videos")
    if not videos:
        raise ValueError(f"No 'videos' field in annotation {ann_path}")
    video_path = videos[0].get("video_path")
    if not video_path:
        raise ValueError(f"No 'video_path' in annotation {ann_path}")

    ann_dir = ann_path.parent
    video_abs = (ann_dir / video_path).resolve()
    parts = list(video_abs.parts)

    # Prefer sub-path starting at "data_raw" up to "lerobot"
    if "data_raw" in parts and "lerobot" in parts:
        start = parts.index("data_raw")
        end = parts.index("lerobot")
        dataset_abs = repo_root / Path(*parts[start : end + 1])
    else:
        # fallback: cut at "lerobot" or "videos" using absolute path
        if "lerobot" in parts:
            idx = parts.index("lerobot")
            dataset_abs = Path(*parts[: idx + 1])
        else:
            if "videos" not in parts:
                raise ValueError(f"Cannot infer dataset_dir from path {video_abs}")
            idx = parts.index("videos")
            dataset_abs = Path(*parts[:idx])

    # store path relative to repo root, to match examples in evaluate_robocasa.py
    try:
        dataset_rel = os.path.relpath(dataset_abs, repo_root)
    except ValueError:
        # different drive or unexpected case – just use absolute path
        dataset_rel = str(dataset_abs)
    return dataset_rel


def _collect_episodes_from_dir(
    ann_dir: Path,
    distribution: str,
    num_samples: int = -1,
    repo_root: Path = Path("."),
    rng: random.Random = random,
) -> List[Dict[str, Any]]:
    if not ann_dir.exists():
        raise FileNotFoundError(f"Annotation directory not found: {ann_dir}")

    ann_files = sorted(p for p in ann_dir.glob("*.json") if p.is_file())
    if not ann_files:
        raise FileNotFoundError(f"No annotation json files under {ann_dir}")

    # determine dataset_dir from the first file
    dataset_dir = _infer_dataset_dir(ann_files[0], repo_root)

    if num_samples < 0 or num_samples >= len(ann_files):
        selected = ann_files
    else:
        selected = rng.sample(ann_files, num_samples)

    episodes: List[Dict[str, Any]] = []
    for p in selected:
        meta = _load_episode_meta(p)
        episode_id = meta.get("episode_id")
        if episode_id is None:
            raise ValueError(f"'episode_id' missing in {p}")

        episodes.append(
            {
                "task_name": ann_dir.parents[1].name,  # data/{task_name}/annotations/{split}
                "dataset_dir": dataset_dir,
                "episode_id": int(episode_id),
                "distribution": distribution,
            }
        )
    return episodes


def main():
    parser = argparse.ArgumentParser(
        description="Prepare json file for evaluate_robocasa.py"
    )
    parser.add_argument(
        "--task_name",
        type=str,
        required=True,
        help="Task name, e.g. CloseCabinet / OpenDrawer",
    )
    parser.add_argument(
        "--num_train",
        type=int,
        default=0,
        help="Number of train episodes to sample from data/{task_name}/annotations/train "
        "(distribution='train'). 0 means no train episodes.",
    )
    parser.add_argument(
        "--num_target",
        type=int,
        default=0,
        help=(
            "Number of target episodes to sample (distribution='target'). "
            "Episode ids are drawn uniformly from [0, 500). "
            "Their dataset_dir is obtained by replacing 'pretrain' with 'target' "
            "in the pretrain dataset_dir."
        ),
    )
    parser.add_argument(
        "--output_json",
        type=str,
        required=True,
        help="Output json path for evaluate_robocasa.py",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for sampling.",
    )

    args = parser.parse_args()
    rng = random.Random(args.seed)

    repo_root = Path(__file__).resolve().parent
    task_root = repo_root / "data" / args.task_name
    train_ann_dir = task_root / "annotations" / "train"
    val_ann_dir = task_root / "annotations" / "val"

    all_entries: List[Dict[str, Any]] = []

    # 1) VAL: always use all val indices (distribution='val')
    if val_ann_dir.exists():
        val_entries = _collect_episodes_from_dir(
            ann_dir=val_ann_dir,
            distribution="val",
            num_samples=-1,  # all
            repo_root=repo_root,
            rng=rng,
        )
        all_entries.extend(val_entries)

    # 2) TRAIN: random sample from train annotations if requested
    if args.num_train > 0:
        train_entries = _collect_episodes_from_dir(
            ann_dir=train_ann_dir,
            distribution="train",
            num_samples=args.num_train,
            repo_root=repo_root,
            rng=rng,
        )
        all_entries.extend(train_entries)

    # 3) TARGET: random episode ids in [0, 500), dataset_dir uses 'target'
    if args.num_target > 0:
        if not all_entries:
            raise ValueError(
                "To create target entries, we need at least one pretrain "
                "entry (from val/train) to infer dataset_dir."
            )

        pretrain_dir = all_entries[0]["dataset_dir"]
        if "pretrain" not in pretrain_dir:
            raise ValueError(
                f"Cannot infer target dataset_dir from '{pretrain_dir}', "
                "expected it to contain 'pretrain'."
            )

        target_dir = pretrain_dir.replace("pretrain", "target", 1)

        for _ in range(args.num_target):
            episode_id = rng.randint(0, 499)  # [0, 500)
            all_entries.append(
                {
                    "task_name": args.task_name,
                    "dataset_dir": target_dir,
                    "episode_id": episode_id,
                    "distribution": "target",
                }
            )

    os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
    with open(args.output_json, "w") as f:
        json.dump(all_entries, f, indent=2)

    print(f"Saved {len(all_entries)} episodes to {args.output_json}")


if __name__ == "__main__":
    main()