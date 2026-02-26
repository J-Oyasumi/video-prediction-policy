"""
Prepare train_all.json and val_all.json for RoboCasa (VPP training).
Mirrors step2_prepare_json.py: sliding-window samples + state_01/99, action_01/99 for normalization.

Expects prepare_latent_robocasa output under data_root:
  data_root/{dataset_name}/annotations/{train,val}/*.json
  each JSON: video_length, states, actions, episode_id, texts, latent_videos, ...
"""

import argparse
import json
import os
import random
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
from tqdm import tqdm


def load_and_process_ann_file(
    data_root: str,
    ann_file: str,
    sequence_interval: int = 2,
    start_interval: int = 1,
    dataset_name: str = "OpenCabinet",
    sequence_length: int = 8,
) -> list:
    """Build sliding-window samples from one annotation file. RoboCasa uses 'actions' and 'video_length'."""
    samples = []
    try:
        with open(os.path.join(data_root, ann_file), "r") as f:
            ann = json.load(f)
    except Exception as e:
        print(f"skip {ann_file}: {e}")
        return samples

    n_frames = ann.get("video_length") or len(ann.get("actions", []))
    if n_frames < sequence_length:
        return samples

    base_idx = np.arange(0, sequence_length, dtype=np.int64) * sequence_interval
    max_idx = np.ones_like(base_idx) * (n_frames - 1)

    for start_frame in range(0, n_frames, start_interval):
        idx = base_idx + start_frame
        idx = np.minimum(idx, max_idx)
        idx = idx.tolist()
        if len(idx) != sequence_length:
            continue

        sample = {
            "dataset_name": dataset_name,
            "ann_file": ann_file,
            "episode_id": ann.get("episode_id", -1),
            "frame_ids": idx,
            "states": np.array(ann["states"], dtype=np.float64)[idx[0] : idx[0] + 1],
            "actions": np.array(ann["actions"], dtype=np.float64)[idx],
        }
        samples.append(sample)

    return samples


def init_anns(dataset_root: str, data_dir: str) -> list:
    """List annotation JSON paths under dataset_root/data_dir (e.g. annotations/train)."""
    final_path = os.path.join(dataset_root, data_dir)
    if not os.path.isdir(final_path):
        return []
    return [
        os.path.join(data_dir, f)
        for f in os.listdir(final_path)
        if f.endswith(".json")
    ]


def init_sequences(
    data_root: str,
    ann_files: list,
    sequence_interval: int,
    start_interval: int,
    dataset_name: str,
    sequence_length: int,
    num_workers: int = 8,
) -> list:
    """Build samples from all annotation files in parallel."""
    samples = []
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {
            executor.submit(
                load_and_process_ann_file,
                data_root,
                af,
                sequence_interval,
                start_interval,
                dataset_name,
                sequence_length,
            ): af
            for af in ann_files
        }
        for future in tqdm(as_completed(futures), total=len(futures), desc=dataset_name):
            samples.extend(future.result())
    return samples


def main():
    parser = argparse.ArgumentParser(description="Prepare RoboCasa train/val JSON and state/action percentiles.")
    parser.add_argument("--data_root", type=str, default="data", help="Root under which task dirs live (e.g. data/OpenCabinet).")
    parser.add_argument("--dataset_names", type=str, default="OpenCabinet", help="Task dirs, '+' separated (e.g. OpenCabinet+CloseDrawer).")
    parser.add_argument("--out_dir", type=str, default="data/annotation_all", help="Output dir for train_all.json, val_all.json, *data.json.")
    parser.add_argument("--sequence_length", type=int, default=8, help="Frames per clip (act_seq_len).")
    parser.add_argument("--sequence_interval", type=int, default=1, help="Frame step within a clip.")
    parser.add_argument("--start_interval", type=int, default=1, help="Step between clip start frames.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for shuffling.")
    parser.add_argument("--num_workers", type=int, default=8, help="Parallel workers for scanning ann files.")
    args = parser.parse_args()

    random.seed(args.seed)
    dataset_names = [s.strip() for s in args.dataset_names.split("+") if s.strip()]

    for data_type in ["train", "val"]:
        data_dir = f"annotations/{data_type}"
        samples_all = []
        ann_files_all = []

        for dataset_name in dataset_names:
            data_root = os.path.join(args.data_root, dataset_name)
            ann_files = init_anns(data_root, data_dir)
            if not ann_files:
                print(f"[{data_type}] no annotations under {data_root}/{data_dir}, skip.")
                continue
            ann_files_all.extend(ann_files)
            samples = init_sequences(
                data_root,
                ann_files,
                args.sequence_interval,
                args.start_interval,
                dataset_name,
                args.sequence_length,
                num_workers=args.num_workers,
            )
            print(f"{dataset_name} {data_type}: {len(samples)} samples")
            samples_all.extend(samples)

        if not samples_all:
            print(f"[{data_type}] no samples, skip writing.")
            continue

        # state 1% / 99% per dimension (state = first-frame state of each clip)
        state_all = np.array([s["states"] for s in samples_all])
        state_all = state_all.reshape(-1, state_all.shape[-1])
        state_01 = np.percentile(state_all, 1, axis=0)
        state_99 = np.percentile(state_all, 99, axis=0)
        print(f"[{data_type}] state_01/99 shape: {state_01.shape}")

        # action 1% / 99% on raw action values
        action_all = np.array([s["actions"] for s in samples_all])
        action_all = action_all.reshape(-1, action_all.shape[-1])
        action_01 = np.percentile(action_all, 1, axis=0)
        action_99 = np.percentile(action_all, 99, axis=0)
        print(f"[{data_type}] action_01/99 shape: {action_01.shape}")

        for s in samples_all:
            del s["states"]
            del s["actions"]

        random.shuffle(samples_all)
        print(f"[{data_type}] step_num={len(samples_all)}, traj_num={len(ann_files_all)}")

        os.makedirs(args.out_dir, exist_ok=True)
        out_all = os.path.join(args.out_dir, f"{data_type}_all.json")
        with open(out_all, "w") as f:
            json.dump(samples_all, f, indent=2)
        print(f"Wrote {out_all}")

        stat = {
            "state_01": state_01.tolist(),
            "state_99": state_99.tolist(),
            "action_01": action_01.tolist(),
            "action_99": action_99.tolist(),
        }
        out_stat = os.path.join(args.out_dir, f"{data_type}data.json")
        with open(out_stat, "w") as f:
            json.dump(stat, f, indent=2)
        print(f"Wrote {out_stat}")

    print("Done. Set data_json_path to", os.path.abspath(args.out_dir), "and copy state_01/99, action_01/99 into VPP_robocasa_train.yaml.")


if __name__ == "__main__":
    main()
