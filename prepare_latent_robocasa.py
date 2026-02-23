"""
Prepare latent videos + annotation JSON for RoboCasa / OpenCabinet (LeRobot format),
following the structure used in `step1_prepare_latent.py`.

Input (fixed in this script):
    data/v1.0/pretrain/atomic/OpenCabinet/20250819/lerobot/
        data/chunk-*/episode_*.parquet
        videos/chunk-*/observation.images.robot0_*/episode_*.mp4

Output structure (relative to repo root):
    data/OpenCabinet/
        annotations/{train,val}/{episode_id}.json
        latent_videos/{train,val}/{episode_id}/{cam_id}.pt

Each JSON roughly matches the schema from `step1_prepare_latent.py`:
    {
        "task": "robot_trajectory_prediction",
        "texts": [...],             # 从 meta/episodes.jsonl 按 episode_index 读取 tasks[0]
        "videos": [...],            # relative paths back to original mp4s
        "episode_id": int,
        "video_length": int,        # 固定为 NUM_FRAMES（49）
        "latent_videos": [...],     # paths to encoded latents
        "states": [...],            # downsampled observation.state
        "actions": [...],           # downsampled action
    }
"""

import os
import json
from pathlib import Path
import argparse

import numpy as np
import pandas as pd
import torch
from diffusers.models import AutoencoderKLTemporalDecoder
import mediapy
from rich import print
from tqdm import tqdm

# Camera order: 3 views, similar于 xbot 的多机位
CAM_KEYS = [
    "observation.images.robot0_agentview_right",
    "observation.images.robot0_agentview_left",
]

# 不论原始轨迹多长，uniform sample 到固定帧数
NUM_FRAMES = 49

# 固定随机选 5 个 episode 作为 val，其余为 train（可复现）
NUM_VAL_SAMPLES = 5
VAL_SPLIT_SEED = 42

VAE_PATH = "stabilityai/stable-video-diffusion-img2vid"


def load_episode_texts(meta_dir: Path) -> dict[int, str]:
    """从 meta/episodes.jsonl 按 episode_index 加载 text（取 tasks[0]）"""
    path = meta_dir / "episodes.jsonl"
    if not path.exists():
        return {}
    episode_texts = {}
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            idx = rec["episode_index"]
            tasks = rec.get("tasks") or []
            episode_texts[idx] = tasks[0]
    return episode_texts


def encode_video_to_latent(vae, frames: np.ndarray, batch_size: int = 64) -> torch.Tensor:
    """
    frames: (T, H, W, C) uint8 in [0, 255]
    return: (T, C', H', W') latent tensor on CPU
    """
    assert frames.ndim == 4 and frames.shape[-1] == 3

    # (T, H, W, C) -> (T, C, H, W), float in [-1, 1]
    x = (
        torch.tensor(frames)
        .permute(0, 3, 1, 2)
        .float()
        .to("cuda")
        / 255.0
        * 2
        - 1
    )

    # 统一 resize 到 256x256，和 xbot 的处理一致
    x = torch.nn.functional.interpolate(
        x, size=(256, 256), mode="bilinear", align_corners=False
    )

    latents = []
    with torch.no_grad():
        for i in range(0, len(x), batch_size):
            batch = x[i : i + batch_size]
            latent = (
                vae.encode(batch)
                .latent_dist.sample()
                .mul_(vae.config.scaling_factor)
                .cpu()
            )
            latents.append(latent)

    return torch.cat(latents, dim=0)


def main(task_name: str):
    # 在 data_raw/{task_name} 下自动搜索唯一的 lerobot 子目录
    task_root = Path("data_raw") / task_name
    candidates = list(task_root.glob("**/lerobot"))
    if not candidates:
        raise FileNotFoundError(f"No 'lerobot' directory found under {task_root}")
    if len(candidates) > 1:
        raise RuntimeError(
            f"Found multiple 'lerobot' directories under {task_root}: {candidates}. "
            "Please keep only one or refine this script."
        )
    data_root = candidates[0]
    # 输出到 data/{task_name}
    output_root = Path("data") / task_name
    output_root.mkdir(parents=True, exist_ok=True)

    parquet_root = data_root / "data"
    video_root = data_root / "videos"
    meta_dir = data_root / "meta"

    # 按 episode_index 加载 text（meta/episodes.jsonl）
    episode_texts = load_episode_texts(meta_dir)
    print(f"[bold green]Loaded {len(episode_texts)} episode texts from meta/episodes.jsonl[/bold green]")

    # 收集所有 episode parquet 文件
    parquet_files = sorted(parquet_root.glob("chunk-*/episode_*.parquet"))
    print(f"[bold green]Found {len(parquet_files)} episodes[/bold green]")

    # 固定随机选 NUM_VAL_SAMPLES 个作为 val
    rng = np.random.default_rng(VAL_SPLIT_SEED)
    all_indices = np.arange(len(parquet_files))
    rng.shuffle(all_indices)
    val_indices = set(all_indices[:NUM_VAL_SAMPLES].tolist())
    print(f"[bold green]Val indices (fixed random): {sorted(val_indices)}[/bold green]")

    # 初始化 VAE
    print("[bold yellow]Loading VAE...[/bold yellow]")
    vae = AutoencoderKLTemporalDecoder.from_pretrained(
        VAE_PATH, subfolder="vae"
    ).to("cuda")
    vae.eval()

    failed_num = 0
    success_num = 0

    for file_num, parquet_path in enumerate(tqdm(parquet_files)):
        # 统一编号规则：固定随机选 5 个为 val
        anno_ind_all = file_num
        data_type = "val" if anno_ind_all in val_indices else "train"

        try:
            df = pd.read_parquet(parquet_path)
        except Exception as e:
            print(f"[red]Failed to read {parquet_path}: {e}[/red]")
            failed_num += 1
            continue

        # 提取状态和动作
        try:
            states_all = np.stack(df["observation.state"].to_numpy(), axis=0)
            action_all = np.stack(df["action"].to_numpy(), axis=0)
        except Exception as e:
            print(f"[red]Failed to parse state/action from {parquet_path}: {e}[/red]")
            failed_num += 1
            continue

        # 根据 episode index 取 text（parquet 顺序即 episode_index）
        text = episode_texts.get(anno_ind_all, "open cabinet pretrain")

        n_orig = len(action_all)
        # 均匀采样到 NUM_FRAMES 帧的索引
        if n_orig >= NUM_FRAMES:
            frame_indices = np.linspace(0, n_orig - 1, NUM_FRAMES, dtype=int)
        else:
            # 不足 NUM_FRAMES 时用最后一帧填充
            frame_indices = np.concatenate([
                np.arange(n_orig),
                np.full(NUM_FRAMES - n_orig, n_orig - 1),
            ])

        actions = action_all[frame_indices]
        states = states_all[frame_indices]
        anno_ind = anno_ind_all

        # episode 基础信息
        chunk_name = parquet_path.parent.name  # e.g. chunk-000
        episode_stem = parquet_path.stem  # e.g. episode_000000

        # 准备输出目录（相对于 data/{task_name}）
        latent_episode_dir = (
            output_root / "latent_videos" / data_type / str(anno_ind)
        )
        latent_episode_dir.mkdir(parents=True, exist_ok=True)

        ann_dir = output_root / "annotations" / data_type
        ann_dir.mkdir(parents=True, exist_ok=True)

        videos_info = []
        latent_info = []

        for cam_id, cam_key in enumerate(CAM_KEYS):
            # 原始视频相对路径（相对于 data/{task_name}）
            # 通过 data_root / "videos" 相对于 output_root 计算
            rel_videos_root = os.path.relpath(data_root / "videos", output_root)
            video_rel_path = (
                Path(rel_videos_root) / chunk_name / cam_key / f"{episode_stem}.mp4"
            )
            videos_info.append({"video_path": str(video_rel_path)})

            # 读取视频并 uniform 采样到 NUM_FRAMES
            video_abs_path = (
                video_root / chunk_name / cam_key / f"{episode_stem}.mp4"
            )
            try:
                frames = mediapy.read_video(str(video_abs_path))
            except Exception as e:
                print(f"[red]Failed to read video {video_abs_path}: {e}[/red]")
                failed_num += 1
                break

            frames = np.array(frames).astype(np.uint8)

            # 与 states/actions 一致的 uniform 采样到 NUM_FRAMES 帧
            frames = frames[frame_indices]

            # 编码为 latent
            latent = encode_video_to_latent(vae, frames)
            torch.save(latent, latent_episode_dir / f"{cam_id}.pt")

            latent_rel_path = (
                Path("latent_videos")
                / data_type
                / str(anno_ind)
                / f"{cam_id}.pt"
            )
            latent_info.append({"latent_video_path": str(latent_rel_path)})

        else:
            # 如果没有在中途 break（视频读取失败），才写 JSON
            info = {
                "task": "robot_trajectory_prediction",
                "texts": [text],
                "videos": videos_info,
                "episode_id": int(anno_ind),
                "video_length": int(len(actions)),
                "latent_videos": latent_info,
                "states": states.tolist(),
                "actions": actions.tolist(),
            }

            ann_path = ann_dir / f"{anno_ind}.json"
            with open(ann_path, "w") as f:
                json.dump(info, f, indent=2)

            success_num += 1
            print(
                "text",
                text,
                "num",
                file_num,
                "total_num",
                len(parquet_files),
            )

    print(
        f"[bold green]Done. success_num={success_num}, failed_num={failed_num}[/bold green]"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task_name",
        type=str,
        default="OpenCabinet",
        help="Task name, used to resolve data_raw/{task_name}/.../lerobot and data/{task_name}",
    )
    args = parser.parse_args()
    main(task_name=args.task_name)

