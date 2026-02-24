# Copyright (2024) Bytedance Ltd. and/or its affiliates
# RoboCasa dataset: single camera only. rgb_static = cam0, rgb_gripper = duplicate of cam0.

import json
import os
import torch
from torch.utils.data import Dataset
import numpy as np


def _load_latent_video(video_path: str, frame_ids):
    with open(video_path, "rb") as f:
        video_tensor = torch.load(f)
        video_tensor.requires_grad = False
    frame_ids = np.asarray(frame_ids)
    assert (frame_ids < video_tensor.size(0)).all() and (frame_ids >= 0).all()
    return video_tensor[frame_ids]


def normalize_bound(data, data_min, data_max, clip_min=-1, clip_max=1, eps=1e-8):
    ndata = 2 * (data - data_min) / (data_max - data_min + eps) - 1
    return np.clip(ndata, clip_min, clip_max).astype(np.float32)


class Dataset_robocasa(Dataset):
    """
    Dataset for RoboCasa (prepare_latent_robocasa output). Single camera only.
    Expects data_json_path/{mode}_all.json with samples:
      { "ann_file": "annotations/train/0.json", "frame_ids": [0,1,...], "dataset_name": "OpenCabinet" }
    data_root_path + dataset_name = per-sample root (video_dir).
    Annotation JSON: texts, states, actions, latent_videos (at least cam0).
    rgb_static = cam0, rgb_gripper = duplicate of cam0 so model input dim is unchanged.
    """

    def __init__(self, args, mode="val"):
        super().__init__()
        self.args = args
        self.mode = mode
        data_json_path = args.data_json_path
        data_root_path = args.data_root_path

        path = f"{data_json_path}/{mode}_all.json"
        with open(path, "r") as f:
            self.samples = json.load(f)
        self.video_path = [os.path.join(data_root_path, s["dataset_name"]) for s in self.samples]

        self.a_min = np.array(args.action_01, dtype=np.float32)[None, :]
        self.a_max = np.array(args.action_99, dtype=np.float32)[None, :]
        self.s_min = np.array(args.state_01, dtype=np.float32)[None, :]
        self.s_max = np.array(args.state_99, dtype=np.float32)[None, :]
        print(f"RoboCasa {mode}: {len(self.samples)} samples, action min/max shape {self.a_min.shape}")

    def __len__(self):
        return len(self.samples)

    def _get_frames(self, label, frame_ids, cam_id, video_dir):
        pre_encode = getattr(self.args, "pre_encode", True)
        assert pre_encode, "RoboCasa dataset expects pre_encode (latent) data"
        item = label["latent_videos"][cam_id]
        rel_path = item["latent_video_path"]
        video_path = os.path.join(video_dir, rel_path)
        if not os.path.exists(video_path):
            video_path = video_path.replace("latent_videos", "latent_videos_svd")
        return _load_latent_video(video_path, frame_ids)

    def _process_action(self, label, frame_ids):
        num_frames = getattr(self.args, "num_frames", len(frame_ids))
        frame_ids = frame_ids[: int(num_frames)]
        states = np.array(label["states"], dtype=np.float32)[frame_ids]
        actions = np.array(label["actions"], dtype=np.float32)[frame_ids]
        state = states[0:1]
        action_scaled = normalize_bound(actions, self.a_min, self.a_max)
        state_scaled = normalize_bound(state, self.s_min, self.s_max)
        return torch.from_numpy(action_scaled).float(), torch.from_numpy(state_scaled).float()

    def __getitem__(self, index, cam_id=None, return_video=False):
        sample = self.samples[index]
        video_dir = self.video_path[index]
        ann_file = sample["ann_file"]
        if not os.path.isabs(ann_file):
            ann_file = os.path.join(video_dir, ann_file)
        frame_ids = sample["frame_ids"]

        with open(ann_file, "r") as f:
            label = json.load(f)

        data = {}
        data["actions"], data["state_obs"] = self._process_action(label, frame_ids)
        data["lang_text"] = label["texts"][0]
        data["ann_file"] = ann_file
        data["frame_ids"] = frame_ids

        if not label.get("latent_videos"):
            raise ValueError(f"No latent_videos in {ann_file}")

        obs_frame_idx = frame_ids[0]
        static_latent = self._get_frames(label, obs_frame_idx, 0, video_dir)
        if static_latent.dim() == 3:
            static_latent = static_latent.unsqueeze(0)  # (1, C, H, W)
        else:
            static_latent = static_latent[:1]
        gripper_latent = static_latent.clone()

        data["rgb_obs"] = {
            "rgb_static": static_latent,
            "rgb_gripper": gripper_latent,
        }
        return data
