"""
Evalute VPP on ONE task in RoboCasa Environment.

Args:
- video_model_path: path to the video model checkpoint
- action_model_path: path to the action model checkpoint
- clip_model_path: path to the clip model checkpoint
- json_path: path to the json file that contains the data to evaluate


Expected Json Structure:
[
    {
        "task_name": "CloseBlenderLid",
        "dataset_dir": "data_raw/v1.0/pretrain/atomic/CloseBlenderLid/20250822/lerobot",
        "episode_id": 18
    },
    {
        "task_name": "CloseBlenderLid",
        "dataset_dir": "data_raw/v1.0/pretrain/atomic/CloseBlenderLid/20250822/lerobot",
        "episode_id": 25
    },
]

Output Structure:
- {output_dir}/
    - {task_name}/
        - {episode_id}.mp4
        - result.json
"""
import json
from argparse import ArgumentParser
import numpy as np
from pathlib import Path
import robosuite
import robocasa.utils.lerobot_utils as LU
from robocasa.scripts.dataset_scripts.playback_dataset import reset_to
import imageio.v3 as iio
from rich import print
from tqdm import tqdm
import torch
import os
from hydra.utils import instantiate
from pytorch_lightning import seed_everything
from diffusers.models import AutoencoderKLTemporalDecoder


CAMERAS = ["robot0_agentview_left", "robot0_agentview_center", "robot0_agentview_right"]

STATE_01 = np.array([-0.8471073344983073, -5.545685094005817, 0.6999459578623527, 0.0, 0.0, -1.0, 5.319110687196371e-07, -0.18826553858304707, -0.6094472837410796, -0.14869284647573502, -0.997725191116333, -0.629423668384552, -0.7791548717021942, 0.0009995199600234628, -3.0091797774880364e-05, -0.04055586245844699], dtype=np.float32)
STATE_99 = np.array([6.711419918072105, -0.3305895016132213, 0.703836780923211, 0.0, 0.0, 1.0, 1.0, 0.6232723876414562, 0.6056495006911888, 0.8914906390281283, 0.9969765591621399, 0.764727573394776, 0.7703708636760713, 0.7068612802028656, 0.04056435594193672, 3.874594788977785e-05], dtype=np.float32)
ACTION_01 = np.array([0.0, 0.0, 0.0, 0.0, -1.0, -1.0, -1.0, -1.0, -0.32857142857142857, -0.4514285714285714, -0.38, -1.0], dtype=np.float32)
ACTION_99 = np.array([0.0, 0.0, 0.0, 0.0, -1.0, 1.0, 1.0, 1.0, 0.3514285714285714, 0.44857142857142857, 0.3857142857142858, 1.0], dtype=np.float32)


def make_env(dataset_dir):
    env_meta = LU.get_env_metadata(dataset_dir)
    env_kwargs = env_meta["env_kwargs"]
    env_kwargs["env_name"] = env_meta["env_name"]
    env_kwargs["has_renderer"] = False
    env_kwargs["renderer"] = "mjviewer"
    env_kwargs["has_offscreen_renderer"] = True
    env_kwargs["use_camera_obs"] = False
    return robosuite.make(**env_kwargs)



def prepare_inputs(env, lang, vae):
    # zero action to get initial observation
    zero_action = {
        "action.gripper_close": np.array([0.0], dtype=np.float32),      # 闭合
        "action.end_effector_position": np.array([0.0, 0.0, 0.0], np.float32),
        "action.end_effector_rotation": np.array([0.0, 0.0, 0.0], np.float32),
        "action.base_motion": np.array([0.0, 0.0, 0.0, 0.0], np.float32),
        "action.control_mode": np.array([1.0], dtype=np.float32),
    }
    obs, _, _, _ = env.step(zero_action)
    
    image = obs["video.robot0_agentview_left"]
    # encode image to latent
    image = torch.from_numpy(image).unsqueeze(0).permute(0, 3, 1, 2).float().to("cuda") / 255.0 * 2 - 1
    image = torch.nn.functional.interpolate(image, size=(256, 256), mode='bilinear', align_corners=False)
    latent = vae.encode(image).latent_dist.sample().mul_(vae.config.scaling_factor)
    
    state = np.concatenate([obs['state.base_position'], obs['state.base_rotation'], obs['state.end_effector_position_relative'], obs['state.end_effector_rotation_relative'], obs['state.gripper_qpos']], axis=0)
    state = state.unsqueeze(0)
    state = normalize(state, STATE_01, STATE_99)
    
    inputs = dict(
        obs=dict(
            rgb_obs=dict(
                rgb_static=latent,
                rgb_gripper=latent,
            ),
            state_obs=state,
        ),
        goal=dict(
            lang_text=lang,
        ),
    )
    return inputs


def denormalize(data, data_min, data_max):
    # from [-1, 1] to [min, max]
    return data * (data_max - data_min) / 2 + data_min
    
    
def normalize(data, data_min, data_max, clip_min=-1, clip_max=1, eps=1e-8):
    ndata = 2 * (data - data_min) / (data_max - data_min + eps) - 1
    return np.clip(ndata, clip_min, clip_max).astype(np.float32)


def main(cfg):
    # Initialize Model
    torch.cuda.set_device(cfg.device)
    seed_everything(0, workers=True)
    state_dict = torch.load(cfg.action_model_path, map_location='cpu')
    model = instantiate(cfg.model)
    model.load_state_dict(state_dict['model'],strict = True)
    model.freeze()
    model = model.cuda(cfg.device)
    print(cfg.num_sampling_steps, cfg.sampler_type, cfg.multistep, cfg.sigma_min, cfg.sigma_max, cfg.noise_scheduler)
    model.num_sampling_steps = cfg.num_sampling_steps
    model.sampler_type = cfg.sampler_type
    model.multistep = cfg.multistep
    if cfg.sigma_min is not None:
        model.sigma_min = cfg.sigma_min
    if cfg.sigma_max is not None:
        model.sigma_max = cfg.sigma_max
    if cfg.noise_scheduler is not None:
        model.noise_scheduler = cfg.noise_scheduler
    model.process_device()
    model.eval()
    
    # Initialize VAE
    vae = AutoencoderKLTemporalDecoder.from_pretrained("stabilityai/stable-video-diffusion-img2vid", subfolder="vae").to("cuda")
    
    # Initialize Environment
    with open(cfg.json_path, "r") as f:
        data = json.load(f)
    
    task_name = data[0]["task_name"]
    dataset_dir = data[0]["dataset_dir"]
    episode_ids = [item["episode_id"] for item in data]
    os.makedirs(f"{cfg.output_dir}/{task_name}", exist_ok=True)
    
    env = make_env(dataset_dir)
    
    for episode_id in tqdm(episode_ids, desc="Evaluating"):
        print(f"Evaluating episode {episode_id} of {task_name}")
        initial_state = dict(
            states=LU.get_episode_states(dataset_dir, episode_id),
            model=LU.get_episode_model_xml(dataset_dir, episode_id),
            ep_meta=json.dumps(LU.get_episode_meta(dataset_dir, episode_id)),
        )
        reset_to(env, initial_state)
        video_writer = iio.get_writer(f"{cfg.output_dir}/{task_name}/{episode_id}.mp4", fps=20)
        
        # Start Rollout
        lang = LU.get_episode_meta(dataset_dir, episode_id)['lang']
        assert lang is not None
        
        success = False
        max_inferences = 10
        
        for infer in tqdm(range(max_inferences), desc="Infering"):
            inputs = prepare_inputs(env, lang, vae)
            with torch.no_grad():
                actions = model.eval_forward(**inputs).squeeze(0).cpu().numpy() # (action_window_size, action_dim)
                for action in actions:
                    action = denormalize(action, ACTION_01, ACTION_99)
                    _, _, _, _, info = env.step(action)
                    
                    # save frame
                    frame = []
                    for camera in CAMERAS:
                        image = env.sim.render(
                            height=480, width=832, camera_name=camera
                        )[::-1]
                        frame.append(image)
                    frame = np.concatenate(frame, axis=1)
                    video_writer.append_data(frame)
                    
                    # check success
                    if info['success']:
                        success = True
                        break
        
        if success:
            print(f"[bold green]Success[/bold green]")
        else:
            print(f"[bold red]Fail[/bold red]")
        
        

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--video_model_path", type=str, required=True)
    parser.add_argument("--action_model_path", type=str, required=True)
    parser.add_argument("--clip_model_path", type=str, required=True)
    parser.add_argument("--json_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()
    
    os.environ["PL_TORCH_DISTRIBUTED_BACKEND"] = "gloo"
    # Set CUDA device IDs
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
    from hydra import compose, initialize
    from omegaconf import OmegaConf
    
    with initialize(config_path="../policy_conf", job_name="robocasa_evaluate.yaml"):
        cfg = compose(config_name="robocasa_evaluate.yaml")
    cfg.model.pretrained_model_path = args.video_model_path
    cfg.action_model_path = args.action_model_path
    cfg.model.text_encoder_path = args.clip_model_path
    cfg.json_path = args.json_path
    cfg.output_dir = args.output_dir
    
    main(cfg)