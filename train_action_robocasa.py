import torch.nn as nn
# import cv2
from torchvision.utils import save_image

import logging
from pathlib import Path
import sys
from typing import List, Union
import os
import wandb
from time import time
# This is for using the locally installed repo clone when using slurm
sys.path.insert(0, Path(__file__).absolute().parents[1].as_posix())
import hydra
from omegaconf import DictConfig, ListConfig, OmegaConf
from accelerate import Accelerator
import torch
from glob import glob
from copy import deepcopy
from collections import OrderedDict

# from pytorch_lightning import Callback, LightningModule, seed_everything, Trainer

# import mdt.models.mdt_agent as models_m
# from mdt.utils.utils import (
#     get_git_commit_hash,
#     get_last_checkpoint,
#     initialize_pretrained_weights,
#     print_system_env_info,
# )

from policy_models.utils.utils import (
    get_git_commit_hash,
    get_last_checkpoint,
    initialize_pretrained_weights,
    print_system_env_info,
)

#from torch.nn.parallel import DistributedDataParallel as DDP

#################################################################################
#                             Training Helper Functions                         #
#################################################################################

@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """
    Step the EMA model towards the current model.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        name = name.replace("module.", "")
        # TODO: Consider applying only to params that require_grad to avoid small numerical changes of pos_embed
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


def requires_grad(model, flag=True):
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag


def create_logger(logging_dir):
    """
    Create a logger that writes to a log file and stdout.
    """
    logging.basicConfig(
        level=logging.INFO,
        format='[\033[34m%(asctime)s\033[0m] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[logging.StreamHandler(), logging.FileHandler(f"{logging_dir}/log.txt")]
    )
    logger = logging.getLogger(__name__)
    return logger



#################################################################################
#                                  Training Loop                                #
#################################################################################


@hydra.main(config_path="./policy_conf", config_name="VPP_robocasa_train")
def train(cfg: DictConfig) -> None:
    os.environ['HYDRA_FULL_ERROR'] = '1'
    accelerator = Accelerator()
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."
    device = accelerator.device
    torch.set_float32_matmul_precision('medium')
    if accelerator.is_main_process:
        os.makedirs(cfg.log_dir, exist_ok=True)  # Make results folder (holds all experiment subfolders)
        from datetime import datetime
        uuid = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        experiment_dir = "robocasa"
        checkpoint_dir = f"{experiment_dir}/checkpoints"  # Stores saved model checkpoints
        os.makedirs(checkpoint_dir, exist_ok=True)
        logger = create_logger(experiment_dir)
        logger.info(f"Experiment directory created at {experiment_dir}")
        logger.info(f"Training with the following config:\n{OmegaConf.to_yaml(cfg)}")
        logger.info(f"Global batch size {cfg.batch_size} num_processes ({accelerator.num_processes})")
        wandb.init(
            project=cfg.benchmark_name,
            name=f"{cfg.uuid}_{uuid}",
        )


    # load datasets (RoboCasa: single camera, same keys as xbot for model compatibility)
    from policy_models.datasets.robocasa_dataset import Dataset_robocasa
    dataset_args = cfg.dataset_args
    train_dataset = Dataset_robocasa(dataset_args, mode="train")
    test_dataset = Dataset_robocasa(dataset_args, mode="val")
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=dataset_args.batch_size,
        shuffle=dataset_args.shuffle,
        num_workers=dataset_args.num_workers,
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, 
        batch_size=dataset_args.test_batch_size,
        shuffle=dataset_args.shuffle,
        num_workers=2,
    )

    # load model
    print("accelerator.device",str(accelerator.device))
    cfg.model.device = str(accelerator.device)
    model = hydra.utils.instantiate(cfg.model)
    if "pretrain_chk" in cfg:
        initialize_pretrained_weights(model, cfg)

    if cfg.use_ckpt_path:
        state_dict = torch.load(cfg.ckpt_path, map_location='cpu')
        print('load_from_ckpt:',cfg.ckpt_path)
        model.load_state_dict(state_dict['model'])

    model = model.to(accelerator.device)
    model.process_device()
    if accelerator.is_main_process:
        logger.info(f"DiT Parameters: {sum(p.numel() for p in model.parameters()):,}")

    opt = model.configure_optimizers()["optimizer"]
    Ir_scheduler = model.configure_optimizers()["lr_scheduler"]["scheduler"]

    model.on_train_start()
    if accelerator.is_main_process:
        logger.info(f"model parameter init")
    # ema = deepcopy(model).to(device)  # Create an EMA of the model for use after training
    # requires_grad(ema, False)
    # update_ema(ema, model, decay=0)  # Ensure EMA is initialized with synced weights
    # ema.eval()
    model.train()

    # accelerator prepare for everything
    model, opt, loader, test_loader = accelerator.prepare(model, opt, train_loader, test_loader)


    ########################## Start Training !!! ##########################
    train_steps = 0
    log_steps = 0
    running_loss = 0
    start_time = time()
    eval_batch = None
    best_eval_loss = 1e8

    if accelerator.is_main_process:
        logger.info(f"Training for {cfg.max_epochs} epochs...")

    running_loss = 0
    for epoch in range(cfg.max_epochs):
        if accelerator.is_main_process:
            logger.info(f"Beginning epoch {epoch}...")
        
        for idx, data_batch in enumerate(loader):
            with accelerator.autocast():
                loss = model(data_batch)
            opt.zero_grad()
            accelerator.backward(loss)
            opt.step()
            Ir_scheduler.step()
            running_loss += loss
            log_steps += 1
            train_steps += 1
            if train_steps % cfg.log_every == 0:
                torch.cuda.synchronize()
                end_time = time()
                steps_per_sec = log_steps / (end_time - start_time)
                avg_loss = (running_loss / cfg.log_every).item()
                if accelerator.is_main_process:
                    logger.info(f"(step={train_steps:07d}) Train Loss: {avg_loss:.6f}, Steps/Sec: {steps_per_sec:.2f}")
                    wandb.log({"train_loss": avg_loss, "step": train_steps})
                running_loss = 0
                log_steps = 0
                start_time = time()
        
        ######################## Evaluate at end of a epoch ########################
        val_log_step = 0
        model.eval()
        if accelerator.is_main_process:
            logger.info(f"Finished training epoch {epoch}")
            logger.info(f"started validation epoch {epoch}")
        total_val_loss = 0
        for test_batch in test_loader:
            print_it = (val_log_step == 0 and accelerator.is_main_process)
            val_loss = model.module.validation_step(test_batch, print_it) if accelerator.num_processes > 1 else model.validation_step(test_batch, print_it)
            total_val_loss += accelerator.gather_for_metrics(val_loss["validation_loss"]).mean().item()
            val_log_step += 1
        model.train()

        if accelerator.is_main_process:
            total_val_loss = total_val_loss / val_log_step
            logger.info(f"Validation Loss: {total_val_loss:.6f}")
            wandb.log({"val_loss": total_val_loss, "step": train_steps})
        
            # log_steps = 0
            checkpoint = {
                "model": model.module.state_dict() if accelerator.num_processes > 1 else model.state_dict(),
                "args": cfg,
            }
            if total_val_loss < best_eval_loss or epoch % 20 == 0:
                checkpoint_path = f"{checkpoint_dir}/{train_steps:07d}_{total_val_loss:.6f}.pt"
                torch.save(checkpoint, checkpoint_path)
                logger.info(f"Saved checkpoint to {checkpoint_path}")
                best_eval_loss = total_val_loss
            last_path = f"{checkpoint_dir}/last.pt"
            torch.save(checkpoint, last_path)


if __name__ == "__main__":
    # os.environ["PL_TORCH_DISTRIBUTED_BACKEND"] = "gloo"
    # Set CUDA device IDs
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    # os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5"
    print(torch.cuda.is_available())
    print(torch.cuda.device_count())
    os.environ["TOKENIZERS_PARALLELISM"] = 'True'
    train()

# Run RoboCasa action policy training (from repo root):
#   CUDA_VISIBLE_DEVICES=0 python train_action_robocasa.py
# Multi-GPU:
#   CUDA_VISIBLE_DEVICES=0,1 accelerate launch train_action_robocasa.py