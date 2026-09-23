import argparse
import math
import os
import pickle
import random
import sys
from contextlib import contextmanager
from collections import deque
from datetime import timedelta
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from omegaconf import DictConfig, ListConfig, OmegaConf
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm

from train_eval_internvl3_mlp_multitask import (
    DEFAULT_TASKS,
    MultiTaskInternVL3MLPPolicy,
    parse_csv,
    parse_task_descriptions,
)


SVD_ROOT = Path(os.environ.get("SVD_ROOT", "generative-models"))
if SVD_ROOT.exists():
    sys.path.insert(0, str(SVD_ROOT.resolve()))
try:
    from sgm.util import instantiate_from_config
except ImportError:
    instantiate_from_config = None


def setup_distributed():
    if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
        return False, 0, 1, 0
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl", timeout=timedelta(hours=6))
    return True, rank, world_size, local_rank


def cleanup_distributed():
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def is_main_process():
    return not dist.is_available() or not dist.is_initialized() or dist.get_rank() == 0


def unwrap_model(model):
    return model.module if isinstance(model, DDP) else model


class LossLogger:
    def __init__(self, log_dir, enabled=True):
        self.enabled = enabled
        self.writer = None
        self.csv_file = None
        if not enabled:
            return
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(str(self.log_dir))
        self.csv_file = open(self.log_dir / "loss.csv", "w", buffering=1)
        self.csv_file.write("step,loss,action_loss,svd_loss,lr\n")

    def log(
        self,
        step,
        loss,
        action_loss,
        svd_loss,
        lr,
        avg_loss=None,
        avg_action_loss=None,
        avg_svd_loss=None,
    ):
        if not self.enabled:
            return
        values = {
            "loss/total": loss,
            "loss/action": action_loss,
            "loss/svd": svd_loss,
            "train/lr": lr,
        }
        if avg_loss is not None:
            values.update(
                {
                    "loss_avg/total": avg_loss,
                    "loss_avg/action": avg_action_loss,
                    "loss_avg/svd": avg_svd_loss,
                }
            )
        for key, value in values.items():
            self.writer.add_scalar(key, value, step)
        self.csv_file.write(f"{step},{loss},{action_loss},{svd_loss},{lr}\n")

    def close(self):
        if self.writer is not None:
            self.writer.close()
        if self.csv_file is not None:
            self.csv_file.close()


def plot_loss_curves(log_dir, output_path=None):
    csv_path = Path(log_dir) / "loss.csv"
    if output_path is None:
        output_path = Path(log_dir) / "loss_curve.png"
    else:
        output_path = Path(output_path)
    if not csv_path.exists():
        return
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"Skip plotting loss curves because matplotlib is unavailable: {exc}")
        return

    data = np.genfromtxt(csv_path, delimiter=",", names=True)
    if data.size == 0:
        return
    if data.shape == ():
        data = np.array([data], dtype=data.dtype)

    steps = data["step"]
    plt.figure(figsize=(10, 6))
    plt.plot(steps, data["loss"], label="total")
    plt.plot(steps, data["action_loss"], label="action")
    plt.plot(steps, data["svd_loss"], label="svd")
    plt.xlabel("step")
    plt.ylabel("loss")
    plt.title("Joint VLA + SVD Training Loss")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=160)
    plt.close()


class JointMetaWorldChunkDataset(Dataset):
    def __init__(
        self,
        data_root,
        task_names,
        task_descriptions,
        max_episodes=25,
        action_horizon=4,
        svd_num_frames=4,
        svd_image_size=128,
        include_svd=True,
    ):
        self.samples = []
        self.task_names = list(task_names)
        self.task_descriptions = task_descriptions
        self.action_horizon = action_horizon
        self.svd_num_frames = svd_num_frames
        self.svd_image_size = svd_image_size
        self.include_svd = include_svd
        self.act_dim = None
        self.obs_dim = 8

        for task_name in self.task_names:
            pkl_path = Path(data_root) / task_name / "expert_demos.pkl"
            if not pkl_path.exists():
                raise FileNotFoundError(f"Missing dataset for {task_name}: {pkl_path}")
            with open(pkl_path, "rb") as f:
                images_list, observations_list, actions_list, _ = pickle.load(f)

            for episode_idx, (images, observations, actions) in enumerate(
                zip(
                    images_list[:max_episodes],
                    observations_list[:max_episodes],
                    actions_list[:max_episodes],
                )
            ):
                self.act_dim = actions.shape[1] if self.act_dim is None else self.act_dim
                max_start = min(
                    len(images) - svd_num_frames,
                    len(actions) - action_horizon,
                    len(observations) - 1,
                )
                for start in range(max_start + 1):
                    self.samples.append(
                        {
                            "task_name": task_name,
                            "episode_idx": episode_idx,
                            "start": start,
                            "images": images,
                            "observations": observations,
                            "actions": actions,
                        }
                    )

        if not self.samples:
            raise ValueError("No valid joint VLA/SVD samples found.")
        print(
            "Loaded joint dataset: "
            f"tasks={self.task_names}, samples={len(self.samples)}, "
            f"action_horizon={action_horizon}, svd_num_frames={svd_num_frames}"
        )

    def __len__(self):
        return len(self.samples)

    def action_stats(self):
        total = np.zeros((self.act_dim,), dtype=np.float64)
        total_sq = np.zeros((self.act_dim,), dtype=np.float64)
        count = 0
        for sample in self.samples:
            start = sample["start"]
            actions = sample["actions"][start : start + self.action_horizon].astype(
                np.float64
            )
            total += actions.sum(axis=0)
            total_sq += np.square(actions).sum(axis=0)
            count += actions.shape[0]
        mean = total / max(count, 1)
        var = np.maximum(total_sq / max(count, 1) - np.square(mean), 1e-6)
        std = np.sqrt(var)
        return (
            torch.tensor(mean, dtype=torch.float32).view(1, 1, self.act_dim),
            torch.tensor(std, dtype=torch.float32).view(1, 1, self.act_dim),
        )

    def action_bounds(self, q_low=0.01, q_high=0.99):
        chunks = []
        for sample in self.samples:
            start = sample["start"]
            chunks.append(
                sample["actions"][start : start + self.action_horizon].astype(
                    np.float32
                )
            )
        actions = np.concatenate(chunks, axis=0)
        low = np.quantile(actions, q_low, axis=0).astype(np.float32)
        high = np.quantile(actions, q_high, axis=0).astype(np.float32)
        narrow = (high - low) < 1e-6
        high[narrow] = low[narrow] + 1e-6
        return (
            torch.tensor(low, dtype=torch.float32).view(1, 1, self.act_dim),
            torch.tensor(high, dtype=torch.float32).view(1, 1, self.act_dim),
        )

    @staticmethod
    def _proprio(obs):
        obs = torch.tensor(obs, dtype=torch.float32)
        state_t = obs[:18]
        state_t1 = obs[18:36]
        return torch.cat([state_t[:4], state_t1[:4]], dim=0)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        start = sample["start"]
        task_name = sample["task_name"]
        images = sample["images"]
        observations = sample["observations"]
        actions = sample["actions"]

        vla_img = torch.tensor(images[start], dtype=torch.float32) / 255.0
        vla_obs = self._proprio(observations[start])
        action_chunk = torch.tensor(
            actions[start : start + self.action_horizon], dtype=torch.float32
        )

        task_text = self.task_descriptions.get(task_name, task_name)
        item = {
            "vla_img": vla_img,
            "vla_obs": vla_obs,
            "action_chunk": action_chunk,
            "task_name": task_name,
            "task_text": task_text,
        }
        if self.include_svd:
            svd_frames = torch.tensor(
                images[start : start + self.svd_num_frames, -3:],
                dtype=torch.float32,
            ) / 255.0
            svd_frames = F.interpolate(
                svd_frames,
                size=(self.svd_image_size, self.svd_image_size),
                mode="bilinear",
                align_corners=False,
            )
            item["svd_video"] = svd_frames * 2.0 - 1.0
        return item


def cosine_beta_schedule(num_timesteps, s=0.008):
    steps = num_timesteps + 1
    x = torch.linspace(0, num_timesteps, steps, dtype=torch.float32)
    alphas_cumprod = torch.cos(((x / num_timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1.0 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return betas.clamp(0.0001, 0.9999)


def extract_schedule_value(values, timesteps, target_shape):
    out = values.gather(0, timesteps)
    return out.view(timesteps.size(0), *([1] * (len(target_shape) - 1)))


def sample_t_beta(batch_size, alpha=1.5, beta=1.0, eps=1e-3, device=None, dtype=None):
    alpha = torch.full((batch_size,), float(alpha), device=device, dtype=torch.float32)
    beta = torch.full((batch_size,), float(beta), device=device, dtype=torch.float32)
    t = torch.distributions.Beta(alpha, beta).sample()
    t = t * (1.0 - 2.0 * eps) + eps
    if dtype is not None:
        t = t.to(dtype=dtype)
    return t


class SinusoidalTimestepEmbedding(nn.Module):
    def __init__(self, dim, max_period=10000):
        super().__init__()
        self.dim = dim
        self.max_period = max_period

    def forward(self, timesteps):
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(self.max_period)
            * torch.arange(half, device=timesteps.device, dtype=torch.float32)
            / max(half, 1)
        )
        args = timesteps.float()[:, None] * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if self.dim % 2 == 1:
            embedding = F.pad(embedding, (0, 1))
        return embedding


class SimpleDiTActionHead(nn.Module):
    def __init__(
        self,
        action_dim,
        horizon,
        cond_dim,
        proprio_dim,
        model_dim=512,
        depth=4,
        num_heads=8,
        mlp_ratio=4.0,
        dropout=0.0,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.horizon = horizon
        self.model_dim = model_dim

        self.action_in = nn.Linear(action_dim, model_dim)
        self.cond_in = nn.Sequential(nn.LayerNorm(cond_dim), nn.Linear(cond_dim, model_dim))
        self.proprio_in = nn.Sequential(
            nn.LayerNorm(proprio_dim), nn.Linear(proprio_dim, model_dim)
        )
        self.time_in = nn.Sequential(
            SinusoidalTimestepEmbedding(model_dim),
            nn.Linear(model_dim, model_dim * 4),
            nn.SiLU(),
            nn.Linear(model_dim * 4, model_dim),
        )
        self.action_pos = nn.Parameter(torch.zeros(1, horizon, model_dim))
        self.prefix_pos = nn.Parameter(torch.zeros(1, 2, model_dim))

        layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=num_heads,
            dim_feedforward=int(model_dim * mlp_ratio),
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.blocks = nn.TransformerEncoder(layer, num_layers=depth)
        self.final_norm = nn.LayerNorm(model_dim)
        self.action_out = nn.Linear(model_dim, action_dim)

        nn.init.normal_(self.action_pos, std=0.02)
        nn.init.normal_(self.prefix_pos, std=0.02)
        nn.init.zeros_(self.action_out.weight)
        nn.init.zeros_(self.action_out.bias)

    def forward(self, noisy_actions, timesteps, action_token_hidden, proprio_hidden):
        time_emb = self.time_in(timesteps)
        cond_token = self.cond_in(action_token_hidden).unsqueeze(1)
        proprio_token = self.proprio_in(proprio_hidden).unsqueeze(1)
        prefix_tokens = torch.cat([cond_token, proprio_token], dim=1)
        prefix_tokens = prefix_tokens + self.prefix_pos + time_emb.unsqueeze(1)

        action_tokens = self.action_in(noisy_actions)
        action_tokens = action_tokens + self.action_pos + time_emb.unsqueeze(1)

        tokens = torch.cat([prefix_tokens, action_tokens], dim=1)
        tokens = self.blocks(tokens)
        action_tokens = self.final_norm(tokens[:, -self.horizon :])
        return self.action_out(action_tokens)


class ActionChunkInternVLPolicy(MultiTaskInternVL3MLPPolicy):
    def __init__(
        self,
        *args,
        action_horizon=4,
        single_action_dim=4,
        diffusion_train_steps=100,
        diffusion_inference_steps=20,
        diffusion_timestep_max=None,
        diffusion_sample_start_step=None,
        diffusion_x0_clip=None,
        action_head_objective="ddpm",
        repeated_diffusion_steps=1,
        action_norm_type="mean_std",
        action_bound_eps=1e-6,
        flow_t_alpha=1.5,
        flow_t_beta=1.0,
        flow_t_eps=1e-3,
        flow_sample_clip=None,
        dit_dim=512,
        dit_layers=4,
        dit_heads=8,
        dit_mlp_ratio=4.0,
        dit_dropout=0.0,
        **kwargs,
    ):
        if action_head_objective not in {"ddpm", "flow"}:
            raise ValueError("action_head_objective must be one of: ddpm, flow")
        if action_norm_type not in {"mean_std", "bounds"}:
            raise ValueError("action_norm_type must be one of: mean_std, bounds")
        self.action_horizon = action_horizon
        self.single_action_dim = single_action_dim
        self.diffusion_train_steps = diffusion_train_steps
        self.diffusion_inference_steps = diffusion_inference_steps
        self.diffusion_timestep_max = diffusion_timestep_max
        self.diffusion_sample_start_step = diffusion_sample_start_step
        self.diffusion_x0_clip = diffusion_x0_clip
        self.action_head_objective = action_head_objective
        self.repeated_diffusion_steps = max(1, int(repeated_diffusion_steps))
        self.action_norm_type = action_norm_type
        self.action_bound_eps = action_bound_eps
        self.flow_t_alpha = flow_t_alpha
        self.flow_t_beta = flow_t_beta
        self.flow_t_eps = flow_t_eps
        self.flow_sample_clip = flow_sample_clip
        if (
            self.action_head_objective == "flow"
            and self.action_norm_type == "bounds"
            and self.flow_sample_clip is None
        ):
            self.flow_sample_clip = 1.0
        self.dit_dim = dit_dim
        self.dit_layers = dit_layers
        self.dit_heads = dit_heads
        self.dit_mlp_ratio = dit_mlp_ratio
        self.dit_dropout = dit_dropout
        super().__init__(*args, act_dim=action_horizon * single_action_dim, **kwargs)
        self.action_head = SimpleDiTActionHead(
            action_dim=single_action_dim,
            horizon=action_horizon,
            cond_dim=self.hidden_size,
            proprio_dim=self.hidden_size,
            model_dim=dit_dim,
            depth=dit_layers,
            num_heads=dit_heads,
            mlp_ratio=dit_mlp_ratio,
            dropout=dit_dropout,
        )
        self.register_buffer(
            "action_mean",
            torch.zeros(1, 1, single_action_dim, dtype=torch.float32),
            persistent=True,
        )
        self.register_buffer(
            "action_std",
            torch.ones(1, 1, single_action_dim, dtype=torch.float32),
            persistent=True,
        )
        self.register_buffer(
            "action_low",
            -torch.ones(1, 1, single_action_dim, dtype=torch.float32),
            persistent=True,
        )
        self.register_buffer(
            "action_high",
            torch.ones(1, 1, single_action_dim, dtype=torch.float32),
            persistent=True,
        )

        betas = cosine_beta_schedule(diffusion_train_steps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        self.register_buffer("diffusion_betas", betas, persistent=True)
        self.register_buffer("diffusion_alphas_cumprod", alphas_cumprod, persistent=True)
        self.register_buffer(
            "diffusion_sqrt_alphas_cumprod",
            torch.sqrt(alphas_cumprod),
            persistent=True,
        )
        self.register_buffer(
            "diffusion_sqrt_one_minus_alphas_cumprod",
            torch.sqrt(1.0 - alphas_cumprod),
            persistent=True,
        )

    def set_action_stats(self, mean, std):
        self.action_mean.copy_(mean.to(device=self.action_mean.device, dtype=torch.float32))
        self.action_std.copy_(
            std.clamp_min(1e-6).to(device=self.action_std.device, dtype=torch.float32)
        )

    def set_action_bounds(self, low, high):
        low = low.to(device=self.action_low.device, dtype=torch.float32)
        high = high.to(device=self.action_high.device, dtype=torch.float32)
        high = torch.maximum(high, low + float(self.action_bound_eps))
        self.action_low.copy_(low)
        self.action_high.copy_(high)

    def _make_prompts(self, batch_size, task_descriptions=None):
        image_tokens = "<img>" + ("<IMG_CONTEXT>" * self.num_image_token) + "</img>"
        if task_descriptions is None:
            task = self.task_description or "What are the next robot actions?"
            task_descriptions = [task] * batch_size
        elif isinstance(task_descriptions, str):
            task_descriptions = [task_descriptions] * batch_size
        elif len(task_descriptions) != batch_size:
            raise ValueError(
                f"Expected {batch_size} task descriptions, got {len(task_descriptions)}"
            )
        action_tokens = " ".join(
            f"<ACTION_CHUNK_{idx}>" for idx in range(self.action_horizon)
        )
        return [
            f"{image_tokens}\n"
            f"According to the instruction '{task}', predict the next "
            f"{self.action_horizon} robot actions. <PROP_CONTEXT> {action_tokens}"
            for task in task_descriptions
        ]

    def encode_context(self, img, state, task_descriptions=None):
        pixel_values = self._preprocess_images(img)
        if self.tune_mode == "frozen":
            with torch.no_grad():
                action_token_hidden = self._extract_action_hidden(
                    pixel_values, task_descriptions
                )
        else:
            action_token_hidden = self._extract_action_hidden(
                pixel_values, task_descriptions
            )

        proprio_hidden = self.proprio_encoder(state.float())
        return action_token_hidden, proprio_hidden

    def _resolve_condition(
        self,
        condition_source,
        action_token_hidden,
        proprio_hidden,
        pred_actions=None,
    ):
        fused = torch.cat([action_token_hidden, proprio_hidden], dim=-1)
        if condition_source == "action_token":
            return action_token_hidden
        elif condition_source == "fused":
            return fused
        elif condition_source == "pred_action":
            if pred_actions is None:
                raise ValueError(
                    "condition_source=pred_action is not supported during DiT "
                    "diffusion training; use action_token or fused."
                )
            return pred_actions
        raise ValueError(f"Unsupported condition_source: {condition_source}")

    def normalize_actions(self, actions):
        if self.action_norm_type == "bounds":
            low = self.action_low.to(device=actions.device, dtype=actions.dtype)
            high = self.action_high.to(device=actions.device, dtype=actions.dtype)
            scale = (high - low).clamp_min(float(self.action_bound_eps))
            normalized = 2.0 * (actions - low) / scale - 1.0
            return normalized.clamp(-1.0, 1.0)
        return (actions - self.action_mean.to(actions.device)) / self.action_std.to(
            actions.device
        )

    def unnormalize_actions(self, actions):
        if self.action_norm_type == "bounds":
            low = self.action_low.to(device=actions.device, dtype=actions.dtype)
            high = self.action_high.to(device=actions.device, dtype=actions.dtype)
            return 0.5 * (actions + 1.0) * (high - low) + low
        return actions * self.action_std.to(actions.device) + self.action_mean.to(
            actions.device
        )

    def _repeat_action_training_batch(
        self, action_token_hidden, proprio_hidden, target_actions
    ):
        if self.repeated_diffusion_steps == 1:
            return action_token_hidden, proprio_hidden, target_actions
        return (
            action_token_hidden.repeat(self.repeated_diffusion_steps, 1),
            proprio_hidden.repeat(self.repeated_diffusion_steps, 1),
            target_actions.repeat(self.repeated_diffusion_steps, 1, 1),
        )

    def diffusion_loss(self, action_token_hidden, proprio_hidden, target_actions):
        target_actions = self.normalize_actions(target_actions.float())
        action_token_hidden, proprio_hidden, target_actions = (
            self._repeat_action_training_batch(
                action_token_hidden, proprio_hidden, target_actions
            )
        )
        batch_size = target_actions.size(0)
        timestep_high = (
            self.diffusion_train_steps
            if self.diffusion_timestep_max is None
            else min(int(self.diffusion_timestep_max) + 1, self.diffusion_train_steps)
        )
        timestep_high = max(timestep_high, 1)
        timesteps = torch.randint(
            0,
            timestep_high,
            (batch_size,),
            device=target_actions.device,
            dtype=torch.long,
        )
        noise = torch.randn_like(target_actions)
        noisy_actions = (
            extract_schedule_value(
                self.diffusion_sqrt_alphas_cumprod, timesteps, target_actions.shape
            )
            * target_actions
            + extract_schedule_value(
                self.diffusion_sqrt_one_minus_alphas_cumprod,
                timesteps,
                target_actions.shape,
            )
            * noise
        )
        pred_noise = self.action_head(
            noisy_actions,
            timesteps,
            action_token_hidden,
            proprio_hidden,
        )
        return F.mse_loss(pred_noise.float(), noise.float())

    def flow_loss(self, action_token_hidden, proprio_hidden, target_actions):
        target_actions = self.normalize_actions(target_actions.float())
        action_token_hidden, proprio_hidden, target_actions = (
            self._repeat_action_training_batch(
                action_token_hidden, proprio_hidden, target_actions
            )
        )
        batch_size = target_actions.size(0)
        x0 = torch.randn_like(target_actions)
        t = sample_t_beta(
            batch_size,
            alpha=self.flow_t_alpha,
            beta=self.flow_t_beta,
            eps=self.flow_t_eps,
            device=target_actions.device,
            dtype=target_actions.dtype,
        )
        t_broadcast = t.view(batch_size, *([1] * (target_actions.dim() - 1)))
        x_t = (1.0 - t_broadcast) * x0 + t_broadcast * target_actions
        target_velocity = target_actions - x0
        pred_velocity = self.action_head(
            x_t,
            t,
            action_token_hidden,
            proprio_hidden,
        )
        return F.mse_loss(pred_velocity.float(), target_velocity.float())

    def action_loss(self, action_token_hidden, proprio_hidden, target_actions):
        if self.action_head_objective == "flow":
            return self.flow_loss(action_token_hidden, proprio_hidden, target_actions)
        return self.diffusion_loss(action_token_hidden, proprio_hidden, target_actions)

    @torch.no_grad()
    def sample_actions(self, action_token_hidden, proprio_hidden, inference_steps=None):
        if self.action_head_objective == "flow":
            return self.sample_actions_flow(
                action_token_hidden,
                proprio_hidden,
                inference_steps=inference_steps,
            )
        inference_steps = inference_steps or self.diffusion_inference_steps
        batch_size = action_token_hidden.size(0)
        device = action_token_hidden.device
        start_step = (
            self.diffusion_train_steps - 1
            if self.diffusion_sample_start_step is None
            else int(self.diffusion_sample_start_step)
        )
        start_step = max(0, min(start_step, self.diffusion_train_steps - 1))
        actions = torch.randn(
            batch_size,
            self.action_horizon,
            self.single_action_dim,
            device=device,
            dtype=torch.float32,
        )
        timesteps = torch.linspace(
            start_step,
            0,
            inference_steps,
            device=device,
        ).long()

        for idx, timestep in enumerate(timesteps):
            t = torch.full((batch_size,), int(timestep.item()), device=device, dtype=torch.long)
            pred_noise = self.action_head(actions, t, action_token_hidden, proprio_hidden)
            alpha_t = self.diffusion_alphas_cumprod[t].view(batch_size, 1, 1).to(
                actions.dtype
            )
            pred_x0 = (actions - torch.sqrt(1.0 - alpha_t) * pred_noise) / torch.sqrt(
                alpha_t
            )
            if self.diffusion_x0_clip is not None and self.diffusion_x0_clip > 0:
                pred_x0 = pred_x0.clamp(
                    -float(self.diffusion_x0_clip),
                    float(self.diffusion_x0_clip),
                )
            if idx == len(timesteps) - 1:
                actions = pred_x0
            else:
                next_t = torch.full(
                    (batch_size,),
                    int(timesteps[idx + 1].item()),
                    device=device,
                    dtype=torch.long,
                )
                alpha_next = self.diffusion_alphas_cumprod[next_t].view(
                    batch_size, 1, 1
                ).to(actions.dtype)
                actions = torch.sqrt(alpha_next) * pred_x0 + torch.sqrt(
                    1.0 - alpha_next
                ) * pred_noise
        return self.unnormalize_actions(actions.float())

    @torch.no_grad()
    def sample_actions_flow(
        self, action_token_hidden, proprio_hidden, inference_steps=None
    ):
        inference_steps = max(2, int(inference_steps or self.diffusion_inference_steps))
        batch_size = action_token_hidden.size(0)
        device = action_token_hidden.device
        actions = torch.randn(
            batch_size,
            self.action_horizon,
            self.single_action_dim,
            device=device,
            dtype=torch.float32,
        )
        t_steps = torch.linspace(
            0.0,
            1.0,
            inference_steps,
            device=device,
            dtype=torch.float32,
        )

        for idx in range(1, inference_steps):
            t_prev = t_steps[idx - 1]
            t_curr = t_steps[idx]
            dt = t_curr - t_prev
            t = torch.full(
                (batch_size,),
                float(t_prev.item()),
                device=device,
                dtype=torch.float32,
            )
            velocity = self.action_head(
                actions,
                t,
                action_token_hidden,
                proprio_hidden,
            )
            actions = actions + velocity.float() * dt

        if self.flow_sample_clip is not None and self.flow_sample_clip > 0:
            clip = float(self.flow_sample_clip)
            actions = actions.clamp(-clip, clip)
        return self.unnormalize_actions(actions.float())

    def forward(
        self,
        img,
        state,
        task_descriptions=None,
        target_actions=None,
        return_condition=False,
        condition_source="action_token",
    ):
        action_token_hidden, proprio_hidden = self.encode_context(
            img, state, task_descriptions
        )

        if target_actions is not None:
            action_loss = self.action_loss(
                action_token_hidden,
                proprio_hidden,
                target_actions,
            )
            if not return_condition:
                return action_loss
            condition = self._resolve_condition(
                condition_source,
                action_token_hidden,
                proprio_hidden,
            )
            return action_loss, condition

        pred_actions = self.sample_actions(action_token_hidden, proprio_hidden)
        if not return_condition:
            return pred_actions
        condition = self._resolve_condition(
            condition_source,
            action_token_hidden,
            proprio_hidden,
            pred_actions=pred_actions,
        )
        return pred_actions, condition

    def trainable_state_dict(self):
        payload = super().trainable_state_dict()
        payload["config"].update(
            {
                "action_head_type": f"simple_dit_{self.action_head_objective}",
                "single_action_dim": self.single_action_dim,
                "action_horizon": self.action_horizon,
                "diffusion_train_steps": self.diffusion_train_steps,
                "diffusion_inference_steps": self.diffusion_inference_steps,
                "diffusion_timestep_max": self.diffusion_timestep_max,
                "diffusion_sample_start_step": self.diffusion_sample_start_step,
                "diffusion_x0_clip": self.diffusion_x0_clip,
                "action_head_objective": self.action_head_objective,
                "repeated_diffusion_steps": self.repeated_diffusion_steps,
                "action_norm_type": self.action_norm_type,
                "action_bound_eps": self.action_bound_eps,
                "flow_t_alpha": self.flow_t_alpha,
                "flow_t_beta": self.flow_t_beta,
                "flow_t_eps": self.flow_t_eps,
                "flow_sample_clip": self.flow_sample_clip,
                "dit_dim": self.dit_dim,
                "dit_layers": self.dit_layers,
                "dit_heads": self.dit_heads,
                "dit_mlp_ratio": self.dit_mlp_ratio,
                "dit_dropout": self.dit_dropout,
            }
        )
        payload["action_mean"] = self.action_mean.detach().cpu()
        payload["action_std"] = self.action_std.detach().cpu()
        payload["action_low"] = self.action_low.detach().cpu()
        payload["action_high"] = self.action_high.detach().cpu()
        return payload


class ActionChunkConditioner(nn.Module):
    def __init__(self, input_dim, out_dim=768):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, 512),
            nn.SiLU(),
            nn.Linear(512, out_dim),
        )

    def forward(self, condition):
        return self.net(condition.flatten(start_dim=1))


class SVDAuxModule(nn.Module):
    def __init__(
        self,
        svd,
        action_conditioner,
        cond_aug,
        fps_id,
        motion_bucket_id,
    ):
        super().__init__()
        self.svd = svd
        self.action_conditioner = action_conditioner
        self.cond_aug = cond_aug
        self.fps_id = fps_id
        self.motion_bucket_id = motion_bucket_id

    def forward(self, svd_video, svd_condition):
        return svd_aux_loss(
            self.svd,
            self.action_conditioner,
            svd_video,
            svd_condition,
            self.cond_aug,
            self.fps_id,
            self.motion_bucket_id,
        )


class LoRALinear(nn.Module):
    def __init__(self, base, rank=4, alpha=8, dropout=0.0):
        super().__init__()
        self.base = base
        self.scaling = alpha / rank
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.lora_down = nn.Linear(base.in_features, rank, bias=False)
        self.lora_up = nn.Linear(rank, base.out_features, bias=False)
        nn.init.kaiming_uniform_(self.lora_down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_up.weight)
        for param in self.base.parameters():
            param.requires_grad_(False)

    def forward(self, x):
        return self.base(x) + self.lora_up(self.lora_down(self.dropout(x))) * self.scaling


@contextmanager
def disable_torch_init():
    old = {}

    def noop(*args, **kwargs):
        return args[0] if args else None

    for name in [
        "uniform_",
        "normal_",
        "trunc_normal_",
        "constant_",
        "ones_",
        "zeros_",
        "kaiming_uniform_",
        "kaiming_normal_",
        "xavier_uniform_",
        "xavier_normal_",
    ]:
        if hasattr(torch.nn.init, name):
            old[name] = getattr(torch.nn.init, name)
            setattr(torch.nn.init, name, noop)
    try:
        yield
    finally:
        for name, fn in old.items():
            setattr(torch.nn.init, name, fn)


def replace_xformers_attention(node):
    if isinstance(node, (dict, DictConfig)):
        for key, value in node.items():
            if isinstance(value, str):
                if value == "vanilla-xformers":
                    node[key] = "vanilla"
                elif value == "softmax-xformers":
                    node[key] = "softmax"
            else:
                replace_xformers_attention(value)
    elif isinstance(node, (list, ListConfig)):
        for value in node:
            replace_xformers_attention(value)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def inject_svd_lora(module, rank=4, alpha=8, dropout=0.0, limit=128):
    selected = []
    suffixes = ("to_q", "to_k", "to_v")
    for name, child in list(module.named_modules()):
        if isinstance(child, nn.Linear) and name.endswith(suffixes):
            selected.append(name)
    if limit > 0:
        selected = selected[:limit]
    module_lookup = dict(module.named_modules())
    for name in selected:
        parent_name, child_name = name.rsplit(".", 1)
        parent = module_lookup[parent_name]
        child = getattr(parent, child_name)
        setattr(parent, child_name, LoRALinear(child, rank, alpha, dropout))
    return selected


def load_svd(device, num_frames, ckpt_path):
    config = OmegaConf.load(SVD_ROOT / "scripts/sampling/configs/svd.yaml")
    config.model.params.ckpt_path = str(ckpt_path)
    config.model.params.sampler_config.params.num_steps = 1
    config.model.params.sampler_config.params.guider_config.params.num_frames = num_frames
    replace_xformers_attention(config)
    with disable_torch_init():
        model = instantiate_from_config(config.model)
    return model.to(device)


def make_svd_conditioning_batch(video, cond_aug, fps_id, motion_bucket_id):
    batch_size, num_frames = video.size(0), video.size(1)
    flat_size = batch_size * num_frames
    cond_frame = video[:, 0]
    device = video.device
    return {
        "cond_frames_without_noise": cond_frame,
        "cond_frames": cond_frame + cond_aug * torch.randn_like(cond_frame),
        "cond_aug": torch.full((flat_size,), cond_aug, device=device),
        "fps_id": torch.full((flat_size,), fps_id, dtype=torch.long, device=device),
        "motion_bucket_id": torch.full(
            (flat_size,), motion_bucket_id, dtype=torch.long, device=device
        ),
        "num_video_frames": num_frames,
        "image_only_indicator": torch.zeros(
            batch_size, num_frames, dtype=torch.long, device=device
        ),
    }


def repeat_conditioning(cond, num_frames):
    out = {}
    for key, value in cond.items():
        if key in {"crossattn", "concat"}:
            value = repeat(value, "b ... -> b t ...", t=num_frames)
            value = rearrange(value, "b t ... -> (b t) ...", t=num_frames)
        out[key] = value
    return out


def svd_aux_loss(
    svd,
    action_conditioner,
    svd_video,
    svd_condition,
    cond_aug,
    fps_id,
    motion_bucket_id,
):
    batch_size, num_frames = svd_video.size(0), svd_video.size(1)
    frames = rearrange(svd_video, "b t c h w -> (b t) c h w")
    with torch.no_grad():
        latents = svd.encode_first_stage(frames)
        cond_batch = make_svd_conditioning_batch(
            svd_video, cond_aug, fps_id, motion_bucket_id
        )
        cond = repeat_conditioning(svd.conditioner(cond_batch), num_frames)

    action_vec = action_conditioner(svd_condition)
    action_vec = repeat(action_vec, "b c -> (b t) c", t=num_frames)
    cond["vector"] = cond["vector"] + action_vec.to(cond["vector"].dtype)

    sigmas = torch.exp(
        torch.randn(latents.size(0), device=latents.device) * 1.6 - 1.2
    ).clamp(0.002, 80.0)
    noise = torch.randn_like(latents)
    noised = latents + noise * sigmas.view(-1, 1, 1, 1)
    noised.requires_grad_(True)
    model_inputs = {
        "num_video_frames": num_frames,
        "image_only_indicator": torch.zeros(
            batch_size, num_frames, device=latents.device
        ),
    }
    with torch.autocast("cuda", dtype=torch.bfloat16):
        pred = svd.denoiser(svd.model, noised, sigmas, cond, **model_inputs)
        return ((pred.float() - latents.float()) ** 2).mean()


def train(args):
    distributed, rank, world_size, local_rank = setup_distributed()
    if distributed:
        args.device = f"cuda:{local_rank}"
    set_seed(args.seed + rank)
    device = torch.device(args.device)
    use_svd = (not args.vla_only) and args.svd_loss_weight > 0

    try:
        task_names = parse_csv(args.tasks)
        task_descriptions = parse_task_descriptions(args.task_descriptions, task_names)
        dataset = JointMetaWorldChunkDataset(
            args.data_root,
            task_names,
            task_descriptions,
            max_episodes=args.max_epi,
            action_horizon=args.action_horizon,
            svd_num_frames=args.svd_num_frames,
            svd_image_size=args.svd_image_size,
            include_svd=use_svd,
        )
        sampler = (
            DistributedSampler(
                dataset,
                num_replicas=world_size,
                rank=rank,
                shuffle=True,
                drop_last=True,
            )
            if distributed
            else None
        )
        dataloader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=sampler is None,
            sampler=sampler,
            num_workers=args.num_workers,
            drop_last=True,
        )
        data_iter = iter(dataloader)
        epoch = 0

        vla = ActionChunkInternVLPolicy(
            obs_dim=dataset.obs_dim,
            single_action_dim=dataset.act_dim,
            action_horizon=args.action_horizon,
            diffusion_train_steps=args.diffusion_train_steps,
            diffusion_inference_steps=args.diffusion_inference_steps,
            diffusion_timestep_max=args.diffusion_timestep_max,
            diffusion_sample_start_step=args.diffusion_sample_start_step,
            diffusion_x0_clip=args.diffusion_x0_clip,
            action_head_objective=args.action_head_objective,
            repeated_diffusion_steps=args.repeated_diffusion_steps,
            action_norm_type=args.action_norm_type,
            action_bound_eps=args.action_bound_eps,
            flow_t_alpha=args.flow_t_alpha,
            flow_t_beta=args.flow_t_beta,
            flow_t_eps=args.flow_t_eps,
            flow_sample_clip=args.flow_sample_clip,
            dit_dim=args.dit_dim,
            dit_layers=args.dit_layers,
            dit_heads=args.dit_heads,
            dit_mlp_ratio=args.dit_mlp_ratio,
            dit_dropout=args.dit_dropout,
            model_name=args.vlm_model,
            task_description="joint robot policy",
            tune_mode=args.vla_tune_mode,
            lora_r=args.vla_lora_r,
            lora_alpha=args.vla_lora_alpha,
            lora_dropout=args.vla_lora_dropout,
            torch_dtype=args.torch_dtype,
            local_files_only=args.local_files_only,
        ).to(device)
        action_mean, action_std = dataset.action_stats()
        action_low, action_high = dataset.action_bounds(
            args.action_quantile_low,
            args.action_quantile_high,
        )
        vla.set_action_stats(action_mean, action_std)
        vla.set_action_bounds(action_low, action_high)

        svd = None
        svd_aux = None
        action_conditioner = None
        svd_lora_modules = []
        svd_condition_dim = None
        if use_svd:
            svd = load_svd(device, args.svd_num_frames, args.svd_ckpt)
            for param in svd.parameters():
                param.requires_grad_(False)
            svd_lora_modules = inject_svd_lora(
                svd.model,
                rank=args.svd_lora_r,
                alpha=args.svd_lora_alpha,
                dropout=args.svd_lora_dropout,
                limit=args.svd_lora_limit,
            )
            svd.to(device)

            if args.svd_condition_source == "action_token":
                svd_condition_dim = vla.hidden_size
            elif args.svd_condition_source == "fused":
                svd_condition_dim = vla.hidden_size * 2
            elif args.svd_condition_source == "pred_action":
                svd_condition_dim = args.action_horizon * dataset.act_dim
            else:
                raise ValueError(
                    f"Unsupported svd_condition_source: {args.svd_condition_source}"
                )

            action_conditioner = ActionChunkConditioner(
                svd_condition_dim,
                out_dim=args.svd_action_cond_dim,
            ).to(device)
            svd_aux = SVDAuxModule(
                svd,
                action_conditioner,
                args.cond_aug,
                args.fps_id,
                args.motion_bucket_id,
            ).to(device)

        trainable = [p for p in vla.parameters() if p.requires_grad]
        if use_svd:
            trainable += [p for p in svd_aux.parameters() if p.requires_grad]
        if is_main_process():
            print(
                "DDP: "
                f"enabled={distributed}, rank={rank}, world_size={world_size}, "
                f"per_rank_batch={args.batch_size}, "
                f"global_batch={args.batch_size * world_size}"
            )
            print(f"Training mode: {'joint_vla_svd' if use_svd else 'vla_only'}")
            print(
                "VLA trainable params: "
                f"{sum(p.numel() for p in vla.parameters() if p.requires_grad):,}"
            )
            print(
                f"Action head: simple_dit_{args.action_head_objective} "
                f"horizon={args.action_horizon}, act_dim={dataset.act_dim}, "
                f"dim={args.dit_dim}, layers={args.dit_layers}, heads={args.dit_heads}, "
                f"repeated_steps={args.repeated_diffusion_steps}, "
                f"norm={args.action_norm_type}, "
                f"train_steps={args.diffusion_train_steps}, "
                f"infer_steps={args.diffusion_inference_steps}, "
                f"timestep_max={args.diffusion_timestep_max}, "
                f"sample_start={args.diffusion_sample_start_step}, "
                f"x0_clip={args.diffusion_x0_clip}, "
                f"flow_t=Beta({args.flow_t_alpha}, {args.flow_t_beta}), "
                f"flow_sample_clip={vla.flow_sample_clip}"
            )
            print(
                "Action mean/std: "
                f"mean={action_mean.flatten().tolist()}, "
                f"std={action_std.flatten().tolist()}"
            )
            print(
                "Action q-bounds: "
                f"q_low={args.action_quantile_low}, "
                f"q_high={args.action_quantile_high}, "
                f"low={action_low.flatten().tolist()}, "
                f"high={action_high.flatten().tolist()}"
            )
            if use_svd:
                print(f"SVD LoRA modules: {len(svd_lora_modules)}")
                print(
                    "SVD condition: "
                    f"source={args.svd_condition_source}, "
                    f"input_dim={svd_condition_dim}, "
                    f"out_dim={args.svd_action_cond_dim}"
                )
                print(
                    "SVD/action-condition trainable params: "
                    f"{sum(p.numel() for p in svd_aux.parameters() if p.requires_grad):,}"
                )

        if distributed:
            vla = DDP(
                vla,
                device_ids=[local_rank],
                output_device=local_rank,
                find_unused_parameters=False,
                static_graph=True,
            )
            if use_svd:
                svd_aux = DDP(
                    svd_aux,
                    device_ids=[local_rank],
                    output_device=local_rank,
                    find_unused_parameters=False,
                    static_graph=True,
                )

        optimizer = torch.optim.AdamW(
            trainable,
            lr=args.lr,
            weight_decay=args.weight_decay,
        )
        logger = LossLogger(args.log_dir, enabled=is_main_process())
        loss_window = deque(maxlen=args.print_every)
        action_window = deque(maxlen=args.print_every)
        svd_window = deque(maxlen=args.print_every)

        vla.train()
        if use_svd:
            svd_aux.train()
            unwrap_model(svd_aux).svd.first_stage_model.eval()
            unwrap_model(svd_aux).svd.conditioner.eval()
        progress = tqdm(
            total=args.total_steps,
            dynamic_ncols=True,
            disable=not is_main_process(),
        )
        for step in range(1, args.total_steps + 1):
            try:
                batch = next(data_iter)
            except StopIteration:
                epoch += 1
                if sampler is not None:
                    sampler.set_epoch(epoch)
                data_iter = iter(dataloader)
                batch = next(data_iter)

            vla_img = batch["vla_img"].to(device, non_blocking=True)
            vla_obs = batch["vla_obs"].to(device, non_blocking=True)
            target_actions = batch["action_chunk"].to(device, non_blocking=True)
            task_texts = list(batch["task_text"])

            if use_svd:
                svd_video = batch["svd_video"].to(device, non_blocking=True)
                action_loss, svd_condition = vla(
                    vla_img,
                    vla_obs,
                    task_texts,
                    target_actions=target_actions,
                    return_condition=True,
                    condition_source=args.svd_condition_source,
                )
            else:
                action_loss = vla(
                    vla_img,
                    vla_obs,
                    task_texts,
                    target_actions=target_actions,
                )
            if use_svd and step % args.svd_every == 0:
                video_loss = svd_aux(svd_video, svd_condition)
                loss = action_loss + args.svd_loss_weight * video_loss
            else:
                video_loss = torch.zeros((), device=device, dtype=action_loss.dtype)
                loss = action_loss

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, args.max_grad_norm)
            optimizer.step()

            progress.update(1)
            lr = optimizer.param_groups[0]["lr"]
            loss_value = loss.item()
            action_value = action_loss.item()
            svd_value = video_loss.item()
            loss_window.append(loss_value)
            action_window.append(action_value)
            svd_window.append(svd_value)
            avg_loss = sum(loss_window) / len(loss_window)
            avg_action = sum(action_window) / len(action_window)
            avg_svd = sum(svd_window) / len(svd_window)
            progress.set_postfix(
                loss=f"{loss_value:.5f}",
                action=f"{action_value:.5f}",
                svd=f"{svd_value:.5f}",
                action_avg=f"{avg_action:.5f}",
            )
            if is_main_process() and step % args.log_every == 0:
                logger.log(
                    step,
                    loss_value,
                    action_value,
                    svd_value,
                    lr,
                    avg_loss=avg_loss,
                    avg_action_loss=avg_action,
                    avg_svd_loss=avg_svd,
                )
            if is_main_process() and step % args.print_every == 0:
                progress.write(
                    f"Step {step}/{args.total_steps} - "
                    f"Loss: {avg_loss:.6f} "
                    f"Action: {avg_action:.6f} "
                    f"SVD: {avg_svd:.6f}"
                )
            if is_main_process() and (
                step % args.save_every == 0 or step == args.total_steps
            ):
                unwrapped_svd_aux = unwrap_model(svd_aux) if use_svd else None
                save_checkpoint(
                    args.save_path,
                    step,
                    unwrap_model(vla),
                    unwrapped_svd_aux.svd if unwrapped_svd_aux is not None else None,
                    (
                        unwrapped_svd_aux.action_conditioner
                        if unwrapped_svd_aux is not None
                        else None
                    ),
                    svd_lora_modules,
                    args,
                )
                if args.plot_loss:
                    plot_loss_curves(args.log_dir)
        progress.close()
        logger.close()
        if is_main_process() and args.plot_loss:
            plot_loss_curves(args.log_dir)
    finally:
        cleanup_distributed()


def save_checkpoint(path, step, vla, svd, action_conditioner, svd_lora_modules, args):
    save_path = Path(path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "step": step,
        "args": vars(args),
        "vla": vla.trainable_state_dict(),
        "svd_lora_state_dict": (
            {
                k: v.detach().cpu()
                for k, v in svd.state_dict().items()
                if "lora_" in k
            }
            if svd is not None
            else {}
        ),
        "action_conditioner": (
            action_conditioner.state_dict()
            if action_conditioner is not None
            else {}
        ),
        "svd_lora_modules": svd_lora_modules,
    }
    torch.save(payload, save_path)
    print(f"Saved joint checkpoint to {save_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", default=",".join(DEFAULT_TASKS))
    parser.add_argument("--task_descriptions", default=None)
    parser.add_argument("--data_root", default="/home/yiyuan/pvrobo/expert_demos/metaworld")
    parser.add_argument("--vlm_model", default="/data/yiyuan/models/InternVL3-2B")
    parser.add_argument("--svd_ckpt", default=str(SVD_ROOT / "checkpoints/svd.safetensors"))
    parser.add_argument("--local_files_only", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_epi", type=int, default=25)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--total_steps", type=int, default=1000)
    parser.add_argument("--save_every", type=int, default=500)
    parser.add_argument("--log_every", type=int, default=1)
    parser.add_argument("--print_every", type=int, default=100)
    parser.add_argument(
        "--log_dir",
        default="/home/yiyuan/pvrobo/bc_code/runs/joint_internvl3_svd_lora",
    )
    parser.add_argument("--plot_loss", dest="plot_loss", action="store_true", default=True)
    parser.add_argument("--no_plot_loss", dest="plot_loss", action="store_false")
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)

    parser.add_argument("--action_horizon", type=int, default=4)
    parser.add_argument("--diffusion_train_steps", type=int, default=100)
    parser.add_argument("--diffusion_inference_steps", type=int, default=20)
    parser.add_argument("--diffusion_timestep_max", type=int, default=None)
    parser.add_argument("--diffusion_sample_start_step", type=int, default=None)
    parser.add_argument("--diffusion_x0_clip", type=float, default=None)
    parser.add_argument(
        "--action_head_objective",
        default="ddpm",
        choices=["ddpm", "flow"],
        help="ddpm keeps the old epsilon objective; flow uses WoG-style flow matching.",
    )
    parser.add_argument(
        "--repeated_diffusion_steps",
        type=int,
        default=1,
        help="Repeat each batch this many times with different diffusion/flow timesteps.",
    )
    parser.add_argument(
        "--action_norm_type",
        default="mean_std",
        choices=["mean_std", "bounds"],
        help="bounds maps q_low/q_high actions to [-1, 1], matching WoG-style action scaling.",
    )
    parser.add_argument("--action_quantile_low", type=float, default=0.01)
    parser.add_argument("--action_quantile_high", type=float, default=0.99)
    parser.add_argument("--action_bound_eps", type=float, default=1e-6)
    parser.add_argument("--flow_t_alpha", type=float, default=1.5)
    parser.add_argument("--flow_t_beta", type=float, default=1.0)
    parser.add_argument("--flow_t_eps", type=float, default=1e-3)
    parser.add_argument("--flow_sample_clip", type=float, default=None)
    parser.add_argument("--dit_dim", type=int, default=512)
    parser.add_argument("--dit_layers", type=int, default=4)
    parser.add_argument("--dit_heads", type=int, default=8)
    parser.add_argument("--dit_mlp_ratio", type=float, default=4.0)
    parser.add_argument("--dit_dropout", type=float, default=0.0)
    parser.add_argument("--svd_num_frames", type=int, default=4)
    parser.add_argument("--svd_image_size", type=int, default=128)
    parser.add_argument(
        "--vla_only",
        action="store_true",
        help="Train the 4-action VLA policy only, without loading or optimizing SVD.",
    )
    parser.add_argument("--svd_loss_weight", type=float, default=0.05)
    parser.add_argument("--svd_every", type=int, default=1)
    parser.add_argument(
        "--svd_condition_source",
        default="action_token",
        choices=["action_token", "fused", "pred_action"],
        help=(
            "SVD condition source. action_token uses the VLA last-token hidden; "
            "fused uses concat(action_token_hidden, proprio_hidden); "
            "pred_action restores the old behavior of conditioning on predicted actions."
        ),
    )
    parser.add_argument("--svd_action_cond_dim", type=int, default=768)
    parser.add_argument("--cond_aug", type=float, default=0.02)
    parser.add_argument("--fps_id", type=int, default=6)
    parser.add_argument("--motion_bucket_id", type=int, default=80)

    parser.add_argument("--vla_tune_mode", default="lora", choices=["frozen", "lora", "full"])
    parser.add_argument("--vla_lora_r", type=int, default=16)
    parser.add_argument("--vla_lora_alpha", type=int, default=32)
    parser.add_argument("--vla_lora_dropout", type=float, default=0.05)
    parser.add_argument("--torch_dtype", default="bfloat16")

    parser.add_argument("--svd_lora_r", type=int, default=4)
    parser.add_argument("--svd_lora_alpha", type=int, default=8)
    parser.add_argument("--svd_lora_dropout", type=float, default=0.0)
    parser.add_argument("--svd_lora_limit", type=int, default=64)
    parser.add_argument(
        "--save_path",
        default="/home/yiyuan/pvrobo/bc_code/checkpoints/joint_internvl3_svd_lora_smoke.pth",
    )
    args = parser.parse_args()
    if not 0.0 <= args.action_quantile_low < args.action_quantile_high <= 1.0:
        raise ValueError(
            "--action_quantile_low and --action_quantile_high must satisfy "
            "0 <= low < high <= 1"
        )
    train(args)


if __name__ == "__main__":
    main()
