# MotionWeave: motion-centric grounding and horizon residual composition.
import argparse
import csv
import math
import os
from collections import deque
from datetime import timedelta
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch import nn
from torch.nn import functional as F
from torch.distributed.fsdp import (
    BackwardPrefetch,
    FullStateDictConfig,
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    ShardingStrategy,
    StateDictType,
)
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm

from train_eval_internvl3_mlp_multitask import (
    DEFAULT_TASKS,
    parse_csv,
    parse_task_descriptions,
)
from train_joint_internvl3_svd_lora_dit import (
    ActionChunkInternVLPolicy,
    JointMetaWorldChunkDataset,
    SinusoidalTimestepEmbedding,
    set_seed,
)


class MotionWeaveDataset(JointMetaWorldChunkDataset):
    """MetaWorld chunks paired with precomputed future robot-arm masks."""

    def __init__(self, *args, motion_mask_root, **kwargs):
        super().__init__(*args, include_svd=False, **kwargs)
        self.motion_mask_root = Path(motion_mask_root)
        self._mask_cache = {}

    def _episode_mask_path(self, task_name, episode_idx):
        task_root = self.motion_mask_root / task_name
        candidates = (
            task_root / f"episode_{episode_idx:03d}.npy",
            task_root / f"episode_{episode_idx}.npy",
            task_root / f"episode_{episode_idx:03d}.npz",
        )
        for path in candidates:
            if path.exists():
                return path
        raise FileNotFoundError(
            f"Missing future mask sequence for {task_name} episode {episode_idx}. "
            f"Expected one of: {', '.join(str(path) for path in candidates)}"
        )

    def _load_episode_masks(self, task_name, episode_idx):
        key = (task_name, int(episode_idx))
        if key not in self._mask_cache:
            path = self._episode_mask_path(*key)
            masks = np.load(path, mmap_mode="r")
            if isinstance(masks, np.lib.npyio.NpzFile):
                if "masks" not in masks:
                    raise KeyError(f"{path} must contain an array named 'masks'.")
                masks = masks["masks"]
            self._mask_cache[key] = masks
        return self._mask_cache[key]

    def __getitem__(self, idx):
        item = super().__getitem__(idx)
        sample = self.samples[idx]
        start = int(sample["start"])
        masks = self._load_episode_masks(
            sample["task_name"],
            sample["episode_idx"],
        )
        future = np.asarray(
            masks[start + 1 : start + 1 + self.action_horizon],
            dtype=np.float32,
        )
        if future.shape[0] != self.action_horizon:
            raise ValueError(
                f"Expected {self.action_horizon} future masks, got {future.shape}."
            )
        if future.ndim == 4 and future.shape[1] == 1:
            future = future[:, 0]
        elif future.ndim == 4 and future.shape[-1] == 1:
            future = future[..., 0]
        if future.ndim != 3:
            raise ValueError(f"Expected mask sequence [T,H,W], got {future.shape}.")
        item["motion_masks"] = torch.from_numpy(future.copy()).unsqueeze(1)
        return item


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


def unwrap_ddp(module):
    return module.module if isinstance(module, DDP) else module


def set_use_cache_false(module):
    for obj in (
        module,
        getattr(module, "vlm", None),
        getattr(getattr(module, "vlm", None), "language_model", None),
    ):
        config = getattr(obj, "config", None)
        if config is not None and hasattr(config, "use_cache"):
            config.use_cache = False


def enable_gradient_checkpointing(policy):
    enabled = []
    targets = [
        getattr(policy, "vlm", None),
        getattr(getattr(policy, "vlm", None), "language_model", None),
        getattr(getattr(policy, "vlm", None), "vision_model", None),
    ]
    for target in targets:
        if target is not None and hasattr(target, "gradient_checkpointing_enable"):
            target.gradient_checkpointing_enable()
            enabled.append(target.__class__.__name__)
    set_use_cache_false(policy)
    return enabled


def fsdp_kwargs(device, mixed_precision_enabled=True):
    mixed_precision = None
    if mixed_precision_enabled:
        mixed_precision = MixedPrecision(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.bfloat16,
            buffer_dtype=torch.bfloat16,
        )
    return dict(
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        mixed_precision=mixed_precision,
        backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
        device_id=device,
        use_orig_params=True,
        limit_all_gathers=True,
    )


def wrap_transformer_layers_fsdp(policy, device, class_names, mixed_precision_enabled):
    wrapped = []
    class_names = set(class_names)

    def visit(module, prefix=""):
        for child_name, child in list(module.named_children()):
            full_name = f"{prefix}.{child_name}" if prefix else child_name
            if child.__class__.__name__ in class_names:
                setattr(
                    module,
                    child_name,
                    FSDP(
                        child,
                        **fsdp_kwargs(
                            device,
                            mixed_precision_enabled=mixed_precision_enabled,
                        ),
                    ),
                )
                wrapped.append(f"{full_name}:{child.__class__.__name__}")
            else:
                visit(child, full_name)

    visit(policy)
    return wrapped


def fsdp_param_ids(module):
    ids = set()
    for child in module.modules():
        if isinstance(child, FSDP):
            for param in child.parameters():
                ids.add(id(param))
    return ids


def non_fsdp_trainable_params(module):
    managed = fsdp_param_ids(module)
    return [
        param
        for param in module.parameters()
        if param.requires_grad and id(param) not in managed
    ]


def average_gradients(params, world_size):
    if world_size <= 1:
        return
    for param in params:
        if param.grad is None:
            continue
        dist.all_reduce(param.grad, op=dist.ReduceOp.SUM)
        param.grad.div_(world_size)


def broadcast_params(params):
    if not dist.is_available() or not dist.is_initialized():
        return
    for param in params:
        dist.broadcast(param.data, src=0)


class MotionWeaveLossLogger:
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
        self.csv_file.write(
            "step,loss,action_loss,motion_loss,lr\n"
        )

    def log(
        self,
        step,
        loss,
        action_loss,
        motion_loss,
        lr,
        avg_loss=None,
        avg_action_loss=None,
        avg_motion_loss=None,
    ):
        if not self.enabled:
            return
        values = {
            "loss/total": loss,
            "loss/action": action_loss,
            "loss/motion": motion_loss,
            "train/lr": lr,
        }
        if avg_loss is not None:
            values.update(
                {
                    "loss_avg/total": avg_loss,
                    "loss_avg/action": avg_action_loss,
                    "loss_avg/motion": avg_motion_loss,
                }
            )
        for key, value in values.items():
            self.writer.add_scalar(key, value, step)
        self.csv_file.write(
            f"{step},{loss},{action_loss},{motion_loss},{lr}\n"
        )

    def close(self):
        if self.writer is not None:
            self.writer.close()
        if self.csv_file is not None:
            self.csv_file.close()


def plot_loss_curves(log_dir, output_path=None):
    csv_path = Path(log_dir) / "loss.csv"
    output_path = Path(output_path) if output_path else Path(log_dir) / "loss_curve.png"
    if not csv_path.exists():
        return
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"Skip plotting loss curves because matplotlib is unavailable: {exc}")
        return

    with open(csv_path, newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        return
    steps = [int(row["step"]) for row in rows]
    plt.figure(figsize=(10, 6))
    for key, label in (
        ("loss", "total"),
        ("action_loss", "action"),
        ("motion_loss", "motion grounding"),
    ):
        plt.plot(steps, [float(row[key]) for row in rows], label=label)
    plt.xlabel("step")
    plt.ylabel("loss")
    plt.title("MotionWeave Training")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=160)
    plt.close()


class MultiTokenDiTActionHead(nn.Module):
    def __init__(
        self,
        action_dim,
        horizon,
        cond_dim,
        proprio_dim,
        num_condition_tokens,
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
        self.num_condition_tokens = num_condition_tokens

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
        self.prefix_pos = nn.Parameter(
            torch.zeros(1, num_condition_tokens + 1, model_dim)
        )

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
        if action_token_hidden.dim() == 2:
            action_token_hidden = action_token_hidden.unsqueeze(1)
        if action_token_hidden.dim() != 3:
            raise ValueError(
                "Expected action_token_hidden shape [B, T, C] or [B, C], "
                f"got {tuple(action_token_hidden.shape)}"
            )
        if action_token_hidden.size(1) != self.num_condition_tokens:
            raise ValueError(
                f"Expected {self.num_condition_tokens} action condition tokens, "
                f"got {action_token_hidden.size(1)}"
            )

        time_emb = self.time_in(timesteps)
        cond_tokens = self.cond_in(action_token_hidden)
        proprio_token = self.proprio_in(proprio_hidden).unsqueeze(1)
        prefix_tokens = torch.cat([cond_tokens, proprio_token], dim=1)
        prefix_tokens = prefix_tokens + self.prefix_pos + time_emb.unsqueeze(1)

        action_tokens = self.action_in(noisy_actions)
        action_tokens = action_tokens + self.action_pos + time_emb.unsqueeze(1)

        tokens = torch.cat([prefix_tokens, action_tokens], dim=1)
        tokens = self.blocks(tokens)
        action_tokens = self.final_norm(tokens[:, -self.horizon :])
        return self.action_out(action_tokens)


class ActionInducedMotionGrounder(nn.Module):
    """AIMG: ground horizon-specific interaction regions in current visual tokens."""

    def __init__(self, input_dim, proprio_dim, horizon, model_dim=512):
        super().__init__()
        self.horizon = int(horizon)
        self.model_dim = int(model_dim)
        self.visual_in = nn.Sequential(nn.LayerNorm(input_dim), nn.Linear(input_dim, model_dim))
        self.action_in = nn.Sequential(nn.LayerNorm(input_dim), nn.Linear(input_dim, model_dim))
        self.proprio_in = nn.Sequential(
            nn.LayerNorm(proprio_dim), nn.Linear(proprio_dim, model_dim)
        )
        self.horizon_queries = nn.Parameter(torch.empty(1, self.horizon, model_dim))
        self.query_mlp = nn.Sequential(
            nn.LayerNorm(model_dim),
            nn.Linear(model_dim, model_dim * 2),
            nn.GELU(),
            nn.Linear(model_dim * 2, model_dim),
        )
        self.interaction_norm = nn.LayerNorm(model_dim)
        nn.init.normal_(self.horizon_queries, std=0.02)

    def forward(self, action_hidden, visual_hidden, proprio_hidden):
        visual = self.visual_in(visual_hidden)
        action_context = self.action_in(action_hidden).mean(dim=1, keepdim=True)
        proprio_context = self.proprio_in(proprio_hidden).unsqueeze(1)
        queries = self.horizon_queries + action_context + proprio_context
        queries = queries + self.query_mlp(queries)
        logits = torch.einsum("bhd,bnd->bhn", queries, visual) / math.sqrt(
            self.model_dim
        )
        attention = logits.softmax(dim=-1)
        interaction = torch.einsum("bhn,bnd->bhd", attention, visual)
        interaction = self.interaction_norm(interaction + queries)
        return interaction, logits


class HorizonResidualComposer(nn.Module):
    """HRC: inject temporal interaction changes through a gated residual."""

    def __init__(
        self,
        action_dim,
        proprio_dim,
        interaction_dim,
        model_dim=768,
        num_heads=8,
    ):
        super().__init__()
        if model_dim % num_heads != 0:
            raise ValueError("composer model_dim must be divisible by num_heads")
        self.action_in = nn.Sequential(nn.LayerNorm(action_dim), nn.Linear(action_dim, model_dim))
        self.interaction_in = nn.Linear(interaction_dim, model_dim)
        self.proprio_in = nn.Sequential(
            nn.LayerNorm(proprio_dim), nn.Linear(proprio_dim, model_dim)
        )
        self.relation_mlp = nn.Sequential(
            nn.LayerNorm(model_dim * 3),
            nn.Linear(model_dim * 3, model_dim * 2),
            nn.GELU(),
            nn.Linear(model_dim * 2, model_dim),
        )
        self.cross_attn = nn.MultiheadAttention(
            model_dim,
            num_heads,
            dropout=0.0,
            batch_first=True,
        )
        self.gate = nn.Sequential(
            nn.LayerNorm(model_dim * 2),
            nn.Linear(model_dim * 2, model_dim),
            nn.SiLU(),
            nn.Linear(model_dim, 1),
            nn.Sigmoid(),
        )
        self.out_norm = nn.LayerNorm(model_dim)
        self.out = nn.Linear(model_dim, action_dim)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, action_hidden, interaction_tokens, proprio_hidden):
        action_queries = self.action_in(action_hidden)
        interaction = self.interaction_in(interaction_tokens)
        previous = torch.cat([interaction[:, :1], interaction[:, :-1]], dim=1)
        temporal_delta = interaction - previous
        proprio = self.proprio_in(proprio_hidden).unsqueeze(1)
        proprio_horizon = proprio.expand(-1, interaction.size(1), -1)
        relation = self.relation_mlp(
            torch.cat([interaction, temporal_delta, proprio_horizon], dim=-1)
        )
        update, _ = self.cross_attn(
            action_queries,
            relation,
            relation,
            need_weights=False,
        )
        action_proprio = proprio.expand(-1, action_queries.size(1), -1)
        gate = self.gate(torch.cat([action_queries, action_proprio], dim=-1))
        return action_hidden + self.out(self.out_norm(update)) * gate


class MotionWeavePolicy(ActionChunkInternVLPolicy):
    """MotionWeave policy with AIMG grounding followed by HRC composition."""

    def __init__(
        self,
        *args,
        action_readout_text="<ACTION> <ACTION>",
        aimg_dim=512,
        hrc_dim=768,
        hrc_heads=8,
        motion_grounding_loss_weight=0.05,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.action_readout_text = action_readout_text
        pattern = self.tokenizer(
            self.action_readout_text,
            add_special_tokens=False,
        )["input_ids"]
        if not pattern:
            raise ValueError("action_readout_text must tokenize to at least one token.")
        self.action_readout_token_ids = pattern
        self.num_action_readout_tokens = len(pattern)
        self.aimg_dim = int(aimg_dim)
        self.hrc_dim = int(hrc_dim)
        self.hrc_heads = int(hrc_heads)
        self.motion_grounding_loss_weight = float(motion_grounding_loss_weight)
        self.action_head = MultiTokenDiTActionHead(
            action_dim=self.single_action_dim,
            horizon=self.action_horizon,
            cond_dim=self.hidden_size,
            proprio_dim=self.hidden_size,
            num_condition_tokens=self.num_action_readout_tokens,
            model_dim=self.dit_dim,
            depth=self.dit_layers,
            num_heads=self.dit_heads,
            mlp_ratio=self.dit_mlp_ratio,
            dropout=self.dit_dropout,
        )
        self.aimg = ActionInducedMotionGrounder(
            input_dim=self.hidden_size,
            proprio_dim=self.hidden_size,
            horizon=self.action_horizon,
            model_dim=self.aimg_dim,
        )
        self.hrc = HorizonResidualComposer(
            action_dim=self.hidden_size,
            proprio_dim=self.hidden_size,
            interaction_dim=self.aimg_dim,
            model_dim=self.hrc_dim,
            num_heads=self.hrc_heads,
        )

    @property
    def motion_grounder(self):
        """Legacy alias for checkpoints and downstream analysis scripts."""
        return self.aimg

    @property
    def action_composer(self):
        """Legacy alias for checkpoints and downstream analysis scripts."""
        return self.hrc

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
        return [
            f"{image_tokens}\n"
            f"According to the instruction '{task}', predict the next "
            f"{self.action_horizon} robot actions. <PROP_CONTEXT>\n"
            f"{self.action_readout_text}"
            for task in task_descriptions
        ]

    @staticmethod
    def _find_subsequence(sequence, pattern, start=0):
        if not pattern:
            return -1
        limit = len(sequence) - len(pattern) + 1
        for idx in range(start, max(start, limit)):
            if sequence[idx : idx + len(pattern)] == pattern:
                return idx
        return -1

    def _action_readout_positions(self, input_ids, attention_mask):
        positions = []
        pattern = self.action_readout_token_ids
        for row, row_mask in zip(input_ids, attention_mask):
            valid_len = int(row_mask.sum().item())
            sequence = row[:valid_len].tolist()
            start = self._find_subsequence(sequence, pattern)
            if start < 0:
                raise ValueError("Could not locate action readout span in prompt.")
            positions.append(list(range(start, start + len(pattern))))
        return torch.tensor(positions, dtype=torch.long, device=input_ids.device)

    def _extract_action_and_visual_hidden(self, pixel_values, task_descriptions=None):
        prompts = self._make_prompts(pixel_values.size(0), task_descriptions)
        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        input_ids = inputs["input_ids"].to(pixel_values.device)
        attention_mask = inputs["attention_mask"].to(pixel_values.device)
        image_flags = torch.ones(
            pixel_values.size(0), dtype=torch.long, device=pixel_values.device
        )

        input_embeds = self.vlm.language_model.get_input_embeddings()(input_ids).clone()
        vit_embeds = self.vlm.extract_feature(pixel_values)
        vit_embeds = vit_embeds[image_flags == 1]

        batch_size, seq_len, hidden_size = input_embeds.shape
        flat_embeds = input_embeds.reshape(batch_size * seq_len, hidden_size)
        flat_input_ids = input_ids.reshape(batch_size * seq_len)
        selected = flat_input_ids == self.vlm.img_context_token_id
        flat_embeds[selected] = flat_embeds[selected] * 0.0 + vit_embeds.reshape(
            -1, hidden_size
        )
        input_embeds = flat_embeds.reshape(batch_size, seq_len, hidden_size)

        language_backbone = getattr(
            self.vlm.language_model, "model", self.vlm.language_model
        )
        outputs = language_backbone(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            use_cache=False,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=True,
        )
        hidden = outputs.last_hidden_state
        readout_positions = self._action_readout_positions(input_ids, attention_mask)
        gather_idx = readout_positions.unsqueeze(-1).expand(
            -1,
            -1,
            hidden.size(-1),
        )
        action_hidden = hidden.gather(1, gather_idx).float()
        visual_hidden = hidden[selected.reshape(batch_size, seq_len)].reshape(
            batch_size,
            self.num_image_token,
            hidden_size,
        )
        return action_hidden, visual_hidden.float()

    def encode_grounded_context(self, img, state, task_descriptions=None):
        pixel_values = self._preprocess_images(img)
        if self.tune_mode == "frozen":
            with torch.no_grad():
                action_hidden, visual_hidden = self._extract_action_and_visual_hidden(
                    pixel_values,
                    task_descriptions,
                )
        else:
            action_hidden, visual_hidden = self._extract_action_and_visual_hidden(
                pixel_values,
                task_descriptions,
            )
        proprio_hidden = self.proprio_encoder(state.float())
        interaction_tokens, motion_logits = self.aimg(
            action_hidden,
            visual_hidden,
            proprio_hidden,
        )
        action_hidden = self.hrc(
            action_hidden,
            interaction_tokens,
            proprio_hidden,
        )
        return action_hidden, proprio_hidden, motion_logits

    def build_motion_targets(self, motion_masks, num_visual_tokens):
        if motion_masks.dim() != 5:
            raise ValueError(
                "Expected motion_masks [B,H,1,Hm,Wm], "
                f"got {tuple(motion_masks.shape)}"
            )
        if motion_masks.size(1) != self.action_horizon:
            raise ValueError(
                f"Motion grounding needs {self.action_horizon} masks, "
                f"got {motion_masks.size(1)}"
            )
        grid_size = math.isqrt(int(num_visual_tokens))
        if grid_size * grid_size != int(num_visual_tokens):
            raise ValueError(
                f"Visual token count {num_visual_tokens} is not a square grid."
            )

        motion = motion_masks.float().clamp(0.0, 1.0)
        batch_size = motion.size(0)
        motion = F.interpolate(
            motion.reshape(-1, 1, motion.size(-2), motion.size(-1)),
            size=(grid_size, grid_size),
            mode="area",
        ).reshape(batch_size, self.action_horizon, -1)
        motion = (motion - motion.mean(dim=-1, keepdim=True)).clamp_min(0.0)
        motion = motion + 1e-6
        return motion / motion.sum(dim=-1, keepdim=True)

    def motion_grounding_loss(self, motion_logits, motion_masks):
        targets = self.build_motion_targets(motion_masks, motion_logits.size(-1))
        kl = F.kl_div(
            F.log_softmax(motion_logits.float(), dim=-1),
            targets,
            reduction="none",
        ).sum(dim=-1)
        return (kl / math.log(float(motion_logits.size(-1)))).mean()

    def forward(
        self,
        img,
        state,
        task_descriptions=None,
        target_actions=None,
        motion_masks=None,
        return_components=False,
        return_condition=False,
        condition_source="action_token",
    ):
        action_hidden, proprio_hidden, motion_logits = self.encode_grounded_context(
            img,
            state,
            task_descriptions,
        )
        motion_loss = torch.zeros(
            (),
            device=action_hidden.device,
            dtype=torch.float32,
        )
        if motion_masks is not None:
            motion_loss = self.motion_grounding_loss(motion_logits, motion_masks)

        if target_actions is not None:
            action_loss = self.action_loss(
                action_hidden,
                proprio_hidden,
                target_actions,
            )
            if return_components:
                return action_loss, motion_loss
            total_loss = (
                action_loss
                + self.motion_grounding_loss_weight * motion_loss
            )
            if not return_condition:
                return total_loss
            condition = self._resolve_condition(
                condition_source,
                action_hidden,
                proprio_hidden,
            )
            return total_loss, condition

        pred_actions = self.sample_actions(action_hidden, proprio_hidden)
        if not return_condition:
            return pred_actions
        condition = self._resolve_condition(
            condition_source,
            action_hidden,
            proprio_hidden,
            pred_actions=pred_actions,
        )
        return pred_actions, condition

    def _repeat_action_training_batch(
        self, action_token_hidden, proprio_hidden, target_actions
    ):
        if self.repeated_diffusion_steps == 1:
            return action_token_hidden, proprio_hidden, target_actions
        if action_token_hidden.dim() == 3:
            action_repeats = action_token_hidden.repeat(
                self.repeated_diffusion_steps, 1, 1
            )
        else:
            action_repeats = action_token_hidden.repeat(
                self.repeated_diffusion_steps, 1
            )
        return (
            action_repeats,
            proprio_hidden.repeat(self.repeated_diffusion_steps, 1),
            target_actions.repeat(self.repeated_diffusion_steps, 1, 1),
        )



# Backward-compatible Python symbols for older evaluation scripts.
HorizonActionResidualComposer = HorizonResidualComposer
MotionGroundedActionPolicy = MotionWeavePolicy


def remap_legacy_motionweave_state_dict(state_dict):
    """Map pre-MotionWeave module paths without changing tensor values."""
    replacements = (
        ("motion_grounder.", "aimg."),
        ("action_composer.", "hrc."),
    )
    remapped = {}
    for key, value in state_dict.items():
        new_key = key
        for old_prefix, new_prefix in replacements:
            if old_prefix in new_key:
                new_key = new_key.replace(old_prefix, new_prefix, 1)
                break
        remapped[new_key] = value
    return remapped


def has_fsdp_modules(module):
    return any(isinstance(child, FSDP) for child in module.modules())


def normalize_state_key(key):
    return key.replace("._fsdp_wrapped_module", "")


def collect_policy_trainable_state(policy, trainable_names, use_fsdp_state):
    if use_fsdp_state:
        config = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        with FSDP.state_dict_type(policy, StateDictType.FULL_STATE_DICT, config):
            state = policy.state_dict()
        if not is_main_process():
            return {}
    else:
        if not is_main_process():
            return {}
        state = policy.state_dict()

    trainable = {}
    for key, value in state.items():
        clean_key = normalize_state_key(key)
        if clean_key in trainable_names:
            trainable[clean_key] = value.detach().cpu()
    return trainable


def policy_config(policy):
    config = {
        "model_name": policy.model_name,
        "obs_dim": policy.obs_dim,
        "act_dim": policy.act_dim,
        "task_description": policy.task_description,
        "image_size": policy.image_size,
        "tune_mode": policy.tune_mode,
        "lora_r": policy.lora_r,
        "lora_alpha": policy.lora_alpha,
        "lora_dropout": policy.lora_dropout,
        "lora_target_modules": policy.lora_target_modules,
        "action_head_type": f"simple_dit_{policy.action_head_objective}",
        "single_action_dim": policy.single_action_dim,
        "action_horizon": policy.action_horizon,
        "diffusion_train_steps": policy.diffusion_train_steps,
        "diffusion_inference_steps": policy.diffusion_inference_steps,
        "diffusion_timestep_max": policy.diffusion_timestep_max,
        "diffusion_sample_start_step": policy.diffusion_sample_start_step,
        "diffusion_x0_clip": policy.diffusion_x0_clip,
        "action_head_objective": policy.action_head_objective,
        "repeated_diffusion_steps": policy.repeated_diffusion_steps,
        "action_norm_type": policy.action_norm_type,
        "action_bound_eps": policy.action_bound_eps,
        "flow_t_alpha": policy.flow_t_alpha,
        "flow_t_beta": policy.flow_t_beta,
        "flow_t_eps": policy.flow_t_eps,
        "flow_sample_clip": policy.flow_sample_clip,
        "dit_dim": policy.dit_dim,
        "dit_layers": policy.dit_layers,
        "dit_heads": policy.dit_heads,
        "dit_mlp_ratio": policy.dit_mlp_ratio,
        "dit_dropout": policy.dit_dropout,
    }
    if hasattr(policy, "action_readout_text"):
        config.update(
            {
                "method_name": "MotionWeave",
                "action_token_style": "motionweave_aimg_hrc",
                "action_readout_text": policy.action_readout_text,
                "num_action_readout_tokens": getattr(
                    policy,
                    "num_action_readout_tokens",
                    None,
                ),
                "aimg_dim": getattr(policy, "aimg_dim", None),
                "hrc_dim": getattr(policy, "hrc_dim", None),
                "hrc_heads": getattr(policy, "hrc_heads", None),
                "motion_grounding_loss_weight": getattr(
                    policy,
                    "motion_grounding_loss_weight",
                    None,
                ),
            }
        )
    return config


def cpu_state_dict(module):
    return {key: value.detach().cpu() for key, value in module.state_dict().items()}


def save_checkpoint(
    path,
    step,
    vla,
    trainable_names,
    args,
):
    policy = unwrap_ddp(vla)
    use_fsdp_state = has_fsdp_modules(policy)
    trainable_params = collect_policy_trainable_state(
        policy,
        trainable_names,
        use_fsdp_state=use_fsdp_state,
    )
    if not is_main_process():
        return

    save_path = Path(path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    vla_payload = {
        "trainable_params": trainable_params,
        "config": policy_config(policy),
        "action_mean": policy.action_mean.detach().cpu(),
        "action_std": policy.action_std.detach().cpu(),
        "action_low": policy.action_low.detach().cpu(),
        "action_high": policy.action_high.detach().cpu(),
    }
    payload = {
        "step": step,
        "args": vars(args),
        "vla": vla_payload,
        "svd_lora_state_dict": {},
        "action_conditioner": {},
        "distiller": {},
        "svd_lora_modules": [],
    }
    torch.save(payload, save_path)
    print(
        f"Saved MotionWeave checkpoint to {save_path}",
        flush=True,
    )


def load_init_checkpoint(policy, ckpt_path, device):
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    vla_payload = checkpoint.get("vla")
    if not isinstance(vla_payload, dict):
        raise KeyError(f"{ckpt_path} does not contain a joint 'vla' payload")
    trainable_params = vla_payload.get("trainable_params")
    if trainable_params is None:
        raise KeyError(f"{ckpt_path} does not contain vla.trainable_params")
    trainable_params = remap_legacy_motionweave_state_dict(trainable_params)
    incompatible = policy.load_state_dict(trainable_params, strict=False)
    unexpected = list(incompatible.unexpected_keys)
    if unexpected:
        raise RuntimeError(f"Unexpected checkpoint keys when loading {ckpt_path}: {unexpected[:10]}")
    return int(checkpoint.get("step") or 0), len(trainable_params), len(incompatible.missing_keys)


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Train MotionWeave with the Action-Induced Motion Grounder "
            "and Horizon Residual Composer."
        )
    )
    parser.add_argument("--tasks", default=",".join(DEFAULT_TASKS))
    parser.add_argument("--task_descriptions", default=None)
    parser.add_argument("--data_root", default="/home/yiyuan/pvrobo/expert_demos/metaworld")
    parser.add_argument(
        "--motion_mask_root",
        required=True,
        help="Root containing per-task future robot-arm mask sequences.",
    )
    parser.add_argument("--vlm_model", default="/data/yiyuan/models/InternVL3-2B")
    parser.add_argument("--local_files_only", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_epi", type=int, default=25)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--total_steps", type=int, default=1000)
    parser.add_argument("--save_every", type=int, default=500)
    parser.add_argument("--log_every", type=int, default=1)
    parser.add_argument("--print_every", type=int, default=100)
    parser.add_argument(
        "--log_dir",
        default="/home/yiyuan/pvrobo/bc_code/runs/internvl3_full_vlm_fsdp_action_only_all_action_tokens",
    )
    parser.add_argument("--plot_loss", dest="plot_loss", action="store_true", default=True)
    parser.add_argument("--no_plot_loss", dest="plot_loss", action="store_false")
    parser.add_argument("--skip_save", action="store_true")
    parser.add_argument(
        "--init_ckpt",
        default=None,
        help="Warm-start from a saved checkpoint. Loads only VLA trainable params; optimizer is reset.",
    )
    parser.add_argument(
        "--init_step_offset",
        type=int,
        default=None,
        help="Step offset for logging/saving after --init_ckpt. Defaults to checkpoint['step']; set 0 to reset numbering.",
    )
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)

    parser.add_argument("--strategy", choices=["fsdp", "ddp"], default="fsdp")
    parser.add_argument(
        "--fsdp_wrap_class_names",
        default="Qwen2DecoderLayer,InternVisionEncoderLayer",
    )
    parser.add_argument(
        "--fsdp_mixed_precision",
        dest="fsdp_mixed_precision",
        action="store_true",
        default=True,
    )
    parser.add_argument(
        "--no_fsdp_mixed_precision",
        dest="fsdp_mixed_precision",
        action="store_false",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        dest="gradient_checkpointing",
        action="store_true",
        default=True,
    )
    parser.add_argument(
        "--no_gradient_checkpointing",
        dest="gradient_checkpointing",
        action="store_false",
    )

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
    )
    parser.add_argument("--repeated_diffusion_steps", type=int, default=1)
    parser.add_argument(
        "--action_norm_type",
        default="mean_std",
        choices=["mean_std", "bounds"],
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

    parser.add_argument("--action_readout_text", default="<ACTION> <ACTION>")
    parser.add_argument(
        "--aimg_dim",
        "--motion_grounder_dim",
        dest="aimg_dim",
        type=int,
        default=512,
    )
    parser.add_argument(
        "--hrc_dim",
        "--composer_dim",
        dest="hrc_dim",
        type=int,
        default=768,
    )
    parser.add_argument(
        "--hrc_heads",
        "--composer_heads",
        dest="hrc_heads",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--motion_grounding_loss_weight",
        "--motion_loss_weight",
        dest="motion_grounding_loss_weight",
        type=float,
        default=0.05,
    )
    parser.add_argument("--vla_tune_mode", default="full", choices=["frozen", "lora", "full"])
    parser.add_argument("--vla_lora_r", type=int, default=16)
    parser.add_argument("--vla_lora_alpha", type=int, default=32)
    parser.add_argument("--vla_lora_dropout", type=float, default=0.05)
    parser.add_argument("--torch_dtype", default="bfloat16")

    parser.add_argument(
        "--save_path",
        default="/home/yiyuan/pvrobo/bc_code/checkpoints/internvl3_full_vlm_fsdp_action_only_all_action_tokens_smoke.pth",
    )
    return parser


def train(args):
    distributed, rank, world_size, local_rank = setup_distributed()
    if distributed:
        args.device = f"cuda:{local_rank}"
    device = torch.device(args.device)
    fsdp_enabled = args.strategy == "fsdp"

    if fsdp_enabled and args.vla_tune_mode != "full" and is_main_process():
        print(
            "Warning: this entrypoint is intended for --vla_tune_mode full; "
            f"got {args.vla_tune_mode}.",
            flush=True,
        )

    try:
        set_seed(args.seed)
        task_names = parse_csv(args.tasks)
        task_descriptions = parse_task_descriptions(args.task_descriptions, task_names)
        dataset = MotionWeaveDataset(
            args.data_root,
            task_names,
            task_descriptions,
            max_episodes=args.max_epi,
            action_horizon=args.action_horizon,
            svd_num_frames=args.action_horizon + 1,
            svd_image_size=128,
            motion_mask_root=args.motion_mask_root,
        )
        sampler = (
            DistributedSampler(
                dataset,
                num_replicas=world_size,
                rank=rank,
                shuffle=True,
                seed=args.seed,
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
            pin_memory=True,
        )
        data_iter = iter(dataloader)
        epoch = 0

        policy_cls = MotionWeavePolicy
        policy_kwargs = {
            "action_readout_text": args.action_readout_text,
            "aimg_dim": args.aimg_dim,
            "hrc_dim": args.hrc_dim,
            "hrc_heads": args.hrc_heads,
            "motion_grounding_loss_weight": args.motion_grounding_loss_weight,
        }
        vla = policy_cls(
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
            **policy_kwargs,
        ).to(device)
        action_mean, action_std = dataset.action_stats()
        action_low, action_high = dataset.action_bounds(
            args.action_quantile_low,
            args.action_quantile_high,
        )
        vla.set_action_stats(action_mean, action_std)
        vla.set_action_bounds(action_low, action_high)

        init_step_offset = 0
        if args.init_ckpt:
            ckpt_step, loaded_params, missing_keys = load_init_checkpoint(
                vla,
                args.init_ckpt,
                device,
            )
            init_step_offset = (
                ckpt_step if args.init_step_offset is None else args.init_step_offset
            )
            if is_main_process():
                print(
                    f"Initialized VLA from {args.init_ckpt}: "
                    f"ckpt_step={ckpt_step}, step_offset={init_step_offset}, "
                    f"loaded_params={loaded_params}, missing_keys={missing_keys}",
                    flush=True,
                )
        elif args.init_step_offset is not None:
            init_step_offset = args.init_step_offset

        gc_enabled = []
        if args.gradient_checkpointing:
            gc_enabled = enable_gradient_checkpointing(vla)

        trainable_names = {
            name for name, param in vla.named_parameters() if param.requires_grad
        }

        fsdp_wrapped = []
        manual_sync_policy_params = []
        if distributed and args.strategy == "ddp":
            vla = DDP(
                vla,
                device_ids=[local_rank],
                output_device=local_rank,
                find_unused_parameters=False,
                static_graph=True,
            )
        elif fsdp_enabled:
            class_names = [
                item.strip()
                for item in args.fsdp_wrap_class_names.split(",")
                if item.strip()
            ]
            fsdp_wrapped = wrap_transformer_layers_fsdp(
                vla,
                device,
                class_names,
                mixed_precision_enabled=args.fsdp_mixed_precision,
            )
            manual_sync_policy_params = non_fsdp_trainable_params(vla)
            broadcast_params(manual_sync_policy_params)

        set_seed(args.seed + rank)
        trainable = [param for param in vla.parameters() if param.requires_grad]
        optimizer = torch.optim.AdamW(
            trainable,
            lr=args.lr,
            weight_decay=args.weight_decay,
            foreach=False,
        )

        if is_main_process():
            print(
                "Full-VLM training: "
                f"strategy={args.strategy}, distributed={distributed}, "
                f"world_size={world_size}, per_rank_batch={args.batch_size}, "
                f"global_batch={args.batch_size * world_size}, "
                "mode=motionweave_aimg_hrc"
            )
            print(f"Gradient checkpointing: {gc_enabled}")
            if fsdp_enabled:
                print(f"FSDP wrapped layers: {len(fsdp_wrapped)}")
                print(
                    "Manual DDP grad sync params outside FSDP: "
                    f"{sum(param.numel() for param in manual_sync_policy_params):,}"
                )
            print(
                "VLA trainable params: "
                f"{sum(param.numel() for param in vla.parameters() if param.requires_grad):,}"
            )
            print(
                f"Action head: simple_dit_{args.action_head_objective} "
                f"horizon={args.action_horizon}, act_dim={dataset.act_dim}, "
                f"dim={args.dit_dim}, layers={args.dit_layers}, heads={args.dit_heads}, "
                f"repeated_steps={args.repeated_diffusion_steps}, "
                f"norm={args.action_norm_type}, "
                f"train_steps={args.diffusion_train_steps}, "
                f"infer_steps={args.diffusion_inference_steps}, "
                f"flow_sample_clip={unwrap_ddp(vla).flow_sample_clip}"
            )
            print(
                "MotionWeave (AIMG + HRC): "
                f"action_readout={args.action_readout_text!r}; "
                f"tokens={unwrap_ddp(vla).num_action_readout_tokens}; "
                f"horizons={args.action_horizon}; "
                f"aimg_dim={args.aimg_dim}; "
                f"hrc_dim={args.hrc_dim}; "
                f"motion_weight={args.motion_grounding_loss_weight}."
            )
        logger = MotionWeaveLossLogger(args.log_dir, enabled=is_main_process())
        loss_window = deque(maxlen=args.print_every)
        action_window = deque(maxlen=args.print_every)
        motion_window = deque(maxlen=args.print_every)

        vla.train()

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
            motion_masks = batch["motion_masks"].to(device, non_blocking=True)
            task_texts = list(batch["task_text"])

            optimizer.zero_grad(set_to_none=True)
            action_loss, motion_loss = vla(
                vla_img,
                vla_obs,
                task_texts,
                target_actions=target_actions,
                motion_masks=motion_masks,
                return_components=True,
            )
            loss = (
                action_loss
                + args.motion_grounding_loss_weight * motion_loss
            )

            loss.backward()
            if fsdp_enabled and distributed:
                average_gradients(manual_sync_policy_params, world_size)
            if args.max_grad_norm and args.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(trainable, args.max_grad_norm)
            optimizer.step()

            progress.update(1)
            global_step = init_step_offset + step
            lr = optimizer.param_groups[0]["lr"]
            loss_value = loss.item()
            action_value = action_loss.item()
            motion_value = motion_loss.item()
            loss_window.append(loss_value)
            action_window.append(action_value)
            motion_window.append(motion_value)
            avg_loss = sum(loss_window) / len(loss_window)
            avg_action = sum(action_window) / len(action_window)
            avg_motion = sum(motion_window) / len(motion_window)
            progress.set_postfix(
                loss=f"{loss_value:.5f}",
                action=f"{action_value:.5f}",
                motion=f"{motion_value:.5f}",
                action_avg=f"{avg_action:.5f}",
            )
            if is_main_process() and step % args.log_every == 0:
                logger.log(
                    global_step,
                    loss_value,
                    action_value,
                    motion_value,
                    lr,
                    avg_loss=avg_loss,
                    avg_action_loss=avg_action,
                    avg_motion_loss=avg_motion,
                )
            if is_main_process() and step % args.print_every == 0:
                progress.write(
                    f"Step {global_step}/{init_step_offset + args.total_steps} "
                    f"(local {step}/{args.total_steps}) - "
                    f"Loss: {avg_loss:.6f} "
                    f"Action: {avg_action:.6f} "
                    f"Motion: {avg_motion:.6f}"
                )

            should_save = (
                not args.skip_save
                and args.save_every > 0
                and (step % args.save_every == 0 or step == args.total_steps)
            )
            if should_save:
                save_target = Path(args.save_path).with_name(f"step_{global_step}.pth")
                save_checkpoint(
                    save_target,
                    global_step,
                    vla,
                    trainable_names,
                    args,
                )
                if distributed:
                    dist.barrier()
                if is_main_process() and args.plot_loss:
                    plot_loss_curves(args.log_dir)

            del (
                batch,
                vla_img,
                vla_obs,
                target_actions,
                motion_masks,
                action_loss,
                motion_loss,
                loss,
            )
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        progress.close()
        logger.close()
        if is_main_process() and args.plot_loss:
            plot_loss_curves(args.log_dir)
    finally:
        cleanup_distributed()


def main():
    parser = build_arg_parser()
    args = parser.parse_args()
    if not 0.0 <= args.action_quantile_low < args.action_quantile_high <= 1.0:
        raise ValueError(
            "--action_quantile_low and --action_quantile_high must satisfy "
            "0 <= low < high <= 1"
        )
    train(args)


if __name__ == "__main__":
    main()
