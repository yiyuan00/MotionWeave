import argparse
import os
import random
from datetime import timedelta
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from tqdm.auto import tqdm

from dataset import MetaWorldExpertDataset
from model_internvl3_mlp import InternVL3MLPPolicy


DEFAULT_TASKS = [
    "hammer-v2",
    "dial-turn-v2",
    "door-open-v2",
    "faucet-close-v2",
    "assembly-v2",
]

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


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class MultiTaskMetaWorldDataset(Dataset):
    def __init__(self, data_root, task_names, max_episodes=25):
        self.task_names = list(task_names)
        self.task_to_id = {task: idx for idx, task in enumerate(self.task_names)}

        images, observations, actions, task_ids = [], [], [], []
        for task_name in self.task_names:
            dataset_path = Path(data_root) / task_name / "expert_demos.pkl"
            if not dataset_path.exists():
                raise FileNotFoundError(f"Missing dataset for {task_name}: {dataset_path}")
            dataset = MetaWorldExpertDataset(str(dataset_path), max_episodes=max_episodes)
            images.append(dataset.images)
            observations.append(dataset.observations)
            actions.append(dataset.actions)
            task_ids.append(
                np.full(len(dataset), self.task_to_id[task_name], dtype=np.int64)
            )

        self.images = np.concatenate(images, axis=0)
        self.observations = np.concatenate(observations, axis=0)
        self.actions = np.concatenate(actions, axis=0)
        self.task_ids = np.concatenate(task_ids, axis=0)

        print(
            "Loaded multitask dataset: "
            f"tasks={self.task_names}, "
            f"images={self.images.shape}, "
            f"observations={self.observations.shape}, "
            f"actions={self.actions.shape}"
        )

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = torch.tensor(self.images[idx], dtype=torch.float32) / 255.0
        obs = torch.tensor(self.observations[idx], dtype=torch.float32)
        act = torch.tensor(self.actions[idx], dtype=torch.float32)
        task_id = torch.tensor(self.task_ids[idx], dtype=torch.long)
        return img, obs, act, task_id


class MultiTaskInternVL3MLPPolicy(InternVL3MLPPolicy):
    def _make_prompts(self, batch_size, task_descriptions=None):
        image_tokens = "<img>" + ("<IMG_CONTEXT>" * self.num_image_token) + "</img>"
        if task_descriptions is None:
            task = self.task_description or "What is the next action for the robot?"
            task_descriptions = [task] * batch_size
        elif isinstance(task_descriptions, str):
            task_descriptions = [task_descriptions] * batch_size
        elif len(task_descriptions) != batch_size:
            raise ValueError(
                f"Expected {batch_size} task descriptions, got {len(task_descriptions)}"
            )

        return [
            f"{image_tokens}\n"
            f"According to the instruction '{task}', predict the next robot action. "
            f"<PROP_CONTEXT> <ACTION_CHUNK_0>"
            for task in task_descriptions
        ]

    def _extract_action_hidden(self, pixel_values, task_descriptions=None):
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

        last_token_idx = attention_mask.sum(dim=1) - 1
        batch_idx = torch.arange(hidden.size(0), device=hidden.device)
        return hidden[batch_idx, last_token_idx].view(
            pixel_values.size(0), self.hidden_size
        ).float()

    def forward(self, img, state, task_descriptions=None):
        pixel_values = self._preprocess_images(img)
        if self.tune_mode == "frozen":
            with torch.no_grad():
                action_hidden = self._extract_action_hidden(
                    pixel_values, task_descriptions
                )
        else:
            action_hidden = self._extract_action_hidden(pixel_values, task_descriptions)

        proprio_hidden = self.proprio_encoder(state.float())
        fused = torch.cat([action_hidden, proprio_hidden], dim=-1)
        return self.action_head(fused)


def save_trainable(model, path, step, task_scores, task_descriptions):
    payload = unwrap_model(model).trainable_state_dict()
    payload["step"] = step
    payload["task_success_rates"] = task_scores
    payload["task_descriptions"] = task_descriptions
    torch.save(payload, path)


def parse_csv(value):
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_task_descriptions(value, task_names):
    descriptions = {task: task for task in task_names}
    if value:
        for item in value.split(";"):
            item = item.strip()
            if not item:
                continue
            if "=" not in item:
                raise ValueError(
                    "Expected --task_descriptions format: task=description;task=description"
                )
            task, description = item.split("=", 1)
            descriptions[task.strip()] = description.strip()
    return {task: descriptions.get(task, task) for task in task_names}


def train_bc_multitask(
    task_names,
    data_root,
    save_path,
    total_steps=20000,
    batch_size=4,
    grad_accum_steps=1,
    lr=1e-5,
    device="cuda",
    seed=42,
    max_epi=25,
    vlm_model="OpenGVLab/InternVL3-2B",
    task_descriptions=None,
    tune_mode="lora",
    lora_r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    lora_target_modules=None,
    torch_dtype="bfloat16",
    save_every=5000,
    local_files_only=False,
    skip_save=False,
):
    distributed, rank, world_size, local_rank = setup_distributed()
    if distributed:
        device = f"cuda:{local_rank}"

    set_seed(seed)
    device_obj = torch.device(device)

    dataset = MultiTaskMetaWorldDataset(data_root, task_names, max_episodes=max_epi)
    sampler = (
        DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            seed=seed,
        )
        if distributed
        else None
    )
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=0,
        persistent_workers=False,
    )
    data_iter = iter(dataloader)

    obs_dim = dataset.observations.shape[1]
    act_dim = dataset.actions.shape[1]
    model = MultiTaskInternVL3MLPPolicy(
        obs_dim=obs_dim,
        act_dim=act_dim,
        model_name=vlm_model,
        task_description="multi-task robot policy",
        tune_mode=tune_mode,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        lora_target_modules=lora_target_modules,
        torch_dtype=torch_dtype,
        local_files_only=local_files_only,
    ).to(device_obj)

    if distributed:
        model = DDP(
            model,
            device_ids=[local_rank],
            find_unused_parameters=False,
            static_graph=True,
        )

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if is_main_process():
        trainable_count = sum(p.numel() for p in trainable_params)
        total_count = sum(p.numel() for p in model.parameters())
        print(f"Tasks: {task_names}")
        print(f"Task descriptions: {task_descriptions}")
        print(
            f"Train mode: {tune_mode}; trainable params: "
            f"{trainable_count:,}/{total_count:,}"
        )

    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=lr,
        weight_decay=1e-4,
        foreach=False,
    )
    loss_fn = nn.MSELoss()

    step = 0
    running_loss = 0.0
    save_path = Path(save_path)
    if is_main_process() and not skip_save:
        save_path.parent.mkdir(parents=True, exist_ok=True)

    grad_accum_steps = max(1, int(grad_accum_steps))
    model.train()
    progress = tqdm(
        total=total_steps,
        initial=step,
        desc="train",
        disable=not is_main_process(),
        dynamic_ncols=True,
    )

    try:
        while step < total_steps:
            optimizer.zero_grad(set_to_none=True)
            step_loss = 0.0
            for micro_step in range(grad_accum_steps):
                if sampler is not None and step == 0 and micro_step == 0:
                    sampler.set_epoch(0)
                try:
                    imgs, obs, acts, task_ids = next(data_iter)
                except StopIteration:
                    if sampler is not None:
                        sampler.set_epoch(step)
                    data_iter = iter(dataloader)
                    imgs, obs, acts, task_ids = next(data_iter)

                imgs = imgs.to(device_obj)
                obs = obs.to(device_obj)
                acts = acts.to(device_obj)
                batch_task_texts = [
                    task_descriptions[task_names[int(task_id)]]
                    for task_id in task_ids.cpu().tolist()
                ]

                pred = model(imgs, obs, batch_task_texts)
                loss = loss_fn(pred, acts)
                (loss / grad_accum_steps).backward()
                step_loss += loss.item()

            optimizer.step()

            running_loss += step_loss / grad_accum_steps
            step += 1
            progress.update(1)
            progress.set_postfix(loss=f"{step_loss / grad_accum_steps:.6f}")

            if step % 100 == 0 and is_main_process():
                avg_loss = running_loss / 100
                progress.write(f"Step {step}/{total_steps} - Loss: {avg_loss:.6f}")
                running_loss = 0.0

            if save_every > 0 and step % save_every == 0:
                if is_main_process() and not skip_save:
                    candidate_path = save_path.with_name(
                        f"{save_path.stem}_step{step}{save_path.suffix}"
                    )
                    save_trainable(
                        model,
                        candidate_path,
                        step,
                        {},
                        task_descriptions,
                    )
                    progress.write(f"Saved trainable weights to {candidate_path}")
                if distributed:
                    dist.barrier()
        progress.close()

        if is_main_process() and not skip_save:
            save_trainable(model, save_path, step, {}, task_descriptions)

        if is_main_process():
            print("\n=== Training Finished ===")
            if skip_save:
                print("Checkpoint saving skipped.")
            else:
                print(f"Final trainable weights saved to: {save_path}")
        return save_path
    finally:
        if distributed:
            cleanup_distributed()


def main():
    parser = argparse.ArgumentParser(
        description="Train one InternVL3-MLP policy on multiple MetaWorld tasks."
    )
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--tasks", type=str, default=",".join(DEFAULT_TASKS))
    parser.add_argument(
        "--data_root",
        type=str,
        default="/home/yiyuan/pvrobo/expert_demos/metaworld",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_epi", type=int, default=25)
    parser.add_argument("--total_steps", type=int, default=20000)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--grad_accum_steps", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--save_every", type=int, default=5000)
    parser.add_argument("--vlm_model", type=str, default="OpenGVLab/InternVL3-2B")
    parser.add_argument(
        "--task_descriptions",
        type=str,
        default=None,
        help="Semicolon separated overrides: task=description;task=description",
    )
    parser.add_argument("--torch_dtype", type=str, default="bfloat16")
    parser.add_argument(
        "--tune_mode",
        type=str,
        default="lora",
        choices=["frozen", "lora", "full"],
        help="frozen=head only, lora=LoRA plus head, full=full VLM finetune plus head",
    )
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument(
        "--lora_target_modules",
        type=str,
        default=None,
        help="Comma-separated target module names. Default covers Qwen attention and MLP.",
    )
    parser.add_argument("--local_files_only", action="store_true")
    parser.add_argument("--skip_save", action="store_true")
    parser.add_argument(
        "--save_path",
        type=str,
        default="/home/yiyuan/pvrobo/bc_code/checkpoints/"
        "internvl3_2b_mlp_5task_lora_seed42.pth",
    )
    args = parser.parse_args()

    task_names = parse_csv(args.tasks)
    task_descriptions = parse_task_descriptions(args.task_descriptions, task_names)

    lora_target_modules = None
    if args.lora_target_modules:
        lora_target_modules = parse_csv(args.lora_target_modules)

    train_bc_multitask(
        task_names=task_names,
        data_root=args.data_root,
        save_path=args.save_path,
        total_steps=args.total_steps,
        batch_size=args.batch_size,
        grad_accum_steps=args.grad_accum_steps,
        lr=args.lr,
        device=args.device,
        seed=args.seed,
        max_epi=args.max_epi,
        vlm_model=args.vlm_model,
        task_descriptions=task_descriptions,
        tune_mode=args.tune_mode,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        lora_target_modules=lora_target_modules,
        torch_dtype=args.torch_dtype,
        save_every=args.save_every,
        local_files_only=args.local_files_only,
        skip_save=args.skip_save,
    )


if __name__ == "__main__":
    main()
