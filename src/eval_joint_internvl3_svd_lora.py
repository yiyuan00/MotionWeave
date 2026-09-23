import argparse
import json
import os
import random
import time
from collections import deque
from pathlib import Path

import cv2
import imageio
import numpy as np
import torch

from train_eval_internvl3_mlp_multitask import DEFAULT_TASKS, parse_csv
from train_joint_internvl3_svd_lora_dit import ActionChunkInternVLPolicy


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True


def setup_mujoco_render(device):
    if device.startswith("cuda"):
        os.environ["EGL_DEVICE_ID"] = device.split(":", 1)[1]
    os.environ.setdefault("MUJOCO_GL", "egl")


def normalize_reset_output(reset_output):
    if isinstance(reset_output, tuple):
        return reset_output[0]
    return reset_output


def normalize_step_output(step_output):
    if len(step_output) == 5:
        obs, reward, terminated, truncated, info = step_output
        return obs, reward, terminated or truncated, info
    return step_output


def obs_to_policy_state(obs, device):
    obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
    state_t = obs_tensor[:, :18]
    state_t1 = obs_tensor[:, 18:36]
    return torch.cat([state_t[:, :4], state_t1[:, :4]], dim=1)


class VideoRecorder:
    def __init__(self, root_dir, camera_name, render_size=256, fps=20, enabled=True):
        self.save_dir = Path(root_dir) if root_dir is not None and enabled else None
        if self.save_dir is not None:
            self.save_dir.mkdir(parents=True, exist_ok=True)
        self.camera_name = camera_name
        self.render_size = render_size
        self.fps = fps
        self.frames = []
        self.enabled = enabled

    def init(self, env):
        self.frames = []
        if self.enabled:
            self.record(env)

    def record(self, env):
        if not self.enabled or self.save_dir is None:
            return
        frame = env.render(offscreen=True, camera_name=self.camera_name)
        frame = cv2.resize(frame, (self.render_size, self.render_size))
        self.frames.append(frame)

    def save(self, file_name):
        if self.enabled and self.save_dir is not None and self.frames:
            imageio.mimsave(str(self.save_dir / file_name), self.frames, fps=self.fps)


def infer_action_dims(checkpoint, args):
    ckpt_args = checkpoint.get("args", {})
    vla_config = checkpoint.get("vla", {}).get("config", {})
    action_horizon = args.action_horizon or ckpt_args.get("action_horizon") or 4
    action_horizon = int(action_horizon)
    if args.act_dim is not None:
        act_dim = int(args.act_dim)
    else:
        flat_act_dim = int(vla_config.get("act_dim", action_horizon * 4))
        act_dim = flat_act_dim // action_horizon
    return action_horizon, act_dim


def load_policy(args, checkpoint, device):
    vla_payload = checkpoint.get("vla")
    if not isinstance(vla_payload, dict):
        raise KeyError(f"{args.ckpt} does not contain a joint 'vla' payload")
    trainable_params = vla_payload.get("trainable_params")
    if trainable_params is None:
        raise KeyError(f"{args.ckpt} does not contain vla.trainable_params")

    config = vla_payload.get("config", {})
    action_horizon, act_dim = infer_action_dims(checkpoint, args)
    model_name = args.vlm_model or config.get("model_name") or "/data/yiyuan/models/InternVL3-2B"

    model = ActionChunkInternVLPolicy(
        obs_dim=int(config.get("obs_dim", args.obs_dim)),
        single_action_dim=act_dim,
        action_horizon=action_horizon,
        model_name=model_name,
        task_description=config.get("task_description", "joint robot policy"),
        tune_mode=config.get("tune_mode", args.tune_mode),
        lora_r=int(config.get("lora_r", args.lora_r)),
        lora_alpha=int(config.get("lora_alpha", args.lora_alpha)),
        lora_dropout=float(config.get("lora_dropout", args.lora_dropout)),
        lora_target_modules=config.get("lora_target_modules", None),
        torch_dtype=args.torch_dtype,
        local_files_only=args.local_files_only,
    ).to(device)

    incompatible = model.load_state_dict(trainable_params, strict=False)
    unexpected = list(incompatible.unexpected_keys)
    if unexpected:
        raise RuntimeError(f"Unexpected checkpoint keys: {unexpected[:10]}")
    model.eval()
    return model, action_horizon, act_dim


def evaluate_task(
    model,
    task_name,
    task_description,
    device,
    action_horizon,
    execute_horizon,
    num_episodes,
    max_steps,
    seed,
    save_dir,
    save_video,
    save_video_episodes,
    camera_name,
    clip_action,
):
    import metaworld

    ml1 = metaworld.MT1(task_name)
    env_cls = ml1.train_classes[task_name]
    tasks = ml1.train_tasks
    video_dir = Path(save_dir) / task_name / "eval_video"

    successes = 0
    episode_results = []
    task_start = time.time()
    for ep in range(num_episodes):
        env = env_cls()
        if hasattr(env, "seed"):
            env.seed(seed + ep)
        env.set_task(tasks[ep % len(tasks)])
        obs = normalize_reset_output(env.reset())

        record_episode = save_video and ep < save_video_episodes
        video_recorder = VideoRecorder(video_dir, camera_name, enabled=record_episode)
        img_stack = deque([], maxlen=3)
        video_recorder.init(env)

        goal_achieved = 0
        step_count = 0
        done = False
        while not done and step_count < max_steps:
            frame = env.render(offscreen=True, camera_name=camera_name)
            frame = cv2.resize(frame, (224, 224))
            frame = np.transpose(frame, (2, 0, 1)).astype(np.float32) / 255.0
            img_stack.append(frame)
            while len(img_stack) < 3:
                img_stack.append(frame)
            stacked_img = np.concatenate(img_stack, axis=0)

            img_tensor = torch.from_numpy(stacked_img).to(device).unsqueeze(0)
            state_tensor = obs_to_policy_state(obs, device)
            with torch.inference_mode():
                action_chunk = (
                    model(img_tensor, state_tensor, [task_description])
                    .float()
                    .cpu()
                    .numpy()[0]
                )

            num_to_execute = min(execute_horizon, action_horizon, max_steps - step_count)
            for action_idx in range(num_to_execute):
                action = action_chunk[action_idx]
                if clip_action:
                    action = np.clip(action, -1.0, 1.0)
                obs, reward, done, info = normalize_step_output(env.step(action))
                video_recorder.record(env)
                goal_achieved += info.get("success", 0)
                step_count += 1
                if done or step_count >= max_steps:
                    break

        success = int(goal_achieved > 0)
        successes += success
        episode_results.append({"episode": ep, "success": success, "steps": step_count})
        print(
            f"{task_name} episode {ep}: {'success' if success else 'failure'} "
            f"steps={step_count}",
            flush=True,
        )
        if record_episode:
            video_recorder.save(f"eval_{ep}.mp4")
        env.close()

    return {
        "task": task_name,
        "instruction": task_description,
        "success_rate": successes / float(num_episodes),
        "successes": successes,
        "episodes": num_episodes,
        "elapsed_sec": time.time() - task_start,
        "episode_results": episode_results,
    }


def task_descriptions_from_checkpoint(checkpoint, tasks):
    ckpt_args = checkpoint.get("args", {})
    raw = ckpt_args.get("task_descriptions")
    descriptions = {task: task for task in tasks}
    if raw:
        for item in str(raw).split(";"):
            item = item.strip()
            if not item or "=" not in item:
                continue
            task, description = item.split("=", 1)
            descriptions[task.strip()] = description.strip()
    return descriptions


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a joint InternVL3 + SVD LoRA checkpoint on MetaWorld."
    )
    parser.add_argument(
        "--ckpt",
        default="/home/yiyuan/pvrobo/bc_code/checkpoints/joint_internvl3_svd_lora_smoke.pth",
    )
    parser.add_argument("--tasks", default=None)
    parser.add_argument("--vlm_model", default=None)
    parser.add_argument("--local_files_only", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--torch_dtype", default="bfloat16")
    parser.add_argument("--tune_mode", default="lora")
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--obs_dim", type=int, default=8)
    parser.add_argument("--act_dim", type=int, default=None)
    parser.add_argument("--action_horizon", type=int, default=None)
    parser.add_argument(
        "--execute_horizon",
        type=int,
        default=1,
        help="How many predicted chunk actions to execute open-loop before replanning.",
    )
    parser.add_argument("--num_episodes", "--eval_episodes", type=int, default=25)
    parser.add_argument("--max_steps", type=int, default=175)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--camera", default="corner")
    parser.add_argument(
        "--save_dir",
        default="/home/yiyuan/pvrobo/bc_code/eval_results_joint_internvl3_svd_lora",
    )
    parser.add_argument("--save_video", action="store_true")
    parser.add_argument("--save_video_episodes", type=int, default=1)
    parser.add_argument("--no_clip_action", action="store_true")
    args = parser.parse_args()

    setup_mujoco_render(args.device)
    if args.device.startswith("cuda"):
        torch.cuda.set_device(int(args.device.split(":", 1)[1]))
    set_seed(args.seed)

    checkpoint = torch.load(args.ckpt, map_location="cpu")
    ckpt_args = checkpoint.get("args", {})
    tasks = parse_csv(args.tasks or ckpt_args.get("tasks") or ",".join(DEFAULT_TASKS))
    task_descriptions = task_descriptions_from_checkpoint(checkpoint, tasks)

    device = torch.device(args.device)
    model, action_horizon, act_dim = load_policy(args, checkpoint, device)
    Path(args.save_dir).mkdir(parents=True, exist_ok=True)

    print(f"Checkpoint: {args.ckpt}")
    print(f"Checkpoint step: {checkpoint.get('step')}")
    print(f"Tasks: {tasks}")
    print(f"Device: {args.device}")
    print(f"Action horizon: {action_horizon}; action dim: {act_dim}")
    print(f"Execute horizon: {args.execute_horizon}")

    start = time.time()
    results = []
    for task_name in tasks:
        result = evaluate_task(
            model=model,
            task_name=task_name,
            task_description=task_descriptions.get(task_name, task_name),
            device=device,
            action_horizon=action_horizon,
            execute_horizon=args.execute_horizon,
            num_episodes=args.num_episodes,
            max_steps=args.max_steps,
            seed=args.seed,
            save_dir=args.save_dir,
            save_video=args.save_video,
            save_video_episodes=args.save_video_episodes,
            camera_name=args.camera,
            clip_action=not args.no_clip_action,
        )
        print(
            f"{task_name}: {result['success_rate']:.3f} "
            f"({result['successes']}/{result['episodes']}) "
            f"in {result['elapsed_sec']:.1f}s",
            flush=True,
        )
        results.append(result)

    avg_success = float(np.mean([item["success_rate"] for item in results]))
    summary = {
        "checkpoint": args.ckpt,
        "checkpoint_step": checkpoint.get("step"),
        "tasks": tasks,
        "num_episodes": args.num_episodes,
        "max_steps": args.max_steps,
        "execute_horizon": args.execute_horizon,
        "average_success_rate": avg_success,
        "elapsed_sec": time.time() - start,
        "results": results,
    }
    summary_path = Path(args.save_dir) / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n=== Evaluation Summary ===")
    for item in results:
        print(
            f"{item['task']}: {item['success_rate']:.3f} "
            f"({item['successes']}/{item['episodes']})"
        )
    print(f"Average: {avg_success:.3f}")
    print(f"Saved summary to: {summary_path}")


if __name__ == "__main__":
    main()
