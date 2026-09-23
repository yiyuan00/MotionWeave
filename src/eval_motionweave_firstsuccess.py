# Baseline-compatible evaluator for MotionWeave.
# Rollout behavior is unchanged.
import argparse
import json
import sys
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import torch

SRC = Path(__file__).resolve().parent
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from eval_joint_internvl3_svd_lora import (  # noqa: E402
    VideoRecorder,
    normalize_reset_output,
    normalize_step_output,
    obs_to_policy_state,
    set_seed,
    setup_mujoco_render,
    task_descriptions_from_checkpoint,
)
from train_eval_internvl3_mlp_multitask import DEFAULT_TASKS, parse_csv  # noqa: E402
from train_motionweave import (  # noqa: E402
    MotionWeavePolicy,
    remap_legacy_motionweave_state_dict,
)

MOTIONWEAVE_STYLES = {
    "motionweave_aimg_hrc",
    "motion_grounded_residual_composer",
}


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
        first_success_step = None
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
                step_count += 1
                if info.get("success", 0):
                    goal_achieved += 1
                    if first_success_step is None:
                        first_success_step = step_count
                if done or step_count >= max_steps:
                    break

        success = int(goal_achieved > 0)
        successes += success
        episode_results.append(
            {
                "episode": ep,
                "success": success,
                "steps": step_count,
                "first_success_step": first_success_step,
            }
        )
        print(
            f"{task_name} episode {ep}: {'success' if success else 'failure'} "
            f"steps={step_count} first_success_step={first_success_step}",
            flush=True,
        )
        if record_episode:
            video_recorder.save(f"eval_{ep}.mp4")
        env.close()

    success_steps = [
        item["first_success_step"]
        for item in episode_results
        if item["first_success_step"] is not None
    ]
    return {
        "task": task_name,
        "instruction": task_description,
        "success_rate": successes / float(num_episodes),
        "successes": successes,
        "episodes": num_episodes,
        "avg_rollout_steps": float(np.mean([item["steps"] for item in episode_results])),
        "avg_first_success_step": float(np.mean(success_steps)) if success_steps else None,
        "elapsed_sec": time.time() - task_start,
        "episode_results": episode_results,
    }


def infer_action_dims(checkpoint, args):
    ckpt_args = checkpoint.get("args", {})
    config = checkpoint.get("vla", {}).get("config", {})
    action_horizon = (
        args.action_horizon
        or config.get("action_horizon")
        or ckpt_args.get("action_horizon")
        or 4
    )
    action_horizon = int(action_horizon)
    if args.act_dim is not None:
        act_dim = int(args.act_dim)
    else:
        act_dim = int(
            config.get(
                "single_action_dim",
                int(config.get("act_dim", action_horizon * 4)) // action_horizon,
            )
        )
    return action_horizon, act_dim


def load_policy(args, checkpoint, device):
    vla_payload = checkpoint.get("vla")
    if not isinstance(vla_payload, dict):
        raise KeyError(f"{args.ckpt} does not contain a joint 'vla' payload")
    trainable_params = vla_payload.get("trainable_params")
    if trainable_params is None:
        raise KeyError(f"{args.ckpt} does not contain vla.trainable_params")

    config = vla_payload.get("config", {})
    style = config.get("action_token_style", "motionweave_aimg_hrc")
    if style not in MOTIONWEAVE_STYLES:
        raise ValueError(
            f"Checkpoint is not MotionWeave: action_token_style={style!r}"
        )

    action_horizon, act_dim = infer_action_dims(checkpoint, args)
    model_name = args.vlm_model or config.get("model_name") or "/home/yiyuan/InternVL3-2B"
    policy_kwargs = {
        "action_readout_text": config.get("action_readout_text", args.action_readout_text)
    }
    policy_kwargs.update(
        {
            "aimg_dim": int(
                config.get("aimg_dim", config.get("motion_grounder_dim", 512))
            ),
            "hrc_dim": int(config.get("hrc_dim", config.get("composer_dim", 768))),
            "hrc_heads": int(
                config.get("hrc_heads", config.get("composer_heads", 8))
            ),
            "motion_grounding_loss_weight": float(
                config.get(
                    "motion_grounding_loss_weight",
                    config.get("motion_loss_weight", 0.05),
                )
            ),
        }
    )

    model = MotionWeavePolicy(
        obs_dim=int(config.get("obs_dim", args.obs_dim)),
        single_action_dim=act_dim,
        action_horizon=action_horizon,
        diffusion_train_steps=int(config.get("diffusion_train_steps", 100)),
        diffusion_inference_steps=int(
            args.diffusion_inference_steps
            or config.get("diffusion_inference_steps", 10)
        ),
        diffusion_timestep_max=config.get("diffusion_timestep_max", None),
        diffusion_sample_start_step=(
            args.diffusion_sample_start_step
            if args.diffusion_sample_start_step is not None
            else config.get("diffusion_sample_start_step", None)
        ),
        diffusion_x0_clip=(
            args.diffusion_x0_clip
            if args.diffusion_x0_clip is not None
            else config.get("diffusion_x0_clip", None)
        ),
        action_head_objective=args.action_head_objective
        or config.get("action_head_objective", "flow"),
        repeated_diffusion_steps=int(config.get("repeated_diffusion_steps", 1)),
        action_norm_type=args.action_norm_type
        or config.get("action_norm_type", "bounds"),
        action_bound_eps=float(config.get("action_bound_eps", 1e-6)),
        flow_t_alpha=float(config.get("flow_t_alpha", 1.5)),
        flow_t_beta=float(config.get("flow_t_beta", 1.0)),
        flow_t_eps=float(config.get("flow_t_eps", 1e-3)),
        flow_sample_clip=(
            args.flow_sample_clip
            if args.flow_sample_clip is not None
            else config.get("flow_sample_clip", None)
        ),
        dit_dim=int(config.get("dit_dim", 768)),
        dit_layers=int(config.get("dit_layers", 12)),
        dit_heads=int(config.get("dit_heads", 12)),
        dit_mlp_ratio=float(config.get("dit_mlp_ratio", 4.0)),
        dit_dropout=float(config.get("dit_dropout", 0.0)),
        model_name=model_name,
        task_description=config.get("task_description", "joint robot policy"),
        tune_mode=config.get("tune_mode", args.tune_mode),
        lora_r=int(config.get("lora_r", args.lora_r)),
        lora_alpha=int(config.get("lora_alpha", args.lora_alpha)),
        lora_dropout=float(config.get("lora_dropout", args.lora_dropout)),
        lora_target_modules=config.get("lora_target_modules", None),
        torch_dtype=args.torch_dtype,
        local_files_only=args.local_files_only,
        **policy_kwargs,
    ).to(device)

    if "action_mean" in vla_payload and "action_std" in vla_payload:
        model.set_action_stats(vla_payload["action_mean"], vla_payload["action_std"])
    if "action_low" in vla_payload and "action_high" in vla_payload:
        model.set_action_bounds(vla_payload["action_low"], vla_payload["action_high"])

    if style in MOTIONWEAVE_STYLES:
        trainable_params = remap_legacy_motionweave_state_dict(trainable_params)
    incompatible = model.load_state_dict(trainable_params, strict=False)
    unexpected = list(incompatible.unexpected_keys)
    if unexpected:
        raise RuntimeError(f"Unexpected checkpoint keys: {unexpected[:10]}")
    model.eval()
    return model, action_horizon, act_dim, style


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate a MotionWeave (AIMG + HRC) checkpoint on MetaWorld."
        )
    )
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--tasks", default=None)
    parser.add_argument("--vlm_model", default=None)
    parser.add_argument("--local_files_only", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--torch_dtype", default="bfloat16")
    parser.add_argument("--tune_mode", default="full")
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--obs_dim", type=int, default=8)
    parser.add_argument("--act_dim", type=int, default=None)
    parser.add_argument("--action_horizon", type=int, default=None)
    parser.add_argument("--diffusion_inference_steps", type=int, default=None)
    parser.add_argument("--diffusion_sample_start_step", type=int, default=None)
    parser.add_argument("--diffusion_x0_clip", type=float, default=None)
    parser.add_argument("--action_head_objective", choices=["ddpm", "flow"], default=None)
    parser.add_argument("--action_norm_type", choices=["mean_std", "bounds"], default=None)
    parser.add_argument("--flow_sample_clip", type=float, default=None)
    parser.add_argument("--execute_horizon", type=int, default=1)
    parser.add_argument("--num_episodes", "--eval_episodes", type=int, default=25)
    parser.add_argument("--max_steps", type=int, default=175)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--camera", default="corner")
    parser.add_argument(
        "--save_dir",
        default="/kzs_data2/yiyuan/pvrobo_runs/eval_motionweave",
    )
    parser.add_argument("--save_video", action="store_true")
    parser.add_argument("--save_video_episodes", type=int, default=1)
    parser.add_argument("--no_clip_action", action="store_true")
    parser.add_argument("--dry_run_load", action="store_true")
    parser.add_argument("--action_readout_text", default="<ACTION> <ACTION>")
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
    model, action_horizon, act_dim, style = load_policy(args, checkpoint, device)
    Path(args.save_dir).mkdir(parents=True, exist_ok=True)

    print(f"Checkpoint: {args.ckpt}")
    print(f"Checkpoint step: {checkpoint.get('step')}")
    print(f"Action token style: {style}")
    print(f"Tasks: {tasks}")
    print(f"Device: {args.device}")
    print(f"Action horizon: {action_horizon}; action dim: {act_dim}")
    print(f"Execute horizon: {args.execute_horizon}")
    print(f"Diffusion inference steps: {model.diffusion_inference_steps}")
    print(f"Action head objective: {model.action_head_objective}")
    print(f"Action normalization: {model.action_norm_type}")
    print(f"Flow sample clip: {model.flow_sample_clip}")
    print(f"Task descriptions: {task_descriptions}")

    if args.dry_run_load:
        peak_mem_mb = 0.0
        if args.device.startswith("cuda"):
            torch.cuda.synchronize()
            peak_mem_mb = torch.cuda.max_memory_allocated(device) / 1024**2
        print(f"Dry-run load OK. Peak allocated: {peak_mem_mb:.1f} MiB")
        return

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
    first_success_means = [
        item["avg_first_success_step"]
        for item in results
        if item.get("avg_first_success_step") is not None
    ]
    avg_first_success_step = (
        float(np.mean(first_success_means)) if first_success_means else None
    )
    summary = {
        "checkpoint": args.ckpt,
        "checkpoint_step": checkpoint.get("step"),
        "action_token_style": style,
        "tasks": tasks,
        "task_descriptions": task_descriptions,
        "num_episodes": args.num_episodes,
        "max_steps": args.max_steps,
        "execute_horizon": args.execute_horizon,
        "diffusion_inference_steps": model.diffusion_inference_steps,
        "average_success_rate": avg_success,
        "average_first_success_step": avg_first_success_step,
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
            f"({item['successes']}/{item['episodes']}) "
            f"avg_first_success_step={item.get('avg_first_success_step')}"
        )
    print(f"Average: {avg_success:.3f}")
    print(f"Average first success step: {avg_first_success_step}")
    print(f"Saved summary to: {summary_path}")


if __name__ == "__main__":
    main()
