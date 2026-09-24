"""Generate episode-aligned robot masks for MotionWeave with RoboEngine."""

import argparse
import pickle
from pathlib import Path

import numpy as np
from PIL import Image


DEFAULT_TASKS = (
    "pick-place-v2",
    "disassemble-v2",
    "stick-pull-v2",
    "assembly-v2",
    "shelf-place-v2",
    "hand-insert-v2",
)


def episode_to_rgb(episode, camera_index):
    frames = np.asarray(episode)
    if frames.ndim != 4:
        raise ValueError(f"Expected an episode with four dimensions, got {frames.shape}.")

    start = 3 * int(camera_index)
    stop = start + 3
    if frames.shape[1] >= stop:
        frames = frames[:, start:stop].transpose(0, 2, 3, 1)
    elif frames.shape[-1] >= stop:
        frames = frames[..., start:stop]
    else:
        raise ValueError(
            f"Camera {camera_index} is unavailable in episode shape {frames.shape}."
        )

    if np.issubdtype(frames.dtype, np.floating) and frames.max(initial=0.0) <= 1.0:
        frames = frames * 255.0
    return np.clip(frames, 0, 255).astype(np.uint8)


def canonicalize_masks(masks, num_frames, mask_size):
    masks = np.asarray(masks)
    if masks.ndim == 4 and masks.shape[1] == 1:
        masks = masks[:, 0]
    elif masks.ndim == 4 and masks.shape[-1] == 1:
        masks = masks[..., 0]
    if masks.ndim != 3 or masks.shape[0] != num_frames:
        raise ValueError(
            f"RoboEngine returned {masks.shape}; expected [{num_frames}, H, W]."
        )
    threshold = 0.5 if masks.max(initial=0.0) <= 1.0 else 127.5
    masks = (masks >= threshold).astype(np.uint8)
    if mask_size > 0 and masks.shape[-2:] != (mask_size, mask_size):
        masks = np.stack(
            [
                np.asarray(
                    Image.fromarray(mask).resize(
                        (mask_size, mask_size),
                        resample=Image.Resampling.NEAREST,
                    )
                )
                for mask in masks
            ],
            axis=0,
        ).astype(np.uint8)
    return masks


def main():
    parser = argparse.ArgumentParser(
        description="Cache RoboEngine robot masks for MetaWorld expert episodes."
    )
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--tasks", default=",".join(DEFAULT_TASKS))
    parser.add_argument("--episodes", type=int, default=25)
    parser.add_argument("--camera_index", type=int, default=0)
    parser.add_argument("--anchor_frequency", type=int, default=8)
    parser.add_argument(
        "--mask_size",
        type=int,
        default=128,
        help="Saved square mask size; use 0 to retain RoboEngine output size.",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    try:
        from robo_engine.infer_engine import RoboEngineRobotSegmentation
    except ImportError as error:
        raise SystemExit(
            "RoboEngine is not importable. Install the official repository and "
            "add it to PYTHONPATH before running this preprocessing script."
        ) from error

    data_root = Path(args.data_root)
    output_root = Path(args.output_root)
    tasks = [task.strip() for task in args.tasks.split(",") if task.strip()]
    segmenter = RoboEngineRobotSegmentation()

    for task in tasks:
        with open(data_root / task / "expert_demos.pkl", "rb") as stream:
            episodes, _, _, _ = pickle.load(stream)
        task_output = output_root / task
        task_output.mkdir(parents=True, exist_ok=True)

        for episode_idx, episode in enumerate(episodes[: args.episodes]):
            output_path = task_output / f"episode_{episode_idx:03d}.npy"
            if output_path.exists() and not args.overwrite:
                print(f"skip {output_path}", flush=True)
                continue

            frames = episode_to_rgb(episode, args.camera_index)
            masks = segmenter.gen_video(
                image_np_list=list(frames),
                prompt="robot",
                anchor_frequency=args.anchor_frequency,
            )
            masks = canonicalize_masks(masks, len(frames), args.mask_size)
            np.save(output_path, masks)
            print(
                f"saved {task} episode {episode_idx:03d}: "
                f"{masks.shape} -> {output_path}",
                flush=True,
            )


if __name__ == "__main__":
    main()
