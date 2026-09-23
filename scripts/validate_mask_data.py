import argparse
import pickle
from pathlib import Path

import numpy as np


DEFAULT_TASKS = (
    "pick-place-v2",
    "disassemble-v2",
    "stick-pull-v2",
    "assembly-v2",
    "shelf-place-v2",
    "hand-insert-v2",
)


def find_mask_path(root, task, episode_idx):
    task_root = root / task
    candidates = (
        task_root / f"episode_{episode_idx:03d}.npy",
        task_root / f"episode_{episode_idx}.npy",
        task_root / f"episode_{episode_idx:03d}.npz",
    )
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(f"No mask file found for {task} episode {episode_idx}.")


def load_masks(path):
    masks = np.load(path, mmap_mode="r")
    if isinstance(masks, np.lib.npyio.NpzFile):
        if "masks" not in masks:
            raise KeyError(f"{path} must contain an array named 'masks'.")
        masks = masks["masks"]
    return masks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--mask_root", required=True)
    parser.add_argument("--tasks", default=",".join(DEFAULT_TASKS))
    parser.add_argument("--episodes", type=int, default=25)
    args = parser.parse_args()

    data_root = Path(args.data_root)
    mask_root = Path(args.mask_root)
    tasks = [task.strip() for task in args.tasks.split(",") if task.strip()]
    checked = 0

    for task in tasks:
        demo_path = data_root / task / "expert_demos.pkl"
        with open(demo_path, "rb") as stream:
            images, _, _, _ = pickle.load(stream)
        for episode_idx, episode in enumerate(images[: args.episodes]):
            path = find_mask_path(mask_root, task, episode_idx)
            masks = load_masks(path)
            if len(masks) != len(episode):
                raise ValueError(
                    f"Length mismatch for {task} episode {episode_idx}: "
                    f"images={len(episode)}, masks={len(masks)}"
                )
            if masks.ndim not in (3, 4):
                raise ValueError(f"Invalid mask shape in {path}: {masks.shape}")
            checked += 1
        print(f"{task}: OK")

    print(f"Validated {checked} expert episodes.")


if __name__ == "__main__":
    main()

