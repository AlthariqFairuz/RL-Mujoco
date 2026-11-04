"""Render the camera-based student: the scene, plus the two 84x84 images the policy sees.

    # from the repository root; default output so101/videos/pick_lift_vision.mp4
    uv run so101/vision/play.py so101/vision/runs/dagger/best_student.pt --episodes 4
"""
import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np
import torch
from distill import EVAL_SEED, load_student
from vision_env import CAMERAS, IMG, SO101VisionEnv

SO101 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SO101.parent / "low_cost_robot_arm"))
from play import write_video  # noqa: E402  (low_cost_robot_arm/play.py)

INSET = 2 * IMG  # each camera image is shown at 168x168


def frame(env: SO101VisionEnv, image: np.ndarray) -> np.ndarray:
    """Scene render with the policy's camera images in the top-left corner."""
    out = env.render().copy()
    for i in range(len(CAMERAS)):
        cam = image[3 * i : 3 * i + 3].transpose(1, 2, 0).repeat(2, axis=0).repeat(2, axis=1)
        x = 8 + i * (INSET + 8)
        out[6:8 + INSET + 2, x - 2 : x + INSET + 2] = 255  # white border
        out[8 : 8 + INSET, x : x + INSET] = cam
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("student", help="path to a saved student .pt")
    p.add_argument("--episodes", type=int, default=4)
    p.add_argument("--seed", type=int, default=EVAL_SEED)
    p.add_argument("--out", default=str(SO101 / "videos" / "pick_lift_vision.mp4"))
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    student = load_student(args.student, device)
    env = SO101VisionEnv(render_mode="rgb_array")
    fps = env.metadata["render_fps"]
    frames = []
    for ep in range(args.episodes):
        obs, _ = env.reset(seed=args.seed + ep)
        frames.append(frame(env, obs["image"]))
        done, info = False, {}
        while not done:
            batch = {k: v[None] for k, v in obs.items()}
            obs, _, terminated, truncated, info = env.step(student.act(batch, device)[0])
            done = terminated or truncated
            frames.append(frame(env, obs["image"]))
        frames.extend([frames[-1]] * fps)  # 1 s pause between episodes
        print(f"episode {ep + 1}: {'success' if info['is_success'] else 'fail'}")
    env.close()
    write_video(frames, Path(args.out), fps)
    print(f"wrote {args.out} ({len(frames) / fps:.0f} s)")


if __name__ == "__main__":
    main()
