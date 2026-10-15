"""Render a trained Diffusion Policy on pick-and-lift (headless via EGL).

    # from the repository root; default output low_cost_robot_arm/videos/pick_lift_diffusion.mp4
    uv run low_cost_robot_arm/diffusion/play.py \
        low_cost_robot_arm/diffusion/runs/dp_200_hold/model.pt
"""
import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np
import torch
from policy import ACTION_HORIZON, OBS_HORIZON
from record import load_env_module
from train import EVAL_SEED, load_policy

ARM = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ARM))
from play import write_video  # noqa: E402  (low_cost_robot_arm/play.py)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("model", help="path to a saved .pt")
    p.add_argument("--episodes", type=int, default=4)
    p.add_argument("--seed", type=int, default=EVAL_SEED)
    p.add_argument("--out", default=str(ARM / "videos" / "pick_lift_diffusion.mp4"))
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    policy = load_policy(args.model, device).eval()
    env_module = load_env_module()
    env = env_module.LowCostArmPickLiftEnv(render_mode="rgb_array")
    frames = []
    for ep in range(args.episodes):
        obs, _ = env.reset(seed=args.seed + ep)
        hist, done, info = [obs] * OBS_HORIZON, False, {}
        frames.append(env.render())
        while not done:
            o = torch.as_tensor(np.array(hist[-OBS_HORIZON:])[None], device=device)
            for target in policy.sample(o)[0, :ACTION_HORIZON].cpu().numpy():
                desired = env._mid + env._half * target
                action = np.clip((desired - env._target) / env_module.ACTION_SCALE, -1, 1)
                obs, _, terminated, truncated, info = env.step(action)
                hist.append(obs)
                frames.append(env.render())
                done = terminated or truncated
                if done:
                    break
        print(f"episode {ep}: success={info['is_success']}")
    env.close()
    write_video(frames, Path(args.out), env.metadata["render_fps"])
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
