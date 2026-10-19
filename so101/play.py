"""Render a trained SO-101 policy to a video (works headless over SSH via EGL).

    # from the repository root; video goes to so101/videos/pick_lift.mp4 by default
    uv run so101/play.py so101/pick_lift/runs/so101_pick_lift/best_model.zip --episodes 4

Output format follows the extension: .mp4, .webm or .gif (see low_cost_robot_arm/play.py).
"""
import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")  # offscreen GPU rendering, no display needed

from stable_baselines3 import SAC

PROJECT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT / "pick_lift"))
sys.path.insert(0, str(PROJECT.parent / "low_cost_robot_arm"))
from play import write_video  # noqa: E402  (low_cost_robot_arm/play.py)

from env import SO101PickLiftEnv  # noqa: E402  (so101/pick_lift/env.py)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("model", help="path to a saved SAC .zip")
    p.add_argument("--episodes", type=int, default=3)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--out", default=str(PROJECT / "videos" / "pick_lift.mp4"))
    args = p.parse_args()

    env = SO101PickLiftEnv(render_mode="rgb_array")
    model = SAC.load(args.model, device="auto")
    fps = env.metadata["render_fps"]
    frames = []
    for ep in range(args.episodes):
        obs, _ = env.reset(seed=args.seed + ep)
        frames.append(env.render())
        done, ret, info = False, 0.0, {}
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, r, terminated, truncated, info = env.step(action)
            ret += r
            done = terminated or truncated
            frames.append(env.render())
        frames.extend([frames[-1]] * fps)  # 1 s pause between episodes
        print(f"episode {ep + 1}: {'success' if info['is_success'] else 'fail'} | return {ret:.1f}")

    env.close()
    write_video(frames, Path(args.out), fps)
    print(f"wrote {args.out} ({len(frames) / fps:.0f} s)")


if __name__ == "__main__":
    main()
