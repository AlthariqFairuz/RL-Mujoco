"""Render a trained policy to a video (works headless over SSH via EGL).

    uv run src/play.py reach      src/reach/runs/reach_sac_v1/best_model.zip
    uv run src/play.py pick_lift  src/pick_lift/runs/pick_lift_v2_fixed_reward/best_model.zip \
        --episodes 5 --out videos/pick_lift.mp4

Output format follows the extension: .mp4 (H.264, plays everywhere except VS Code's preview),
.webm (VP9, plays inside VS Code), .gif (no ffmpeg needed, larger file).
"""
import argparse
import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")  # offscreen GPU rendering, no display needed

import numpy as np
from stable_baselines3 import PPO, SAC, TD3

SRC = Path(__file__).resolve().parent
TASKS = {"reach": "LowCostArmReachEnv", "pick_lift": "LowCostArmPickLiftEnv"}
ALGOS = {"sac": SAC, "td3": TD3, "ppo": PPO}


def run_config(model_path: str) -> dict:
    """config.json written by train.py next to the model (empty for older runs)."""
    for d in Path(model_path).resolve().parents[:2]:
        if (d / "config.json").exists():
            return json.loads((d / "config.json").read_text())
    return {}


def load_env_class(task: str):
    path = SRC / task / "env.py"
    spec = importlib.util.spec_from_file_location(f"{task}_env", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, TASKS[task])


def write_video(frames: list[np.ndarray], out: Path, fps: int):
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.suffix == ".gif":
        from PIL import Image

        imgs = [Image.fromarray(f) for f in frames[::2]]  # halve the frame rate to keep it small
        imgs[0].save(out, save_all=True, append_images=imgs[1:], duration=2000 // fps, loop=0)
        return
    if shutil.which("ffmpeg") is None:
        sys.exit("ffmpeg not found - install it or use --out something.gif")
    codec = ["-c:v", "libvpx-vp9", "-b:v", "2M"] if out.suffix == ".webm" else [
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20"]
    h, w, _ = frames[0].shape
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
           "-s", f"{w}x{h}", "-r", str(fps), "-i", "-", *codec, str(out)]
    subprocess.run(cmd, input=np.stack(frames).tobytes(), check=True)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("task", choices=TASKS)
    p.add_argument("model", help="path to a saved .zip (algorithm/env settings come from the"
                   " run's config.json)")
    p.add_argument("--episodes", type=int, default=3)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--out", help="video path (default: videos/<task>.mp4)")
    p.add_argument("--stochastic", action="store_true", help="sample actions instead of the mean")
    args = p.parse_args()

    out = Path(args.out or f"videos/{args.task}.mp4")
    cfg = run_config(args.model)
    env_kwargs = dict(reward_type=cfg["reward"], goal_obs=cfg["her"]) if "reward" in cfg else {}
    env = load_env_class(args.task)(render_mode="rgb_array", **env_kwargs)
    model = ALGOS[cfg.get("algo", "sac")].load(args.model, env=env, device="auto")
    fps = env.metadata["render_fps"]

    frames = []
    for ep in range(args.episodes):
        obs, _ = env.reset(seed=args.seed + ep)
        frames.append(env.render())
        done, ret, info = False, 0.0, {}
        while not done:
            action, _ = model.predict(obs, deterministic=not args.stochastic)
            obs, r, terminated, truncated, info = env.step(action)
            ret += r
            done = terminated or truncated
            frames.append(env.render())
        frames.extend([frames[-1]] * fps)  # 1 s pause between episodes
        print(f"episode {ep + 1}: {'success' if info['is_success'] else 'fail'} | return {ret:.1f}")

    env.close()
    write_video(frames, out, fps)
    print(f"wrote {out} ({len(frames) / fps:.0f} s)")


if __name__ == "__main__":
    main()
