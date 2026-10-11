"""Render a trained Go2 walking policy to a video (works headless over SSH via EGL).

The robot is driven through a fixed tour of velocity commands (forward, turn, sideways,
backward, stand) instead of random ones, and the tracking error of each segment is printed.

    # from the repository root; video goes to unitree_go2/videos/walk.mp4 by default
    uv run unitree_go2/play.py unitree_go2/walk/runs/walk_ppo/best_model.zip
    uv run unitree_go2/play.py <model.zip> --out unitree_go2/videos/walk.gif

Output format follows the extension: .mp4 (H.264), .webm (VP9, plays inside VS Code),
.gif (no ffmpeg needed, larger file).
"""
import argparse
import importlib.util
import os
import shutil
import subprocess
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")  # offscreen GPU rendering, no display needed

import numpy as np
from stable_baselines3 import PPO

PROJECT = Path(__file__).resolve().parent

# (seconds, [vx m/s, vy m/s, wz rad/s], label)
TOUR = [
    (1.0, [0.0, 0.0, 0.0], "stand"),
    (4.0, [0.8, 0.0, 0.0], "forward 0.8 m/s"),
    (4.0, [0.5, 0.0, 0.8], "forward + turn left"),
    (3.0, [0.0, 0.4, 0.0], "sideways left 0.4 m/s"),
    (3.0, [-0.4, 0.0, 0.0], "backward 0.4 m/s"),
    (3.0, [0.0, 0.0, -1.0], "turn on the spot"),
    (2.0, [0.0, 0.0, 0.0], "stand"),
]


def load_env_class():
    spec = importlib.util.spec_from_file_location("walk_env", PROJECT / "walk" / "env.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.Go2WalkEnv


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
    p.add_argument("model", help="path to a saved .zip")
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--out", help="video path (default: unitree_go2/videos/walk.mp4)")
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=480)
    args = p.parse_args()

    out = Path(args.out) if args.out else PROJECT / "videos" / "walk.mp4"
    total = sum(seconds for seconds, _, _ in TOUR)
    env = load_env_class()(render_mode="rgb_array", width=args.width, height=args.height,
                           max_episode_steps=10**9, resample_every=10**9)
    model = PPO.load(args.model, device="cpu")
    fps = env.metadata["render_fps"]

    obs, _ = env.reset(seed=args.seed)
    frames = [env.render()]
    print(f"{'segment':<24}{'command (vx, vy, wz)':>24}{'measured':>24}")
    for seconds, command, label in TOUR:
        env.command = np.array(command, dtype=float)
        measured = []
        for _ in range(int(round(seconds / env.dt))):
            obs = env._get_obs()  # observation with the new command
            action, _ = model.predict(obs, deterministic=True)
            obs, r, terminated, truncated, info = env.step(action)
            frames.append(env.render())
            measured.append([*info["lin_vel"], info["ang_vel_z"]])
            if terminated:
                print(f"fell during '{label}'")
                break
        m = np.mean(measured[len(measured) // 3:], axis=0)  # skip the transition
        print(f"{label:<24}{str(np.round(command, 2)):>24}{str(np.round(m, 2)):>24}")
        if terminated:
            break

    env.close()
    write_video(frames, out, fps)
    print(f"wrote {out} ({len(frames) / fps:.0f} s of {total:.0f} s tour)")


if __name__ == "__main__":
    main()
