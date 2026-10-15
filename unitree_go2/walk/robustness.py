"""Stress-test walking policies: pushes, payload, foot friction and sensor noise.

Each condition changes one thing at a time from the nominal robot and runs `--episodes` 20 s
episodes with random velocity commands (same seeds for every model). Pushes are horizontal
velocity kicks of a fixed size in a random direction every 3 s.

    # from the repository root
    uv run unitree_go2/walk/robustness.py \
        v2=unitree_go2/walk/runs/walk_ppo_v2/best_model.zip \
        v4=unitree_go2/walk/runs/walk_ppo_v4_robust/best_model.zip

Prints a markdown table (fall rate | velocity error per condition) and writes a plot to
unitree_go2/videos/robustness.png.
"""
import argparse
import os
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import PPO

from env import FRICTION_RANGE, Go2WalkEnv

VIDEOS = Path(__file__).resolve().parents[1] / "videos"
PUSH_EVERY = 3.0  # s
SEED = 20_000

# (group, x value, env kwargs, push speed m/s)
NOMINAL = {"randomize": True, "friction_range": (0.8, 0.8), "payload_range": (0.0, 0.0),
           "kp_range": (1.0, 1.0), "push_speed": 0.0, "obs_noise": 0.0}
CONDITIONS = (
    [("push (m/s)", v, NOMINAL, v) for v in (0.0, 0.5, 1.0, 1.5, 2.0)]
    + [("payload (kg)", kg, {**NOMINAL, "payload_range": (kg, kg)}, 0.0) for kg in (3, 6, 9)]
    + [("foot friction", f, {**NOMINAL, "friction_range": (f, f)}, 0.0) for f in (0.2, 0.4)]
    + [("sensor noise (x legged_gym)", n, {**NOMINAL, "obs_noise": float(n)}, 0.0)
       for n in (1, 2)]
)


def run(job):
    label, path, (group, x, kwargs, push), episodes = job
    torch.set_num_threads(1)
    model = PPO.load(path, device="cpu")
    env = Go2WalkEnv(**kwargs)
    push_steps = int(round(PUSH_EVERY / env.dt))
    falls, errors = [], []
    for ep in range(episodes):
        obs, _ = env.reset(seed=SEED + ep)
        rng = np.random.default_rng(SEED + ep)
        done, err, info = False, [], {}
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, info = env.step(action)
            if push and env.steps % push_steps == 0:
                angle = rng.uniform(-np.pi, np.pi)
                env.data.qvel[0:2] += push * np.array([np.cos(angle), np.sin(angle)])
            err.append(info["lin_vel_error"])
            done = terminated or truncated
        falls.append(info["fell"])
        errors.append(np.mean(err))
    return label, group, x, float(np.mean(falls)), float(np.mean(errors))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("models", nargs="+", metavar="LABEL=MODEL.zip")
    p.add_argument("--episodes", type=int, default=20)
    p.add_argument("--workers", type=int, default=os.cpu_count() // 2)
    p.add_argument("--out", default=str(VIDEOS / "robustness.png"))
    args = p.parse_args()

    models = dict(m.split("=", 1) for m in args.models)
    jobs = [(label, path, c, args.episodes) for label, path in models.items() for c in CONDITIONS]
    with Pool(args.workers) as pool:
        results = pool.map(run, jobs)
    res = {(label, group, x): (fall, err) for label, group, x, fall, err in results}

    print(f"nominal foot friction 0.8 (training range {FRICTION_RANGE}); "
          f"{args.episodes} episodes x 20 s per cell; cells: fall rate | velocity error (m/s)")
    print("| Condition | " + " | ".join(models) + " |")
    print("|---|" + "---|" * len(models))
    for group, x, _, _ in CONDITIONS:
        cells = [f"{res[(m, group, x)][0]:.0%} \\| {res[(m, group, x)][1]:.2f}" for m in models]
        print(f"| {group} = {x} | " + " | ".join(cells) + " |")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    groups = list(dict.fromkeys(g for g, *_ in CONDITIONS))
    fig, axes = plt.subplots(1, len(groups), figsize=(4 * len(groups), 3.2), sharey=True)
    for ax, group in zip(axes, groups):
        xs = [x for g, x, *_ in CONDITIONS if g == group]
        for m in models:
            ax.plot(xs, [100 * res[(m, group, x)][0] for x in xs], "o-", label=m)
        ax.set_xlabel(group)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("fall rate (%)")
    axes[0].legend()
    fig.tight_layout()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=120)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
