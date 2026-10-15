"""Record pick-and-lift demonstrations from a trained SAC policy (the "expert").

Only successful episodes are kept. For every step the dataset stores the observation and the
joint-position targets the expert commanded *after* that step (absolute, scaled to [-1, 1] by
the joint ranges). Diffusion Policy learns these absolute targets instead of the env's per-step
deltas; `policy.py` turns them back into env actions.

    # from the repository root
    uv run low_cost_robot_arm/diffusion/record.py --episodes 500
    # hybrid expert: SAC grasps and lifts, then the arm holds still (no post-lift wiggle)
    uv run low_cost_robot_arm/diffusion/record.py --episodes 500 --hold

Output: low_cost_robot_arm/diffusion/data/<name>.npz with
    obs (N, 35), target (N, 6), episode_ends (E,)  - cumulative step counts per episode
"""
import argparse
import importlib.util
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import SAC

ARM = Path(__file__).resolve().parents[1]
DATA_DIR = Path(__file__).resolve().parent / "data"
EXPERT = ARM / "pick_lift" / "runs" / "pick_lift_v2_fixed_reward" / "best_model.zip"
SEED_OFFSET = 100_000  # demo seeds never overlap the evaluation seeds (10_000 + i)


def load_env_module():
    spec = importlib.util.spec_from_file_location("pick_lift_env", ARM / "pick_lift" / "env.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--episodes", type=int, default=200, help="successful demos to keep")
    p.add_argument("--expert", default=str(EXPERT))
    p.add_argument("--hold", action="store_true",
                   help="freeze the joint targets once the cube is lifted (SAC wiggles there)")
    p.add_argument("--name", help="dataset name (default: demos_<episodes>[_hold])")
    args = p.parse_args()

    torch.set_num_threads(2)
    env = load_env_module().LowCostArmPickLiftEnv()
    expert = SAC.load(args.expert, device="cpu")
    obs_buf, target_buf, ends = [], [], []
    tried = 0
    while len(ends) < args.episodes:
        obs, _ = env.reset(seed=SEED_OFFSET + tried)
        tried += 1
        ep_obs, ep_target, info, done = [], [], {}, False
        while not done:
            action, _ = expert.predict(obs, deterministic=True)
            if args.hold and info.get("is_success"):
                action = np.zeros_like(action)  # zero change = keep the current targets
            ep_obs.append(obs)
            obs, _, terminated, truncated, info = env.step(action)
            ep_target.append((env._target - env._mid) / env._half)
            done = terminated or truncated
        if not info["is_success"]:
            continue
        obs_buf += ep_obs
        target_buf += ep_target
        ends.append(len(obs_buf))
        if len(ends) % 50 == 0:
            print(f"{len(ends)}/{args.episodes} demos ({tried} episodes tried)")

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    out = DATA_DIR / f"{args.name or f'demos_{args.episodes}' + ('_hold' if args.hold else '')}.npz"
    np.savez_compressed(
        out,
        obs=np.array(obs_buf, dtype=np.float32),
        target=np.array(target_buf, dtype=np.float32),
        episode_ends=np.array(ends),
    )
    print(f"wrote {out}: {len(ends)} demos, {len(obs_buf)} steps "
          f"(expert success {len(ends) / tried:.0%})")


if __name__ == "__main__":
    main()
