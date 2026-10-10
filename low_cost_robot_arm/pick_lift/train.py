"""Train SAC on the low-cost-arm pick-and-lift task (see env.py).

    # train (GPU is used automatically if torch sees one)
    uv run low_cost_robot_arm/pick_lift/train.py --timesteps 2000000 --n-envs 16

    # evaluate a saved model headless: success rate over N episodes
    uv run low_cost_robot_arm/pick_lift/train.py \
        --eval low_cost_robot_arm/pick_lift/runs/<name>/best_model.zip

Everything goes to low_cost_robot_arm/pick_lift/runs/<name>/:
    best_model.zip      best checkpoint by evaluation return (EvalCallback)
    final_model.zip     model at the end of training
    checkpoints/        periodic snapshots
    monitor/*.csv       per-episode return / length / is_success of the training envs
    evaluations.npz     evaluation history
    tb/                 TensorBoard logs (only if `tensorboard` is installed):
                        tensorboard --logdir low_cost_robot_arm/pick_lift/runs
                        watch rollout/success_rate and eval/mean_reward
"""
import argparse
import importlib.util
import time
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from env import LowCostArmPickLiftEnv

RUNS_DIR = Path(__file__).resolve().parent / "runs"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--name", default=time.strftime("sac_%Y%m%d_%H%M%S"), help="run folder name")
    p.add_argument("--timesteps", type=int, default=2_000_000)
    p.add_argument("--n-envs", type=int, default=8, help="parallel simulators (CPU processes)")
    p.add_argument("--device", default="auto", help="auto | cuda | cpu")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--utd", type=float, default=1.0, help="gradient steps per env transition")
    p.add_argument("--learning-starts", type=int, default=10_000)
    p.add_argument("--eval-freq", type=int, default=25_000, help="env steps between evaluations")
    p.add_argument("--eval-episodes", type=int, default=20)
    p.add_argument("--eval", metavar="MODEL.zip", help="only evaluate this saved model and exit")
    return p.parse_args()


def resolve_device(requested: str) -> str:
    cuda = torch.cuda.is_available()
    if requested == "auto":
        requested = "cuda" if cuda else "cpu"
    if requested.startswith("cuda") and not cuda:
        print("WARNING: --device cuda requested but this torch build has no CUDA; using CPU.")
        print(f"         torch {torch.__version__} - install a CUDA build of torch on the VM.")
        requested = "cpu"
    name = torch.cuda.get_device_name(0) if requested.startswith("cuda") else "CPU"
    print(f"torch {torch.__version__} | device: {requested} ({name})")
    return requested


def evaluate(model_path: str, episodes: int, device: str, seed: int):
    model = SAC.load(model_path, device=device)
    env = LowCostArmPickLiftEnv()
    successes, returns = [], []
    for ep in range(episodes):
        obs, _ = env.reset(seed=seed + ep)
        done, ret, info = False, 0.0, {}
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, r, terminated, truncated, info = env.step(action)
            ret += r
            done = terminated or truncated
        successes.append(info["is_success"])
        returns.append(ret)
    print(f"success rate: {np.mean(successes):.0%} ({sum(successes)}/{episodes}) | "
          f"mean return: {np.mean(returns):.1f} +/- {np.std(returns):.1f}")


def main():
    args = parse_args()
    device = resolve_device(args.device)

    if args.eval:
        evaluate(args.eval, args.eval_episodes, device, args.seed + 10_000)
        return

    run_dir = RUNS_DIR / args.name
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"run dir: {run_dir}")

    vec_cls = SubprocVecEnv if args.n_envs > 1 else DummyVecEnv
    train_env = make_vec_env(
        LowCostArmPickLiftEnv,
        n_envs=args.n_envs,
        seed=args.seed,
        vec_env_cls=vec_cls,
        monitor_dir=str(run_dir / "monitor"),
        monitor_kwargs=dict(info_keywords=("is_success",)),
    )
    eval_env = make_vec_env(LowCostArmPickLiftEnv, n_envs=1, seed=args.seed + 1_000)

    use_tb = importlib.util.find_spec("tensorboard") is not None
    if not use_tb:
        print("tensorboard not installed -> no learning curves (`uv add tensorboard`);"
              " per-episode results are still written to monitor/*.csv")

    model = SAC(
        "MlpPolicy",
        train_env,
        learning_rate=3e-4,
        buffer_size=1_000_000,
        learning_starts=args.learning_starts,
        batch_size=args.batch_size,
        gamma=0.98,
        tau=0.005,
        train_freq=1,  # one update round per vec-env step (= n_envs transitions)
        gradient_steps=max(1, round(args.utd * args.n_envs)),
        ent_coef="auto",
        policy_kwargs=dict(net_arch=[256, 256, 256]),
        tensorboard_log=str(run_dir / "tb") if use_tb else None,
        seed=args.seed,
        device=device,
        verbose=1,
    )

    # callbacks count vec-env steps, so divide env-step intervals by n_envs
    per_call = max(args.n_envs, 1)
    callbacks = [
        EvalCallback(
            eval_env,
            n_eval_episodes=args.eval_episodes,
            eval_freq=max(args.eval_freq // per_call, 1),
            deterministic=True,
            best_model_save_path=str(run_dir),
            log_path=str(run_dir),
        ),
        CheckpointCallback(
            save_freq=max(100_000 // per_call, 1),
            save_path=str(run_dir / "checkpoints"),
            name_prefix="sac",
        ),
    ]

    try:
        model.learn(total_timesteps=args.timesteps, callback=callbacks, progress_bar=False)
    except KeyboardInterrupt:
        print("interrupted - saving current model")
    model.save(run_dir / "final_model")
    train_env.close()
    eval_env.close()
    print(f"saved {run_dir / 'final_model.zip'} (best: {run_dir / 'best_model.zip'})")


if __name__ == "__main__":
    main()
