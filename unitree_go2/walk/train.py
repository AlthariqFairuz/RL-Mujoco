"""Train PPO on Go2 velocity-tracking locomotion (see env.py).

    # train (16 parallel simulators on the CPU; the policy is small, so the CPU also trains it)
    uv run unitree_go2/walk/train.py --timesteps 30000000 --name walk_ppo
    # with domain randomization, pushes and sensor noise (see env.py)
    uv run unitree_go2/walk/train.py --randomize --name walk_ppo_robust \
        --init unitree_go2/walk/runs/walk_ppo/best_model.zip --timesteps 15000000

    # evaluate a saved model headless: fall rate and velocity-tracking error
    uv run unitree_go2/walk/train.py --eval unitree_go2/walk/runs/<name>/best_model.zip

Everything goes to unitree_go2/walk/runs/<name>/:
    best_model.zip      best checkpoint by evaluation return (EvalCallback)
    final_model.zip     model at the end of training
    checkpoints/        periodic snapshots
    monitor/*.csv       per-episode return / length of the training envs
    evaluations.npz     evaluation history
    config.json         command-line arguments
    tb/                 TensorBoard logs: tensorboard --logdir unitree_go2/walk/runs
                        rollout/ep_rew_mean, reward/<term> (each reward term per episode),
                        walk/lin_vel_error, walk/fall_rate
"""
import argparse
import importlib.util
import json
import time
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from env import Go2WalkEnv

RUNS_DIR = Path(__file__).resolve().parent / "runs"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--name", help="run folder name (default: ppo_<date>_<time>)")
    p.add_argument("--timesteps", type=int, default=30_000_000)
    p.add_argument("--n-envs", type=int, default=16, help="parallel simulators (CPU processes)")
    p.add_argument("--device", default="cpu", help="cpu | cuda | auto (cpu was faster: the"
                   " MLP is small and every env step needs a policy forward pass)")
    p.add_argument("--torch-threads", type=int, default=4,
                   help="leave the other cores to the simulators")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--randomize", action="store_true",
                   help="domain randomization, pushes and sensor noise (train and eval)")
    p.add_argument("--n-steps", type=int, default=1024, help="steps per env per rollout")
    p.add_argument("--batch-size", type=int, default=4096, help="minibatch size")
    p.add_argument("--n-epochs", type=int, default=5, help="passes over each rollout")
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--eval-freq", type=int, default=500_000, help="env steps between evaluations")
    p.add_argument("--eval-episodes", type=int, default=10)
    p.add_argument("--init", metavar="MODEL.zip",
                   help="continue training this model (e.g. a nominal walker before --randomize)")
    p.add_argument("--eval", metavar="MODEL.zip", help="only evaluate this saved model and exit")
    return p.parse_args()


def resolve_device(requested: str) -> str:
    cuda = torch.cuda.is_available()
    if requested == "auto":
        requested = "cuda" if cuda else "cpu"
    if requested.startswith("cuda") and not cuda:
        print("WARNING: --device cuda requested but this torch build has no CUDA; using CPU.")
        requested = "cpu"
    name = torch.cuda.get_device_name(0) if requested.startswith("cuda") else "CPU"
    print(f"torch {torch.__version__} | device: {requested} ({name})")
    return requested


class WalkMetricsCallback(BaseCallback):
    """Log each reward term, velocity-tracking error and fall rate to TensorBoard."""

    def _on_step(self) -> bool:
        for info, done in zip(self.locals["infos"], self.locals["dones"]):
            self.logger.record_mean("walk/lin_vel_error", info["lin_vel_error"])
            if done:
                self.logger.record_mean("walk/fall_rate", float(info["fell"]))
                for k, v in info["episode_sums"].items():
                    self.logger.record_mean(f"reward/{k}", v)
        return True


def evaluate(model_path: str, episodes: int, device: str, seed: int, randomize: bool):
    env = Go2WalkEnv(randomize=randomize)
    model = PPO.load(model_path, device=device)
    falls, returns, errors, lengths = [], [], [], []
    for ep in range(episodes):
        obs, _ = env.reset(seed=seed + ep)
        done, ret, err, info = False, 0.0, [], {}
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, r, terminated, truncated, info = env.step(action)
            ret += r
            err.append(info["lin_vel_error"])
            done = terminated or truncated
        falls.append(info["fell"])
        returns.append(ret)
        errors.append(np.mean(err))
        lengths.append(env.steps * env.dt)
    print(f"fall rate: {np.mean(falls):.0%} ({sum(falls)}/{episodes}) | "
          f"mean return: {np.mean(returns):.2f} +/- {np.std(returns):.2f} | "
          f"velocity error: {np.mean(errors):.3f} m/s | "
          f"mean episode: {np.mean(lengths):.1f} s")


def main():
    args = parse_args()
    torch.set_num_threads(args.torch_threads)
    device = resolve_device(args.device)

    if args.eval:
        evaluate(args.eval, args.eval_episodes, device, args.seed + 10_000, args.randomize)
        return

    run_dir = RUNS_DIR / (args.name or time.strftime("ppo_%Y%m%d_%H%M%S"))
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"run dir: {run_dir}")
    (run_dir / "config.json").write_text(json.dumps(vars(args), indent=2))

    vec_cls = SubprocVecEnv if args.n_envs > 1 else DummyVecEnv
    env_kwargs = {"randomize": args.randomize}
    train_env = make_vec_env(
        Go2WalkEnv,
        n_envs=args.n_envs,
        seed=args.seed,
        vec_env_cls=vec_cls,
        monitor_dir=str(run_dir / "monitor"),
        env_kwargs=env_kwargs,
    )
    eval_env = make_vec_env(Go2WalkEnv, n_envs=1, seed=args.seed + 1_000, env_kwargs=env_kwargs)

    use_tb = importlib.util.find_spec("tensorboard") is not None
    tb_log = str(run_dir / "tb") if use_tb else None
    if args.init:  # same hyperparameters as the saved model; only the environment changes
        model = PPO.load(args.init, env=train_env, device=device, tensorboard_log=tb_log)
    else:
        model = PPO(
            "MlpPolicy",
            train_env,
            learning_rate=args.lr,
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            n_epochs=args.n_epochs,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            # no entropy bonus: actions are clipped to [-1, 1], so a wider Gaussian costs
            # nothing and the bonus alone grew the policy std from 1 to 15 in the first run
            ent_coef=0.0,
            max_grad_norm=1.0,
            policy_kwargs=dict(
                net_arch=dict(pi=[512, 256, 128], vf=[512, 256, 128]),
                activation_fn=torch.nn.ELU,
                log_std_init=-0.5,  # initial std 0.6
            ),
            tensorboard_log=tb_log,
            seed=args.seed,
            device=device,
            verbose=1,
        )

    # callbacks count vec-env steps, so divide env-step intervals by n_envs
    per_call = max(args.n_envs, 1)
    callbacks = [
        WalkMetricsCallback(),
        EvalCallback(
            eval_env,
            n_eval_episodes=args.eval_episodes,
            eval_freq=max(args.eval_freq // per_call, 1),
            deterministic=True,
            best_model_save_path=str(run_dir),
            log_path=str(run_dir),
        ),
        CheckpointCallback(
            save_freq=max(2_000_000 // per_call, 1),
            save_path=str(run_dir / "checkpoints"),
            name_prefix="ppo",
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
