"""Train SAC, TD3 or PPO on the low-cost-arm reach task (see env.py).

    # train (GPU is used automatically if torch sees one)
    uv run src/reach/train.py --timesteps 300000 --n-envs 8
    uv run src/reach/train.py --algo ppo --timesteps 3000000 --name reach_ppo
    uv run src/reach/train.py --reward sparse --her --name reach_sac_sparse_her

    # compare runs (learning curves vs env steps and vs wall-clock time)
    uv run src/reach/compare.py reach_sac reach_td3 reach_ppo

    # evaluate a saved model headless: success rate over N episodes
    uv run src/reach/train.py --eval src/reach/runs/<name>/best_model.zip

Everything goes to src/reach/runs/<name>/:
    best_model.zip      best checkpoint by evaluation return (EvalCallback)
    final_model.zip     model at the end of training
    checkpoints/        periodic snapshots
    monitor/*.csv       per-episode return / length / is_success of the training envs
    evaluations.npz     evaluation history
    config.json         command-line arguments (used by --eval and play.py to rebuild the env)
    tb/                 TensorBoard logs (only if `tensorboard` is installed):
                        tensorboard --logdir src/reach/runs
                        watch rollout/success_rate and eval/mean_reward
"""
import argparse
import importlib.util
import json
import time
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import PPO, SAC, TD3, HerReplayBuffer
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from env import LowCostArmReachEnv

RUNS_DIR = Path(__file__).resolve().parent / "runs"
ALGOS = {"sac": SAC, "td3": TD3, "ppo": PPO}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--algo", choices=ALGOS, default="sac")
    p.add_argument("--name", help="run folder name (default: <algo>_<date>_<time>)")
    p.add_argument("--timesteps", type=int, default=300_000)
    p.add_argument("--n-envs", type=int, default=8, help="parallel simulators (CPU processes)")
    p.add_argument("--device", default="auto", help="auto | cuda | cpu")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--batch-size", type=int, default=256, help="SAC/TD3: sampled from the buffer;"
                   " PPO: minibatch size within a rollout")
    p.add_argument("--utd", type=float, default=1.0,
                   help="SAC/TD3: gradient steps per env transition")
    p.add_argument("--learning-starts", type=int, default=5_000, help="SAC/TD3 only")
    p.add_argument("--n-steps", type=int, default=512, help="PPO: steps per env per rollout")
    p.add_argument("--n-epochs", type=int, default=10, help="PPO: passes over each rollout")
    p.add_argument("--reward", choices=["dense", "sparse"], default="dense",
                   help="dense = shaped (distance every step), sparse = -1 until within 1 cm")
    p.add_argument("--her", action="store_true",
                   help="SAC/TD3: Hindsight Experience Replay (goal-conditioned observations)")
    p.add_argument("--eval-freq", type=int, default=10_000, help="env steps between evaluations")
    p.add_argument("--eval-episodes", type=int, default=50)
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


def env_kwargs_for(model_path: str) -> dict:
    """Env settings the model was trained with, from config.json in its run folder."""
    for d in Path(model_path).resolve().parents[:2]:
        if (d / "config.json").exists():
            cfg = json.loads((d / "config.json").read_text())
            return dict(reward_type=cfg.get("reward", "dense"), goal_obs=cfg.get("her", False))
    return {}


def evaluate(algo: str, model_path: str, episodes: int, device: str, seed: int):
    env = LowCostArmReachEnv(**env_kwargs_for(model_path))
    model = ALGOS[algo].load(model_path, env=env, device=device)  # HER models need the env
    successes, returns, dists = [], [], []
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
        dists.append(info["dist"])
    print(f"success rate: {np.mean(successes):.0%} ({sum(successes)}/{episodes}) | "
          f"mean return: {np.mean(returns):.1f} +/- {np.std(returns):.1f} | "
          f"final distance: median {1000 * np.median(dists):.1f} mm, "
          f"max {1000 * np.max(dists):.1f} mm")


def main():
    args = parse_args()
    device = resolve_device(args.device)

    if args.eval:
        evaluate(args.algo, args.eval, args.eval_episodes, device, args.seed + 10_000)
        return

    run_dir = RUNS_DIR / (args.name or time.strftime(f"{args.algo}_%Y%m%d_%H%M%S"))
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"run dir: {run_dir}")
    if args.her and args.algo == "ppo":
        raise SystemExit("--her needs an off-policy algorithm (sac or td3)")
    (run_dir / "config.json").write_text(json.dumps(vars(args), indent=2))
    env_kwargs = dict(reward_type=args.reward, goal_obs=args.her)

    vec_cls = SubprocVecEnv if args.n_envs > 1 else DummyVecEnv
    train_env = make_vec_env(
        LowCostArmReachEnv,
        n_envs=args.n_envs,
        seed=args.seed,
        env_kwargs=env_kwargs,
        vec_env_cls=vec_cls,
        monitor_dir=str(run_dir / "monitor"),
        monitor_kwargs=dict(info_keywords=("is_success",)),
    )
    eval_env = make_vec_env(LowCostArmReachEnv, n_envs=1, seed=args.seed + 1_000,
                            env_kwargs=env_kwargs)

    use_tb = importlib.util.find_spec("tensorboard") is not None
    if not use_tb:
        print("tensorboard not installed -> no learning curves (`uv add tensorboard`);"
              " per-episode results are still written to monitor/*.csv")

    common = dict(
        gamma=0.95,
        tensorboard_log=str(run_dir / "tb") if use_tb else None,
        seed=args.seed,
        device=device,
        verbose=1,
    )
    if args.algo == "ppo":
        # on-policy: collect n_steps x n_envs transitions, train n_epochs over them, discard
        model = PPO(
            "MlpPolicy",
            train_env,
            learning_rate=3e-4,
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            n_epochs=args.n_epochs,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.0,
            policy_kwargs=dict(net_arch=dict(pi=[256, 256], vf=[256, 256])),
            **common,
        )
    else:
        # off-policy: replay buffer, one update round per vec-env step (= n_envs transitions)
        off_policy = dict(
            learning_rate=3e-4,
            buffer_size=1_000_000,
            learning_starts=args.learning_starts,
            batch_size=args.batch_size,
            tau=0.005,
            train_freq=1,
            gradient_steps=max(1, round(args.utd * args.n_envs)),
            policy_kwargs=dict(net_arch=[256, 256]),
        )
        policy = "MlpPolicy"
        if args.her:
            # relabel each stored transition with 4 goals the arm actually reached later in
            # the same episode ("future" strategy), recomputing the reward for each
            policy = "MultiInputPolicy"
            off_policy.update(
                replay_buffer_class=HerReplayBuffer,
                replay_buffer_kwargs=dict(n_sampled_goal=4, goal_selection_strategy="future"),
            )
        if args.algo == "sac":
            model = SAC(policy, train_env, ent_coef="auto", **off_policy, **common)
        else:  # TD3: deterministic policy + Gaussian exploration noise
            noise = NormalActionNoise(np.zeros(5), 0.1 * np.ones(5))
            model = TD3(policy, train_env, action_noise=noise, **off_policy, **common)
    print(f"algo: {args.algo.upper()} | reward: {args.reward} | HER: {args.her}")

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
            save_freq=max(50_000 // per_call, 1),
            save_path=str(run_dir / "checkpoints"),
            name_prefix=args.algo,
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
