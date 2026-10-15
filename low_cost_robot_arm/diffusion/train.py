"""Train a state-based Diffusion Policy on SAC pick-and-lift demos (see record.py, policy.py).

    # from the repository root
    uv run low_cost_robot_arm/diffusion/train.py --demos 200 --name dp_200

    # evaluate a saved policy: success rate over N episodes (same seeds as the SAC evaluation)
    uv run low_cost_robot_arm/diffusion/train.py \
        --eval low_cost_robot_arm/diffusion/runs/dp_200/model.pt

Everything goes to low_cost_robot_arm/diffusion/runs/<name>/:
    model.pt        EMA weights at the end of training
    best_model.pt   EMA weights with the best evaluation success rate
    evals.json      success rate during training
    config.json     command-line arguments
    tb/             TensorBoard logs (train/loss, eval/success_rate)
"""
import argparse
import json
import time
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
from policy import ACTION_HORIZON, OBS_HORIZON, PRED_HORIZON, DiffusionPolicy
from record import DATA_DIR, load_env_module

RUNS_DIR = Path(__file__).resolve().parent / "runs"
EVAL_SEED = 10_000  # pick_lift/train.py evaluates SAC on seeds 10_000 + i as well


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--name", default=time.strftime("dp_%Y%m%d_%H%M%S"), help="run folder name")
    p.add_argument("--data", default=str(DATA_DIR / "demos_500.npz"))
    p.add_argument("--demos", type=int, default=200, help="use the first N demos of the dataset")
    p.add_argument("--iters", type=int, default=40_000, help="gradient steps")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--ema", type=float, default=0.999, help="EMA decay of the weights")
    p.add_argument("--eval-every", type=int, default=5_000)
    p.add_argument("--eval-episodes", type=int, default=50, help="during training")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--eval", metavar="MODEL.pt", help="only evaluate this saved policy and exit")
    p.add_argument("--final-episodes", type=int, default=200, help="for --eval")
    return p.parse_args()


# ------------------------------------------------------------------ data
def make_windows(data: np.lib.npyio.NpzFile, demos: int):
    """Every step t of every demo -> (obs[t-1..t], target[t..t+15]), padded at episode edges."""
    obs, target, ends = data["obs"], data["target"], data["episode_ends"][:demos]
    o_win, t_win = [], []
    start = 0
    for end in ends:
        idx = np.arange(start, end)
        o_idx = np.clip(idx[:, None] + np.arange(-OBS_HORIZON + 1, 1), start, end - 1)
        t_idx = np.clip(idx[:, None] + np.arange(PRED_HORIZON), start, end - 1)
        o_win.append(obs[o_idx])
        t_win.append(target[t_idx])
        start = end
    return np.concatenate(o_win), np.concatenate(t_win), obs[: ends[-1]]


# ------------------------------------------------------------------ evaluation
def evaluate(policy: DiffusionPolicy, episodes: int, device: str, seed: int = EVAL_SEED):
    """Run all episodes in lockstep so that every replanning is one batched sampling call."""
    env_module = load_env_module()
    envs = [env_module.LowCostArmPickLiftEnv() for _ in range(episodes)]
    scale = env_module.ACTION_SCALE
    hist = []
    for i, env in enumerate(envs):
        o, _ = env.reset(seed=seed + i)
        hist.append([o] * OBS_HORIZON)
    alive = np.ones(episodes, bool)
    success = np.zeros(episodes, bool)
    policy.eval()
    while alive.any():
        ids = np.flatnonzero(alive)
        obs = torch.as_tensor(np.array([hist[i][-OBS_HORIZON:] for i in ids]), device=device)
        plan = policy.sample(obs).cpu().numpy()[:, :ACTION_HORIZON]
        for j, i in enumerate(ids):
            env = envs[i]
            for target in plan[j]:
                desired = env._mid + env._half * target
                action = np.clip((desired - env._target) / scale, -1, 1)
                o, _, terminated, truncated, info = env.step(action)
                hist[i].append(o)
                if terminated or truncated:
                    alive[i], success[i] = False, info["is_success"]
                    break
    for env in envs:
        env.close()
    return float(success.mean())


def load_policy(path: str, device: str) -> DiffusionPolicy:
    ckpt = torch.load(path, map_location=device)
    policy = DiffusionPolicy(ckpt["obs_dim"], ckpt["action_dim"]).to(device)
    policy.load_state_dict(ckpt["state_dict"])
    return policy


def save_policy(policy: DiffusionPolicy, path: Path):
    torch.save(dict(obs_dim=policy.obs_dim, action_dim=policy.action_dim,
                    state_dict=policy.state_dict()), path)


# ------------------------------------------------------------------ training
def main():
    args = parse_args()
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    if args.eval:
        policy = load_policy(args.eval, args.device)
        rate = evaluate(policy, args.final_episodes, args.device)
        print(f"success rate: {rate:.0%} over {args.final_episodes} episodes")
        return

    run_dir = RUNS_DIR / args.name
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps(vars(args), indent=2))
    try:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(run_dir / "tb")
    except ImportError:
        writer = None

    o_win, t_win, obs_all = make_windows(np.load(args.data), args.demos)
    print(f"{args.demos} demos -> {len(o_win)} training windows | device {args.device}")
    o_win = torch.as_tensor(o_win, device=args.device)
    t_win = torch.as_tensor(t_win, device=args.device)

    policy = DiffusionPolicy(o_win.shape[-1], t_win.shape[-1],
                             obs_all.min(0), obs_all.max(0)).to(args.device)
    ema = deepcopy(policy).requires_grad_(False)
    print(f"{sum(p.numel() for p in policy.parameters()) / 1e6:.1f}M parameters")
    opt = torch.optim.AdamW(policy.parameters(), lr=args.lr, weight_decay=1e-6)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.iters,
                                                pct_start=0.02, anneal_strategy="cos")

    evals, best, t0 = [], -1.0, time.time()
    for it in range(1, args.iters + 1):
        idx = torch.randint(0, len(o_win), (args.batch_size,), device=args.device)
        loss = policy.loss(o_win[idx], t_win[idx])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        with torch.no_grad():
            for pe, p in zip(ema.parameters(), policy.parameters()):
                pe.lerp_(p, 1 - args.ema)

        if writer and it % 100 == 0:
            writer.add_scalar("train/loss", loss.item(), it)
        if it % args.eval_every == 0 or it == args.iters:
            rate = evaluate(ema, args.eval_episodes, args.device)
            evals.append(dict(iter=it, success_rate=rate, minutes=(time.time() - t0) / 60))
            print(f"iter {it:>6} | loss {loss.item():.4f} | success {rate:.0%} | "
                  f"{(time.time() - t0) / 60:.1f} min", flush=True)
            if writer:
                writer.add_scalar("eval/success_rate", rate, it)
            if rate > best:
                best = rate
                save_policy(ema, run_dir / "best_model.pt")
            (run_dir / "evals.json").write_text(json.dumps(evals, indent=2))

    save_policy(ema, run_dir / "model.pt")
    print(f"saved {run_dir / 'model.pt'} (best eval {best:.0%})")


if __name__ == "__main__":
    main()
