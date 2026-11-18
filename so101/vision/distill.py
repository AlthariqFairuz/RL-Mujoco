"""Distil the state-based SAC expert into a camera-based student policy with DAgger.

The teacher (SAC from `so101/pick_lift`) reads the true cube pose; the student only sees two
camera images and the joint readings (see vision_env.py). Every iteration:
    1. collect episodes in 16 parallel simulators. At each step the teacher's action is used with
       probability beta (beta = beta_decay ** iteration), otherwise the student's; the teacher
       labels every visited state with its own action either way
    2. add them to the dataset (all iterations are kept) and train the student to output the
       teacher's actions (mean-squared error, random-shift image augmentation as in DrQ)
With beta_decay = 1 the teacher always drives: plain behaviour cloning, the student only ever
sees states the teacher visits. With beta_decay < 1 (DAgger) the student increasingly drives
and is corrected in the states its own mistakes lead to, so it learns to recover.

    # from the repository root
    uv run so101/vision/distill.py --name dagger
    uv run so101/vision/distill.py --name bc --beta-decay 1      # behaviour-cloning baseline
    uv run so101/vision/distill.py --name dagger_s1 --seed 1      # another training seed
    # evaluate a saved student: success rate over N episodes (same seeds as the SAC evaluation)
    uv run so101/vision/distill.py --eval so101/vision/runs/dagger/student.pt --episodes 200

Everything goes to so101/vision/runs/<name>/: student.pt, log.json, config.json.
"""
import argparse
import json
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parent / "pick_lift"), str(HERE)]
from vision_env import CAMERAS, IMG, SO101VisionEnv  # noqa: E402

TEACHER = HERE.parent / "pick_lift" / "runs" / "so101_pick_lift" / "final_model.zip"
RUNS_DIR = HERE / "runs"
EVAL_SEED = 10_000  # same evaluation seeds as so101/pick_lift/train.py --eval
TRAIN_SEED = 200_000  # collection episodes never overlap the evaluation seeds
PERTURB_STEP = 40  # --perturb: move the cube 4 cm (env.PERTURB_DIST) at step 40 (2 s)


# ---------------------------------------------------------------------- parallel simulators
def _worker(conn, perturb: bool):
    torch.set_num_threads(1)
    env = SO101VisionEnv()
    rng = np.random.default_rng()
    while True:
        cmd, arg = conn.recv()
        if cmd == "reset":
            obs = env.reset(seed=arg)[0]
            rng = np.random.default_rng(arg)
            conn.send(obs)
        elif cmd == "step":
            if perturb and env._steps == PERTURB_STEP:
                env.move_cube(rng.uniform(-np.pi, np.pi))
            obs, _, terminated, truncated, info = env.step(arg)
            conn.send((obs, terminated or truncated, info["is_success"]))
        elif cmd == "perturb_prob":  # random-step perturbations of the env itself (collection)
            env.perturb_prob = arg
        else:
            env.close()
            conn.close()
            return


class Envs:
    """N simulators in worker processes, stepped in lockstep."""

    def __init__(self, n: int, perturb: bool = False):
        ctx = mp.get_context("spawn")
        self.conns, self.procs = [], []
        for _ in range(n):
            parent, child = ctx.Pipe()
            p = ctx.Process(target=_worker, args=(child, perturb), daemon=True)
            p.start()
            self.conns.append(parent)
            self.procs.append(p)

    def reset(self, i: int, seed: int):
        self.conns[i].send(("reset", seed))
        return self.conns[i].recv()

    def step(self, actions: np.ndarray, active: list[int]):
        for i in active:
            self.conns[i].send(("step", actions[i]))
        return {i: self.conns[i].recv() for i in active}

    def set_perturb_prob(self, prob: float):
        for c in self.conns:
            c.send(("perturb_prob", prob))

    def close(self):
        for c in self.conns:
            c.send(("close", None))
        for p in self.procs:
            p.join()


def stack(obs_list: list[dict]) -> dict:
    return {k: np.stack([o[k] for o in obs_list]) for k in obs_list[0]}


# ------------------------------------------------------------------------------ student
class Encoder(nn.Module):
    """DrQ-style conv encoder for one 84x84 RGB camera."""

    def __init__(self, feat: int = 64):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2), nn.ReLU(),
            nn.Conv2d(32, 32, 3), nn.ReLU(),
            nn.Conv2d(32, 32, 3), nn.ReLU(),
            nn.Conv2d(32, 32, 3), nn.ReLU(),
        )
        self.head = nn.Sequential(nn.Linear(32 * 35 * 35, feat), nn.LayerNorm(feat), nn.Tanh())

    def forward(self, x):
        return self.head(self.conv(x).flatten(1))


class Student(nn.Module):
    def __init__(self, proprio_dim: int = 18, act_dim: int = 6):
        super().__init__()
        self.encoders = nn.ModuleList(Encoder() for _ in CAMERAS)
        self.mlp = nn.Sequential(
            nn.Linear(64 * len(CAMERAS) + proprio_dim, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, act_dim), nn.Tanh(),
        )

    def forward(self, image, proprio):
        """image: uint8 (B, 3 * cameras, 84, 84); proprio: (B, 18)."""
        x = image.float() / 255.0 - 0.5
        feats = [enc(x[:, 3 * i : 3 * i + 3]) for i, enc in enumerate(self.encoders)]
        return self.mlp(torch.cat([*feats, proprio], dim=1))

    @torch.no_grad()
    def act(self, obs: dict, device) -> np.ndarray:
        image = torch.as_tensor(obs["image"], device=device)
        proprio = torch.as_tensor(obs["proprio"], device=device)
        return self(image, proprio).cpu().numpy()


def random_shift(image: torch.Tensor, pad: int = 4) -> torch.Tensor:
    """Pad by replicating the border, then crop back to 84x84 at a random offset (DrQ)."""
    b = image.shape[0]
    padded = F.pad(image.float(), (pad,) * 4, mode="replicate")
    ox, oy = torch.randint(0, 2 * pad + 1, (2, b), device=image.device)
    rows = (oy[:, None] + torch.arange(IMG, device=image.device)[None])[:, :, None]
    cols = (ox[:, None] + torch.arange(IMG, device=image.device)[None])[:, None, :]
    idx = torch.arange(b, device=image.device)[:, None, None]
    return padded.permute(0, 2, 3, 1)[idx, rows, cols].permute(0, 3, 1, 2)


# ---------------------------------------------------------------------------- rollouts
def rollout(envs: Envs, seeds: list[int], student, teacher, beta: float, device, rng,
            record=None) -> list[bool]:
    """Run one episode per seed; `record(obs, labels)` receives every step's data."""
    n = len(envs.conns)
    queue = list(seeds)
    obs, active = {}, []
    for i in range(min(n, len(queue))):
        obs[i] = envs.reset(i, queue.pop(0))
        active.append(i)
    successes = []
    actions = np.zeros((n, 6), dtype=np.float32)
    while active:
        batch = stack([obs[i] for i in active])
        need_teacher = record is not None or beta > 0
        teacher_act = None
        if need_teacher:
            teacher_act = teacher.predict(batch["state"], deterministic=True)[0]
        student_act = student.act(batch, device) if beta < 1 else None
        if record is not None:
            record(batch, teacher_act)
        for j, i in enumerate(active):
            use_teacher = beta >= 1 or (beta > 0 and rng.random() < beta)
            actions[i] = teacher_act[j] if use_teacher else student_act[j]
        results = envs.step(actions, active)
        for i, (o, done, success) in results.items():
            obs[i] = o
            if done:
                successes.append(bool(success))
                if queue:
                    obs[i] = envs.reset(i, queue.pop(0))
                else:
                    active.remove(i)
    return successes


class Dataset:
    def __init__(self, capacity: int):
        self.image = np.zeros((capacity, 3 * len(CAMERAS), IMG, IMG), dtype=np.uint8)
        self.proprio = np.zeros((capacity, 18), dtype=np.float32)
        self.action = np.zeros((capacity, 6), dtype=np.float32)
        self.size = 0

    def add(self, batch: dict, labels: np.ndarray):
        k = min(len(labels), len(self.action) - self.size)
        s = slice(self.size, self.size + k)
        self.image[s] = batch["image"][:k]
        self.proprio[s] = batch["proprio"][:k]
        self.action[s] = labels[:k]
        self.size += k

    def sample(self, batch_size: int, device):
        idx = np.random.randint(0, self.size, batch_size)
        return (torch.as_tensor(self.image[idx], device=device),
                torch.as_tensor(self.proprio[idx], device=device),
                torch.as_tensor(self.action[idx], device=device))


def load_student(path, device) -> Student:
    student = Student().to(device)
    student.load_state_dict(torch.load(path, map_location=device))
    return student.eval()


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--name", default=time.strftime("dagger_%Y%m%d_%H%M%S"))
    p.add_argument("--teacher", default=str(TEACHER))
    p.add_argument("--iters", type=int, default=25)
    p.add_argument("--episodes-per-iter", type=int, default=64)
    p.add_argument("--updates-per-iter", type=int, default=1500)
    p.add_argument("--beta-decay", type=float, default=0.7,
                   help="teacher drives with probability beta_decay**iteration"
                   " (1 = behaviour cloning)")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--n-envs", type=int, default=16)
    p.add_argument("--seed", type=int, default=0,
                   help="training seed: network init, beta draws and collection episodes")
    p.add_argument("--eval-episodes", type=int, default=50,
                   help="student-only episodes per iteration")
    p.add_argument("--eval", metavar="STUDENT.pt",
                   help="only evaluate this student and exit"
                   " ('teacher' evaluates the SAC teacher)")
    p.add_argument("--collect-perturb-prob", type=float, default=0.0,
                   help="fraction of collection episodes in which the cube is moved once, at a "
                        "random step (pick_lift/env.py); the per-iteration evaluation stays "
                        "unperturbed")
    p.add_argument("--perturb", action="store_true",
                   help="with --eval: move the cube 4 cm at step 40"
                   " (pulls it out of the gripper)")
    p.add_argument("--episodes", type=int, default=200,
                   help="episodes for --eval and the final evaluation")
    args = p.parse_args()

    from stable_baselines3 import SAC

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_num_threads(4)
    teacher = SAC.load(args.teacher, device=device)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    envs = Envs(args.n_envs, perturb=args.perturb)
    eval_seeds = [EVAL_SEED + i for i in range(args.episodes)]

    if args.eval:
        if args.eval == "teacher":
            succ = rollout(envs, eval_seeds, None, teacher, 1.0, device, rng)
        else:
            succ = rollout(envs, eval_seeds, load_student(args.eval, device), teacher, 0.0,
                           device, rng)
        print(f"{args.eval}{' (perturbed)' if args.perturb else ''}: success "
              f"{np.mean(succ):.0%} ({sum(succ)}/{len(succ)})")
        envs.close()
        return

    run_dir = RUNS_DIR / args.name
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps(vars(args), indent=2))
    print(f"run dir: {run_dir}")

    student = Student().to(device)
    opt = torch.optim.Adam(student.parameters(), lr=args.lr)
    data = Dataset(args.iters * args.episodes_per_iter * 200)
    # each run uses at most iters * (episodes_per_iter + eval_episodes) episode seeds
    log, seed, best = [], TRAIN_SEED + 100_000 * args.seed, -1.0
    start = time.time()
    for it in range(args.iters):
        beta = args.beta_decay ** it
        seeds = list(range(seed, seed + args.episodes_per_iter))
        seed += args.episodes_per_iter
        student.eval()
        envs.set_perturb_prob(args.collect_perturb_prob)
        collect_succ = rollout(envs, seeds, student, teacher, beta, device, rng, record=data.add)
        envs.set_perturb_prob(0.0)

        student.train()
        losses = []
        for _ in range(args.updates_per_iter):
            image, proprio, action = data.sample(args.batch_size, device)
            loss = F.mse_loss(student(random_shift(image), proprio), action)
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(loss.item())

        student.eval()
        eval_succ = rollout(envs, list(range(seed, seed + args.eval_episodes)), student, teacher,
                            0.0, device, rng)
        seed += args.eval_episodes
        entry = dict(iter=it, beta=round(beta, 3), samples=data.size,
                     collect_success=float(np.mean(collect_succ)), loss=float(np.mean(losses)),
                     student_success=float(np.mean(eval_succ)),
                     minutes=round((time.time() - start) / 60, 1))
        log.append(entry)
        print(json.dumps(entry), flush=True)
        (run_dir / "log.json").write_text(json.dumps(log, indent=1))
        torch.save(student.state_dict(), run_dir / "student.pt")
        if entry["student_success"] >= best:
            best = entry["student_success"]
            torch.save(student.state_dict(), run_dir / "best_student.pt")

    for name in ("student.pt", "best_student.pt"):
        succ = rollout(envs, eval_seeds, load_student(run_dir / name, device), teacher, 0.0,
                       device, rng)
        print(f"final {name}: success {np.mean(succ):.0%} ({sum(succ)}/{len(succ)})", flush=True)
    envs.close()


if __name__ == "__main__":
    main()
