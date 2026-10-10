"""Compare reach runs: learning curves vs env steps and vs wall-clock time.

    uv run low_cost_robot_arm/reach/compare.py reach_sac reach_td3 reach_ppo [--out PATH.png]

The plot goes to low_cost_robot_arm/videos/reach_compare.png unless --out is given.

Reads runs/<name>/evaluations.npz (deterministic evaluation, success per eval) and
runs/<name>/monitor/*.csv (training episodes, with wall-clock timestamps).
"""
import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RUNS_DIR = Path(__file__).resolve().parent / "runs"
VIDEOS_DIR = Path(__file__).resolve().parents[1] / "videos"


def load_monitor(run_dir: Path) -> pd.DataFrame:
    """All training episodes of all envs, in finishing order, with cumulative env steps."""
    df = pd.concat(pd.read_csv(f, skiprows=1) for f in sorted((run_dir / "monitor").glob("*.csv")))
    df = df.sort_values("t").reset_index(drop=True)
    df["steps"] = df["l"].cumsum()
    df["success"] = df["is_success"].astype(float).rolling(200, min_periods=1).mean()
    return df


def first_reach(x: np.ndarray, y: np.ndarray, level: float):
    idx = np.flatnonzero(y >= level)
    return x[idx[0]] if len(idx) else None


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("runs", nargs="+", help="run folder names under low_cost_robot_arm/reach/runs/")
    p.add_argument("--out", default=str(VIDEOS_DIR / "reach_compare.png"))
    args = p.parse_args()

    fig, axes = plt.subplots(1, 3, figsize=(17, 4.8))
    header = f"{"run":<22}{'eval steps':>11}{'final eval':>11}{'best eval':>10}" \
             f"{'1st >=50%':>11}{'1st >=90%':>11}{'train time':>12}{'90% at time':>12}"
    print(header + "\n" + "-" * len(header))
    for name in args.runs:
        run_dir = RUNS_DIR / name
        ev = np.load(run_dir / "evaluations.npz")
        steps, succ = ev["timesteps"], ev["successes"].mean(axis=1)
        mon = load_monitor(run_dir)
        hours = mon["t"] / 3600

        axes[0].plot(steps / 1e3, 100 * succ, label=name)
        axes[1].plot(mon["steps"] / 1e3, 100 * mon["success"], label=name)
        axes[2].plot(hours, 100 * mon["success"], label=name)

        s50, s90 = first_reach(steps, succ, 0.5), first_reach(steps, succ, 0.9)
        t90 = first_reach(hours.to_numpy(), mon["success"].to_numpy(), 0.9)
        def fmt(v):
            return f"{v / 1e3:.0f}k" if v is not None else "never"

        print(f"{name:<22}{steps[-1] / 1e3:>10.0f}k{succ[-1]:>11.0%}{succ.max():>10.0%}"
              f"{fmt(s50):>11}{fmt(s90):>11}{hours.iloc[-1]:>11.2f}h"
              f"{(f'{t90:.2f}h' if t90 is not None else 'never'):>12}")

    titles = ["evaluation (deterministic policy)", "training episodes (rolling 200)",
              "training episodes vs wall-clock"]
    xlabels = ["env steps (thousands)", "env steps (thousands)", "hours"]
    for ax, title, xl in zip(axes, titles, xlabels):
        ax.set(title=title, xlabel=xl, ylabel="success rate (%)", ylim=(-2, 102))
        ax.grid(alpha=0.3, which="both")
        ax.legend()
    for ax in axes[:2]:  # PPO's 10x longer run would squash the off-policy curves otherwise
        ax.set_xscale("log")
        ax.set_xlim(left=5)
    fig.tight_layout()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=110)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
