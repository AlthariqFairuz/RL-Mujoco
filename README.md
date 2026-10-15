# MuJoCo RL simulations

A collection of reinforcement learning experiments in [MuJoCo](https://mujoco.org) simulation.
Each project trains agents on a simulated robot with custom
[Gymnasium](https://gymnasium.farama.org) environments and
[Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3).
All projects share one Python environment.

## Projects

| Project | Robot | Tasks | Highlights |
|---|---|---|---|
| [**Low-cost robot arm**](low_cost_robot_arm/) | Koch v1.1, 5-DoF + gripper | reach, pick-and-lift | 96% pick-and-lift success; SAC vs TD3 vs PPO; sparse reward + HER (12% → 96%); Diffusion Policy imitating SAC (78%) |
| [**Unitree Go2**](unitree_go2/) | quadruped, 12 joints | velocity-tracking walking | PPO, 0.06 m/s tracking error, 0% falls; forward, sideways, backward, turning; domain randomization cuts falls under 1 m/s pushes from 62% to 10% |

<p align="center">
  <img src="low_cost_robot_arm/videos/pick_lift.gif" width="45%">
  <img src="low_cost_robot_arm/videos/reach_sparse_vs_her.gif" width="53%">
</p>
<p align="center">
  <img src="unitree_go2/videos/walk.gif" width="45%">
</p>

## Setup

Requires Python ≥ 3.12 and [uv](https://docs.astral.sh/uv/). A CUDA GPU is used automatically if
PyTorch sees one.

```bash
uv sync

# robot models come from MuJoCo Menagerie: clone it sparsely, then add each robot a project needs
git clone --filter=blob:none --sparse https://github.com/google-deepmind/mujoco_menagerie.git
git -C mujoco_menagerie sparse-checkout set low_cost_robot_arm unitree_go2
```

Run every command from the repository root, e.g.
`uv run low_cost_robot_arm/reach/train.py`. Each project's README lists its commands.

## Repository layout

```
.
├── pyproject.toml, uv.lock     shared dependencies (one environment for all projects)
├── mujoco_menagerie/           robot models, downloaded (git-ignored)
└── <project>/                  one folder per robot / project
    ├── README.md               results, environment details, usage
    ├── <task>/
    │   ├── env.py              Gymnasium environment (observation, action, reward, termination)
    │   ├── train.py            training + evaluation
    │   └── runs/               trained models, logs, TensorBoard (git-ignored)
    ├── play.py                 render trained policies to video (headless via EGL)
    ├── reports/                experiment write-ups
    └── videos/                 videos, GIFs and plots
```

Projects are independent: each has its own environments, training scripts and results, and
loads its robot model from `mujoco_menagerie/`.
