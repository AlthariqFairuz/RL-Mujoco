# MuJoCo RL simulations

A collection of reinforcement learning experiments in [MuJoCo](https://mujoco.org) simulation.
Each project trains agents on a simulated robot with custom
[Gymnasium](https://gymnasium.farama.org) environments and
[Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3).
All projects share one Python environment.

## Projects

<table>
  <tr>
    <th width="33%"><a href="low_cost_robot_arm/">Low-cost robot arm</a></th>
    <th width="33%"><a href="so101/">SO-101</a></th>
    <th width="33%"><a href="unitree_go2/">Unitree Go2</a></th>
  </tr>
  <tr>
    <td><img src="low_cost_robot_arm/videos/pick_lift.gif" width="100%"></td>
    <td><img src="so101/videos/pick_lift.gif" width="100%"></td>
    <td><img src="unitree_go2/videos/walk.gif" width="100%"></td>
  </tr>
  <tr>
    <td align="center">Koch v1.1 arm, pick-and-lift (SAC)</td>
    <td align="center">LeRobot's SO-101 arm, pick-and-lift (SAC)</td>
    <td align="center">quadruped, velocity-tracking walking (PPO)</td>
  </tr>
</table>

| Project | Robot | Highlights |
|---|---|---|
| [**Low-cost robot arm**](low_cost_robot_arm/) | Koch v1.1, 5-DoF + gripper | reach and pick-and-lift, **96%** success; SAC vs TD3 vs PPO; sparse reward + HER (12% → 96%); Diffusion Policy imitating SAC (78%) |
| [**SO-101**](so101/) | SO-101 (LeRobot), 5-DoF + gripper | pick-and-lift recipe ported from the Koch arm unchanged: **92%** success |
| [**Unitree Go2**](unitree_go2/) | quadruped, 12 joints | walks at commanded speed and turn rate: 0.06 m/s error, 0% falls; domain randomization cuts falls under 1 m/s pushes from 62% to 10% |

All results are in simulation; no policy has been run on a real robot yet.

## Setup

Requires Python ≥ 3.12 and [uv](https://docs.astral.sh/uv/). A CUDA GPU is used automatically if
PyTorch sees one.

```bash
uv sync

# robot models come from MuJoCo Menagerie: clone it sparsely, then add each robot a project needs
git clone --filter=blob:none --sparse https://github.com/google-deepmind/mujoco_menagerie.git
git -C mujoco_menagerie sparse-checkout set low_cost_robot_arm robotstudio_so101 unitree_go2
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
