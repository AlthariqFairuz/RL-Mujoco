"""Reach environment for the Koch v1.1 "low-cost robot arm" (MuJoCo Menagerie).

A target point (green ball, visual only) is placed somewhere the arm can actually reach:
a random arm configuration is sampled, its grasp-centre position is used as the target, and
configurations that touch the floor / themselves or end up too low are rejected. The agent
must move the grasp centre (midpoint of the finger pads) onto the target and keep it there.

Action (5,) in [-1, 1]: per-step *change* of the joint-position targets of the five arm
actuators (base_rotation, pitch, elbow, wrist_pitch, wrist_roll). The gripper is held open.

Observation (24,), roughly in [-1, 1]:
    arm joint positions (5, scaled to joint range), joint velocities (5), grasp-centre
    position (3), target position (3), target - grasp-centre (3), current joint targets (5).

Reward (`reward_type`):
    "dense"  (shaped): -10 * distance (m) + 1 when within `success_tol`, minus a small action
             penalty - every step says how close the arm is.
    "sparse": 0 when within `success_tol`, else -1 - only says "there yet or not".
Episodes have a fixed length (no early termination); `is_success` is whether the grasp
centre is within `success_tol` of the target, so the value at the last step counts.

`goal_obs=True` returns the goal-conditioned Dict observation that Hindsight Experience Replay
(HER) needs: {"observation": arm state (18,), "achieved_goal": grasp centre (3,),
"desired_goal": target (3,)} (positions scaled by POS_SCALE), plus a vectorised
`compute_reward(achieved_goal, desired_goal, info)` so HER can recompute rewards for
substituted goals. The dense reward's action penalty is left out there (it does not depend on
the goal, and HER cannot recompute it).
"""
from pathlib import Path

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces

SCENE_XML = (
    Path(__file__).resolve().parents[2] / "mujoco_menagerie" / "low_cost_robot_arm" / "scene.xml"
)

ARM_JOINTS = ["base_rotation", "pitch", "elbow", "wrist_pitch", "wrist_roll"]
PADS = [
    "gripper_static_finger_pad_1",
    "gripper_static_finger_pad_3",
    "gripper_moving_finger_pad_1",
    "gripper_moving_finger_pad_3",
]
BASE_XY = np.array([0.012, 0.0])  # world xy of the base_rotation axis
POS_SCALE = 5.0  # metres -> ~O(1)
GRIPPER_OPEN = -1.6

# per-step change of the joint targets at |action| = 1 (rad)
ACTION_SCALE = np.array([0.12, 0.12, 0.12, 0.15, 0.20])

W_DIST, W_SUCCESS, W_ACTION = 10.0, 1.0, 0.01


def _build_model() -> mujoco.MjModel:
    """Menagerie scene + a mocap target marker that collides with nothing."""
    spec = mujoco.MjSpec.from_file(str(SCENE_XML))
    target = spec.worldbody.add_body(name="target", mocap=True, pos=[-0.2, 0.0, 0.1])
    target.add_geom(
        name="target_geom",
        type=mujoco.mjtGeom.mjGEOM_SPHERE,
        size=[0.01, 0, 0],
        contype=0,
        conaffinity=0,
        rgba=[0.1, 0.9, 0.2, 0.6],
    )
    return spec.compile()


class LowCostArmReachEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 20}

    def __init__(
        self,
        render_mode: str | None = None,
        frame_skip: int = 25,  # 25 * 0.002 s = 20 Hz control
        max_episode_steps: int = 100,
        success_tol: float = 0.01,
        base_range: float = 1.2,  # +/- rad of base rotation used when sampling targets
        min_height: float = 0.03,  # lowest allowed target z
        min_radius: float = 0.08,  # closest allowed target distance from the base axis
        init_noise: float = 0.05,  # rad noise on the arm's start pose
        reward_type: str = "dense",  # "dense" (shaped) | "sparse"
        goal_obs: bool = False,  # Dict observation for HER
    ):
        super().__init__()
        self.render_mode = render_mode
        self.frame_skip = frame_skip
        self.max_episode_steps = max_episode_steps
        self.success_tol = success_tol
        self.base_range = base_range
        self.min_height = min_height
        self.min_radius = min_radius
        self.init_noise = init_noise
        if reward_type not in ("dense", "sparse"):
            raise ValueError(f"reward_type must be 'dense' or 'sparse', got {reward_type!r}")
        self.reward_type = reward_type
        self.goal_obs = goal_obs

        self.model = _build_model()
        self.data = mujoco.MjData(self.model)
        self._scratch = mujoco.MjData(self.model)  # for forward kinematics when sampling targets

        jnt = [self.model.joint(n) for n in ARM_JOINTS]
        self._qadr = np.array([j.qposadr[0] for j in jnt])
        self._dadr = np.array([j.dofadr[0] for j in jnt])
        self._lo = self.model.jnt_range[[j.id for j in jnt], 0].copy()
        self._hi = self.model.jnt_range[[j.id for j in jnt], 1].copy()
        self._mid, self._half = (self._lo + self._hi) / 2, (self._hi - self._lo) / 2
        self._gripper_qadr = self.model.joint("gripper").qposadr[0]
        self._gripper_act = self.model.actuator("gripper").id
        self._arm_act = np.array([self.model.actuator(n).id for n in ARM_JOINTS])
        self._pad_ids = np.array([self.model.geom(n).id for n in PADS])

        self.action_space = spaces.Box(-1.0, 1.0, shape=(5,), dtype=np.float32)
        if goal_obs:
            def box(n):
                return spaces.Box(-np.inf, np.inf, shape=(n,), dtype=np.float32)

            self.observation_space = spaces.Dict(
                dict(observation=box(18), achieved_goal=box(3), desired_goal=box(3))
            )
        else:
            self.observation_space = spaces.Box(-np.inf, np.inf, shape=(24,), dtype=np.float32)

        self._renderer = None
        self._ctrl_target = np.zeros(5)
        self._goal = np.zeros(3)
        self._steps = 0

    # ------------------------------------------------------------------ helpers
    def _grasp_center(self, data: mujoco.MjData | None = None) -> np.ndarray:
        data = self.data if data is None else data
        return data.geom_xpos[self._pad_ids].mean(axis=0)

    def _sample_goal(self) -> np.ndarray:
        """Grasp-centre position of a random collision-free arm configuration."""
        rng, d = self.np_random, self._scratch
        lo, hi = self._lo.copy(), self._hi.copy()
        lo[0], hi[0] = -self.base_range, self.base_range
        for _ in range(1000):
            mujoco.mj_resetData(self.model, d)
            d.qpos[self._qadr] = rng.uniform(lo, hi)
            d.qpos[self._gripper_qadr] = GRIPPER_OPEN
            mujoco.mj_forward(self.model, d)
            p = self._grasp_center(d)
            if (
                d.ncon == 0
                and d.geom_xpos[self._pad_ids, 2].min() >= self.min_height
                and np.linalg.norm(p[:2] - BASE_XY) >= self.min_radius
            ):
                return p.copy()
        raise RuntimeError("could not sample a reachable target")

    def _obs(self):
        q = self.data.qpos[self._qadr]
        qd = self.data.qvel[self._dadr]
        grasp = self._grasp_center()
        if self.goal_obs:
            arm = np.concatenate(
                [
                    (q - self._mid) / self._half,
                    0.2 * qd,
                    POS_SCALE * grasp,
                    (self._ctrl_target - self._mid) / self._half,
                ]
            )
            return dict(
                observation=arm.astype(np.float32),
                achieved_goal=(POS_SCALE * grasp).astype(np.float32),
                desired_goal=(POS_SCALE * self._goal).astype(np.float32),
            )
        return np.concatenate(
            [
                (q - self._mid) / self._half,
                0.2 * qd,
                POS_SCALE * grasp,
                POS_SCALE * self._goal,
                POS_SCALE * (self._goal - grasp),
                (self._ctrl_target - self._mid) / self._half,
            ]
        ).astype(np.float32)

    def compute_reward(self, achieved_goal, desired_goal, info=None) -> np.ndarray:
        """Goal-only reward on POS_SCALE-scaled positions, batched over leading dims (for HER)."""
        dist = np.linalg.norm(np.asarray(achieved_goal) - np.asarray(desired_goal), axis=-1)
        dist = dist / POS_SCALE
        success = (dist < self.success_tol).astype(np.float32)
        if self.reward_type == "sparse":
            return success - 1.0
        return (-W_DIST * dist + W_SUCCESS * success).astype(np.float32)

    # ---------------------------------------------------------------- gym API
    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self._goal = self._sample_goal()

        mujoco.mj_resetData(self.model, self.data)
        q = self.np_random.uniform(-self.init_noise, self.init_noise, 5)
        q = np.clip(q, self._lo, self._hi)
        self.data.qpos[self._qadr] = q
        self.data.qpos[self._gripper_qadr] = GRIPPER_OPEN
        self._ctrl_target = q.copy()
        self.data.ctrl[self._arm_act] = self._ctrl_target
        self.data.ctrl[self._gripper_act] = GRIPPER_OPEN
        self.data.mocap_pos[0] = self._goal
        mujoco.mj_forward(self.model, self.data)

        self._steps = 0
        return self._obs(), {}

    def step(self, action):
        action = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        self._ctrl_target = np.clip(
            self._ctrl_target + action * ACTION_SCALE, self._lo, self._hi
        )
        self.data.ctrl[self._arm_act] = self._ctrl_target
        mujoco.mj_step(self.model, self.data, nstep=self.frame_skip)
        self._steps += 1

        dist = float(np.linalg.norm(self._goal - self._grasp_center()))
        success = dist < self.success_tol
        if self.reward_type == "sparse":
            reward = float(success) - 1.0
        else:
            reward = -W_DIST * dist + W_SUCCESS * float(success)
            if not self.goal_obs:
                reward -= W_ACTION * float(np.sum(np.square(action)))

        truncated = self._steps >= self.max_episode_steps
        info = dict(dist=dist, is_success=bool(success))
        return self._obs(), float(reward), False, truncated, info

    def render(self):
        if self.render_mode != "rgb_array":
            return None
        if self._renderer is None:
            self._renderer = mujoco.Renderer(self.model, height=480, width=640)
            self._cam = mujoco.MjvCamera()
            self._cam.lookat[:] = [-0.12, 0.0, 0.08]
            self._cam.distance, self._cam.azimuth, self._cam.elevation = 0.75, 150.0, -25.0
        self._renderer.update_scene(self.data, self._cam)
        return self._renderer.render()

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
