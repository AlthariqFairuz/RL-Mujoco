"""Pick-and-lift environment for the SO-101 arm (LeRobot's arm; model from MuJoCo Menagerie).

Port of `low_cost_robot_arm/pick_lift/env.py` (Koch v1.1) to the SO-101: same task, observation
layout, action interface and staged reward, only the robot changes.

A 2.5 cm cube sits on the table in front of the arm. The agent must grasp it and lift it
`LIFT_HEIGHT` above the table and hold it there for a few steps.

Action (6,) in [-1, 1]: per-step *change* of the joint-position targets of the six position
actuators (shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper). Targets are
clipped to the joint ranges. Gripper: -0.17 = closed, GRIPPER_OPEN = open (the full 1.75 rad
opening is 13 cm, far more than the cube needs).

Observation (35,), roughly in [-1, 1]:
    arm joint positions (6, scaled to range), joint velocities (6), grasp-centre position (3),
    cube position (3), cube - grasp-centre (3), cube orientation (6, first two columns of its
    rotation matrix), jaw-touching-cube flags (2), current joint targets (6, scaled to range).

Reward (dense, staged): reach + close-near-cube + both-jaws-on-cube + lift-while-grasped,
a per-step bonus while the cube is held lifted, and a small action penalty. See `_reward`.
Episodes run their full length; `is_success` is whether the cube has been held lifted for
`hold_steps` at the current step.
"""
from pathlib import Path

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces

SCENE_XML = (
    Path(__file__).resolve().parents[2] / "mujoco_menagerie" / "robotstudio_so101" / "scene.xml"
)

ARM_JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
# small collision boxes near the jaw tips; their mean is the grasp centre
PADS = ["fixed_jaw_box4", "fixed_jaw_box5", "moving_jaw_box2", "moving_jaw_box3"]
FIXED_JAW, MOVING_JAW = "gripper", "moving_jaw_so101_v1"  # bodies
BASE_XY = np.array([0.0388, 0.0])  # world xy of the shoulder_pan axis
GRIPPER_OPEN = 1.0  # rad, ~9 cm between the jaw tips
POS_SCALE = 5.0  # metres -> ~O(1)

# per-step change of the joint targets at |action| = 1 (rad)
ACTION_SCALE = np.array([0.10, 0.10, 0.10, 0.15, 0.20, 0.25])

# reward weights (W_SUCCESS is paid on every step the cube is held lifted)
W_REACH, W_CLOSE, W_GRASP, W_LIFT, W_SUCCESS, W_ACTION = 1.0, 0.5, 1.0, 5.0, 2.5, 0.005


def _build_model(cube_half: float, cube_mass: float) -> mujoco.MjModel:
    """Menagerie scene + a free-floating cube."""
    spec = mujoco.MjSpec.from_file(str(SCENE_XML))
    cube = spec.worldbody.add_body(name="cube", pos=[0.25, 0.0, cube_half])
    cube.add_freejoint(name="cube_joint")
    cube.add_geom(
        name="cube_geom",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=[cube_half] * 3,
        mass=cube_mass,
        friction=[1.0, 0.005, 0.0001],
        condim=4,
        rgba=[0.9, 0.15, 0.1, 1.0],
    )
    return spec.compile()


class SO101PickLiftEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 20}

    def __init__(
        self,
        render_mode: str | None = None,
        frame_skip: int = 10,  # 10 * 0.005 s = 20 Hz control
        max_episode_steps: int = 200,
        cube_half: float = 0.0125,
        cube_mass: float = 0.02,
        lift_height: float = 0.05,
        hold_steps: int = 3,
        spawn_radius: tuple[float, float] = (0.13, 0.25),  # from the shoulder_pan axis
        spawn_angle: float = 0.6,  # +/- rad around the arm's forward (+x) direction
        yaw_range: float = 0.5,  # +/- rad random cube yaw
        init_noise: float = 0.05,  # rad noise on the arm's start pose
    ):
        super().__init__()
        self.render_mode = render_mode
        self.frame_skip = frame_skip
        self.max_episode_steps = max_episode_steps
        self.cube_half = cube_half
        self.lift_height = lift_height
        self.hold_steps = hold_steps
        self.spawn_radius = spawn_radius
        self.spawn_angle = spawn_angle
        self.yaw_range = yaw_range
        self.init_noise = init_noise

        self.model = _build_model(cube_half, cube_mass)
        self.data = mujoco.MjData(self.model)

        jnt = [self.model.joint(n) for n in ARM_JOINTS]
        self._qadr = np.array([j.qposadr[0] for j in jnt])
        self._dadr = np.array([j.dofadr[0] for j in jnt])
        self._lo = self.model.jnt_range[[j.id for j in jnt], 0].copy()
        self._hi = self.model.jnt_range[[j.id for j in jnt], 1].copy()
        self._hi[5] = GRIPPER_OPEN
        self._mid, self._half = (self._lo + self._hi) / 2, (self._hi - self._lo) / 2

        cube_jnt = self.model.joint("cube_joint")
        self._cube_qadr = cube_jnt.qposadr[0]
        self._cube_body = self.model.body("cube").id
        self._cube_geom = self.model.geom("cube_geom").id
        self._pad_ids = np.array([self.model.geom(n).id for n in PADS])
        self._fixed_geoms = self._body_geoms(FIXED_JAW)
        self._moving_geoms = self._body_geoms(MOVING_JAW)

        self.action_space = spaces.Box(-1.0, 1.0, shape=(6,), dtype=np.float32)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(35,), dtype=np.float32)

        self._renderer = None
        self._target = self._mid.copy()
        self._steps = 0
        self._hold = 0

    # ------------------------------------------------------------------ helpers
    def _body_geoms(self, body_name: str) -> np.ndarray:
        bid = self.model.body(body_name).id
        return np.flatnonzero(self.model.geom_bodyid == bid)

    def _grasp_center(self) -> np.ndarray:
        return self.data.geom_xpos[self._pad_ids].mean(axis=0)

    def _cube_pos(self) -> np.ndarray:
        return self.data.xpos[self._cube_body]

    def _jaw_contacts(self) -> tuple[bool, bool]:
        """(fixed jaw touches cube, moving jaw touches cube)."""
        n = self.data.ncon
        if n == 0:
            return False, False
        g1, g2 = self.data.contact.geom1[:n], self.data.contact.geom2[:n]
        other = np.concatenate([g2[g1 == self._cube_geom], g1[g2 == self._cube_geom]])
        return bool(np.isin(other, self._fixed_geoms).any()), bool(
            np.isin(other, self._moving_geoms).any()
        )

    def _closure(self) -> float:
        """0 = gripper open, 1 = closed."""
        q = self.data.qpos[self._qadr[5]]
        return float(np.clip((self._hi[5] - q) / (self._hi[5] - self._lo[5]), 0.0, 1.0))

    def _obs(self) -> np.ndarray:
        q = self.data.qpos[self._qadr]
        qd = self.data.qvel[self._dadr]
        grasp, cube = self._grasp_center(), self._cube_pos()
        rot = self.data.xmat[self._cube_body].reshape(3, 3)[:, :2].ravel()
        touch = np.array(self._jaw_contacts(), dtype=np.float64)
        return np.concatenate(
            [
                (q - self._mid) / self._half,
                0.2 * qd,
                POS_SCALE * grasp,
                POS_SCALE * cube,
                POS_SCALE * (cube - grasp),
                rot,
                touch,
                (self._target - self._mid) / self._half,
            ]
        ).astype(np.float32)

    def _reward(self, action: np.ndarray):
        grasp, cube = self._grasp_center(), self._cube_pos()
        dist = float(np.linalg.norm(cube - grasp))
        touch_a, touch_b = self._jaw_contacts()
        grasped = touch_a and touch_b
        lift = float(np.clip((cube[2] - self.cube_half) / self.lift_height, 0.0, 1.0))
        lifted = (cube[2] - self.cube_half) >= self.lift_height

        reward = W_REACH * (1.0 - np.tanh(10.0 * dist))
        reward += W_CLOSE * float(dist < 0.04) * self._closure()
        reward += W_GRASP * float(grasped)
        reward += W_LIFT * lift * float(grasped)
        reward -= W_ACTION * float(np.sum(np.square(action)))

        self._hold = self._hold + 1 if (lifted and grasped) else 0
        success = self._hold >= self.hold_steps
        if success:
            reward += W_SUCCESS
        return reward, success, dict(dist=dist, lift=lift, grasped=grasped)

    # ---------------------------------------------------------------- gym API
    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        rng = self.np_random

        # arm: stretched out forward (all zeros) + noise, gripper open
        q = np.zeros(6)
        q[:5] = rng.uniform(-self.init_noise, self.init_noise, 5)
        q[5] = GRIPPER_OPEN
        q = np.clip(q, self._lo, self._hi)
        self.data.qpos[self._qadr] = q
        self._target = q.copy()
        self.data.ctrl[:] = self._target

        # cube: random polar position in front of the arm, random yaw
        r = rng.uniform(*self.spawn_radius)
        th = rng.uniform(-self.spawn_angle, self.spawn_angle)
        xy = BASE_XY + r * np.array([np.cos(th), np.sin(th)])
        yaw = rng.uniform(-self.yaw_range, self.yaw_range)
        a = self._cube_qadr
        self.data.qpos[a : a + 3] = [xy[0], xy[1], self.cube_half]
        self.data.qpos[a + 3 : a + 7] = [np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)]

        mujoco.mj_forward(self.model, self.data)
        for _ in range(10):  # let the cube settle on the table
            mujoco.mj_step(self.model, self.data)

        self._steps = 0
        self._hold = 0
        return self._obs(), {}

    def step(self, action):
        action = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        self._target = np.clip(self._target + action * ACTION_SCALE, self._lo, self._hi)
        self.data.ctrl[:] = self._target
        mujoco.mj_step(self.model, self.data, nstep=self.frame_skip)
        self._steps += 1

        reward, success, info = self._reward(action)
        cube_r = np.linalg.norm(self._cube_pos()[:2] - BASE_XY)
        terminated = bool(cube_r > 0.38)  # knocked away, cannot be recovered
        truncated = self._steps >= self.max_episode_steps
        info["is_success"] = bool(success)
        return self._obs(), float(reward), terminated, truncated, info

    def render(self):
        if self.render_mode != "rgb_array":
            return None
        if self._renderer is None:
            self._renderer = mujoco.Renderer(self.model, height=480, width=640)
            self._cam = mujoco.MjvCamera()
            self._cam.lookat[:] = [0.2, 0.0, 0.05]
            self._cam.distance, self._cam.azimuth, self._cam.elevation = 0.75, 140.0, -25.0
        self._renderer.update_scene(self.data, self._cam)
        return self._renderer.render()

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
