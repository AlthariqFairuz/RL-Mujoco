"""Pick-and-lift environment for the Koch v1.1 "low-cost robot arm" (MuJoCo Menagerie).

A 2.5 cm cube sits on the floor in front of the arm. The agent must grasp it and lift it
`LIFT_HEIGHT` above the floor and hold it there for a few steps.

Action (6,) in [-1, 1]: per-step *change* of the joint-position targets of the six position
actuators (base_rotation, pitch, elbow, wrist_pitch, wrist_roll, gripper). Targets are clipped
to the joint ranges. Gripper: -1.6 = fully open, 0 = closed.

Observation (35,), roughly in [-1, 1] (no external normalisation needed):
    arm joint positions (6, scaled to joint range), joint velocities (6), grasp-centre position (3),
    cube position (3), cube - grasp-centre (3), cube orientation (6, first two columns of its
    rotation matrix), finger-touching-cube flags (2), current joint targets (6, scaled to range).

Reward (dense, staged): reach + close-near-cube + both-fingers-on-cube + lift-while-grasped,
a per-step bonus while the cube is held lifted, and a small action penalty. See `_reward`.

Episodes run their full length - success does not terminate them. (Terminating on success
with positive per-step rewards made success the worse option: hovering just below the lift
height earned more than the one-off success bonus.) `is_success` is whether the cube has been
held lifted for `hold_steps` at the current step, so the value at the last step counts.
"""
from pathlib import Path

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces

SCENE_XML = (
    Path(__file__).resolve().parents[2] / "mujoco_menagerie" / "low_cost_robot_arm" / "scene.xml"
)

ARM_JOINTS = ["base_rotation", "pitch", "elbow", "wrist_pitch", "wrist_roll", "gripper"]
PADS = [
    "gripper_static_finger_pad_1",
    "gripper_static_finger_pad_3",
    "gripper_moving_finger_pad_1",
    "gripper_moving_finger_pad_3",
]
BASE_XY = np.array([0.012, 0.0])  # world xy of the base_rotation axis
POS_SCALE = 5.0  # metres -> ~O(1)

# per-step change of the joint targets at |action| = 1 (rad)
ACTION_SCALE = np.array([0.12, 0.12, 0.12, 0.15, 0.20, 0.30])

# reward weights (W_SUCCESS is paid on every step the cube is held lifted)
W_REACH, W_CLOSE, W_GRASP, W_LIFT, W_SUCCESS, W_ACTION = 1.0, 0.5, 1.0, 5.0, 2.5, 0.005


def _build_model(cube_half: float, cube_mass: float) -> mujoco.MjModel:
    """Menagerie scene + a free-floating cube."""
    spec = mujoco.MjSpec.from_file(str(SCENE_XML))
    cube = spec.worldbody.add_body(name="cube", pos=[-0.2, 0.0, cube_half])
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


class LowCostArmPickLiftEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 20}

    def __init__(
        self,
        render_mode: str | None = None,
        frame_skip: int = 25,  # 25 * 0.002 s = 20 Hz control
        max_episode_steps: int = 200,
        cube_half: float = 0.0125,
        cube_mass: float = 0.015,
        lift_height: float = 0.05,
        hold_steps: int = 3,
        spawn_radius: tuple[float, float] = (0.15, 0.23),  # from the base axis
        spawn_angle: float = 0.5,  # +/- rad around the arm's forward (-x) direction
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
        self._mid, self._half = (self._lo + self._hi) / 2, (self._hi - self._lo) / 2

        cube_jnt = self.model.joint("cube_joint")
        self._cube_qadr = cube_jnt.qposadr[0]
        self._cube_dadr = cube_jnt.dofadr[0]
        self._cube_body = self.model.body("cube").id
        self._cube_geom = self.model.geom("cube_geom").id
        self._pad_ids = np.array([self.model.geom(n).id for n in PADS])
        self._static_geoms = self._body_geoms("gripper_static_finger")
        self._moving_geoms = self._body_geoms("gripper_moving_finger")

        self.action_space = spaces.Box(-1.0, 1.0, shape=(6,), dtype=np.float32)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(35,), dtype=np.float32)

        self._renderer = None
        self._target = self._lo.copy()
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

    def _finger_contacts(self) -> tuple[bool, bool]:
        """(static finger touches cube, moving finger touches cube)."""
        n = self.data.ncon
        if n == 0:
            return False, False
        g1, g2 = self.data.contact.geom1[:n], self.data.contact.geom2[:n]
        other = np.concatenate([g2[g1 == self._cube_geom], g1[g2 == self._cube_geom]])
        return bool(np.isin(other, self._static_geoms).any()), bool(
            np.isin(other, self._moving_geoms).any()
        )

    def _closure(self) -> float:
        """0 = gripper open, 1 = closed."""
        q = self.data.qpos[self._qadr[5]]
        return float((q - self._lo[5]) / (self._hi[5] - self._lo[5]))

    def _obs(self) -> np.ndarray:
        q = self.data.qpos[self._qadr]
        qd = self.data.qvel[self._dadr]
        grasp, cube = self._grasp_center(), self._cube_pos()
        rot = self.data.xmat[self._cube_body].reshape(3, 3)[:, :2].ravel()
        touch = np.array(self._finger_contacts(), dtype=np.float64)
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
        touch_a, touch_b = self._finger_contacts()
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

        # arm: home pose + noise, gripper fully open
        q = np.zeros(6)
        q[:5] = rng.uniform(-self.init_noise, self.init_noise, 5)
        q[5] = self._lo[5]
        q = np.clip(q, self._lo, self._hi)
        self.data.qpos[self._qadr] = q
        self._target = q.copy()
        self.data.ctrl[:] = self._target

        # cube: random polar position in front of the arm, random yaw
        r = rng.uniform(*self.spawn_radius)
        th = np.pi + rng.uniform(-self.spawn_angle, self.spawn_angle)
        xy = BASE_XY + r * np.array([np.cos(th), np.sin(th)])
        yaw = rng.uniform(-self.yaw_range, self.yaw_range)
        a = self._cube_qadr
        self.data.qpos[a : a + 3] = [xy[0], xy[1], self.cube_half]
        self.data.qpos[a + 3 : a + 7] = [np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)]

        mujoco.mj_forward(self.model, self.data)
        for _ in range(10):  # let the cube settle on the floor
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
        out_of_reach = cube_r > 0.32  # knocked away, cannot be recovered
        terminated = bool(out_of_reach)
        truncated = self._steps >= self.max_episode_steps
        info["is_success"] = bool(success)
        return self._obs(), float(reward), terminated, truncated, info

    def render(self):
        if self.render_mode != "rgb_array":
            return None
        if self._renderer is None:
            self._renderer = mujoco.Renderer(self.model, height=480, width=640)
            self._cam = mujoco.MjvCamera()
            self._cam.lookat[:] = [-0.15, 0.0, 0.05]
            self._cam.distance, self._cam.azimuth, self._cam.elevation = 0.7, 150.0, -25.0
        self._renderer.update_scene(self.data, self._cam)
        return self._renderer.render()

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
