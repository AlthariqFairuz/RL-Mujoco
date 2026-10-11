"""Velocity-tracking locomotion environment for the Unitree Go2 quadruped (MuJoCo Menagerie).

Every episode samples a velocity command (forward vx, sideways vy, turning rate wz) and the
robot must walk at that velocity. The command is re-sampled every `resample_every` seconds,
so one episode also teaches the robot to change gait/direction on the fly.

The Menagerie model uses torque motors; the env turns them into joint PD controllers (like the
real robot's low-level controller, torque = KP (target - q) - KD qdot, clipped to the motor
limits): the policy outputs joint-position targets at 50 Hz and MuJoCo runs the PD loop at
250 Hz. The PD loop is a MuJoCo position actuator, so it runs in C rather than in Python.

Action (12,) in [-1, 1]: joint-position target = default standing pose + ACTION_SCALE * action.
Joint order: FL, FR, RL, RR legs x (hip abduction, thigh, calf).

Observation (48,), roughly O(1), all in the robot's own (base) frame - nothing the real robot
could not measure with its IMU and joint encoders (plus base velocity, usually estimated):
    base linear velocity (3), base angular velocity (3), projected gravity (3: which way is
    "down" seen from the body -> tilt), command (3), joint positions - default pose (12),
    joint velocities (12), previous action (12).

Reward: a weighted sum of terms, each multiplied by the control period dt (legged_gym style):
    + track_lin_vel   exp(-|v_cmd_xy - v_xy|^2 / 0.25)     follow the commanded velocity
    + track_ang_vel   exp(-(wz_cmd - wz)^2 / 0.25)          follow the commanded turn rate
    + feet_air_time   reward steps that last longer than 0.5 s -> real steps, not shuffling
    - lin_vel_z, ang_vel_xy, orientation                    no bouncing, rolling or tilting
    - torques, action_rate, joint_acc                        smooth and energy-efficient
    - collision                                              thighs/calves must not hit the floor
Episodes terminate when the robot falls (body tilted > ~60 deg or base too low) and are
truncated after `max_episode_steps` (20 s).
"""
from pathlib import Path

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces

SCENE_XML = Path(__file__).resolve().parents[2] / "mujoco_menagerie" / "unitree_go2" / "scene.xml"

LEGS = ["FL", "FR", "RL", "RR"]
JOINTS = [f"{leg}_{part}_joint" for leg in LEGS for part in ("hip", "thigh", "calf")]
DEFAULT_POSE = np.array([0.0, 0.9, -1.8] * 4)  # Menagerie "home" keyframe
ACTION_SCALE = 0.5  # rad of target offset at |action| = 1 (0.25 was too tight to step)
KP, KD = 35.0, 0.5  # PD gains (Nm/rad, Nm s/rad); 20 lets the robot sag to 0.19 m



def _build_model(sim_dt: float) -> mujoco.MjModel:
    """Menagerie scene with each torque motor turned into a torque-limited PD controller."""
    spec = mujoco.MjSpec.from_file(str(SCENE_XML))
    spec.option.timestep = sim_dt
    spec.option.cone = mujoco.mjtCone.mjCONE_PYRAMIDAL  # 1.7x faster than elliptic, same gait
    for act in spec.actuators:
        limit = act.ctrlrange[1]  # the motor's torque limit
        act.set_to_position(kp=KP, kv=KD)
        act.forcelimited = mujoco.mjtLimited.mjLIMITED_TRUE
        act.forcerange = [-limit, limit]
        act.ctrllimited = mujoco.mjtLimited.mjLIMITED_FALSE
    return spec.compile()


# observation scales
OBS_LIN_VEL, OBS_ANG_VEL, OBS_JOINT_VEL = 2.0, 0.25, 0.05

REWARD_WEIGHTS = {
    "track_lin_vel": 1.0,
    "track_ang_vel": 0.5,
    "feet_air_time": 1.0,
    "lin_vel_z": -2.0,
    "ang_vel_xy": -0.05,
    "orientation": -1.0,
    "torques": -2e-4,
    "joint_acc": -2.5e-7,
    "action_rate": -0.01,
    "collision": -1.0,
}
TRACKING_SIGMA = 0.25


class Go2WalkEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 50}

    def __init__(
        self,
        render_mode: str | None = None,
        sim_dt: float = 0.004,
        decimation: int = 5,  # 5 * 0.004 s = 0.02 s -> 50 Hz policy
        max_episode_steps: int = 1000,  # 20 s
        resample_every: float = 10.0,  # s between new commands
        cmd_vx: tuple[float, float] = (-0.5, 1.0),
        cmd_vy: tuple[float, float] = (-0.5, 0.5),
        cmd_wz: tuple[float, float] = (-1.0, 1.0),
        init_noise: float = 0.1,  # rad noise on the start pose
        width: int = 640,
        height: int = 480,
    ):
        super().__init__()
        self.render_mode = render_mode
        self.decimation = decimation
        self.max_episode_steps = max_episode_steps
        self.cmd_low = np.array([cmd_vx[0], cmd_vy[0], cmd_wz[0]])
        self.cmd_high = np.array([cmd_vx[1], cmd_vy[1], cmd_wz[1]])
        self.init_noise = init_noise

        self.model = _build_model(sim_dt)
        self.data = mujoco.MjData(self.model)
        self.dt = sim_dt * decimation
        self.resample_steps = int(round(resample_every / self.dt))

        m = self.model
        self._qadr = np.array([m.joint(n).qposadr[0] for n in JOINTS])
        self._dadr = np.array([m.joint(n).dofadr[0] for n in JOINTS])
        self._base = m.body("base").id
        self._floor = m.geom("floor").id
        self._feet = np.array([m.geom(leg).id for leg in LEGS])
        # thigh / calf collision geoms (everything on the leg except the foot sphere)
        leg_bodies = {m.body(f"{leg}_{part}").id for leg in LEGS for part in ("thigh", "calf")}
        self._leg_geoms = np.array([
            g for g in range(m.ngeom)
            if m.geom_bodyid[g] in leg_bodies and g not in self._feet and m.geom_contype[g]
        ])

        self.action_space = spaces.Box(-1.0, 1.0, shape=(12,), dtype=np.float32)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(48,), dtype=np.float32)

        self._renderer = None
        self._width, self._height = width, height

    # ---------------------------------------------------------------- state helpers
    def _base_rot(self) -> np.ndarray:
        return self.data.xmat[self._base].reshape(3, 3)  # body -> world

    def _base_lin_vel(self) -> np.ndarray:
        return self._base_rot().T @ self.data.qvel[0:3]  # free-joint linear vel is in world frame

    def _base_ang_vel(self) -> np.ndarray:
        return self.data.qvel[3:6].copy()  # free-joint angular vel is already in body frame

    def _projected_gravity(self) -> np.ndarray:
        return self._base_rot().T @ np.array([0.0, 0.0, -1.0])

    def _contacts(self):
        """(foot on floor (4,), any thigh/calf on floor)."""
        pairs = self.data.contact.geom[: self.data.ncon]  # (ncon, 2) geom ids
        other = pairs[pairs[:, 0] == self._floor, 1]
        other = np.concatenate([other, pairs[pairs[:, 1] == self._floor, 0]])
        return np.isin(self._feet, other), bool(np.isin(other, self._leg_geoms).any())

    def _sample_command(self) -> np.ndarray:
        cmd = self.np_random.uniform(self.cmd_low, self.cmd_high)
        if np.linalg.norm(cmd[:2]) < 0.2:  # tiny commands are just noise: stand still instead
            cmd[:2] = 0.0
        if self.np_random.random() < 0.1:  # sometimes: stand still
            cmd[:] = 0.0
        return cmd

    def _get_obs(self) -> np.ndarray:
        return np.concatenate([
            self._base_lin_vel() * OBS_LIN_VEL,
            self._base_ang_vel() * OBS_ANG_VEL,
            self._projected_gravity(),
            self.command * np.array([OBS_LIN_VEL, OBS_LIN_VEL, OBS_ANG_VEL]),
            self.data.qpos[self._qadr] - DEFAULT_POSE,
            self.data.qvel[self._dadr] * OBS_JOINT_VEL,
            self.last_action,
        ]).astype(np.float32)

    # ---------------------------------------------------------------- gym API
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetDataKeyframe(self.model, self.data, self.model.key("home").id)
        self.data.qpos[self._qadr] = DEFAULT_POSE + self.np_random.uniform(
            -self.init_noise, self.init_noise, 12)
        yaw = self.np_random.uniform(-np.pi, np.pi)
        self.data.qpos[3:7] = [np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)]
        self.data.ctrl[:] = self.data.qpos[self._qadr]
        mujoco.mj_forward(self.model, self.data)

        self.command = self._sample_command()
        self.last_action = np.zeros(12)
        self.last_joint_vel = np.zeros(12)
        self.feet_air_time = np.zeros(4)
        self.last_contact = np.ones(4, dtype=bool)
        self.steps = 0
        self.episode_sums = {k: 0.0 for k in REWARD_WEIGHTS}
        return self._get_obs(), {}

    def step(self, action):
        action = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        self.data.ctrl[:] = DEFAULT_POSE + ACTION_SCALE * action
        mujoco.mj_step(self.model, self.data, nstep=self.decimation)
        self.steps += 1

        lin_vel, ang_vel = self._base_lin_vel(), self._base_ang_vel()
        gravity = self._projected_gravity()
        joint_vel = self.data.qvel[self._dadr]
        feet_contact, leg_hit = self._contacts()

        # feet air time: reward each foot touchdown by how long that foot was in the air
        contact = feet_contact | self.last_contact  # MuJoCo contacts flicker; smooth over 2 steps
        first_contact = (self.feet_air_time > 0.0) & contact
        self.feet_air_time += self.dt
        air_reward = np.sum((self.feet_air_time - 0.5) * first_contact)
        if np.linalg.norm(self.command[:2]) < 0.1:
            air_reward = 0.0  # standing still: no need to step
        self.feet_air_time[contact] = 0.0
        self.last_contact = feet_contact

        terms = {
            "track_lin_vel": np.exp(-np.sum((self.command[:2] - lin_vel[:2]) ** 2)
                                    / TRACKING_SIGMA),
            "track_ang_vel": np.exp(-((self.command[2] - ang_vel[2]) ** 2) / TRACKING_SIGMA),
            "feet_air_time": air_reward,
            "lin_vel_z": lin_vel[2] ** 2,
            "ang_vel_xy": np.sum(ang_vel[:2] ** 2),
            "orientation": np.sum(gravity[:2] ** 2),
            "torques": np.sum(self.data.actuator_force**2),
            "joint_acc": np.sum(((joint_vel - self.last_joint_vel) / self.dt) ** 2),
            "action_rate": np.sum((action - self.last_action) ** 2),
            "collision": float(leg_hit),
        }
        reward = 0.0
        for k, v in terms.items():
            r = REWARD_WEIGHTS[k] * v * self.dt
            self.episode_sums[k] += r
            reward += r

        self.last_action = action
        self.last_joint_vel = joint_vel.copy()
        if self.steps % self.resample_steps == 0:
            self.command = self._sample_command()

        base_z = self.data.qpos[2]
        terminated = bool(gravity[2] > -0.5 or base_z < 0.15)  # tilted > 60 deg or collapsed
        truncated = self.steps >= self.max_episode_steps
        info = {
            "lin_vel": lin_vel[:2].copy(),
            "ang_vel_z": float(ang_vel[2]),
            "command": self.command.copy(),
            "lin_vel_error": float(np.linalg.norm(self.command[:2] - lin_vel[:2])),
            "fell": terminated,
        }
        if terminated or truncated:
            info["episode_sums"] = dict(self.episode_sums)
        return self._get_obs(), float(reward), terminated, truncated, info

    def render(self):
        if self._renderer is None:
            self._renderer = mujoco.Renderer(self.model, self._height, self._width)
            self._camera = mujoco.MjvCamera()
            self._camera.type = mujoco.mjtCamera.mjCAMERA_TRACKING
            self._camera.trackbodyid = self._base
            self._camera.distance = 1.6
            self._camera.elevation = -20.0
            self._camera.azimuth = 135.0
        self._renderer.update_scene(self.data, self._camera)
        return self._renderer.render()

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
