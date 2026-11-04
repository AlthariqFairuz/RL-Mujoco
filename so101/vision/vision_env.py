"""Camera-based SO-101 pick-and-lift: `so101/pick_lift/env.py` with images instead of the cube pose.

Same robot, task, reward and actions. The observation is a dict:
    image    (6, 84, 84) uint8   two RGB cameras stacked on the channel axis:
                                 the wrist camera (on the gripper) and a fixed front camera
    proprio  (18,) float32       joint positions, joint velocities, joint targets (all scaled)
    state    (35,) float32       the full state observation of the state-based env; only the
                                 teacher reads it (cube pose, contacts), the student never does

So the student sees what a real SO-101 provides: cameras and servo readings.
"""
import mujoco
import numpy as np
from gymnasium import spaces

from env import SO101PickLiftEnv  # so101/pick_lift/env.py (the scripts put it on sys.path)

IMG = 84
CAMERAS = ["wrist_cam", "front"]
FRONT_POS, FRONT_LOOKAT = np.array([0.55, 0.0, 0.32]), np.array([0.2, 0.0, 0.0])


def _xyaxes(pos: np.ndarray, lookat: np.ndarray) -> list[float]:
    """Camera frame (x right, y up) looking from `pos` at `lookat`."""
    f = (lookat - pos) / np.linalg.norm(lookat - pos)
    x = np.cross(f, [0.0, 0.0, 1.0])
    x /= np.linalg.norm(x)
    return [*x, *np.cross(x, f)]


def _add_front_camera(spec: mujoco.MjSpec):
    spec.worldbody.add_camera(name="front", pos=FRONT_POS, fovy=55,
                              xyaxes=_xyaxes(FRONT_POS, FRONT_LOOKAT))


class SO101VisionEnv(SO101PickLiftEnv):
    def __init__(self, **kwargs):
        super().__init__(spec_edit=_add_front_camera, **kwargs)
        self._cam_renderer = None
        self.observation_space = spaces.Dict(
            image=spaces.Box(0, 255, shape=(3 * len(CAMERAS), IMG, IMG), dtype=np.uint8),
            proprio=spaces.Box(-np.inf, np.inf, shape=(18,), dtype=np.float32),
            state=self.observation_space,
        )

    def _images(self) -> np.ndarray:
        if self._cam_renderer is None:
            self._cam_renderer = mujoco.Renderer(self.model, IMG, IMG)
        imgs = []
        for cam in CAMERAS:
            self._cam_renderer.update_scene(self.data, camera=cam)
            imgs.append(self._cam_renderer.render())
        return np.concatenate(imgs, axis=2).transpose(2, 0, 1).copy()

    def _obs(self) -> dict:
        state = super()._obs()
        # state layout: joint pos (0:6), 0.2 * joint vel (6:12), ..., joint targets (29:35)
        proprio = np.concatenate([state[0:6], state[6:12], state[29:35]])
        return {"image": self._images(), "proprio": proprio, "state": state}

    def close(self):
        if self._cam_renderer is not None:
            self._cam_renderer.close()
            self._cam_renderer = None
        super().close()
