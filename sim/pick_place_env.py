import math
from typing import Dict, Optional, Tuple

import gymnasium as gym
import mujoco
import numpy as np


class HardcodedPickPlaceEnv(gym.Env):
    """Gym-style MuJoCo env with a scripted pick-and-place controller.

    The environment ignores the incoming action and drives the SO-100 arm
    through a hardcoded finite-state sequence to pick the red block and
    place it on the goal marker.
    """

    metadata = {"render_modes": ["rgb_array", "human"], "render_fps": 50}

    def __init__(
        self,
        xml_path: str = "sim/scene.xml",
        max_steps: int = 800,
        control_dt: float = 0.02,
        render_mode: Optional[str] = None,
        image_size: Tuple[int, int] = (480, 640),
    ) -> None:
        super().__init__()
        if render_mode not in (None, "rgb_array", "human"):
            raise ValueError(f"Unsupported render_mode: {render_mode}")

        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)

        self.max_steps = max_steps
        self.render_mode = render_mode
        self.image_h, self.image_w = image_size
        self.frame_skip = max(1, int(round(control_dt / self.model.opt.timestep)))

        self.joint_names = [
            "shoulder_pan",
            "shoulder_lift",
            "elbow_flex",
            "wrist_flex",
            "wrist_roll",
            "gripper_jaw",
        ]

        self._joint_qadr: Dict[str, int] = {
            name: int(np.asarray(self.model.joint(name).qposadr).reshape(-1)[0])
            for name in self.joint_names
        }
        self._actuator_id: Dict[str, int] = {
            name: int(mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name))
            for name in self.joint_names
        }

        self._body_block = int(mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "block"))
        self._site_goal = int(mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "place_centre"))
        self._site_ee = int(mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "ee_site"))

        obs_dim = int(self.model.nq + self.model.nv + self.model.nsensordata)
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32,
        )

        # Action is accepted for API compatibility but ignored by policy.
        self.action_space = gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(len(self.joint_names),),
            dtype=np.float32,
        )

        self._renderer = None
        self._cv2 = None
        if self.render_mode is not None:
            self._renderer = mujoco.Renderer(self.model, height=self.image_h, width=self.image_w)

        self._phase_idx = 0
        self._phase_ticks = 0
        self._steps = 0
        self._success_count = 0

        self._phase_names = [
            "pregrasp",
            "descend",
            "close",
            "lift",
            "move_to_goal",
            "lower_to_goal",
            "release",
            "retreat",
        ]

        # Joint target waypoints for scripted policy.
        self._targets = {
            "pregrasp": np.array([0.00, 1.25, -1.95, -0.90, 0.00, 1.50], dtype=np.float64),
            "descend": np.array([0.00, 1.45, -2.15, -1.05, 0.00, 1.50], dtype=np.float64),
            "close": np.array([0.00, 1.45, -2.15, -1.05, 0.00, 0.10], dtype=np.float64),
            "lift": np.array([0.00, 1.20, -1.80, -0.75, 0.00, 0.10], dtype=np.float64),
            "move_to_goal": np.array([0.65, 1.20, -1.85, -0.75, 0.00, 0.10], dtype=np.float64),
            "lower_to_goal": np.array([0.65, 1.40, -2.10, -0.95, 0.00, 0.10], dtype=np.float64),
            "release": np.array([0.65, 1.40, -2.10, -0.95, 0.00, 1.50], dtype=np.float64),
            "retreat": np.array([0.35, 1.15, -1.70, -0.60, 0.00, 1.50], dtype=np.float64),
        }

    def _current_target(self) -> np.ndarray:
        return self._targets[self._phase_names[self._phase_idx]]

    def _joint_vector(self) -> np.ndarray:
        q = np.zeros(len(self.joint_names), dtype=np.float64)
        for i, name in enumerate(self.joint_names):
            q[i] = self.data.qpos[self._joint_qadr[name]]
        return q

    def _set_actuator_targets(self, target: np.ndarray) -> None:
        for i, name in enumerate(self.joint_names):
            self.data.ctrl[self._actuator_id[name]] = float(target[i])

    def _update_phase(self) -> None:
        q = self._joint_vector()
        target = self._current_target()

        joint_err = float(np.max(np.abs(q - target)))
        block_pos = self.data.xpos[self._body_block]
        goal_pos = self.data.site_xpos[self._site_goal]

        at_target = joint_err < 0.06
        near_goal = float(np.linalg.norm(block_pos - goal_pos)) < 0.05

        min_hold = 8
        if self._phase_names[self._phase_idx] in ("close", "release"):
            min_hold = 20

        if at_target:
            self._phase_ticks += 1
        else:
            self._phase_ticks = 0

        if self._phase_ticks >= min_hold:
            if self._phase_idx < len(self._phase_names) - 1:
                self._phase_idx += 1
            self._phase_ticks = 0

        if near_goal and self._phase_idx >= self._phase_names.index("release"):
            self._success_count += 1
        else:
            self._success_count = 0

    def _get_obs(self) -> np.ndarray:
        return np.concatenate(
            [
                self.data.qpos[: self.model.nq].copy(),
                self.data.qvel[: self.model.nv].copy(),
                self.data.sensordata[: self.model.nsensordata].copy(),
            ]
        ).astype(np.float32)

    def _reward(self) -> float:
        ee_pos = self.data.site_xpos[self._site_ee]
        block_pos = self.data.xpos[self._body_block]
        goal_pos = self.data.site_xpos[self._site_goal]

        d_ee_block = float(np.linalg.norm(ee_pos - block_pos))
        d_block_goal = float(np.linalg.norm(block_pos - goal_pos))

        lifted = 1.0 if block_pos[2] > 0.50 else 0.0
        success = 1.0 if self._success_count > 15 else 0.0

        reward = -0.6 * d_ee_block - 1.8 * d_block_goal + 0.7 * lifted + 8.0 * success
        return float(reward)

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)

        # Stable home pose before scripted sequence starts.
        home = {
            "shoulder_pan": 0.0,
            "shoulder_lift": 1.2,
            "elbow_flex": -1.8,
            "wrist_flex": -0.5,
            "wrist_roll": 0.0,
            "gripper_jaw": 1.5,
        }
        for name, val in home.items():
            self.data.qpos[self._joint_qadr[name]] = val
            self.data.ctrl[self._actuator_id[name]] = val

        mujoco.mj_forward(self.model, self.data)

        self._steps = 0
        self._phase_idx = 0
        self._phase_ticks = 0
        self._success_count = 0

        obs = self._get_obs()
        info = {
            "phase": self._phase_names[self._phase_idx],
            "scripted": True,
        }
        return obs, info

    def step(self, action):
        del action  # intentionally ignored: behavior is hardcoded.

        self._set_actuator_targets(self._current_target())

        for _ in range(self.frame_skip):
            mujoco.mj_step(self.model, self.data)

        self._update_phase()

        obs = self._get_obs()
        reward = self._reward()

        self._steps += 1
        success = self._success_count > 15

        terminated = bool(success)
        truncated = bool(self._steps >= self.max_steps)

        info = {
            "phase": self._phase_names[self._phase_idx],
            "scripted": True,
            "success": success,
            "steps": self._steps,
        }
        return obs, reward, terminated, truncated, info

    def render(self):
        if self.render_mode is None:
            return None

        self._renderer.update_scene(self.data)
        frame = self._renderer.render()

        if self.render_mode == "rgb_array":
            return frame

        if self.render_mode == "human":
            if self._cv2 is None:
                import cv2  # local import to avoid hard dependency for rgb_array mode

                self._cv2 = cv2
            bgr = frame[..., ::-1]
            self._cv2.imshow("HardcodedPickPlaceEnv", bgr)
            self._cv2.waitKey(1)
        return None

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None

        if self._cv2 is not None:
            self._cv2.destroyAllWindows()
            self._cv2 = None


def make_env(**kwargs) -> HardcodedPickPlaceEnv:
    """Convenience constructor."""
    return HardcodedPickPlaceEnv(**kwargs)
