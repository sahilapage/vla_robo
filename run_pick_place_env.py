"""
SO-100 Pick-and-Place — Gym-style Environment
══════════════════════════════════════════════
A hardcoded pick-and-place task:
  • Pick the red block from its fixed start position
  • Place it on the green goal zone

Install:
    pip install mujoco gymnasium numpy

Usage:
    python so100_env.py              # runs a scripted demo
    python so100_env.py --random     # runs random actions

API (gymnasium-compatible):
    env = SO100PickPlaceEnv(render_mode="human")
    obs, info = env.reset()
    obs, reward, terminated, truncated, info = env.step(action)
    env.close()
"""

import os
import time
import numpy as np
import mujoco
import mujoco.viewer
import gymnasium as gym
from gymnasium import spaces


# ─── Hardcoded task parameters ─────────────────────────────────────────────
BLOCK_START_POS   = np.array([0.0, -0.08, 0.485])   # world xyz
GOAL_POS          = np.array([0.15, -0.08, 0.462])  # centre of place zone
GOAL_RADIUS       = 0.04                             # success threshold (m)
LIFT_HEIGHT       = 0.55                             # z to lift block above table

# Joint names in order (matches actuator order)
JOINT_NAMES = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper_jaw",
]

# Home pose — arm raised, gripper open
HOME_QPOS = np.array([0.0, 1.2, -1.8, -0.5, 0.0, 1.5])

# Joint limits [low, high] — must match ctrlrange in so100.xml
JOINT_LIMITS = np.array([
    [-2.0,     2.0    ],   # shoulder_pan
    [ 0.0,     3.5    ],   # shoulder_lift
    [-3.14158, 0.0    ],   # elbow_flex
    [-2.5,     1.2    ],   # wrist_flex
    [-3.14158, 3.14158],   # wrist_roll
    [-0.2,     2.0    ],   # gripper_jaw
])

# Scene XML path (relative to this file)
SCENE_XML = os.path.join(os.path.dirname(__file__), "sim/scene.xml")


class SO100PickPlaceEnv(gym.Env):
    """
    Observation space (35-dim):
        [0:6]   joint positions
        [6:12]  joint velocities
        [12:15] end-effector position (world xyz)
        [15:18] block position (world xyz)
        [18:21] block-to-goal vector
        [21:22] gripper touch left/right (0 or 1)
        [22:24] touch sensors (2)
        [24:27] goal position (constant but included for convenience)
        [27:35] block quaternion

    Action space (6-dim, continuous):
        Target joint positions, clipped to joint limits.
        Actions are in [-1, 1] and scaled to joint ranges.

    Reward:
        Dense:
          +reach:   1 - tanh(5 * dist(ee, block))
          +grasp:   1.0 if both fingers touching block
          +lift:    1 - tanh(5 * |block_z - LIFT_HEIGHT|)  (when grasped)
          +place:   1 - tanh(5 * dist(block_xy, goal_xy))  (when lifted)
        Sparse:
          +10.0  success bonus (block within GOAL_RADIUS of goal)
        Penalty:
          -0.01  per step (time pressure)

    Termination:
        success: block centre within GOAL_RADIUS of goal AND block on table
        truncated: max_steps reached
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}

    def __init__(
        self,
        render_mode: str | None = None,
        max_steps: int = 1000,
        control_freq: int = 10,     # env steps per sim step group (10 Hz control)
        sim_steps_per_ctrl: int = 5, # mujoco steps per control step (timestep=0.002 → 10ms)
        camera_name: str = "side_cam",
        render_backend: str = "opencv",
    ):
        super().__init__()

        self.render_mode       = render_mode
        self.max_steps         = max_steps
        self.control_freq      = control_freq
        self.sim_steps_per_ctrl = sim_steps_per_ctrl
        self.camera_name       = camera_name
        self.render_backend    = render_backend

        # Load model
        self.model = mujoco.MjModel.from_xml_path(SCENE_XML)
        self.data  = mujoco.MjData(self.model)
        self._ik_data = mujoco.MjData(self.model)

        # Cache IDs once
        self._cache_ids()

        # Observation / action spaces
        obs_dim = 35
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        # Actions: normalised joint position targets in [-1, 1]
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(6,), dtype=np.float32
        )

        # Viewer
        self._viewer     = None
        self._renderer   = None
        self._cv2        = None
        self._step_count = 0

    # ── Internal helpers ──────────────────────────────────────────────

    def _cache_ids(self):
        """Cache all MuJoCo IDs we'll use every step."""
        m = self.model

        # Joint qpos addresses
        self._jnt_qadr = np.array([
            int(m.joint(name).qposadr[0]) for name in JOINT_NAMES
        ], dtype=int)

        # Joint dof (velocity) addresses  
        self._jnt_dadr = np.array([
            int(m.joint(name).dofadr[0]) for name in JOINT_NAMES
        ], dtype=int)

        # Actuator IDs
        self._act_ids = np.array([
            mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
            for name in JOINT_NAMES
        ], dtype=int)

        # Body IDs
        self._block_bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "block")

        # Site IDs
        self._ee_sid       = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "ee_site")
        self._grasp_sid    = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "grasp_center")
        self._block_sid    = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "block_centre")
        self._place_sid    = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "place_centre")
        self._tip_l_sid    = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "finger_tip_l")
        self._tip_r_sid    = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "finger_tip_r")

        # Sensor addresses
        self._sens_touch_l = int(m.sensor("touch_finger_l").adr[0])
        self._sens_touch_r = int(m.sensor("touch_finger_r").adr[0])

        # Free joint qpos address for block (7 values: xyz + quat)
        self._block_fj_qadr = int(m.joint("block_joint").qposadr[0])

    def _get_qpos(self) -> np.ndarray:
        return np.array([self.data.qpos[a] for a in self._jnt_qadr], dtype=np.float32)

    def _get_qvel(self) -> np.ndarray:
        return np.array([self.data.qvel[a] for a in self._jnt_dadr], dtype=np.float32)

    def _get_ee_pos(self) -> np.ndarray:
        return self.data.site_xpos[self._ee_sid].copy().astype(np.float32)

    def _get_block_pos(self) -> np.ndarray:
        return self.data.site_xpos[self._block_sid].copy().astype(np.float32)

    def _get_grasp_pos(self) -> np.ndarray:
        return self.data.site_xpos[self._grasp_sid].copy().astype(np.float32)

    def _get_block_quat(self) -> np.ndarray:
        # xquat of the block body
        return self.data.xquat[self._block_bid].copy().astype(np.float32)

    def _get_touch(self):
        tl = float(float(self.data.sensordata[self._sens_touch_l]) > 0.01)
        tr = float(float(self.data.sensordata[self._sens_touch_r]) > 0.01)
        return tl, tr

    def _action_to_ctrl(self, action: np.ndarray) -> np.ndarray:
        """Scale normalised [-1,1] action to actual joint-position targets."""
        lo  = JOINT_LIMITS[:, 0]
        hi  = JOINT_LIMITS[:, 1]
        mid = (hi + lo) / 2.0
        rng = (hi - lo) / 2.0
        return mid + action * rng

    def _set_ctrl(self, targets: np.ndarray):
        for i, aid in enumerate(self._act_ids):
            self.data.ctrl[aid] = np.clip(
                targets[i], JOINT_LIMITS[i, 0], JOINT_LIMITS[i, 1]
            )

    def _reset_block(self):
        """Teleport block back to start pose."""
        qadr = self._block_fj_qadr
        self.data.qpos[qadr:qadr+3]   = BLOCK_START_POS
        self.data.qpos[qadr+3:qadr+7] = [1, 0, 0, 0]   # identity quaternion
        self.data.qvel[int(self.model.joint("block_joint").dofadr[0]) :
                       int(self.model.joint("block_joint").dofadr[0]) + 6] = 0.0

    def _reset_arm(self):
        """Set arm to home pose."""
        for i, qadr in enumerate(self._jnt_qadr):
            self.data.qpos[qadr] = HOME_QPOS[i]
            self.data.qvel[self._jnt_dadr[i]] = 0.0
        # Seed ctrl so position actuators don't fight
        for i, aid in enumerate(self._act_ids):
            self.data.ctrl[aid] = HOME_QPOS[i]

    def solve_ik(
        self,
        ee_target: np.ndarray,
        q_seed: np.ndarray | None = None,
        site_id: int | None = None,
        n_restarts: int = 14,
        max_iters: int = 120,
        tol: float = 2e-3,
        damping: float = 2e-3,
        step_size: float = 0.55,
    ) -> np.ndarray:
        """Solve position-only IK for arm joints [0:5] using damped least squares."""
        lo = JOINT_LIMITS[:5, 0]
        hi = JOINT_LIMITS[:5, 1]
        arm_qadr = self._jnt_qadr[:5]
        arm_dadr = self._jnt_dadr[:5]
        solve_site = self._ee_sid if site_id is None else int(site_id)

        # Keep a full-state copy so site/world transforms are correct in ik_data.
        base_qpos = self.data.qpos.copy()
        self._ik_data.qpos[:] = base_qpos
        self._ik_data.qvel[:] = 0.0

        if q_seed is None:
            q_seed_full = self._get_qpos().astype(np.float64)
        else:
            q_seed_full = np.array(q_seed, dtype=np.float64).copy()

        rng = np.random.default_rng(0)
        seed_list = [q_seed_full[:5].copy(), HOME_QPOS[:5].copy(), self._get_qpos().astype(np.float64)[:5].copy()]
        for _ in range(n_restarts):
            seed_list.append(rng.uniform(lo, hi))

        jacp = np.zeros((3, self.model.nv), dtype=np.float64)
        jacr = np.zeros((3, self.model.nv), dtype=np.float64)

        best_q = seed_list[0].copy()
        best_err = 1e9

        for seed in seed_list:
            q_arm = np.clip(seed.copy(), lo, hi)

            for _ in range(max_iters):
                self._ik_data.qpos[:] = base_qpos
                for i, qadr in enumerate(arm_qadr):
                    self._ik_data.qpos[qadr] = q_arm[i]
                mujoco.mj_forward(self.model, self._ik_data)

                ee_pos = self._ik_data.site_xpos[solve_site]
                err_vec = ee_target - ee_pos
                err_norm = float(np.linalg.norm(err_vec))
                if err_norm < tol:
                    break

                mujoco.mj_jacSite(self.model, self._ik_data, jacp, jacr, solve_site)
                j = jacp[:, arm_dadr]

                h = j @ j.T + damping * np.eye(3)
                dq = j.T @ np.linalg.solve(h, err_vec)
                q_arm += step_size * dq
                q_arm = np.clip(q_arm, lo, hi)

            self._ik_data.qpos[:] = base_qpos
            for i, qadr in enumerate(arm_qadr):
                self._ik_data.qpos[qadr] = q_arm[i]
            mujoco.mj_forward(self.model, self._ik_data)
            final_err = float(np.linalg.norm(ee_target - self._ik_data.site_xpos[solve_site]))
            if final_err < best_err:
                best_err = final_err
                best_q = q_arm.copy()

        out = q_seed_full.copy()
        out[:5] = best_q
        return out

    # ── Gym API ───────────────────────────────────────────────────────

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        mujoco.mj_resetData(self.model, self.data)
        self._reset_arm()
        self._reset_block()
        mujoco.mj_forward(self.model, self.data)

        # Settle for a few steps so arm doesn't start flailing
        for _ in range(50):
            mujoco.mj_step(self.model, self.data)

        self._step_count = 0
        obs  = self._get_obs()
        info = self._get_info()
        return obs, info

    def step(self, action: np.ndarray):
        assert self.action_space.contains(action.astype(np.float32)), \
            f"Action out of bounds: {action}"

        targets = self._action_to_ctrl(action)
        self._set_ctrl(targets)

        # Run several sim steps per control step
        for _ in range(self.sim_steps_per_ctrl):
            mujoco.mj_step(self.model, self.data)

        self._step_count += 1

        obs        = self._get_obs()
        reward     = self._compute_reward()
        terminated = self._is_success()
        truncated  = self._step_count >= self.max_steps
        info       = self._get_info()

        if self.render_mode == "human":
            self.render()

        return obs, reward, terminated, truncated, info

    def _get_obs(self) -> np.ndarray:
        qpos       = self._get_qpos()          # 6
        qvel       = self._get_qvel()          # 6
        ee_pos     = self._get_ee_pos()        # 3
        block_pos  = self._get_block_pos()     # 3
        to_goal    = GOAL_POS.astype(np.float32) - block_pos   # 3
        tl, tr     = self._get_touch()         # 2
        goal       = GOAL_POS.astype(np.float32)               # 3
        block_quat = self._get_block_quat()    # 4  (padded to make 35 total)
        touch      = np.array([tl, tr], dtype=np.float32)      # 2 (already counted above)

        # Total: 6+6+3+3+3+3+2+4 = 30... let's add ee→block vector (3) + gripper width (1) + block_z(1)
        ee_to_block   = block_pos - ee_pos                      # 3
        gripper_width = float(self.data.qpos[int(self._jnt_qadr[5])])  # gripper_jaw angle (proxy for width)
        block_z       = float(block_pos[2])

        obs = np.concatenate([
            qpos,          # [0:6]   joint positions
            qvel,          # [6:12]  joint velocities
            ee_pos,        # [12:15] EE world pos
            block_pos,     # [15:18] block world pos
            to_goal,       # [18:21] block→goal vector
            touch,         # [21:23] finger touch
            goal,          # [23:26] goal pos (constant)
            block_quat,    # [26:30] block orientation
            ee_to_block,   # [30:33] EE→block vector
            [gripper_width],  # [33]
            [block_z],        # [34]
        ]).astype(np.float32)

        assert obs.shape == (35,), f"obs shape mismatch: {obs.shape}"
        return obs

    def _compute_reward(self) -> float:
        ee_pos    = self._get_ee_pos()
        block_pos = self._get_block_pos()
        tl, tr    = self._get_touch()
        grasped   = (tl > 0) and (tr > 0)

        # Distance-based sub-rewards
        dist_ee_block  = float(np.linalg.norm(ee_pos    - block_pos))
        dist_block_goal = float(np.linalg.norm(block_pos[:2] - GOAL_POS[:2]))  # XY only
        block_lifted   = float(block_pos[2]) > (BLOCK_START_POS[2] + 0.03)

        # --- Reward shaping ---
        # 1. Reach: drive EE toward block
        r_reach = 1.0 - np.tanh(5.0 * dist_ee_block)

        # 2. Grasp: reward touching with both fingers
        r_grasp = float(grasped)

        # 3. Lift: reward getting block above table (only if grasped)
        r_lift = (1.0 - np.tanh(5.0 * abs(float(block_pos[2]) - LIFT_HEIGHT))) * float(grasped)

        # 4. Place: reward moving block XY toward goal (only if lifted)
        r_place = (1.0 - np.tanh(5.0 * dist_block_goal)) * float(block_lifted)

        # 5. Success bonus
        r_success = 10.0 if self._is_success() else 0.0

        # 6. Time penalty
        r_time = -0.01

        total = (
            0.3 * r_reach  +
            0.5 * r_grasp  +
            1.0 * r_lift   +
            1.5 * r_place  +
            r_success      +
            r_time
        )
        return float(total)

    def _is_success(self) -> bool:
        block_pos = self._get_block_pos()
        dist_xy   = float(np.linalg.norm(block_pos[:2] - GOAL_POS[:2]))
        # Must be close to goal and close to tabletop height (not mid-air).
        on_table_height = 0.462 <= float(block_pos[2]) <= 0.535
        gripper_open = float(self.data.qpos[int(self._jnt_qadr[5])]) > 1.0
        return dist_xy < GOAL_RADIUS and on_table_height and gripper_open

    def _get_info(self) -> dict:
        block_pos = self._get_block_pos()
        ee_pos    = self._get_ee_pos()
        tl, tr    = self._get_touch()
        return {
            "block_pos":         block_pos.tolist(),
            "ee_pos":            ee_pos.tolist(),
            "dist_block_goal":   float(np.linalg.norm(block_pos[:2] - GOAL_POS[:2])),
            "dist_ee_block":     float(np.linalg.norm(ee_pos - block_pos)),
            "touch_l":           tl,
            "touch_r":           tr,
            "is_success":        self._is_success(),
            "step":              self._step_count,
        }

    # ── Rendering ─────────────────────────────────────────────────────

    def render(self):
        if self.render_mode == "human":
            if self.render_backend == "mujoco":
                if self._viewer is None:
                    self._viewer = mujoco.viewer.launch_passive(self.model, self.data)
                    self._viewer.cam.lookat[:] = [0.06, -0.10, 0.52]
                    self._viewer.cam.distance = 0.65
                    self._viewer.cam.elevation = -28
                    self._viewer.cam.azimuth = 150
                self._viewer.sync()
                return None

            if self._renderer is None:
                self._renderer = mujoco.Renderer(self.model, height=480, width=640)

            self._renderer.update_scene(self.data, camera=self.camera_name)
            frame = self._renderer.render()

            if self._cv2 is None:
                import cv2

                self._cv2 = cv2
            self._cv2.imshow("SO100PickPlaceEnv", frame[..., ::-1])
            self._cv2.waitKey(1)

        elif self.render_mode == "rgb_array":
            if self._renderer is None:
                self._renderer = mujoco.Renderer(self.model, height=480, width=640)
            self._renderer.update_scene(self.data, camera=self.camera_name)
            return self._renderer.render()

    def close(self):
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
        if self._cv2 is not None:
            self._cv2.destroyAllWindows()
            self._cv2 = None


# ─── Scripted hardcoded demo ───────────────────────────────────────────────

def normalise_action(targets: np.ndarray) -> np.ndarray:
    """Convert absolute joint targets → normalised [-1,1] action."""
    lo  = JOINT_LIMITS[:, 0]
    hi  = JOINT_LIMITS[:, 1]
    mid = (hi + lo) / 2.0
    rng = (hi - lo) / 2.0
    return np.clip((targets - mid) / rng, -1.0, 1.0)


def _run_joint_target_phase(
    env: SO100PickPlaceEnv,
    total_reward: float,
    name: str,
    q_target: np.ndarray,
    steps: int,
    render: bool,
):
    print(f"\n  Phase: {name}")
    action = normalise_action(q_target.astype(np.float32))

    terminated = False
    truncated = False
    info = env._get_info()
    for _ in range(steps):
        _, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        if render:
            time.sleep(0.01)
        if terminated or truncated:
            break

    print(
        f"    block@{[f'{v:.3f}' for v in info['block_pos']]}  "
        f"dist_goal={info['dist_block_goal']:.3f}  "
        f"touch=({info['touch_l']:.0f},{info['touch_r']:.0f})"
    )
    return total_reward, terminated, truncated, info


def run_scripted_demo(
    render: bool = False,
    render_backend: str = "opencv",
    camera_name: str = "side_cam",
):
    """Run a scripted pick-and-place demo."""
    env = SO100PickPlaceEnv(
        render_mode="human" if render else None,
        max_steps=2000,
        render_backend=render_backend,
        camera_name=camera_name,
    )
    obs, info = env.reset()

    print("=" * 60)
    print("  SO-100 Scripted Pick-and-Place Demo")
    print("=" * 60)

    total_reward = 0.0
    terminated = False
    truncated = False
    info = env._get_info()

    block = env._get_block_pos().astype(np.float64)
    q_home = HOME_QPOS.astype(np.float64).copy()
    q_home[5] = 1.75

    # Empirical offset between ee_site and grasp center for this gripper model.
    grasp_offset = np.array([-0.05, 0.00, -0.08], dtype=np.float64)

    q_pre = env.solve_ik(block + grasp_offset + np.array([0.0, 0.0, 0.08]), q_seed=q_home)
    q_pre[5] = 1.75
    q_descend = env.solve_ik(block + np.array([0.0, 0.0, 0.028]), q_seed=q_pre)
    q_descend = env.solve_ik(block + grasp_offset + np.array([0.0, 0.0, 0.015]), q_seed=q_pre)
    q_descend[5] = 1.75
    q_close = q_descend.copy()
    q_close[5] = -0.05

    initial_phases = [
        ("Home", q_home, 90),
        ("Pre-grasp above block", q_pre, 130),
        ("Descend to block", q_descend, 130),
        ("Close gripper", q_close, 150),
    ]

    for desc, q_target, steps in initial_phases:
        total_reward, terminated, truncated, info = _run_joint_target_phase(
            env, total_reward, desc, q_target, steps, render
        )
        if terminated or truncated:
            break

    if not (terminated or truncated):
        # Calibrate carry relation from actual grasp, then solve IK to place block at goal.
        ee_now = env._get_ee_pos().astype(np.float64)
        block_now = env._get_block_pos().astype(np.float64)
        carry_rel = block_now - ee_now

        ee_goal_center = GOAL_POS.astype(np.float64) - carry_rel
        ee_goal_center[:2] += np.array([0.02, 0.04], dtype=np.float64)

        q_lift = env.solve_ik(np.array([ee_goal_center[0], ee_goal_center[1], 0.66]), q_seed=q_close)
        q_lift[5] = -0.05
        q_goal_above = env.solve_ik(np.array([ee_goal_center[0], ee_goal_center[1], 0.58]), q_seed=q_lift)
        q_goal_above[5] = -0.05
        q_goal_place = env.solve_ik(np.array([ee_goal_center[0], ee_goal_center[1], 0.53]), q_seed=q_goal_above)
        q_goal_place[5] = -0.05
        q_release = q_goal_place.copy()
        q_release[5] = 1.75
        q_retreat = env.solve_ik(np.array([ee_goal_center[0], ee_goal_center[1], 0.64]), q_seed=q_release)
        q_retreat[5] = 1.75

        for desc, q_target, steps in [
            ("Lift", q_lift, 170),
            ("Move above goal", q_goal_above, 160),
            ("Lower to goal", q_goal_place, 130),
            ("Release", q_release, 120),
            ("Retreat", q_retreat, 120),
        ]:
            total_reward, terminated, truncated, info = _run_joint_target_phase(
                env, total_reward, desc, q_target, steps, render
            )
            if terminated:
                print(f"  ✓ SUCCESS! block at {info['block_pos']}")
                break
            if truncated:
                break

    print(f"\n  Total reward: {total_reward:.2f}")
    print(f"  Success: {info['is_success']}")
    env.close()


def run_random(
    render: bool = True,
    render_backend: str = "opencv",
    camera_name: str = "side_cam",
):
    """Run random policy to verify env works."""
    env = SO100PickPlaceEnv(
        render_mode="human" if render else None,
        render_backend=render_backend,
        camera_name=camera_name,
    )
    obs, _ = env.reset()
    total_r = 0.0
    for step in range(500):
        action = env.action_space.sample()
        obs, r, term, trunc, info = env.step(action)
        total_r += r
        if render:
            time.sleep(0.01)
        if term or trunc:
            break
    print(f"Random policy: steps={step+1}  reward={total_r:.2f}  success={info['is_success']}")
    env.close()


if __name__ == "__main__":
    import sys
    use_render = "--render" in sys.argv
    use_viewer = "--viewer" in sys.argv

    camera_name = "side_cam"
    if "--camera" in sys.argv:
        cam_i = sys.argv.index("--camera")
        if cam_i + 1 < len(sys.argv):
            camera_name = sys.argv[cam_i + 1]

    backend = "mujoco" if use_viewer else "opencv"
    if "--random" in sys.argv:
        run_random(render=use_render, render_backend=backend, camera_name=camera_name)
    else:
        run_scripted_demo(render=use_render, render_backend=backend, camera_name=camera_name)