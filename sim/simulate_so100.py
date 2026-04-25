"""
SO-100 Arm — MuJoCo Viewer Launcher
=====================================
Just opens the MuJoCo viewer. Use the built-in panels to control the arm:

  • Expand "Control"  → sliders set position targets for each joint
  • Expand "Joint"    → shows live joint positions (read-only)

No keyboard overrides — MuJoCo's native UI handles everything.

Usage:
    python launch_so100.py
"""

import mujoco
import mujoco.viewer

MODEL_PATH = "so100.xml"

model = mujoco.MjModel.from_xml_path(MODEL_PATH)
data  = mujoco.MjData(model)

# Set a sensible home pose so the arm doesn't collapse under gravity
import numpy as np

home = {
    "shoulder_pan":  0.0,
    "shoulder_lift": 1.57,
    "elbow_flex":   -1.57,
    "wrist_flex":    0.0,
    "wrist_roll":    0.0,
    "gripper_jaw":   0.5,
}

for name, val in home.items():
    jid = model.joint(name).qposadr
    data.qpos[jid] = val
    # also seed the ctrl so the position actuator targets the same angle
    aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
    if aid >= 0:
        data.ctrl[aid] = val

mujoco.mj_forward(model, data)

print("Launching MuJoCo viewer...")
print("  → Expand 'Control' panel to move joints with sliders")
print("  → Expand 'Joint'   panel to read joint positions")

# launch() is blocking and gives you the full interactive viewer
mujoco.viewer.launch(model, data)