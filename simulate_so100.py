"""
SO-100 Pick-and-Place Scene Launcher
=====================================
Run:  python launch_so100.py

In viewer:
  • Expand "Control"  → 6 joint sliders
  • Expand "Joint"    → live joint positions
  • Press F           → toggle collision geometry visible (group 3)
"""
import mujoco, mujoco.viewer, numpy as np

model = mujoco.MjModel.from_xml_path("sim/scene.xml")
data  = mujoco.MjData(model)

# Home pose — arm raised, reaching toward block
home = {
    "shoulder_pan":   0.0,   # centred
    "shoulder_lift":  1.2,   # arm bent forward
    "elbow_flex":    -1.8,   # elbow down toward table
    "wrist_flex":    -0.5,   # wrist level
    "wrist_roll":     0.0,
    "gripper_jaw":    1.5,   # open
}
for name, val in home.items():
    data.qpos[model.joint(name).qposadr] = val
    aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
    if aid >= 0:
        data.ctrl[aid] = val

mujoco.mj_forward(model, data)

# Print sensor map
print("=" * 58)
print("  Sensor map  (index into data.sensordata)")
print("=" * 58)
for i in range(model.nsensor):
    s   = model.sensor(i)
    adr = int(np.asarray(s.adr).reshape(-1)[0])
    dim = int(np.asarray(s.dim).reshape(-1)[0])
    end = adr + dim
    print(f"  [{adr:2d}:{end:2d}]  {s.name}")
print("=" * 58)

mujoco.viewer.launch(model, data)