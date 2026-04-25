import mujoco
import numpy as np
import cv2

class SimpleEnv:
    def __init__(self, xml_path):
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)

        self.renderer = mujoco.Renderer(self.model)

    def reset(self):
        mujoco.mj_resetData(self.model, self.data)
        return self.get_obs()

    def step(self, action):
        # apply action (joint control)
        self.data.ctrl[:] = action

        mujoco.mj_step(self.model, self.data)

        obs = self.get_obs()
        reward = 0
        done = False

        return obs, reward, done

    def get_obs(self):
        self.renderer.update_scene(self.data)
        image = self.renderer.render()

        return image

    def render(self):
        img = self.get_obs()
        cv2.imshow("sim", img)
        cv2.waitKey(1)