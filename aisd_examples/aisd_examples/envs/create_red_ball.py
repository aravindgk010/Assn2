import random
import gymnasium as gym
from gymnasium import spaces
import numpy as np

import rclpy
from rclpy.node import Node
import sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge
import cv2


class RedBallEnv(gym.Env):
    metadata = {"render_modes": "rgb_array", "render_fps":4}

    def __init__(self, render_mode=None, size=5):
        rclpy.init(args=None)
        self.redball  = RedBall()

        self.step_count = 0

        self.observation_space = spaces.Dict({
            "position": spaces.Box(low=0, high=640, shape=(1,), dtype=np.int32)
        })

        self.action_space = spaces.Discrete(641)

    def _get_obs(self):
        return {"position": self.redball.redball_position if self.redball.redball_position is not None else 0}
    
    def _get_info(self):
        return {"position": self.redball.redball_position}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.step_count = 0
        observation = self._get_obs()

        return observation

    def step(self, action):
        self.redball.step(action)
        rclpy.spin_once(self.redball)
        self.step_count += 1

        observation = self._get_obs()
        info = self._get_info()

        reward = 0
        terminate = False


        return observation, reward, terminate, False, info

    def render(self):
        return
    
    def close(self):
        self.mqi.stop()
