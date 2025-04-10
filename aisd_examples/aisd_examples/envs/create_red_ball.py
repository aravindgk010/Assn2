import random
import gymnasium as gym #type: ignore
from gymnasium import spaces #type: ignore
import numpy as np

import rclpy 
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge
import cv2

class RedBall(Node):
    """
    A Node to analyse red balls in images and publish the results
    """
    def __init__(self):
        super().__init__('redball')
        self.subscription = self.create_subscription(
            Image,
            'custom_ns/camera1/image_raw',
            self.listener_callback,
            10)
        self.subscription # prevent unused variable warning
        
        self.redball_position = None

        # A converter between ROS and OpenCV images
        self.br = CvBridge()
        self.target_publisher = self.create_publisher(Image, 'target_redball', 10)
        self.twist_publisher = self.create_publisher(Twist, 'cmd_vel', 10)

    def listener_callback(self, msg):
        frame = self.br.imgmsg_to_cv2(msg)

        # convert image to BGR format (red ball becomes blue)
        hsv_conv_img = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        bright_red_lower_bounds = (110, 100, 100)
        bright_red_upper_bounds = (130, 255, 255)
        bright_red_mask = cv2.inRange(hsv_conv_img, bright_red_lower_bounds, bright_red_upper_bounds)

        blurred_mask = cv2.GaussianBlur(bright_red_mask,(9,9),3,3)
    
        # some morphological operations (closing) to remove small blobs
        erode_element = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        dilate_element = cv2.getStructuringElement(cv2.MORPH_RECT, (8, 8))
        eroded_mask = cv2.erode(blurred_mask,erode_element)
        dilated_mask = cv2.dilate(eroded_mask,dilate_element)

        # on the color-masked, blurred and morphed image I apply the cv2.HoughCircles-method to detect circle-shaped objects
        detected_circles = cv2.HoughCircles(dilated_mask, cv2.HOUGH_GRADIENT, 1, 150, param1=100, param2=20, minRadius=2, maxRadius=2000)
        the_circle = None
        """ 
        if detected_circles is not None:
            for circle in detected_circles[0, :]:
                circled_orig = cv2.circle(frame, (int(circle[0]), int(circle[1])), int(circle[2]), (0,255,0),thickness=3)
                the_circle = (int(circle[0]), int(circle[1]))
            self.target_publisher.publish(self.br.cv2_to_imgmsg(circled_orig))
        else:
            self.get_logger().info('no ball detected')
        """ 
        if detected_circles is not None:
            for circle in detected_circles[0, :]:
                the_circle = (int(circle[0]), int(circle[1]))
                self.redball_position = the_circle[0]  # horizontal position for Q-learning
                circled_orig = cv2.circle(frame, (int(circle[0]), int(circle[1])), int(circle[2]), (0,255,0), thickness=3)
            self.target_publisher.publish(self.br.cv2_to_imgmsg(circled_orig))
        else:
            self.redball_position = None  # Important to reset when nothing is found
            self.get_logger().info('no ball detected')


class RedBallEnv(gym.Env):
    metadata = {"render_modes": "rgb_array", "render_fps":4}

    def __init__(self, render_mode=None, size=5):
        rclpy.init(args=None)
        self.redball  = RedBall()

        self.step_count = 0

        #self.observation_space = spaces.Discrete(4) #previously 641
        """ for PPO & DQN"""
        self.observation_space = spaces.Dict({
            "position": spaces.Discrete(4)
        })


        self.action_space = spaces.Discrete(3) #previously 641

        self.rotation_direction = 1

    """ edited """
    def _get_obs(self):
        pos = self.redball.redball_position
        if pos is None:
            return {"position": 0}  # Treat as 'left-most'
        
        # Assuming image width is 640 pixels
        if pos < 160:
            return {"position": 0}  # Left
        elif pos < 320:
            return {"position": 1}  # Center-left
        elif pos < 480:
            return {"position": 2}  # Center-right
        else:
            return {"position": 3}  # Right
    
    def _get_info(self):
        return {"position": self.redball.redball_position}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.step_count = 0
        observation = self._get_obs()

        return observation

    """ edited """
    def step(self, action):
        twist = Twist()

        if action == 0:
            twist.angular.z = 0.5  # rotate left
        elif action == 1:
            twist.angular.z = 0.0  # stop
        elif action == 2:
            twist.angular.z = -0.5  # rotate right

        self.redball.twist_publisher.publish(twist)

        rclpy.spin_once(self.redball)

        self.step_count += 1
        observation = self._get_obs()
        info = self._get_info()

        # Reward based on how close the ball is to the center (position == 1 or 2 is ideal)
       
        #reward = 1.0 if observation["position"] in [1, 2] else -1.0
        if observation["position"] == 1 or observation["position"] == 2:
            reward = 1.0  # favorable
        elif observation["position"] == 0 or observation["position"] == 3:
            reward = -0.5  # less favorable
        else:
            reward = -1.0  # no detection (if handled this way)

        terminated = self.step_count >= 100

        return observation, reward, terminated, False, info


    def render(self):
        return
    
    def close(self):
        self.redball.destroy_node()
        rclpy.shutdown()
