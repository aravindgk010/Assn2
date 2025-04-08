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
        if detected_circles is not None:
            for circle in detected_circles[0, :]:
                circled_orig = cv2.circle(frame, (int(circle[0]), int(circle[1])), int(circle[2]), (0,255,0),thickness=3)
                the_circle = (int(circle[0]), int(circle[1]))
            self.target_publisher.publish(self.br.cv2_to_imgmsg(circled_orig))
        else:
            self.get_logger().info('no ball detected')



class RedBallEnv(gym.Env):
    metadata = {"render_modes": "rgb_array", "render_fps":4}

    def __init__(self, render_mode=None, size=5):
        rclpy.init(args=None)
        self.redball  = RedBall()

        self.step_count = 0

        self.observation_space = spaces.Discrete(641)

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
        twist = Twist()

        if self.redball.redball_position is None:
            twist.angular.z = 0.5
        else:
            twist.angular.z = (action -320) / 320 *(np.pi /2) #move towards the ball

        self.redball.twist_publisher.publish(twist)

        rclpy.spin_once(self.redball)
        self.step_count += 1

        observation = self._get_obs()
        info = self._get_info()

        reward = -abs(observation["position"] - 320) / 320
        terminate = self.step_count == 100


        return observation, reward, terminate, False, info

    def render(self):
        return
    
    def close(self):
        self.redball.destroy_node()
        rclpy.shutdown()
