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

        # Convert to HSV
        hsv_conv_img = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Full red range (wraps around HSV)
        lower_red_1 = (0, 100, 100)
        upper_red_1 = (10, 255, 255)
        lower_red_2 = (160, 100, 100)
        upper_red_2 = (180, 255, 255)

        mask1 = cv2.inRange(hsv_conv_img, lower_red_1, upper_red_1)
        mask2 = cv2.inRange(hsv_conv_img, lower_red_2, upper_red_2)
        red_mask = cv2.bitwise_or(mask1, mask2)

        # Smoothing & Morphological operations
        blurred_mask = cv2.GaussianBlur(red_mask, (9, 9), 3)
        eroded_mask = cv2.erode(blurred_mask, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
        dilated_mask = cv2.dilate(eroded_mask, cv2.getStructuringElement(cv2.MORPH_RECT, (8, 8)))

        # Circle detection
        detected_circles = cv2.HoughCircles(
            dilated_mask, 
            cv2.HOUGH_GRADIENT, 
            dp=1, 
            minDist=100, 
            param1=100, 
            param2=20, 
            minRadius=10, 
            maxRadius=100
        )

        self.redball_position = None  # Reset initially

        if detected_circles is not None:
            for circle in detected_circles[0, :]:
                x, y, r = int(circle[0]), int(circle[1]), int(circle[2])

                # Filter: near image center, reasonable size
                if r < 10 or r > 100:
                    continue
                if abs(x - 320) > 300:  # skip far edges
                    continue

                self.redball_position = x
                circled_frame = cv2.circle(frame, (x, y), r, (0, 255, 0), thickness=3)
                self.target_publisher.publish(self.br.cv2_to_imgmsg(circled_frame))
                break  # only use the first valid detection
        else:
            self.get_logger().info('No red ball detected')




class RedBallEnv(gym.Env):
    metadata = {"render_modes": "rgb_array", "render_fps":4}

    def __init__(self, render_mode=None, size=5):
        rclpy.init(args=None)
        self.redball  = RedBall()

        self.step_count = 0

        self.observation_space = spaces.Discrete(11) #previously 641

        self.action_space = spaces.Discrete(5) #previously 641

        self.rotation_direction = 1

    def _get_obs(self):
        return {"position": self.redball.redball_position if self.redball.redball_position is not None else 0}
    
    def _get_info(self):
        return {"position": self.redball.redball_position}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.step_count = 0
        observation = self._get_obs()

        return observation

    """ 
    def step(self, action):
        twist = Twist()

        if self.redball.redball_position is None:
            twist.angular.z = 0.5
        else:
            twist.angular.z = (action -320) / 320 *(np.pi /2) #move towards the ball
       
        rclpy.spin_once(self.redball)
        self.step_count += 1

        observation = self._get_obs()
        info = self._get_info()

        reward = -abs(observation["position"] - 320) / 320
        terminate = self.step_count == 100


        return observation, reward, terminate, False, info
    """ 

    def step(self, action):
        rclpy.spin_once(self.redball, timeout_sec=0.1)

        twist = Twist()

        if self.redball.redball_position is None:
            twist.angular.z = self.rotation_direction * 0.5

            # change direction periodically to simulate pendulum
            if self.step_count % 20 == 0:
                self.rotation_direction *= -1
        else:
            self.rotation_direction = 1  # reset when ball is found
            twist.angular.z = (action - 320) / 320 * (np.pi / 2)

        print(f"Twist Command: {twist.angular.z}")
        self.redball.twist_publisher.publish(twist)

        self.step_count += 1
        observation = self._get_obs()
        info = self._get_info()

        reward = -abs(observation["position"] - 320) / 320
        terminated = self.step_count == 100

        return observation, reward, terminated, False, info

    def render(self):
        return
    
    def close(self):
        self.redball.destroy_node()
        rclpy.shutdown()
