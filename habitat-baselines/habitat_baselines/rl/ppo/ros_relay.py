#!/usr/bin/env python

import rospy
from sensor_msgs.msg import CompressedImage
from cv_bridge import CvBridge, CvBridgeError
import cv2
import numpy as np
import tf
import tf2_ros
from tf.transformations import euler_from_quaternion
import actionlib
from mbf_msgs.msg import MoveBaseAction, MoveBaseGoal, MoveBaseActionResult
from collections import OrderedDict
from typing import List

class ROSRelay:
    def __init__(self):
        # Initialize the node
        rospy.init_node('ros_relay', anonymous=True)

        # Initialize the CvBridge class
        self.bridge = CvBridge()

        # Initialize a variable to store the latest image
        self.latest_image = np.zeros((480,640,3), dtype=np.uint8)

        # Initialize a variable to store the latest depth image
        self.latest_depth = np.zeros((480,640,1), dtype=np.uint8)

        # Subscribe to the image topic
        self.image_sub = rospy.Subscriber("/camera/color/image_raw/compressed", CompressedImage, self.image_callback)
        self.depth_sub = rospy.Subscriber("/camera/depth/image_raw/compressed", CompressedImage, self.depth_callback)

        # Create an action client
        self.mbf_client = actionlib.SimpleActionClient('move_base_flex/move_base', MoveBaseAction)

        # Wait for the action server to start
        rospy.loginfo("Waiting for move_base_flex action server to start...")
        self.mbf_client.wait_for_server()
        rospy.loginfo("Connected to move_base_flex action server")


    def image_callback(self, msg):
        try:
            # Convert the compressed image data to a numpy array
            np_arr = np.frombuffer(msg.data, np.uint8)

            # Decode the numpy array to an OpenCV image
            self.latest_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        except CvBridgeError as e:
            rospy.logerr("CvBridge Error: {0}".format(e))

    def depth_callback(self, msg):
        try:
            # Convert the compressed image data to a numpy array
            np_arr = np.frombuffer(msg.data, np.uint8)

            # Decode the numpy array to an OpenCV image
            self.latest_depth = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        except CvBridgeError as e:
            rospy.logerr("CvBridge Error: {0}".format(e))

    def get_ros_observations(self) -> List[OrderedDict]:
        pose = self.get_robot_pose();

        obs = OrderedDict({
            'compass': np.array([pose[-1]], dtype=np.float32),
            'depth': self.get_latest_depth(),
            'gps': np.array(pose[:2], dtype=np.float32),
            'objectgoal': np.array([5]), #dummy
            'rgb': self.get_latest_image()
        })

        return [obs, obs, obs, obs]

    def get_latest_image(self) -> np.ndarray:
        # Return the latest image received (or None if no image has been received yet)
        self.latest_image = cv2.resize(self.latest_image, (640, 480))
        return self.latest_image

    def get_latest_depth(self) -> np.ndarray:
        # Return the latest image received (or None if no image has been received yet)
        self.latest_depth = cv2.resize(self.latest_depth, (640, 480)).reshape((480, 640, 1))
        return self.latest_depth

    def get_robot_pose(self) -> np.ndarray:
        listener = tf.TransformListener()
        pose = np.zeros(3)
        rate = rospy.Rate(10.0)
        while not rospy.is_shutdown():
            try:
                # Wait for the transform to become available
                listener.waitForTransform("map", "base_link", rospy.Time(0), rospy.Duration(1.0))

                # Lookup the transform
                (trans, rot) = listener.lookupTransform("map", "base_link", rospy.Time(0))

                pose[0] = trans[0]
                pose[1] = trans[1]
                roll, pitch, pose[2] = euler_from_quaternion(rot)

            except (tf2_ros.TransformException, tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException) as e:
                rospy.logwarn("Exception: %s", str(e))
                rate.sleep()

            print("robot pose is: ", pose)
            return pose

    def ros_step(self, action):
        # Create a new goal
        goal = MoveBaseGoal()
        goal.planner = "navfn"
        goal.controller = "eband"
        goal.target_pose.header.frame_id = "base_link"
        goal.target_pose.header.stamp = rospy.Time.now()

        if action == 0:
            print("stop")
            goal.target_pose.pose.position.x = 0.0
            goal.target_pose.pose.position.y = 0.0
            goal.target_pose.pose.orientation.x = 0.0
            goal.target_pose.pose.orientation.y = 0.0
            goal.target_pose.pose.orientation.z = 0.0
            goal.target_pose.pose.orientation.w = 1.0

        if action == 1:
            print("go forward")
            goal.target_pose.pose.position.x = 3.0
            goal.target_pose.pose.position.y = 0.0
            goal.target_pose.pose.orientation.x = 0.0
            goal.target_pose.pose.orientation.y = 0.0
            goal.target_pose.pose.orientation.z = 0.0
            goal.target_pose.pose.orientation.w = 1.0

        elif action == 2:
            print("turn left")
            goal.target_pose.pose.position.x = 0.0
            goal.target_pose.pose.position.y = 0.0
            goal.target_pose.pose.orientation.x = 0.0
            goal.target_pose.pose.orientation.y = 0.0
            goal.target_pose.pose.orientation.z = 0.7071
            goal.target_pose.pose.orientation.w = 0.7071

        elif action == 3:
            print("turn right")
            goal.target_pose.pose.position.x = 0.0
            goal.target_pose.pose.position.y = 0.0
            goal.target_pose.pose.orientation.x = 0.0
            goal.target_pose.pose.orientation.y = 0.0
            goal.target_pose.pose.orientation.z = -0.7071
            goal.target_pose.pose.orientation.w = 0.7071

        # Send the goal to the action server
        self.mbf_client.send_goal(goal)

        # Wait for the result with a timeout
        rospy.loginfo("Waiting for result with timeout...")
        finished_before_timeout = self.mbf_client.wait_for_result(rospy.Duration(20.0))

        if finished_before_timeout:
            # Action completed within the timeout
            result = self.mbf_client.get_result()
            rospy.loginfo("Action result: %s", result)
            return result
        else:
            # Action did not complete within the timeout
            rospy.logwarn("Action did not finish before the timeout.")
            self.mbf_client.cancel_goal()  # Cancel the goal
            return None

    def spin(self):
        # Keep the node running
        rospy.spin()

