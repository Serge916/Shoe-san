#!/usr/bin/env python3
import os
import time
from typing import Optional, Tuple

import numpy as np
import rospy
import yaml
from duckietown.dtros import DTROS, NodeType, TopicType
from duckietown_msgs.msg import Twist2DStamped, WheelEncoderStamped, Pose2DStamped
from nav_msgs.msg import Odometry
from std_msgs.msg import String, Bool

from constants import *


class OdometryNode(DTROS):
    """
    Computes an estimate of the Duckiebot pose using a combination of wheel encoders and April tags.
    Args:
        node_name (:obj:`str`): a unique, descriptive name for the ROS node
    Configuration:

    Publisher:
        ~robot_localization (:obj:`PoseStamped`): The computed position
    Subscribers:
        ~/encoder_localization (:obj:`WheelEncoderStamped`):
            encoder based odometry
        ~/april_tags/obtained_data (:obj:`WheelEncoderStamped`):
            distance to the tag, tag id and angle of view
    """

    def __init__(self, node_name):
        # Initialize the DTROS parent class
        super(OdometryNode, self).__init__(
            node_name=node_name, node_type=NodeType.LOCALIZATION
        )
        self.log("Initializing...")
        # get the name of the robot
        self.veh = os.environ["VEHICLE_NAME"]

        # internal state
        # - encoders
        self.delta_phi_left = 0.0
        self.left_tick_prev = None

        self.delta_phi_right = 0.0
        self.right_tick_prev = None

        self.time_now: float = 0.0
        self.time_last_step: float = 0.0

        self.LEFT_RECEIVED = False
        self.RIGHT_RECEIVED = False

        # Init the parameters
        self.resetParameters()

        # In case we want to load from a file:
        # self.log("Loading odometry calibration...")
        # self.read_params_from_calibration_file()  # must have a custom robot calibration

        # Defining subscribers:
        # Wheel encoder subscriber:
        left_encoder_topic = f"/{self.veh}/left_wheel_encoder_node/tick"
        rospy.Subscriber(
            left_encoder_topic, WheelEncoderStamped, self.cbLeftEncoder, queue_size=1
        )

        # Wheel encoder subscriber:
        right_encoder_topic = f"/{self.veh}/right_wheel_encoder_node/tick"
        rospy.Subscriber(
            right_encoder_topic, WheelEncoderStamped, self.cbRightEncoder, queue_size=1
        )

        # April tag subscriber:
        left_encoder_topic = f"/{self.veh}/april_tags/obtained_data"
        rospy.Subscriber(
            left_encoder_topic, Pose2DStamped, self.cbAprilTagReading, queue_size=1
        )
        # Time-of-flight subscriber:
        left_encoder_topic = f"/{self.veh}/ToF/reading"
        rospy.Subscriber(
            left_encoder_topic, Pose2DStamped, self.cbToFReading, queue_size=1
        )

        # Odometry publisher
        self.db_estimated_pose = rospy.Publisher(
            f"/{self.veh}/robot_odometry",
            Odometry,
            queue_size=1,
            dt_topic_type=TopicType.LOCALIZATION,
        )

        self.log("Initialized.")

    def cbLeftEncoder(self, encoder_msg):
        """
        Wheel encoder callback
        Args:
            encoder_msg (:obj:`WheelEncoderStamped`) encoder ROS message.
        """
        # initializing ticks to stored absolute value
        if self.left_tick_prev is None:
            self.left_tick_prev = encoder_msg.data
            return

        ticks = encoder_msg.data - self.left_tick_prev
        dphi = ticks / encoder_msg.resolution
        self.delta_phi_left += dphi
        self.left_tick_prev += ticks

        # update time
        self.time_now = max(self.time_now, encoder_msg.header.stamp.to_sec())

        # compute the new pose
        self.LEFT_RECEIVED = True
        self.poseEstimator()

    def cbRightEncoder(self, encoder_msg):
        """
        Wheel encoder callback, the rotation of the wheel.
        Args:
            encoder_msg (:obj:`WheelEncoderStamped`) encoder ROS message.
        """

        if self.right_tick_prev is None:
            self.right_tick_prev = encoder_msg.data
            return

        ticks = encoder_msg.data - self.right_tick_prev
        dphi = ticks / encoder_msg.resolution
        self.delta_phi_right += dphi
        self.right_tick_prev += ticks

        # update time
        self.time_now = max(self.time_now, encoder_msg.header.stamp.to_sec())

        # compute the new pose
        self.RIGHT_RECEIVED = True
        self.poseEstimator()

    def resetParameters(self):
        # Initialize the Kalman filter
        # Initial state [x, y, theta]
        self.estimate = np.array([0, 0, 0])
        # Initial covariance matrix
        self.P = np.zeros((3, 3))
        # Process noise covariance matrix
        self.Q = Q
        # Measurement noise covariance
        self.R = R
        # Measurement prediction
        self.H = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

    def poseEstimator(self):
        """
        Publish the pose of the Duckiebot given by the aggregated odometry.
        Publish:
            ~/robot_localization (:obj:`PoseStamped`): Duckiebot pose.
        """
        if not self.LEFT_RECEIVED or not self.RIGHT_RECEIVED:
            return

        if self.delta_phi_left == 0 and self.delta_phi_right == 0:
            return

        self.LEFT_RECEIVED = False
        self.RIGHT_RECEIVED = False

        left_wheel_distance = self.delta_phi_left * (2 * np.pi * RADIUS)
        right_wheel_distance = self.delta_phi_right * (2 * np.pi * RADIUS)
        delta_distance = (right_wheel_distance + left_wheel_distance) / 2
        delta_theta = (right_wheel_distance - left_wheel_distance) / BASELINE
        delta_theta = self.angle_clamp(delta_theta)

        # Calculate new odometry only when new data from encoders arrives
        self.delta_phi_left = self.delta_phi_right = 0

        # Prediction step:
        # Publisher reads estimated position. Difference to get increment
        u = np.array(
            [
                delta_distance * np.cos(delta_theta),
                delta_distance * np.sin(delta_theta),
                delta_theta,
            ]
        )

        self.F = np.eye(3)
        # Control input model
        self.B = np.eye(3)
        # Predict the state
        self.estimate = self.F @ self.estimate + self.B @ u
        self.estimate[2] = self.angle_clamp(self.estimate[2])
        # Predict the error covariance
        self.P = self.F @ self.P @ np.transpose(self.F) + self.Q
        self.publishOdometry()

    def cbAprilTagReading(self, msg):
        """
        Gives a reliable estimation of the robot position and orientation.
        Args:
            phi: Angle measured from the direction of looking straight into the tag and the orientation of the boot
            alpha: Angle measured from the perspective of the tag. Direction to the robot
            distance: Distance from robot to tag
            id: Tag unique identifier
        """
        tag_orientation = TAG_ORIENTATIONS.get(id)
        theta = msg.alpha - np.pi + msg.phi + tag_orientation
        x = msg.distance * np.cos(theta)
        y = msg.distance * np.sin(theta)

        z = np.array([x, y, theta])
        # Measurement model
        y = z - self.H @ self.estimate
        # Measurement covariance
        S = self.H @ self.P @ np.transpose(self.H) + self.R
        # Kalman gain
        K = self.P @ np.transpose(self.H) @ np.linalg.inv(S)
        # Update the state
        self.estimate = self.estimate + K @ y
        # Update the error covariance
        self.P = self.P - K @ self.H @ self.P

        self.publishOdometry()

    def publishOdometry(self):
        # Creating message to plot pose in RVIZ
        odom = Odometry()
        odom.header.frame_id = "map"
        odom.header.stamp = rospy.Time.now()

        odom.pose.pose.position.x = self.estimate[0]  # x position - estimate
        odom.pose.pose.position.y = self.estimate[1]  # y position - estimate
        odom.pose.pose.position.z = 0  # z position - no flying allowed in Duckietown

        # these are quaternions - stuff for a different course!
        odom.pose.pose.orientation.x = 0
        odom.pose.pose.orientation.y = 0
        odom.pose.pose.orientation.z = np.sin(self.estimate[2] / 2)
        odom.pose.pose.orientation.w = np.cos(self.estimate[2] / 2)

        padded = np.pad(
            self.P, pad_width=((0, 3), (0, 3)), mode="constant", constant_values=0
        )
        odom.pose.covariance = padded.flatten().tolist()

        self.log(
            f"Robot position is estimated to be: {self.estimate}. Covariance is {odom.pose.covariance[0]} of type {type(odom.pose.covariance[0])}"
            # f"Robot position is estimated to be: x:{self.estimate[0]}, y:{self.estimate[1]}, theta:{self.estimate[2]}"
        )

        self.db_estimated_pose.publish(odom)

    def cbToFReading(self, msg):
        pass

    def onShutdown(self):
        super(OdometryNode, self).on_shutdown()

    def read_params_from_calibration_file(self):
        """
        Reads the saved parameters from `/data/config/calibrations/kinematics/DUCKIEBOTNAME.yaml`
        or uses the default values if the file doesn't exist. Adjsuts the ROS paramaters for the
        node with the new values.
        """
        # Check file existence
        file_path = (
            "/code/catkin_ws/src/mobility/assets/calibrations/odometry/default.yaml"
        )

        if not os.path.isfile(file_path):
            self.logfatal("Odometry calibration %s not found!" % file_path)
            rospy.signal_shutdown("")
        else:
            self.readFile(file_path)

    def readFile(self, fname):
        with open(fname, "r") as in_file:
            try:
                yaml_dict = yaml.load(in_file, Loader=yaml.FullLoader)
                self.log(yaml_dict)
                self.R = yaml_dict["radius"]

            except yaml.YAMLError as exc:
                self.logfatal(
                    "YAML syntax error. File: %s fname. Exc: %s" % (fname, exc)
                )
                rospy.signal_shutdown("")
                return

    @staticmethod
    def angle_clamp(theta):
        while theta > np.pi:
            theta -= 2 * np.pi
        while theta < -np.pi:
            theta += 2 * np.pi
        return theta

    # def angle_clamp(theta):
    #     if theta > 2 * np.pi:
    #         return theta - 2 * np.pi
    #     elif theta < 0:
    #         return theta + 2 * np.pi
    #     else:
    #         return theta


if __name__ == "__main__":
    # Initialize the node
    encoder_pose_node = OdometryNode(node_name="odometry_node")
    # Keep it spinning
    rospy.spin()
