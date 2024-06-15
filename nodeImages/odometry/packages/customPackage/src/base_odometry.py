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

        # Init the parameters
        self.resetParameters()

        # In case we want to load from a file:
        # self.log("Loading odometry calibration...")
        # self.read_params_from_calibration_file()  # must have a custom robot calibration

        # Defining subscribers:

        # Wheel encoder subscriber:
        left_encoder_topic = f"/{self.veh}/encoder_odometry"
        rospy.Subscriber(
            left_encoder_topic, Pose2DStamped, self.cbEncoderReading, queue_size=1
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

    def resetParameters(self):
        # Initialize the Kalman filter
        # Initial state [x, y, theta]
        self.estimate = np.array([0, 0, 0])
        # Initial covariance matrix
        self.P = np.eye(3)
        # Process noise covariance matrix
        self.Q = np.diag([0.1, 0.1, 0.01])
        # Measurement noise covariance
        self.R = np.diag([0.5, 0.5])
        # Measurement prediction
        self.H = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

    def cbEncoderReading(self, msg):
        """
        Publish the pose of the Duckiebot given by the aggregated odometry.
        Publish:
            ~/robot_localization (:obj:`PoseStamped`): Duckiebot pose.
        """
        # Prediction step:
        # Publisher reads estimated position. Difference to get increment
        u = np.array([msg.x, msg.y, msg.theta]) - self.estimate
        # State transition model
        self.F = np.eye(3)
        # Control input model
        self.B = np.eye(3)
        # Predict the state
        self.estimate = self.F @ self.estimate + self.B @ u
        # Predict the error covariance
        self.P = self.F @ self.P @ np.transpose(self.F) + self.Q
        self.publishOdometry()

    def cbAprilTagReading(self, msg):
        distanceToTag = msg.distance
        angleOfView = msg.angle
        tagOrientation = self.tagPlacement(msg.tagId)
        # If angleOfView belongs in (0,180) and tagOrientation in (0,360)
        x = np.cos(angleOfView) * distanceToTag
        y = np.sin(angleOfView) * distanceToTag
        theta = self.angle_clamp(angleOfView - np.pi / 2 + tagOrientation)

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

        self.log(
            f"Robot position is estimated to be: x:{self.estimate[0]}, y:{self.estimate[1]}, theta:{self.estimate[2]}"
        )

        self.db_estimated_pose.publish(odom)

    def cbToFReading():
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
        if theta > 2 * np.pi:
            return theta - 2 * np.pi
        elif theta < 0:
            return theta + 2 * np.pi
        else:
            return theta


if __name__ == "__main__":
    # Initialize the node
    encoder_pose_node = OdometryNode(node_name="odometry_node")
    # Keep it spinning
    rospy.spin()
