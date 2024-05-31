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


class MobilityNode(DTROS):
    """
    Computes an estimate of the Duckiebot pose using the wheel encoders.
    Args:
        node_name (:obj:`str`): a unique, descriptive name for the ROS node
    Configuration:

    Publisher:
        ~encoder_localization (:obj:`PoseStamped`): The computed position
    Subscribers:
        ~/left_wheel_encoder_node/tick (:obj:`WheelEncoderStamped`):
            encoder ticks
        ~/right_wheel_encoder_node/tick (:obj:`WheelEncoderStamped`):
            encoder ticks
    """

    right_tick_prev: Optional[int]
    left_tick_prev: Optional[int]
    delta_phi_left: float
    delta_phi_right: float

    def __init__(self, node_name):
        # Initialize the DTROS parent class
        super(MobilityNode, self).__init__(
            node_name=node_name, node_type=NodeType.CONTROL
        )
        self.log("Initializing...")
        # get the name of the robot
        self.veh = rospy.get_namespace().strip("/")

        # internal state
        # - odometry
        self.x_prev = 0.0
        self.y_prev = 0.0
        self.theta_prev = 0.0
        self.x_curr = 0.0
        self.y_curr = 0.0
        self.theta_curr = 0.0
        self.x_target = 0.0
        self.y_target = 0.0
        self.theta_target = 0.0

        # - PID controller
        self.kp_angular = 0
        self.ki_angular = 0
        self.kd_angular = 0

        self.error_x = 0.0
        self.error_y = 0.0
        self.error_theta = 0.0
        self.prev_error_x = 0.0
        self.prev_error_x_int = 0.0
        self.prev_error_y = 0.0
        self.prev_error_y_int = 0.0
        self.prev_error_theta = 0.0
        self.prev_error_theta_int = 0.0

        self.time_now: float = 0.0
        self.time_last_step: float = 0.0

        # fixed robot linear velocity - starts at zero so the activities start on command
        self.v = 0.0

        # initializing omega command to the robot
        self.omega = 0.0

        # Init the parameters
        self.resetParameters()

        # nominal R and L:
        self.log("Loading kinematics calibration...")
        self.read_params_from_calibration_file()  # must have a custom robot calibration

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

        # Coordinate subscriber:
        target_coordinate_topic = f"/{self.veh}/path_planner/coordinates"
        rospy.Subscriber(
            target_coordinate_topic, Pose2DStamped, self.triggerController, queue_size=1
        )

        # Odometry publisher
        self.db_estimated_pose = rospy.Publisher(
            f"/{self.veh}/encoder_localization",
            Odometry,
            queue_size=1,
            dt_topic_type=TopicType.LOCALIZATION,
        )

        # Command publisher
        car_cmd_topic = f"/{self.veh}/joy_mapper_node/car_cmd"
        self.pub_car_cmd = rospy.Publisher(
            car_cmd_topic, Twist2DStamped, queue_size=1, dt_topic_type=TopicType.CONTROL
        )

        # For encoders syncronization:
        self.RIGHT_RECEIVED = False
        self.LEFT_RECEIVED = False

        self.log("Initialized.")

    def resetParameters(self):
        # Add the node parameters to the parameters dictionary
        self.delta_phi_left = 0.0
        self.left_tick_prev = None

        self.delta_phi_right = 0.0
        self.right_tick_prev = None

        # Initializing the odometry
        self.x_prev = 0.0
        self.y_prev = 0.0
        self.theta_prev = 0.0

        self.x_curr = 0.0
        self.y_curr = 0.0
        self.theta_curr = 0.0

        # Initializing the PID controller parameters
        self.prev_e = 0.0  # previous tracking error, starts at 0
        self.prev_int = 0.0  # previous tracking error integral, starts at 0
        self.time_now: float = 0.0
        self.time_last_step: float = 0.0
        self.kp_angular: float = 0.0
        self.ki_angular: float = 0.0
        self.kd_angular: float = 0.0

        # fixed robot linear velocity - starts at zero so the activities start on command
        self.velocity = 0.0
        self.y_ref = 0.0
        self.v_ref = 0.0

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

        ticks = encoder_msg.data - self.left_tick_prev
        dphi = ticks / encoder_msg.resolution
        self.delta_phi_left += dphi

        # update time
        self.time_now = max(self.time_now, encoder_msg.header.stamp.to_sec())

        # compute the new pose
        self.RIGHT_RECEIVED = True
        self.poseEstimator()

    def poseEstimator(self):
        """
        Publish the pose of the Duckiebot given by the kinematic model
            using the encoders.
        Publish:
            ~/encoder_localization (:obj:`PoseStamped`): Duckiebot pose.
        """

        left_wheel_distance = self.delta_phi_left * self.R
        right_wheel_distance = self.delta_phi_right * self.R
        distance = (right_wheel_distance + left_wheel_distance) / 2
        delta_theta = (right_wheel_distance - left_wheel_distance) / self.baseline
        # These are random values, replace with your own
        self.x_curr = self.x_prev + distance * np.cos(self.theta_prev)
        self.y_curr = self.y_prev + distance * np.sin(self.theta_prev)
        self.theta_curr = self.theta_prev + delta_theta

        self.theta_curr = self.angle_clamp(
            self.theta_curr
        )  # angle always between 0,2pi

        # Current estimate becomes previous estimate at next iteration
        self.x_prev = self.x_curr
        self.y_prev = self.y_curr
        self.theta_prev = self.theta_curr

        # Calculate new odometry only when new data from encoders arrives
        self.delta_phi_left = self.delta_phi_right = 0

        # Current estimate becomes previous estimate at next iteration
        self.x_prev = self.x_curr
        self.y_prev = self.y_curr
        self.theta_prev = self.theta_curr

        # Creating message to plot pose in RVIZ
        odom = Odometry()
        odom.header.frame_id = "map"
        odom.header.stamp = rospy.Time.now()

        odom.pose.pose.position.x = self.x_curr  # x position - estimate
        odom.pose.pose.position.y = self.y_curr  # y position - estimate
        odom.pose.pose.position.z = 0  # z position - no flying allowed in Duckietown

        # these are quaternions - stuff for a different course!
        odom.pose.pose.orientation.x = 0
        odom.pose.pose.orientation.y = 0
        odom.pose.pose.orientation.z = np.sin(self.theta_curr / 2)
        odom.pose.pose.orientation.w = np.cos(self.theta_curr / 2)

        self.db_estimated_pose.publish(odom)
        self.Controller()

    def triggerController(self, coordinate_msg):
        self.x_target = coordinate_msg.x
        self.y_target = coordinate_msg.y
        self.theta_target = coordinate_msg.theta

        self.Controller()

    def Controller(self):
        """
        Calculate theta and perform the control actions given by the PID
        """

        delta_time = self.time_now - self.time_last_step
        self.time_last_step = self.time_now

        self.error_x = self.x_target - self.x_curr
        self.prev_error_x_int = self.error_x * delta_time + self.prev_error_x_int
        error_x_der = (self.error_x - self.prev_error_x) / delta_time

        self.error_y = self.y_target - self.y_curr
        self.prev_error_y_int = self.error_y * delta_time + self.prev_error_y_int
        error_y_der = (self.error_y - self.prev_error_y) / delta_time

        self.error_theta = self.theta_target - self.theta_curr
        self.prev_error_theta_int = (
            self.error_theta * delta_time + self.prev_error_theta_int
        )
        error_theta_der = (self.error_theta - self.prev_error_theta) / delta_time

        if abs(self.error_x) <= 0.2 or abs(self.error_y) <= 0.2:
            # Already in place, just turn around
            omega = (
                self.kp_angular * self.error_theta
                + self.ki_angular * self.prev_error_theta_int
                + self.kd_angular * error_theta_der
            )
            velocity = 0.2
        else:
            # TO DO: Have orientation and change omega to follow it
            omega = (
                self.kp_angular * self.error
                + self.ki_angular * self.error_int
                + self.kd_angular * self.error_der
            )
            velocity = (
                self.kp_linear * self.error
                + self.ki_linear * self.error_int
                + self.kd_linear * self.error_der
            )

        self.publishCmd(velocity, omega)

    def publishCmd(self, v, omega):
        """
        Publishes a car command message.

        Args:
            v       (:obj:`double`): linear velocity
            omega   (:obj:`double`): angular velocity
        """

        car_control_msg = Twist2DStamped()
        car_control_msg.header.stamp = rospy.Time.from_sec(self.time_now)

        car_control_msg.v = v
        car_control_msg.omega = omega

        self.pub_car_cmd.publish(car_control_msg)

    def onShutdown(self):
        super(MobilityNode, self).on_shutdown()

    def read_params_from_calibration_file(self):
        """
        Reads the saved parameters from `/data/config/calibrations/kinematics/DUCKIEBOTNAME.yaml`
        or uses the default values if the file doesn't exist. Adjsuts the ROS paramaters for the
        node with the new values.
        """
        # Check file existence
        file_path = (
            "/code/catkin_ws/src/mobility/assets/calibrations/kinematics/default.yaml"
        )
        if not os.path.isfile(file_path):
            self.logfatal("Kinematics calibration %s not found!" % file_path)
            rospy.signal_shutdown("")
        else:
            self.readFile(file_path)

    def readFile(self, fname):
        with open(fname, "r") as in_file:
            try:
                yaml_dict = yaml.load(in_file, Loader=yaml.FullLoader)
                self.log(yaml_dict)
                self.R = yaml_dict["radius"]
                self.baseline = yaml_dict["baseline"]
                self.kd_angular = yaml_dict["kd_angular"]
                self.ki_angular = yaml_dict["ki_angular"]
                self.kp_angular = yaml_dict["kp_angular"]
                self.kd_linear = yaml_dict["kd_linear"]
                self.ki_linear = yaml_dict["ki_linear"]
                self.kp_linear = yaml_dict["kp_linear"]
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
        elif theta < -2 * np.pi:
            return theta + 2 * np.pi
        else:
            return theta


if __name__ == "__main__":
    # Initialize the node
    encoder_pose_node = MobilityNode(node_name="mobility_node")
    # Keep it spinning
    rospy.spin()
