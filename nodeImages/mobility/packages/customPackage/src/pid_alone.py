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
        self.veh = os.environ["VEHICLE_NAME"]

        # internal state
        # - Past values
        self.orientation_curr = 0.0
        self.orientation_prev = 0.0
        self.distance_curr = 0.0
        self.distance_prev = 0.0
        self.prev_error_orientation_int = 0.0
        self.prev_error_orientation = 0.0
        self.prev_error_distance_int = 0.0
        self.prev_error_distance = 0.0

        self.x_curr = 0.0
        self.y_curr = 0.0
        self.x_prev = 0.0
        self.y_prev = 0.0
        self.theta_curr = 0.0
        self.theta_prev = 0.0
        # - commanded
        self.x_target = 0.0
        self.y_target = 0.0
        self.theta_target = 0.0

        # - PID controller
        self.kp_angular = 0
        self.ki_angular = 0
        self.kd_angular = 0
        self.kp_linear = 0
        self.ki_linear = 0
        self.kd_linear = 0

        self.time_now: float = 0.0
        self.time_last_step: float = 0.0

        # fixed robot linear velocity - starts at zero so the activities start on command
        self.velocity = 0.0
        self.omega = 0.0

        # Init the parameters
        self.resetErrorParameters()

        # nominal R and L:
        self.log("Loading kinematics calibration...")
        self.read_params_from_calibration_file()  # must have a custom robot calibration

        # Defining subscribers:
        # Coordinate subscriber:
        target_coordinate_topic = f"/{self.veh}/path_planner/coordinates"
        rospy.Subscriber(
            target_coordinate_topic, Pose2DStamped, self.triggerController, queue_size=1
        )
        # Coordinate subscriber:
        kill_switch_topic = f"/{self.veh}/kill_switch"
        rospy.Subscriber(kill_switch_topic, Bool, self.cbKillSwitch, queue_size=1)

        # Odometry publisher
        self.db_estimated_pose = rospy.Publisher(
            f"/{self.veh}/encoder_odometry",
            Pose2DStamped,
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

        # Shutdown flag
        self.is_shutdown = False

        # Register shutdown callback
        rospy.on_shutdown(self.shutdown_hook)

    def resetErrorParameters(self):
        # Add the node parameters to the parameters dictionary
        self.delta_phi_left = 0.0
        self.left_tick_prev = None

        self.delta_phi_right = 0.0
        self.right_tick_prev = None

        # - commanded
        self.x_target = None
        self.y_target = None
        self.theta_target = None

        # - PID controller
        self.kp_angular = 0
        self.ki_angular = 0
        self.kd_angular = 0
        self.kp_linear = 0
        self.ki_linear = 0
        self.kd_linear = 0

        self.prev_error_theta_int = 0.0
        self.prev_error_theta = 0.0
        self.prev_error_orientation_int = 0.0
        self.prev_error_orientation = 0.0
        self.prev_error_distance_int = 0.0
        self.prev_error_distance = 0.0

        self.time_now: float = 0.0
        self.time_last_step: float = 0.0

        # fixed robot linear velocity - starts at zero so the activities start on command
        self.velocity = 0.0
        self.omega = 0.0

    def cbKillSwitch(self, msg):
        if msg.data == True:
            self.log("Received an abort movement command!")
            self.publishCmd(0, 0)

    def triggerController(self, coordinate_msg):

        if self.is_shutdown:
            return

        self.resetErrorParameters()

        self.x_target = coordinate_msg.x
        self.y_target = coordinate_msg.y
        self.theta_target = self.angle_clamp(coordinate_msg.theta)
        self.log(
            f"Received command!: Moving from ({self.x_curr},{self.y_curr}, {self.theta_curr}) to ({self.x_target},{self.y_target},{self.theta_target})"
        )

        self.Controller()

    def Controller(self):
        """
        Calculate theta and perform the control actions given by the PID
        """
        if self.is_shutdown:
            return  # Skip control loop if shutting down
        if self.x_target == None or self.y_target == None or self.theta_target == None:
            self.publishCmd(0, 0)
            return

        delta_time = self.time_now - self.time_last_step
        # Avoid division by zero
        if delta_time <= 0:
            delta_time = 0.01

        self.time_last_step = self.time_now

        error_x = self.x_target - self.x_curr
        error_y = self.y_target - self.y_curr

        error_distance = np.sqrt(error_x**2 + error_y**2)

        error_distance_int = error_distance * delta_time + self.prev_error_distance_int
        error_distance_der = (error_distance - self.prev_error_distance) / delta_time

        # If far to the coordinates
        if abs(error_distance) > 0.2:
            error_orientation = self.angle_clamp(
                np.arctan2(error_y, error_x) - self.theta_curr
            )
            error_orientation_int = (
                error_orientation * delta_time + self.prev_error_orientation_int
            )
            error_orientation_der = (
                error_orientation - self.prev_error_orientation
            ) / delta_time

            omega = (
                self.kp_angular * error_orientation
                + self.ki_angular * error_orientation_int
                + self.kd_angular * error_orientation_der
            )
        # If far from the coordinates
        else:
            error_theta = self.theta_target - self.theta_curr
            error_theta_int = error_theta * delta_time + self.prev_error_theta_int
            error_theta_der = (error_theta - self.prev_error_theta) / delta_time

            omega = (
                self.kp_angular * error_theta
                + self.ki_angular * error_theta_int
                + self.kd_angular * error_theta_der
            )

        velocity = (
            self.kp_linear * error_distance
            + self.ki_linear * error_distance_int
            + self.kd_linear * error_distance_der
        )
        # velocity = min(velocity, 0.05)
        # self.publishCmd(velocity, omega)
        self.publishCmd(0.3, 2)
        # self.publishCmd(velocity, omega)

        self.prev_error_distance = error_distance
        self.prev_error_distance_int = error_distance_int

        # If far to the coordinates
        if abs(error_distance) > 0.2:
            self.prev_error_orientation = error_orientation
            self.prev_error_orientation_int = error_orientation_int
        # If far from the coordinates
        else:
            self.prev_error_theta = error_theta
            self.prev_error_theta_int = error_theta_int

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
        self.log(f"Publishing to car, v:{v}, omega:{omega}")
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
        while theta > np.pi:
            theta -= 2 * np.pi
        while theta < -np.pi:
            theta += 2 * np.pi
        return theta

    def shutdown_hook(self):
        # Set the shutdown flag
        self.is_shutdown = True

        # Stop the Duckiebot by sending zero velocities
        rospy.loginfo("Shutting down... Sending stop command.")
        self.publishCmd(0, 0)

        # Sleep to ensure the message is sent before shutting down
        rospy.sleep(1)


if __name__ == "__main__":
    try:
        # Initialize the node
        encoder_pose_node = MobilityNode(node_name="mobility_node")
        # Keep it spinning
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
