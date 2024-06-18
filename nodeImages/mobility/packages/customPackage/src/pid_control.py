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

ON_DEST_THRESHOLD_OUT = 0.05
ON_DEST_THRESHOLD_IN = 0.025
ON_ANGLE_THRESHOLD_OUT = 0.2
ON_ANGLE_THRESHOLD_IN = 0.1

# State of the PID
IDLE                  = 0
FIXING_THETA          = 1
CLOSING_GAP           = 2
ADJUSTING_ORIENTATION = 3


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
        # - encoders
        self.delta_phi_left = 0.0
        self.left_tick_prev = None

        self.delta_phi_right = 0.0
        self.right_tick_prev = None

        # - odometry
        self.orientation_curr = 0.0
        self.orientation_prev = 0.0
        self.distance_curr = 0.0
        self.distance_prev = 0.0

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

        self.prev_error_orientation_int = 0.0
        self.prev_error_orientation = 0.0
        self.prev_error_distance_int = 0.0
        self.prev_error_distance = 0.0

        # Variable keeping track of the state of the PID
        self.state = IDLE

        # Create a switcher for the different functions depending on the state
        self.state_switch = {
            IDLE                  : self.idle,
            FIXING_THETA          : self.fixing_theta,
            CLOSING_GAP           : self.closing_gap,
            ADJUSTING_ORIENTATION : self.adjusting_oreintation
        }

        self.time_now: float = 0.0
        self.time_last_step: float = 0.0

        # fixed robot linear velocity - starts at zero so the activities start on command
        self.velocity = 0.0
        self.omega = 0.0

        # For encoders syncronization:
        self.RIGHT_RECEIVED = False
        self.LEFT_RECEIVED = False

        # Init the parameters
        self.resetErrorParameters()

        # nominal R and L:
        self.log("Loading kinematics calibration...")
        self.read_params_from_calibration_file()  # must have a custom robot calibration
        self.log(f"After read!: {self.kp_angular}")

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
        # Coordinate subscriber:
        kill_switch_topic = f"/{self.veh}/kill_switch"
        rospy.Subscriber(kill_switch_topic, Bool, self.cbKillSwitch, queue_size=1)

        # Odometry publisher
        self.db_estimated_pose = rospy.Publisher(
            f"/{self.veh}/encoder_odometry",
            Odometry,
            queue_size=1,
            dt_topic_type=TopicType.LOCALIZATION,
        )

        # Command publisher
        car_cmd_topic = f"/{self.veh}/joy_mapper_node/car_cmd"
        self.pub_car_cmd = rospy.Publisher(
            car_cmd_topic, Twist2DStamped, queue_size=1, dt_topic_type=TopicType.CONTROL
        )

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

        # - odometry
        self.orientation_curr = 0.0
        self.orientation_prev = 0.0
        self.ditance_curr = 0.0
        self.distance_prev = 0.0

        self.x_curr = 0.0
        self.y_curr = 0.0
        self.theta_curr = 0.0

        self.x_target = None
        self.y_target = None
        self.theta_target = None

        self.state = IDLE

        self.prev_error_theta_int = 0.0
        self.prev_error_theta = 0.0
        self.prev_error_orientation_int = 0.0
        self.prev_error_orientation = 0.0
        self.prev_error_distance_int = 0.0
        self.prev_error_distance = 0.0

        self.time_now: float = 0.0
        self.time_last_step: float = 0.0

        # fixed robot linear velocity - starts at zero so the activities start on command TODO check!
        self.velocity = 0.0
        self.omega = 0.0
    
    # TODO make a smarter kill switch check cbPIDparam
    def cbKillSwitch(self, msg):
        if msg.data == True:
            self.log("Received an abort movement command!")
            self.resetErrorParameters()
            self.publishCmd(0, 0)

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

        if self.is_shutdown:
            return

        ticks = encoder_msg.data - self.left_tick_prev
        dphi = (ticks * (2 * np.pi)) / encoder_msg.resolution
        self.delta_phi_left += dphi
        self.left_tick_prev += ticks

        # update time
        self.time_now = max(self.time_now, encoder_msg.header.stamp.to_sec())

        # compute the new pose
        self.LEFT_RECEIVED = True
        print(f"Left encoder signaled {ticks}")
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

        if self.is_shutdown:
            return

        ticks = encoder_msg.data - self.right_tick_prev
        dphi = (ticks * (2 * np.pi)) / encoder_msg.resolution
        self.delta_phi_right += dphi
        self.right_tick_prev += ticks

        # update time
        self.time_now = max(self.time_now, encoder_msg.header.stamp.to_sec())

        # compute the new pose
        self.RIGHT_RECEIVED = True
        print(f"Right encoder signaled {ticks}")
        self.poseEstimator()

    def poseEstimator(self):
        """
        Publish the pose of the Duckiebot given by the kinematic model
            using the encoders.
        Publish:
            ~/encoder_localization (:obj:`PoseStamped`): Duckiebot pose.
        """
        if not self.LEFT_RECEIVED or not self.RIGHT_RECEIVED:
            return

        if self.is_shutdown:
            return
        
        # synch incoming messages from encoders
        self.LEFT_RECEIVED = self.RIGHT_RECEIVED = False

        ## Operations to calculate the new pose
        left_wheel_distance = self.delta_phi_left * self.R
        right_wheel_distance = self.delta_phi_right * self.R

        distance = (right_wheel_distance + left_wheel_distance) / 2
        delta_theta = (right_wheel_distance - left_wheel_distance) / self.baseline

        self.x_curr = self.x_prev + distance * np.cos(self.theta_prev)
        self.y_curr = self.y_prev + distance * np.sin(self.theta_prev)

        self.theta_curr = self.theta_prev + delta_theta
        
        self.theta_curr = self.angle_clamp(self.theta_curr)  # angle always between -pi,pi

        # Calculate new odometry only when new data from encoders arrives
        self.delta_phi_left = self.delta_phi_right = 0

        # Current estimate becomes previous estimate at next iteration
        self.x_prev = self.x_curr
        self.y_prev = self.y_curr
        self.theta_prev = self.theta_curr

        # # Creating message to plot pose in RVIZ
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

        print(f"Pose estimated to be x:{odom.pose.pose.position.x}, y: {odom.pose.pose.position.y}, theta: {self.theta_curr}")
        self.db_estimated_pose.publish(odom)
        self.Controller()

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

        self.idle()

    def idle(self):
        # TODO: Check if the state has to be changed
        # The previous error is set to the current error there is between the measurements so that there isn't a spike in the error
        error_x = self.x_target - self.x_curr
        error_y = self.y_target - self.y_curr

        error_distance = np.sqrt(error_x**2 + error_y**2)
        error_orientation = self.angle_clamp(
            np.arctan2(error_y, error_x) - self.theta_curr
        )
        error_theta = self.theta_target - self.theta_curr

        self.prev_error_distance = error_distance
        self.prev_error_orientation = error_orientation
        self.prev_error_theta = error_theta

        if abs(error_distance) > ON_DEST_THRESHOLD_OUT and abs(error_orientation) > ON_ANGLE_THRESHOLD_OUT:
            self.state = ADJUSTING_ORIENTATION
        elif abs(error_distance) > ON_ANGLE_THRESHOLD_OUT:
            self.state = CLOSING_GAP
        elif abs(error_theta) > ON_ANGLE_THRESHOLD_OUT:
            self.state = FIXING_THETA
        else:
            self.state = IDLE
            self.log("The duckiebot is close enough, stopping the robot to avoid jitter")


    def fixing_theta(self):
        #Adjust theta by changing the angular velocity. Check if the state has to be changed
        self.log("On destination point but aligning with desired theta")

        delta_time = self.time_now - self.time_last_step
        # Avoid division by zero
        if delta_time <= 0:
            delta_time = 0.01

        self.time_last_step = self.time_now

        # Calculation of all errors
        error_x = self.x_target - self.x_curr
        error_y = self.y_target - self.y_curr

        error_distance = np.sqrt(error_x**2 + error_y**2)
        error_orientation = self.angle_clamp(
            np.arctan2(error_y, error_x) - self.theta_curr
        )

        # PID adjusting theta
        error_theta = self.theta_target - self.theta_curr
        error_theta_int = error_theta * delta_time + self.prev_error_theta_int
        error_theta_der = (error_theta - self.prev_error_theta) / delta_time

        # Clamping the integrar error just in case
        error_theta_int = max(min(error_theta_int, 2), -2)

        omega = (
            self.kp_angular * error_theta
            + self.ki_angular * error_theta_int
            + self.kd_angular * error_theta_der
        )

        self.prev_error_theta = error_theta
        self.prev_error_theta_int = error_theta_int

        self.publishCmd(0, omega)

        if abs(error_distance) > ON_DEST_THRESHOLD_OUT and abs(error_orientation) > ON_ANGLE_THRESHOLD_OUT:
            self.state = ADJUSTING_ORIENTATION
            self.prev_error_orientation = error_orientation
        elif abs(error_distance) > ON_ANGLE_THRESHOLD_OUT:
            self.state = CLOSING_GAP
            self.prev_error_distance = error_distance
        elif abs(error_theta) > ON_ANGLE_THRESHOLD_IN:
            self.state = FIXING_THETA
            self.prev_error_theta = error_theta
        else:
            self.state = IDLE

    def closing_gap(self):
        self.log("Aligned with destinition point, closing the gap!")

        delta_time = self.time_now - self.time_last_step
        # Avoid division by zero
        if delta_time <= 0:
            delta_time = 0.01

        self.time_last_step = self.time_now

        # Calculation of all errors
        error_x = self.x_target - self.x_curr
        error_y = self.y_target - self.y_curr

        error_distance = np.sqrt(error_x**2 + error_y**2)
        error_orientation = self.angle_clamp(
            np.arctan2(error_y, error_x) - self.theta_curr
        )
        error_theta = self.theta_target - self.theta_curr

        # PID adjusting distance
        error_distance_int = error_distance * delta_time + self.prev_error_distance_int
        error_distance_der = (error_distance - self.prev_error_distance) / delta_time

        # Clamping the integrar error just in case
        error_distance_int = max(min(error_distance_int, 2), -2)

        velocity = (
            self.kp_linear * error_distance
            + self.ki_linear * error_distance_int
            + self.kd_linear * error_distance_der
        )

        self.prev_error_distance = error_distance
        self.prev_error_distance_int = error_distance_int

        self.publishCmd(velocity, 0)

        if abs(error_distance) > ON_DEST_THRESHOLD_IN and abs(error_orientation) > ON_ANGLE_THRESHOLD_OUT:
            self.state = ADJUSTING_ORIENTATION
            self.prev_error_orientation = error_orientation
        elif abs(error_distance) > ON_ANGLE_THRESHOLD_IN:
            self.state = CLOSING_GAP
            self.prev_error_distance = error_distance
        elif abs(error_theta) > ON_ANGLE_THRESHOLD_OUT:
            self.state = FIXING_THETA
            self.prev_error_theta = error_theta
        else:
            self.state = IDLE

    def adjusting_oreintation(self):
        self.log("Aligning the robot to the destination point")

        delta_time = self.time_now - self.time_last_step
        # Avoid division by zero
        if delta_time <= 0:
            delta_time = 0.01

        self.time_last_step = self.time_now

        # Calculation of all errors
        error_x = self.x_target - self.x_curr
        error_y = self.y_target - self.y_curr

        error_distance = np.sqrt(error_x**2 + error_y**2)
        error_orientation = self.angle_clamp(
            np.arctan2(error_y, error_x) - self.theta_curr
        )
        error_theta = self.theta_target - self.theta_curr

        error_orientation_int = (
            error_orientation * delta_time + self.prev_error_orientation_int
        )
        error_orientation_der = (
            error_orientation - self.prev_error_orientation
        ) / delta_time

        # Clamping the integrar error just in case
        error_orientation_int = max(min(error_orientation_int, 2), -2)

        omega = (
            self.kp_angular * error_orientation
            + self.ki_angular * error_orientation_int
            + self.kd_angular * error_orientation_der
        )

        self.prev_error_orientation = error_orientation
        self.prev_error_orientation_int = error_orientation_int

        self.publishCmd(0, omega)

        if abs(error_distance) > ON_DEST_THRESHOLD_OUT and abs(error_orientation) > ON_ANGLE_THRESHOLD_IN:
            self.state = ADJUSTING_ORIENTATION
            self.prev_error_orientation = error_orientation
        elif abs(error_distance) > ON_ANGLE_THRESHOLD_OUT:
            self.state = CLOSING_GAP
            self.prev_error_distance = error_distance
        elif abs(error_theta) > ON_ANGLE_THRESHOLD_OUT:
            self.state = FIXING_THETA
            self.prev_error_theta = error_theta
        else:
            self.state = IDLE

    def Controller(self):
        """
        Calculate theta and perform the control actions given by the PID
        """
        if self.is_shutdown:
            return  # Skip control loop if shutting down
        if self.x_target == None or self.y_target == None or self.theta_target == None:
            self.log(f"Target is None")
            self.publishCmd(0, 0)
            return

        # The controller has three different behaviours. First if the robot is far it will try to align itself with the destination by controlling angular velocity
        # Once it is aligned then it will try to reduce the distance by controlling the linear velocity
        # Lastly after arriving at the destination it alinges to the desired angular velocity
        state_func = self.state_switch.get(self.state)
        state_func()


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
        # self.log(f"Publishing to car, v:{v}, omega:{omega}")
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
