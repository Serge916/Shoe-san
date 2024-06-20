#!/usr/bin/env python3
import os
import time
from typing import Optional, Tuple

import tf
import numpy as np
import rospy
import yaml
from duckietown.dtros import DTROS, NodeType, TopicType
from duckietown_msgs.msg import Twist2DStamped, WheelEncoderStamped, Pose2DStamped
from nav_msgs.msg import Odometry
from std_msgs.msg import String, Bool
from geometry_msgs.msg import Point32, Quaternion
from sensor_msgs.msg import Range, PointCloud, ChannelFloat32
from visualization_msgs.msg import MarkerArray, Marker

from constants import *


class OdometryNode(DTROS):
    """
    Computes an estimate of the Duckiebot pose using a combination of wheel encoders and April tags.
    Args:
        node_name (:obj:`str`): a unique, descriptive name for the ROS node
    Configuration:

    Publisher:
        ~robot_localization (:local_shoe_posestamped`): The computed position
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

        self.LEFT_RECEIVED = False
        self.RIGHT_RECEIVED = False
        # Initialize local_shoe_poses
        self.local_shoe_poses = PointCloud()
        self.local_shoe_poses.header.stamp = rospy.Time.now()
        self.local_shoe_poses.header.frame_id = "map"
        for i in range(10):
            newpoint = Point32()
            newpoint.x = 0
            newpoint.y = 0
            newpoint.z = -1
            self.local_shoe_poses.points.append(newpoint)
        self.shoe_counter = [0 for i in range(10)]
        self.local_shoe_poses.channels = [ChannelFloat32()]
        self.local_shoe_poses.channels[0].name = "rgb"
        self.local_shoe_poses.channels[0].values.append(WHITE)
        self.local_shoe_poses.channels[0].values.append(WHITE)
        self.local_shoe_poses.channels[0].values.append(GREEN)
        self.local_shoe_poses.channels[0].values.append(GREEN)
        self.local_shoe_poses.channels[0].values.append(BLUE)
        self.local_shoe_poses.channels[0].values.append(BLUE)
        self.local_shoe_poses.channels[0].values.append(GREY)
        self.local_shoe_poses.channels[0].values.append(GREY)
        self.local_shoe_poses.channels[0].values.append(YELLOW)
        self.local_shoe_poses.channels[0].values.append(YELLOW)

        # Init the parameters
        self.resetParameters()

        # In case we want to load from a file:
        # self.log("Loading odometry calibration...")
        # self.read_params_from_calibration_file()  # must have a custom robot calibration

        # Odometry publisher
        self.db_estimated_pose = rospy.Publisher(
            f"/{self.veh}/robot_odometry/odometry",
            Odometry,
            queue_size=1,
            dt_topic_type=TopicType.LOCALIZATION,
        )

        # Shoe position publisher
        self.db_shoe_pose = rospy.Publisher(
            f"/{self.veh}/robot_odometry/shoe_positions",
            PointCloud,
            queue_size=10,
            dt_topic_type=TopicType.LOCALIZATION,
        )

        # Construct publishers
        april_tags_pos_topic = f"/{self.veh}/odometry_node/april_tags_position"
        self.pub_tag_position = rospy.Publisher(
            april_tags_pos_topic,
            MarkerArray,
            queue_size=1,
            dt_topic_type=TopicType.PERCEPTION,
        )

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
        april_tags_topic = f"/{self.veh}/april_tags_node/april_tags"
        rospy.Subscriber(
            april_tags_topic, Quaternion, self.cbAprilTagReading, queue_size=1
        )
        # Time-of-flight subscriber:
        left_encoder_topic = f"/{self.veh}/front_center_tof_driver_node/range"
        rospy.Subscriber(left_encoder_topic, Range, self.cbToFReading, queue_size=1)

        # Shoe classifier subscriber:
        left_encoder_topic = f"/{self.veh}/shoe_class_node/shoe_class"
        rospy.Subscriber(
            left_encoder_topic, PointCloud, self.cbShoePosition, queue_size=1
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
            ~/robot_localization (:stamped`): Duckiebot pose.
        """
        if not self.LEFT_RECEIVED or not self.RIGHT_RECEIVED:
            return

        # IDEA
        # if self.delta_phi_left == 0 and self.delta_phi_right == 0:
        #     return

        self.LEFT_RECEIVED = False
        self.RIGHT_RECEIVED = False

        left_wheel_distance = self.delta_phi_left * (2 * np.pi * RADIUS)
        right_wheel_distance = self.delta_phi_right * (2 * np.pi * RADIUS)
        delta_distance = (right_wheel_distance + left_wheel_distance) / 2
        delta_theta = (right_wheel_distance - left_wheel_distance) / BASELINE
        delta_theta = self.angle_clamp(delta_theta)

        # Calculate new odometry only when new data from encoders arrives
        self.delta_phi_left = self.delta_phi_right = 0

        # Publisher reads estimated position. Difference to get increment IDEA
        u = np.array(
            [
                delta_distance * np.cos(self.estimate[2] + delta_theta),
                delta_distance * np.sin(self.estimate[2] + delta_theta),
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
            psi: Angle measured from the perspective of the tag. Direction to the robot
            distance: Distance from robot to tag
            id: Tag unique identifier
        """
        tag_id = int(msg.x)
        distance = msg.y
        phi = msg.z
        psi = msg.w

        beta = psi - phi

        tag_orientation = TAG_POSES.get(tag_id)[2]
        tag_x_position = TAG_POSES.get(tag_id)[0]
        tag_y_position = TAG_POSES.get(tag_id)[1]

        theta = psi + np.pi / 2 * (tag_id - 1)

        coord_x = tag_x_position + distance * np.sin(beta + tag_orientation)
        coord_y = tag_y_position - distance * np.cos(beta + tag_orientation)

        new_update = np.array([coord_x, coord_y, theta])
        # Measurement model
        difference = new_update - self.H @ self.estimate
        # Measurement covariance
        # S = self.H @ self.P @ np.transpose(self.H) + self.R * (
        #     distance * 10 + abs(phi) * 100
        # )
        S = self.H @ self.P @ np.transpose(self.H) + self.R
        # Kalman gain
        K = self.P @ np.transpose(self.H) @ np.linalg.inv(S)
        # Update the state
        self.estimate = self.estimate + K @ difference
        # Update the error covariance
        self.P = self.P - K @ self.H @ self.P

        self.log(
            f"April tag reading. id: {tag_id} phi: {np.rad2deg(phi)}, psi: {np.rad2deg(psi)}, beta: {np.rad2deg(beta)}, dist: {distance}. Robot in ({self.estimate[0]}, {self.estimate[1]}, {np.rad2deg(self.estimate[2])}, theta not estimate {theta})"
        )

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

        br = tf.TransformBroadcaster()
        br.sendTransform(
            (self.estimate[0], self.estimate[1], 0),
            tf.transformations.quaternion_from_euler(0, 0, self.estimate[2]),
            rospy.Time.now(),
            f"{self.veh}/base",
            "map",
        )

        padded = np.zeros((6, 6))
        padded[:, 0] = np.append(self.P[:, 0], np.zeros(3))
        padded[:, 1] = np.append(self.P[:, 1], np.zeros(3))
        padded[:, 5] = np.append(np.zeros(3), self.P[:, 2])

        # padded = np.pad(
        #     self.P, pad_width=((0, 3), (0, 3)), mode="constant", constant_values=0
        # )
        odom.pose.covariance = padded.flatten().tolist()

        # self.log(
        #     f"Robot position is estimated to be: {self.estimate}. Covariance is {odom.pose.covariance}"
        #     # f"Robot position is estimated to be: x:{self.estimate[0]}, y:{self.estimate[1]}, theta:{self.estimate[2]}"
        # )

        self.db_estimated_pose.publish(odom)

    def cbShoePosition(self, msg):

        # self.log(f"{msg}")
        for i, each_pose in enumerate(msg.points):
            if each_pose.z > 0:
                self.shoe_counter[i] = VALID_TIME
                shoe_dist = each_pose.x
                shoe_theta = each_pose.y

                # Check if the shoe detected is within the FOV of the ToF sensor
                if abs(shoe_theta) < self.tof_fov / 2:
                    shoe_dist = self.dist_to_object

                # Get the shoe position
                shoe_pos_x = shoe_dist * np.cos(shoe_theta)
                shoe_pos_y = shoe_dist * np.sin(shoe_theta)

                duckie_pos_x = self.estimate[0]
                duckie_pos_y = self.estimate[1]
                duckie_theta = self.estimate[2]

                # Apply the transformations
                # From origin to duckiebot
                t_mat_od = np.array(
                    (
                        [np.cos(duckie_theta), -np.sin(duckie_theta), duckie_pos_x],
                        [np.sin(duckie_theta), np.cos(duckie_theta), duckie_pos_y],
                        [0, 0, 1],
                    )
                )
                # From duckie to shoe
                t_mat_ds = np.array(
                    (
                        [np.cos(shoe_theta), -np.sin(shoe_theta), shoe_pos_x],
                        [np.sin(shoe_theta), np.cos(shoe_theta), shoe_pos_y],
                        [0, 0, 1],
                    )
                )
                # Transform
                t = np.dot(t_mat_od, t_mat_ds)

                # Update with global coordinates
                self.local_shoe_poses.points[i].x = t[0, 2]
                self.local_shoe_poses.points[i].y = t[1, 2]
                self.local_shoe_poses.points[i].z = 0
                print(f"Shoe is at [{t[0, 2]} , {t[1, 2]}]")

        self.publishShoes(self.local_shoe_poses)

    def publishShoes(self, shoe_pos):
        # print("Got into publisher!")
        for i, each_pose in enumerate(shoe_pos.points):
            if each_pose.z == 0:
                # self.log(f"Shoe {i} is at global coordinates : {each_pose}")
                self.db_shoe_pose.publish(shoe_pos)

    def cbToFReading(self, msg):
        self.dist_to_object = msg.range
        self.tof_fov = msg.field_of_view
        # self.log(f"Object detected at {self.dist_to_object} fov : {self.tof_fov}")

    def onShutdown(self):
        super(OdometryNode, self).on_shutdown()

    @staticmethod
    def angle_clamp(theta):
        while theta > np.pi:
            theta -= 2 * np.pi
        while theta < -np.pi:
            theta += 2 * np.pi
        return theta

    def run(self):
        # publish message every 0.5 second (2 Hz)
        rate = rospy.Rate(2)

        while not rospy.is_shutdown():
            # TODO: for loop, countdown, invalidate shoes
            for i in range(10):
                if self.shoe_counter[i] == 0:
                    self.local_shoe_poses.points[i].x = 0
                    self.local_shoe_poses.points[i].y = 0
                    self.local_shoe_poses.points[i].z = -1
                else:
                    self.shoe_counter[i] -= 1

            update_msg = MarkerArray()

            for id in range(4):
                marker = Marker()
                marker.header.frame_id = "map"  # Marker type
                marker.id = id  # Tag id
                marker.type = 1
                marker.action = 0
                marker.pose.position.x = TAG_POSES[id][0]  # x-axis location
                marker.pose.position.y = TAG_POSES[id][1]  # y-axis location
                marker.pose.position.z = 0
                marker.pose.orientation.x = 0
                marker.pose.orientation.y = 0
                marker.pose.orientation.z = np.sin(TAG_POSES[id][2] / 2)
                marker.pose.orientation.w = np.cos(TAG_POSES[id][2] / 2)

                marker.color.r = 1 if id == 0 else 0
                marker.color.g = 1 if id == 1 else 0
                marker.color.b = 1 if id == 2 else 0
                marker.color.a = 1

                marker.scale.x = 0.01
                marker.scale.y = 0.11
                marker.scale.z = 0.11

                update_msg.markers.append(marker)

            self.pub_tag_position.publish(update_msg)
            self.db_shoe_pose.publish(self.local_shoe_poses)
            rate.sleep()


if __name__ == "__main__":
    # Initialize the node
    encoder_pose_node = OdometryNode(node_name="odometry_node")

    encoder_pose_node.run()

    # Keep it spinning
    rospy.spin()
