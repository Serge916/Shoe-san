#!/usr/bin/env python3

import os
import rospy
from std_msgs.msg import String
from duckietown.dtros import DTROS, NodeType
from duckietown_msgs.msg import Twist2DStamped, WheelEncoderStamped, Pose2DStamped


class MyPublisherNode(DTROS):

    def __init__(self, node_name):
        # initialize the DTROS parent class
        super(MyPublisherNode, self).__init__(
            node_name=node_name, node_type=NodeType.GENERIC
        )
        # static parameters
        self._vehicle_name = os.environ["VEHICLE_NAME"]
        # construct publisher
        self._publisher = rospy.Publisher(
            f"/{self._vehicle_name}/path_planner/coordinates",
            Pose2DStamped,
            queue_size=1,
        )

    def run(self):
        while not rospy.is_shutdown():
            # Get user input for x and y coordinates
            x, y, theta = self.get_user_input()

            # Create the Pose2DStamped message to publish
            msg = Pose2DStamped()
            msg.x = x
            msg.y = y
            msg.theta = theta

            # Log and publish the message
            rospy.loginfo(f"Publishing message: x={x}, y={y}, theta={theta}")
            rospy.loginfo(f"Topic is: /{self._vehicle_name}/path_planner/coordinates")
            self._publisher.publish(msg)

    def get_user_input(self):
        x = float(input("Enter x coordinate: "))
        y = float(input("Enter y coordinate: "))
        theta = float(input("Enter theta (orientation): "))
        return x, y, theta


if __name__ == "__main__":
    # create the node
    node = MyPublisherNode(node_name="my_publisher_node")
    # run node
    node.run()
    # keep the process from terminating
    rospy.spin()
