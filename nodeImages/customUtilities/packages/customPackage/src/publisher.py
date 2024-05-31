#!/usr/bin/env python3

import os
import rospy
from std_msgs.msg import String
from duckietown.dtros import DTROS, NodeType


class MyPublisherNode(DTROS):

    def __init__(self, node_name):
        # initialize the DTROS parent class
        super(MyPublisherNode, self).__init__(
            node_name=node_name, node_type=NodeType.GENERIC
        )
        # static parameters
        self._vehicle_name = os.environ["VEHICLE_NAME"]
        # construct publisher
        self._publisher = rospy.Publisher("chatter", String, queue_size=10)

    def run(self):
        # publish message every 1 second (1 Hz)
        rate = rospy.Rate(1)
        message = f"Hello from {self._vehicle_name}!"
        while not rospy.is_shutdown():
            rospy.loginfo("Publishing message: '%s'" % message)
            self._publisher.publish(message)
            rate.sleep()


if __name__ == "__main__":
    # create the node
    node = MyPublisherNode(node_name="my_publisher_node")
    # run node
    node.run()
    # keep the process from terminating
    rospy.spin()
