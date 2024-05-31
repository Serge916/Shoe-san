#!/usr/bin/env python3

import os
import rospy
from std_msgs.msg import Bool
from duckietown.dtros import DTROS, NodeType


class MyPublisherNode(DTROS):
    def cbDetected(self, stop):
        self._publisher.publish(stop)

    def __init__(self, node_name):
        # initialize the DTROS parent class
        super(MyPublisherNode, self).__init__(
            node_name=node_name, node_type=NodeType.GENERIC
        )
        # static parameters
        self._vehicle_name = os.environ["VEHICLE_NAME"]
        # construct publisher
        self._publisher = rospy.Publisher("~command_stop", Bool, queue_size=1)
        self._subscriber = rospy.Subscriber(
            f"{self._vehicle_name}/object_detection_node/detected_duckie",
            Bool,
            callback=self.cbDetected,
        )


if __name__ == "__main__":
    # create the node
    node = MyPublisherNode(node_name="custom_stop_node")
    # run node
    # keep the process from terminating
    rospy.spin()
