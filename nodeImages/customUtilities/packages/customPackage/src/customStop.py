#!/usr/bin/env python3

import os
import rospy
from std_msgs.msg import Bool
from duckietown.dtros import DTROS, NodeType, TopicType
from duckietown_msgs.msg import Twist2DStamped, WheelEncoderStamped, Pose2DStamped


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
        car_cmd_topic = f"/{self._vehicle_name}/joy_mapper_node/car_cmd"
        self.pub_car_cmd = rospy.Publisher(
            car_cmd_topic, Twist2DStamped, queue_size=1, dt_topic_type=TopicType.CONTROL
        )

        car_control_msg = Twist2DStamped()

        car_control_msg.v = 0
        car_control_msg.omega = 0
        self.log(f"Published STOP")
        self.pub_car_cmd.publish(car_control_msg)


if __name__ == "__main__":
    # create the node
    node = MyPublisherNode(node_name="custom_stop_node")
    # run node
    # keep the process from terminating
    rospy.spin()
