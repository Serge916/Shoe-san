#!/usr/bin/env python3

import rospy
from duckietown.dtros import DTROS, NodeType
from std_msgs.msg import String
from duckietown_msgs.msg import Pose2DStamped


class MySubscriberNode(DTROS):

    def __init__(self, node_name):
        # initialize the DTROS parent class
        super(MySubscriberNode, self).__init__(
            node_name=node_name, node_type=NodeType.GENERIC
        )
        # construct subscriber
        self.sub = rospy.Subscriber("/db3/path_planner_node/path_planner/coordinates", Pose2DStamped, self.callback)

    def callback(self, data):
        rospy.loginfo("I heard x: %d, y: %d. theta: %d, \n", data.x, data.y, data.theta)


if __name__ == "__main__":
    # create the node
    node = MySubscriberNode(node_name="my_subscriber_node")
    # keep spinning
    rospy.spin()
