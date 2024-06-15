#!/usr/bin/env python3
import os
import time

import numpy as np

import rospy
from duckietown.dtros import DTROS, NodeType, TopicType
from geometry_msgs.msg import Polygon, Point32, PoseStamped


from common.constants import *

class OrchestratorNode(DTROS):

    def __init__(self, node_name):
        # initialize the DTROS parent class
        super(OrchestratorNode, self).__init__(
            node_name=node_name, node_type=NodeType.BEHAVIOR
        )

        ## Construct publishers
        # Mission to execute
        self.decision = rospy.Publisher(
            "~mission",
            Polygon,
            queue_size=1,
            dt_topic_type=TopicType.BEHAVIOR,
        )

        # Pose
        self.debug = rospy.Publisher(
            "~/visualization/Posedebug",
            PoseStamped,
            queue_size=1,
            dt_topic_type=TopicType.BEHAVIOR,
        )

        self.veh = os.environ["VEHICLE_NAME"]

        # construct subscriber
        # self.sub = rospy.Subscriber(f"/{self.veh}/front_center_tof_driver_node/range", Range, self.callback)

    #def callback(self, data):
        #rospy.loginfo("Saw range '%f'", data.range)

    def trySomething(self):
        mission = Polygon()
        left_shoe = Point32()
        left_shoe.x = 0
        left_shoe.y = 0
        left_shoe.z = NOTHING_IN_SIGHT
        right_shoe = Point32()
        right_shoe.x = 0
        right_shoe.y = 0
        right_shoe.z = NOTHING_IN_SIGHT
        mission.points = [left_shoe, right_shoe]
        self.decision.publish(mission)

        p = PoseStamped()
        p.header.stamp = rospy.Time.now()
        p.header.frame_id = "db3/base"
        p.pose.position.x = 0
        p.pose.position.y = 0
        p.pose.position.z = 0

        theta = 0

        p.pose.orientation.x = 0
        p.pose.orientation.y = 0
        p.pose.orientation.z = np.sin(theta / 2)
        p.pose.orientation.w = np.cos(theta / 2)

        self.debug.publish(p)


if __name__ == "__main__":
    # create the node
    node = OrchestratorNode(node_name="orchestrator_node")

    while(True):
        node.trySomething()
        time.sleep(4)
    # keep spinning
    rospy.spin()
