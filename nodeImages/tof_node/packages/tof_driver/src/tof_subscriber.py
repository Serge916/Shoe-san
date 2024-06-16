#!/usr/bin/env python3

import rospy
import os
from std_msgs.msg import String, Header
from dt_vl53l0x import VL53L0X
from duckietown.dtros import DTROS, NodeType
from sensor_msgs.msg import Range, PointCloud, ChannelFloat32
from duckietown_msgs.msg import Rects
from geometry_msgs.msg import Point32

WHITE = 2.350988561514728583455766E-38
GREEN = 9.147676375112405838990107E-41
BLUE = 3.573311084028283530855510E-43
GREY = 8.931498061021016309458703E-40
YELLOW = 2.350952828403888300620457E-38

class ToFSubscriber(DTROS):

    def __init__(self, node_name):
        # initialize the DTROS parent class
        super(ToFSubscriber, self).__init__(
            node_name=node_name, node_type=NodeType.PERCEPTION
        )

        self.veh = os.environ["VEHICLE_NAME"]

        # construct publisher
        self._publisher = rospy.Publisher(f"/{self.veh}/shoe_positions/shoes", PointCloud, queue_size=10, dt_topic_type=TopicType.VISUALIZATION)

        self.shoe_poses = PointCloud()
        self.shoe_poses.header.stamp = rospy.Time.now()
        self.shoe_poses.header.frame_id = f"{self.veh}/base"
        for i in range(10):
            new_point = Point32()
            new_point.x = i
            new_point.y = i
            new_point.z = 0
            self.shoe_poses.points.append(new_point)

        self.shoe_poses.channels = [ChannelFloat32()]
        self.shoe_poses.channels[0].name = "rgb"
        self.shoe_poses.channels[0].values.append(WHITE)
        self.shoe_poses.channels[0].values.append(WHITE)
        self.shoe_poses.channels[0].values.append(GREEN) 
        self.shoe_poses.channels[0].values.append(GREEN)
        self.shoe_poses.channels[0].values.append(BLUE) 
        self.shoe_poses.channels[0].values.append(BLUE)
        self.shoe_poses.channels[0].values.append(GREY) 
        self.shoe_poses.channels[0].values.append(GREY)
        self.shoe_poses.channels[0].values.append(YELLOW) 
        self.shoe_poses.channels[0].values.append(YELLOW)


        # construct subscribers
        #Subscribe from the ToF node
        self.sub_tof = rospy.Subscriber(f"/{self.veh}/front_center_tof_driver_node/range", Range, self.callback)

        #Subscribe from the Shoe classifier node
        self.sub_class = rospy.Subscriber(f"/{self.veh}/shoe_class_node/shoe_class", Rects, self.callback)
    

    def callback(self, data):
        rospy.loginfo("Range: '%0.f'", data.range)


if __name__ == "__main__":
    # create the node
    node = ToFSubscriber(node_name="tof_subscriber")
    # keep spinning
    rospy.spin()
