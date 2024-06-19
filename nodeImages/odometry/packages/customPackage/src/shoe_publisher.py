#!/usr/bin/env python3

import os
import rospy
from std_msgs.msg import String
from duckietown.dtros import DTROS, NodeType, TopicType
from sensor_msgs.msg import PointCloud, ChannelFloat32
from geometry_msgs.msg import Point32

WHITE = 2.350988561514728583455766e-38
GREEN = 9.147676375112405838990107e-41
BLUE = 3.573311084028283530855510e-43
GREY = 9.133129495754249912976847e-39
YELLOW = 2.350952828403888300620457e-38


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
            "/db3/shoe_positions_sh/shoes",
            PointCloud,
            queue_size=10,
            dt_topic_type=TopicType.VISUALIZATION,
        )

        self.shoe_poses = PointCloud()
        self.shoe_poses.header.stamp = rospy.Time.now()
        self.shoe_poses.header.frame_id = "map"
        for i in range(10):
            if i == 6:
                new_point = Point32()
                new_point.x = 0.23
                new_point.y = 0.5235987762
                new_point.z = 0
                self.shoe_poses.points.append(new_point)
            else:
                new_point = Point32()
                new_point.x = 0
                new_point.y = 0
                new_point.z = -1
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

    def run(self):
        # publish message every 1 second (1 Hz)
        rate = rospy.Rate(1.0 / 2)

        while not rospy.is_shutdown():
            self._publisher.publish(self.shoe_poses)
            rate.sleep()


if __name__ == "__main__":
    # create the node
    node = MyPublisherNode(node_name="shoe_publiseer")
    # run node
    node.run()
    # keep the process from terminating
    rospy.spin()
