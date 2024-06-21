#!/usr/bin/env python3

import os
import rospy
from duckietown.dtros import DTROS, NodeType
from sensor_msgs.msg import CompressedImage

import cv2
from cv_bridge import CvBridge


class CameraReaderNode(DTROS):

    def __init__(self, node_name):
        # initialize the DTROS parent class
        super(CameraReaderNode, self).__init__(
            node_name=node_name, node_type=NodeType.VISUALIZATION
        )
        # static parameters
        self._vehicle_name = os.environ["VEHICLE_NAME"]
        obj_det_topic = f"/{self._vehicle_name}/object_detection_node/image/debug_compressed"
        shoe_class_topic = f"/{self._vehicle_name}/shoe_class_node/debug_topic"
        april_tag_topic = f"/{self._vehicle_name}/april_tags_node/image/debug_compressed"
        raw_camera_topic = f"/{self._vehicle_name}/camera_node/image/compressed"

        
        self._camera_topic = april_tag_topic

        # bridge between OpenCV and ROS
        self._bridge = CvBridge()
        # create window
        self._window = "camera-reader"
        cv2.namedWindow(self._window, cv2.WINDOW_AUTOSIZE)
        # construct subscriber
        self.sub = rospy.Subscriber(self._camera_topic, CompressedImage, self.callback)

    def callback(self, msg):
        # convert JPEG bytes to CV image
        image = self._bridge.compressed_imgmsg_to_cv2(msg)
        # display frame
        cv2.imshow(self._window, image)
        cv2.waitKey(1)


if __name__ == "__main__":
    # create the node
    node = CameraReaderNode(node_name="camera_feed_node")
    # keep spinning
    rospy.spin()
