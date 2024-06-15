#!/usr/bin/env python3

import cv2
import numpy as np
import rospy
import os
from duckietown.dtros import DTROS, NodeType, TopicType
from duckietown_msgs.msg import Pose2DStamped
from cv_bridge import CvBridge
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import Bool

from camera_intrinsic.constants import TAG_SIZE
from camera_intrinsic.model import Wrapper


class AprilTagsNode(DTROS):
    def __init__(self, node_name):

        # Initialize the DTROS parent class
        super(AprilTagsNode, self).__init__(
            node_name=node_name, node_type=NodeType.PERCEPTION
        )
        self.initialized = False
        self.log("Initializing!")

        self.veh = rospy.get_namespace().strip("/")
        os.environ['VEHICLE_NAME']

        # Construct publishers
        april_tags_topic = f"/{self.veh}/april_tags_node/april_tags"
        self.april_tags_cmd = rospy.Publisher(
            april_tags_topic, Pose2DStamped, queue_size=1, dt_topic_type=TopicType.PERCEPTION
        )

        # Construct subscribers
        self.sub_image = rospy.Subscriber(
            f"/{self.veh}/camera_node/image/compressed",
            CompressedImage,
            self.image_cb,
            buff_size=10000000,
            queue_size=1,
        )

        self.bridge = CvBridge()
        self.log("Initializing April Tags Detector ...")
        self.detector = Wrapper()

        self.first_image_received = False
        self.initialized = True
        self.log("Initialized!")

    def image_cb(self, img):
        if not self.initialized:
            return

        # Access Image and bounding boxes
        try:
            bgr = self.bridge.compressed_imgmsg_to_cv2(img)
        except ValueError as e:
            self.logerr("Could not decode image: %s" % e)
            return  
        rgb = bgr[..., ::-1]
        im_gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
        
        tags = self.detector.predict(im_gray)
        
        for tag in tags:
            pose = Pose2DStamped()
            pose.header = img.header
            pose.x = tag.pose_t[2][0]
            pose.y = TAG_SIZE
            pose.theta = np.arcsin(tag.pose_R[0,1])
            self.log(pose)

            self.april_tags_cmd.publish(pose)

        return


if __name__ == "__main__":
    # Initialize the node
    april_tags_node = AprilTagsNode(node_name="april_tags_node")
    # Keep it spinning
    rospy.spin()
