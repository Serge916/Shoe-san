#!/usr/bin/env python3

import cv2
import numpy as np
import rospy
import os
from duckietown.dtros import DTROS, NodeType, TopicType
from duckietown_msgs.msg import Rect
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
            april_tags_topic, Rect, queue_size=1, dt_topic_type=TopicType.PERCEPTION
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
        self.pose = Rect()
        self.pose.x = -1    # Cardinal Direction (0 -> North, 1 -> East, 2 -> South, 3 -> West)
        self.pose.y = 0     # Distance from AprilTag
        self.pose.w = 0     # Angle of the April Tag with respect to the robot
        self.pose.h = 0     # Angle of the robot with respect to the orientation of the April Tag
        
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

        detections = self.detector.predict(im_gray)
        
        for detection in detections:        
            # tvec gives the position of the tag in the camera coordinate system
            x, z = detection.pose_t[0], detection.pose_t[2]
            
            self.pose.x = detection.tag_id
            self.pose.y = np.sqrt(x**2+z**2)[0]
            self.pose.w = np.arctan2(x,z)[0]*180/np.pi
            self.pose.h = np.arcsin(detection.pose_R[2][0])*180/np.pi
            self.log(self.pose)

            self.april_tags_cmd.publish(self.pose)

        return


if __name__ == "__main__":
    # Initialize the node
    april_tags_node = AprilTagsNode(node_name="april_tags_node")
    # Keep it spinning
    rospy.spin()
