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

        self.pub_tag_image = rospy.Publisher(
            f"/{self.veh}/april_tags_node/image/debug_compressed",
            CompressedImage,
            queue_size=1,
            dt_topic_type=TopicType.VISUALIZATION
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
        self._debug = True
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

        array_det = self.detector.predict(im_gray)
        names = {0: "North", 1: "East", 2: "South", 3: "West"}
        font = cv2.FONT_HERSHEY_SIMPLEX

        if len(array_det) == 0:
            if self._debug:
                bgr = rgb[..., ::-1]
                apr_tag_img = self.bridge.cv2_to_compressed_imgmsg(bgr)
                self.pub_tag_image.publish(apr_tag_img)
            return
        
        for msg_det in array_det:            
            self.pose.x, self.pose.y, self.pose.w, self.pose.h = msg_det[0], msg_det[1], msg_det[2], msg_det[3]
            self.log(self.pose)
            self.april_tags_cmd.publish(self.pose)

            if self._debug:
                colors = {0: (0, 255, 255), 1: (255, 0, 255), 2: (255, 255, 0), 3: (255, 255, 255)}

                pt1 = msg_det[4]
                pt2 = msg_det[5]
                color = tuple(reversed(colors[int(self.pose.x)]))
                distance = self.pose.y

                name = f"{names[self.pose.x]}: ({distance:.2f} m))"
                # draw bounding box
                # rgb_crop = cv2.rectangle(rgb_crop, pt1, pt2, color, 2)
                rgb = cv2.rectangle(rgb.astype(np.uint8), pt1, pt2, color, 2)
                # label location
                text_location = (pt1[0], min(pt2[1] + 10, pt1[1] - pt2[1]))
                # draw label underneath the bounding box
                rgb = cv2.putText(rgb, name, text_location, font, 1, color, thickness=2)

        if self._debug:
            bgr = rgb[..., ::-1]
            apr_tag_img = self.bridge.cv2_to_compressed_imgmsg(bgr)
            self.pub_tag_image.publish(apr_tag_img)
        return


if __name__ == "__main__":
    # Initialize the node
    april_tags_node = AprilTagsNode(node_name="april_tags_node")
    # Keep it spinning
    rospy.spin()
