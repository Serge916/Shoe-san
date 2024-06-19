#!/usr/bin/env python3

import cv2
import numpy as np
import rospy
import os

from duckietown.dtros import DTROS, NodeType, TopicType
from duckietown_msgs.msg import Rect, Rects, SceneSegments
from cv_bridge import CvBridge
from sensor_msgs.msg import CompressedImage

IMAGE_SIZE = 416

class DataCollectionNode(DTROS):
    def __init__(self, node_name):

        # Initialize the DTROS parent class
        super(DataCollectionNode, self).__init__(
            node_name=node_name, node_type=NodeType.VISUALIZATION
        )
        self.initialized = False
        self.log("Initializing!")

        self.veh = rospy.get_namespace().strip("/")
        os.environ['VEHICLE_NAME']

        # Construct subscribers
        self.sub_detection_scene = rospy.Subscriber(
            f"/{self.veh}/object_detection_node/image/img_and_bboxes",
            SceneSegments,
            self.segment_cb,
            queue_size=1,
            buff_size=10000000)

        self.bridge = CvBridge()
        self.initialized = True
        self.image_count = 0
        self.save_path = "/dataset/images"
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        self.log("Initialized!")

    def segment_cb(self, image_segment):
        if not self.initialized:
            return

        # Access Image and bounding boxes
        try:
            rgb = self.bridge.compressed_imgmsg_to_cv2(image_segment.segimage)
        except ValueError as e:
            self.logerr("Could not decode image: %s" % e)
            return
        
        # Find how many bounding boxes were sent
        bboxes_msg = image_segment.rects
        num_images = int(len(bboxes_msg)/2)
        bbox_list = []
        # Decompress the bounding box coordinates to a list
        # for rect in bboxes_msg:
        #     bbox_list.append([rect.x, rect.y, rect.w, rect.h])
        for idx in range(0, len(bboxes_msg), 2):
            rect = bboxes_msg[idx]
            bbox_list.append([rect.x, rect.y, rect.w, rect.h])

        for i in range(num_images):
            self.image_count += 1
            image_path = os.path.join(self.save_path, f"image_{self.image_count:04d}.jpg")
            bbox = bbox_list[i]
            # Crop image based on the bounding boxes
            cropped_image = self.cropImage(rgb, bbox)
            try:
                cv2.imwrite(image_path, cropped_image)
                rospy.loginfo(f"Saved image {self.image_count} at {os.path.abspath(image_path)}")
            except Exception as e:
                rospy.loginfo(f"Failed to save image: {e}")

        return
            
    def cropImage(self, image, bbox):
        
        image_arr = image[bbox[1]:bbox[1]+bbox[3], bbox[0]:bbox[0]+bbox[2]] 

        cropped_image = cv2.resize(image_arr, (IMAGE_SIZE, IMAGE_SIZE))
        return cropped_image


if __name__ == "__main__":
    # Initialize the node
    data_collection_node = DataCollectionNode(node_name="data_collection_node")
    # Keep it spinning
    rospy.spin()
