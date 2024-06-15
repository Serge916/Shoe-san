#!/usr/bin/env python3

import cv2
import numpy as np
import rospy
import os

from duckietown.dtros import DTROS, NodeType, TopicType
from duckietown_msgs.msg import Rect, Rects, SceneSegments
from cv_bridge import CvBridge
from sensor_msgs.msg import CompressedImage

from nn_model.constants import IMAGE_SIZE, NUM_OF_CLASSES
from nn_model.model import Wrapper, SimpleCNN


class ShoeClassificationNode(DTROS):
    def __init__(self, node_name):

        # Initialize the DTROS parent class
        super(ShoeClassificationNode, self).__init__(
            node_name=node_name, node_type=NodeType.PERCEPTION
        )
        self.initialized = False
        self.log("Initializing!")

        self.veh = rospy.get_namespace().strip("/")
        os.environ['VEHICLE_NAME']

        # Construct publishers
        shoe_class_topic = f"/{self.veh}/shoe_class_node/shoe_class"
        self.shoe_class_cmd = rospy.Publisher(
            shoe_class_topic, Rects, queue_size=1, dt_topic_type=TopicType.PERCEPTION
        )

        # Debug image with rectangles
        self.pub_detections_image = rospy.Publisher(
            "~image/debug_compressed",
            CompressedImage,
            queue_size=1,
            dt_topic_type=TopicType.DEBUG,
        )


        # Construct subscribers
        self.sub_detection_scene = rospy.Subscriber(
            f"/{self.veh}/object_detection_node/image/img_and_bboxes",
            SceneSegments,
            self.segment_cb,
            queue_size=1,
            buff_size=10000000)

        self.bridge = CvBridge()

        self.log("Starting model loading!")
        self.model_wrapper = Wrapper()
        self.log("Finished model loading!")
        self.first_image_received = False
        self.initialized = True
        self.log("Initialized!")

    def segment_cb(self, image_segment):
        if not self.initialized:
            return

        # Access Image and bounding boxes
        try:
            bgr = self.bridge.compressed_imgmsg_to_cv2(image_segment.segimage)
        except ValueError as e:
            self.logerr("Could not decode image: %s" % e)
            return

        rgb = bgr[..., ::-1]
        rgb_xyb_rescaled = cv2.resize(rgb, (IMAGE_SIZE, IMAGE_SIZE))
        
        # Find how many bounding boxes were sent
        bboxes_msg = image_segment.rects
        num_images = len(bboxes_msg)
        bbox_list = []
        # Decompress the bounding box coordinates to a list
        for rect in bboxes_msg:
            bbox_list.append([rect.x, rect.y, rect.w, rect.h])

        shoe_bbox_list = [ [0,0,0,0] for _ in range(2*NUM_OF_CLASSES)]

        for i in range(num_images):
            bbox = bbox_list[i]
            # Crop image based on the bounding boxes
            cropped_image = self.cropImage(rgb_xyb_rescaled, bbox)
            cropped_image_rescaled = cv2.resize(cropped_image, (IMAGE_SIZE, IMAGE_SIZE))

            # Classify image
            shoe_class = self.model_wrapper.predict(cropped_image_rescaled)
            self.log(f"Detected {self.convertInt2Str(shoe_class)}'s shoe.")
            # Depending on the classification of the image, set that list ID's values to the bounding boxes

            ###################
            # Class   | Index #
            # Sergio  | 0     #
            # Shashank| 1     #
            # Tom     | 2     #
            # Varun   | 3     #
            # Vasilis | 4     #
            ###################
            if shoe_bbox_list[2*shoe_class] != [0,0,0,0]:
                shoe_bbox_list[2*shoe_class+1] = bbox
            else:
                shoe_bbox_list[2*shoe_class] = bbox

            bgr = cropped_image_rescaled[..., ::-1]
            obj_det_img = self.bridge.cv2_to_compressed_imgmsg(bgr)
            self.pub_detections_image.publish(obj_det_img)
            
        self.log(shoe_bbox_list)
        self.pub_class_bboxes(shoe_bbox_list)
        return

    def convertInt2Str(self, class_idx: int):

        if class_idx == 0: return "Sergio"
        if class_idx == 1: return "Shashank"
        if class_idx == 2: return "Tom"
        if class_idx == 3: return "Varun"
        if class_idx == 4: return "Vasilis"
        return "Ghost"
            
            
    def cropImage(self, image, bbox):

        image_arr = image[bbox[1]:bbox[1]+bbox[3], bbox[0]:bbox[0]+bbox[2]] 

        cropped_image = cv2.resize(image_arr, (IMAGE_SIZE, IMAGE_SIZE))
        return cropped_image

    def pub_class_bboxes(self, classified_bboxes):

        bboxes_array_msg = Rects()
        for class_bbox in classified_bboxes:
            rect = Rect()
            rect.x = class_bbox[0]
            rect.y = class_bbox[1]
            rect.w = class_bbox[2]
            rect.h = class_bbox[3]
            bboxes_array_msg.rects.append(rect)
        
        # Publish the RectArray message
        self.shoe_class_cmd.publish(bboxes_array_msg)
        return


if __name__ == "__main__":
    # Initialize the node
    shoe_classification_node = ShoeClassificationNode(node_name="shoe_classification_node")
    # Keep it spinning
    rospy.spin()
