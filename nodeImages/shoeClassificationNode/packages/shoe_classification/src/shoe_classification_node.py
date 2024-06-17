#!/usr/bin/env python3

import cv2
import numpy as np
import rospy
import os

from duckietown.dtros import DTROS, NodeType, TopicType
from duckietown_msgs.msg import Rect, Rects, SceneSegments
from cv_bridge import CvBridge
from sensor_msgs.msg import CompressedImage, PointCloud, ChannelFloat32
from geometry_msgs.msg import Point32

from nn_model.constants import *
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
            shoe_class_topic, PointCloud, queue_size=1, dt_topic_type=TopicType.PERCEPTION
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

        self.classifiedShoes = PointCloud()
        self.classifiedShoes.points = [Point32() for _ in range(2*NUM_OF_CLASSES)]
        for idx in range(2*NUM_OF_CLASSES):
            self.classifiedShoes.points[idx].z = -1


        self.classifiedShoes.channels = [ChannelFloat32()]
        self.classifiedShoes.channels[0].name = "rgb"
        self.classifiedShoes.channels[0].values.append(WHITE)
        self.classifiedShoes.channels[0].values.append(WHITE)
        self.classifiedShoes.channels[0].values.append(GREEN) 
        self.classifiedShoes.channels[0].values.append(GREEN)
        self.classifiedShoes.channels[0].values.append(BLUE) 
        self.classifiedShoes.channels[0].values.append(BLUE)
        self.classifiedShoes.channels[0].values.append(GREY) 
        self.classifiedShoes.channels[0].values.append(GREY)
        self.classifiedShoes.channels[0].values.append(YELLOW) 
        self.classifiedShoes.channels[0].values.append(YELLOW)

    def segment_cb(self, image_segment):
        if not self.initialized:
            return

        for idx in range(2*NUM_OF_CLASSES):
            self.classifiedShoes.points[idx].z = -1

        # Access Image and bounding boxes
        try:
            rgb = self.bridge.compressed_imgmsg_to_cv2(image_segment.segimage)
        except ValueError as e:
            self.logerr("Could not decode image: %s" % e)
            return

        image_W = rgb.shape[1]
        image_H = rgb.shape[0]
        
        # Find how many bounding boxes were sent
        bboxes_msg = image_segment.rects
        num_images = len(bboxes_msg)
        bbox_list = []
        # Decompress the bounding box coordinates to a list
        for rect in bboxes_msg:
            bbox_list.append([rect.x, rect.y, rect.w, rect.h])

        for i in range(num_images):
            bbox = bbox_list[i]
            # Crop image based on the bounding boxes
            cropped_image = self.cropImage(rgb, bbox)
            cropped_image_rescaled = cv2.resize(cropped_image, (IMAGE_SIZE, IMAGE_SIZE))

            # Classify image
            shoe_class = self.model_wrapper.predict(cropped_image_rescaled)
            # self.log(f"Detected {self.convertInt2Str(shoe_class)}'s shoe.")
            # Depending on the classification of the image, set that list ID's values to the bounding boxes

            ###################
            # Class   | Index #
            # Sergio  | 0     #
            # Shashank| 1     #
            # Tom     | 2     #
            # Varun   | 3     #
            # Vasilis | 4     #
            ###################

            # Calculate Distance From Object and Orientation
            object_projection = FOCAL_LENGTH * SHOE_HEIGHT[shoe_class] / bbox[3]

            bound_x = bbox[0] + bbox[2]/2   # Center x-coordinate of lower bound
            bound_y = bbox[1] + bbox[3]/2   # Center y-coordinate of lower bound

            try:
                theta = np.arctan((bound_x - image_W/2)/(image_H - bound_y)) * FOV/180
            except ZeroDivisionError:
                theta = FOV/180 * np.pi

            distance = object_projection / np.cos(theta)

            if self.classifiedShoes.points[2*shoe_class].z == -1:
                self.classifiedShoes.points[2*shoe_class].x = distance
                self.classifiedShoes.points[2*shoe_class].y = theta
                self.classifiedShoes.points[2*shoe_class].z = 0
            else: 
                self.classifiedShoes.points[2*shoe_class+1].x = distance
                self.classifiedShoes.points[2*shoe_class+1].y = theta
                self.classifiedShoes.points[2*shoe_class+1].z = 0
            
            # For Debugging
            bgr = cropped_image_rescaled[..., ::-1]
            obj_det_img = self.bridge.cv2_to_compressed_imgmsg(bgr)
            self.pub_detections_image.publish(obj_det_img)
        
        for idx in range(0,2*NUM_OF_CLASSES,2):
            if self.classifiedShoes.points[idx].z == 0:
                self.log(f"Distance from {self.convertInt2Str(idx//2)}'s Shoe: {self.classifiedShoes.points[idx]}")

        self.pub_class_bboxes(image_segment.segimage.header, self.classifiedShoes)
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

    def pub_class_bboxes(self, head, classified_shoes):

        classified_shoes.header = head
        
        # Publish the RectArray message
        self.shoe_class_cmd.publish(classified_shoes)
        return


if __name__ == "__main__":
    # Initialize the node
    shoe_classification_node = ShoeClassificationNode(node_name="shoe_classification_node")
    # Keep it spinning
    rospy.spin()
