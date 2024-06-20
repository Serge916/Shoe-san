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

        # Construct publishers
        debug_topic = f"/{self.veh}/shoe_class_node/debug_topic"
        self.pub_debug_cmd = rospy.Publisher(
            debug_topic, CompressedImage, queue_size=1, dt_topic_type=TopicType.VISUALIZATION
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
            bgr = self.bridge.compressed_imgmsg_to_cv2(image_segment.segimage)
        except ValueError as e:
            self.logerr("Could not decode image: %s" % e)
            return

        # self.log("Received Frame")
        rgb = bgr[..., ::-1]
        # Find how many bounding boxes were sent
        bboxes_distance_msg = image_segment.rects
        num_images = int(len(bboxes_distance_msg)/2)
        bbox_list = np.zeros([num_images, 4], dtype=np.int32)
        distance_list = np.zeros([num_images, 4], dtype=np.float32)
        # Decompress the bounding box coordinates to a list
        for idx in range(0, 2*num_images, 2):
            # self.log(bboxes_distance_msg[idx])
            bbox_list[int(idx/2)] = [bboxes_distance_msg[idx].x, bboxes_distance_msg[idx].y, bboxes_distance_msg[idx].w, bboxes_distance_msg[idx].h]
            distance_list[int(idx/2)] = [bboxes_distance_msg[idx+1].x, bboxes_distance_msg[idx+1].y, bboxes_distance_msg[idx+1].w, bboxes_distance_msg[idx+1].h]

        for i in range(num_images):
            bbox = bbox_list[i]
            distance = distance_list[i]
            # Crop image based on the bounding boxes
            cropped_image = self.cropImage(rgb, bbox)

            bgr_cropped_image = cropped_image[..., ::-1]  
            msg = self.bridge.cv2_to_compressed_imgmsg(bgr_cropped_image)
            self.pub_debug_cmd.publish(msg)



            # Classify image
            shoe_class, confidence = self.model_wrapper.predict(cropped_image)
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



            dist = distance[0] + float(distance[1])/1000
            angle = distance[2] + float(distance[3])/1000


            if self.classifiedShoes.points[2*shoe_class].z == -1:
                self.classifiedShoes.points[2*shoe_class].x = dist
                self.classifiedShoes.points[2*shoe_class].y = angle
                self.classifiedShoes.points[2*shoe_class].z = confidence
            else: 
                self.classifiedShoes.points[2*shoe_class+1].x = dist
                self.classifiedShoes.points[2*shoe_class+1].y = angle
                self.classifiedShoes.points[2*shoe_class+1].z = confidence
            
            # For Debugging
            bgr = cropped_image[..., ::-1]
            obj_det_img = self.bridge.cv2_to_compressed_imgmsg(bgr)
            self.pub_detections_image.publish(obj_det_img)
        
            
            self.log(f"Orientation of {self.convertInt2Str(shoe_class)}'s Shoe: ({dist}m, {angle*180/np.pi}rad) with confidence of {confidence*100}%")

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
