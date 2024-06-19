#!/usr/bin/env python3

import cv2
import numpy as np
import rospy
from typing import Tuple
import os

from duckietown.dtros import DTROS, NodeType, TopicType
from duckietown_msgs.msg import Rects, Rect, SceneSegments
from cv_bridge import CvBridge
from sensor_msgs.msg import CompressedImage


from nn_model.constants import *
from nn_model.model import Wrapper


NUMBER_FRAMES_SKIPPED = 5
NUM_OF_CLASSES = 2


def filter_by_classes(pred_class: int) -> bool:
    """
    Remember the class IDs:

        | Object    | ID    |
        | ---       | ---   |
        | People    | 0     |
        | Shoe      | 1     |


    Args:
        pred_class: the class of a prediction
    """

    return True if pred_class == 0 or pred_class == 1 else False


def filter_by_scores(score: float) -> bool:
    """
    Args:
        score: the confidence score of a prediction
    """

    return True if score > 0.5 else False


def filter_by_bboxes(bbox: Tuple[int, int, int, int]) -> bool:
    """
    Args:
        bbox: is the bounding box of a prediction, in xyxy format
                This means the shape of bbox is (leftmost x pixel, topmost y, rightmost x, bottommost y)
    """

    # TODO: Like in the other cases, return False if the bbox should not be considered.
    return True


class ObjectDetectionNode(DTROS):
    def __init__(self, node_name):

        # Initialize the DTROS parent class
        super(ObjectDetectionNode, self).__init__(
            node_name=node_name, node_type=NodeType.PERCEPTION
        )
        self.initialized = False
        self.log("Initializing!")

        self.veh = os.environ["VEHICLE_NAME"]

        ## Construct publishers
        # Debug image with rectangles
        self.pub_detections_image = rospy.Publisher(
            "~image/debug_compressed",
            CompressedImage,
            queue_size=1,
            dt_topic_type=TopicType.DEBUG,
        )

        # Image with bounding boxes for the clasifier
        self.pub_image_for_class = rospy.Publisher(
            "~image/img_and_bboxes",
            SceneSegments,
            queue_size=1,
            dt_topic_type=TopicType.VISUALIZATION,
        )

        # Bounding boxes people
        self.bboxes_people = rospy.Publisher(
            "~image/bboxes_people",
            Rects,
            queue_size=1,
            dt_topic_type=TopicType.VISUALIZATION,
        )

        ## Construct subscribers
        self.sub_image = rospy.Subscriber(
            f"/{self.veh}/camera_node/image/compressed",
            CompressedImage,
            self.image_cb,
            buff_size=10000000,
            queue_size=1,
        )
        # Distortion variables
        self.newK, self.regionOfInterest = cv2.getOptimalNewCameraMatrix(
            K,
            D,
            (OG_IMAGE_WIDTH, OG_IMAGE_HEIGHT),
            1,
            (OG_IMAGE_WIDTH, OG_IMAGE_HEIGHT),
        )

        self.bridge = CvBridge()

        self.log("Starting model loading!")
        self.model_wrapper = Wrapper()
        self._debug = rospy.get_param("~debug", False)
        self.log("Finished model loading!")
        self.frame_id = 0
        self.initialized = True
        self.log("Initialized!")

    def image_cb(self, image_msg):
        if not self.initialized:
            return

        self.frame_id += 1
        self.frame_id = self.frame_id % (1 + NUMBER_FRAMES_SKIPPED)
        if self.frame_id != 0:
            return

        # Decode from compressed image with OpenCV
        try:
            bgr = self.bridge.compressed_imgmsg_to_cv2(image_msg)
        except ValueError as e:
            self.logerr("Could not decode image: %s" % e)
            return

        rgb = bgr[..., ::-1]
        height_distorted, width_distorted = rgb.shape[:2]

        # Undistort the image
        undistorted_rgb = cv2.undistort(rgb, K, D, None, self.newK)

        # Crop the image (if necessary)
        offset_x, offset_y, width_undistorted, height_undistorted = (
            self.regionOfInterest
        )
        undistorted_rgb = undistorted_rgb[
            offset_y : offset_y + height_undistorted,
            offset_x : offset_x + width_undistorted,
        ]
        resized_distorted = cv2.resize(rgb, (IMAGE_SIZE, IMAGE_SIZE))

        bboxes, classes, scores = self.model_wrapper.predict(resized_distorted)
        classes = [int(classes[x]) % NUM_OF_CLASSES for x in range(len(classes))]

        detection = self.det2bool(bboxes, classes, scores)

        if not detection:
            # self.log("Nothing detected! Sending debug image either way")

            if self._debug:
                bgr = undistorted_rgb[..., ::-1]
                obj_det_img = self.bridge.cv2_to_compressed_imgmsg(bgr)
                self.pub_detections_image.publish(obj_det_img)

            return

        names = {0: "People", 1: "Shoe"}
        font = cv2.FONT_HERSHEY_SIMPLEX

        shoe_bboxes = SceneSegments()
        shoe_bboxes.segimage.header = image_msg.header
        shoe_bboxes.segimage.data = image_msg.data

        shoe_bboxes.rects = []
        people_bboxes = []

        scale_undistorted_x = width_undistorted / IMAGE_SIZE
        scale_undistorted_y = height_undistorted / IMAGE_SIZE
        scale_distorted_x = width_distorted / IMAGE_SIZE
        scale_distorted_y = height_distorted / IMAGE_SIZE
        for clas, box in zip(classes, bboxes):
            box = [box[0]*scale_distorted_x, box[1]*scale_distorted_y, box[2]*scale_distorted_x, box[3]*scale_distorted_y]
            # TODO! Check the signs of these values
            rect_distorted = Rect()
            rect_distorted.x = int(box[0])
            rect_distorted.y = int(box[1])
            rect_distorted.w = int(box[2] - box[0])
            rect_distorted.h = int(box[3] - box[1])

            undistorted_box_points = np.array(
                [[[box[0], box[1]]], [[box[2], box[3]]]], dtype=np.float32
            )
            undistorted_points = cv2.undistortPoints(
                undistorted_box_points, K, D, None, self.newK
            )

            rect_undistorted = Rect()
            rect_undistorted.x = int(undistorted_points[0][0][0] - offset_x)
            rect_undistorted.y = int(undistorted_points[0][0][1] - offset_y)
            rect_undistorted.w = (
                int(undistorted_points[1][0][0] - offset_x) - rect_undistorted.x
            )
            rect_undistorted.h = (
                int(undistorted_points[1][0][1] - offset_y) - rect_undistorted.y
            )

            distance_x = FOCAL_LENGTH * SHOE_HEIGHT / rect_undistorted.h

            try:
                angle = (
                    np.arctan(
                        (   
                            width_undistorted / 2
                            - rect_undistorted.x
                            - rect_undistorted.w / 2
                        )
                        / (
                            height_undistorted
                            - rect_undistorted.y
                            - rect_undistorted.h / 2
                        )
                    )
                    * FOV
                    / 180
                )
            except ZeroDivisionError:
                angle = FOV / 180 * np.pi

            total_distance = distance_x / np.cos(angle)

            extra_data = Rect()
            extra_data.x = int(total_distance)
            extra_data.y = int(total_distance * 1000) - extra_data.x * 1000
            extra_data.w = int(angle)
            extra_data.h = int(angle * 1000) - extra_data.w * 1000

            if clas == 0:
                # Then it has detected people, so we populate the info for the people_bboxes
                people_bboxes.append(rect_distorted)
                people_bboxes.append(extra_data)
            elif clas == 1:
                # Then it has detected a shoe, so populetes the message for the classifier
                shoe_bboxes.rects.append(rect_distorted)
                shoe_bboxes.rects.append(extra_data)

            if self._debug:
                colors = {0: (0, 255, 255), 1: (0, 165, 255)}

                pt1 = np.array(object=[rect_undistorted.x, rect_undistorted.y])
                pt2 = np.array([rect_undistorted.x + rect_undistorted.w, rect_undistorted.y + rect_undistorted.h])
                pt1 = tuple(pt1)
                pt2 = tuple(pt2)
                color = tuple(reversed(colors[clas]))

                name = f"{names[clas]}: ({total_distance:.2f}m, {angle*(180/np.pi):.2f}deg)"
                # draw bounding box
                undistorted_rgb = cv2.rectangle(undistorted_rgb, pt1, pt2, color, 2)
                # label location
                text_location = (pt1[0], min(pt2[1] + 30, height_undistorted))
                # draw label underneath the bounding box
                undistorted_rgb = cv2.putText(
                    undistorted_rgb, name, text_location, font, 0.4, color, thickness=2
                )

        if len(shoe_bboxes.rects) != 0:
            self.pub_image_for_class.publish(shoe_bboxes)

        if len(people_bboxes) != 0:
            self.bboxes_people.publish(people_bboxes)

        if self._debug:
            bgr = undistorted_rgb[..., ::-1]
            obj_det_img = self.bridge.cv2_to_compressed_imgmsg(bgr)
            self.pub_detections_image.publish(obj_det_img)

    def det2bool(self, bboxes, classes, scores):
        box_ids = np.array(list(map(filter_by_bboxes, bboxes))).nonzero()[0]
        cla_ids = np.array(list(map(filter_by_classes, classes))).nonzero()[0]
        sco_ids = np.array(list(map(filter_by_scores, scores))).nonzero()[0]

        box_cla_ids = set(list(box_ids)).intersection(set(list(cla_ids)))
        box_cla_sco_ids = set(list(sco_ids)).intersection(set(list(box_cla_ids)))

        if len(box_cla_sco_ids) > 0:
            return True
        else:
            return False


if __name__ == "__main__":
    # Initialize the node
    object_detection_node = ObjectDetectionNode(node_name="object_detection_node")
    # Keep it spinning
    rospy.spin()
