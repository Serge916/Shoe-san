#!/usr/bin/env python3

import cv2
import numpy as np
import rospy
from typing import Tuple
import os

from duckietown.dtros import DTROS, NodeType, TopicType
from duckietown_msgs.msg import Twist2DStamped, Rects, Rect, SceneSegments
from cv_bridge import CvBridge
from sensor_msgs.msg import CompressedImage


from nn_model.constants import IMAGE_SIZE, OG_IMAGE_WIDTH, OG_IMAGE_HEIGHT, K, D
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

        # Undistort the image
        undistorted_rgb = cv2.undistort(rgb, K, D, None, self.newK)

        # Crop the image (if necessary)
        x, y, w, h = self.regionOfInterest
        undistorted_rgb = undistorted_rgb[y : y + h, x : x + w]
        resized_rgb = cv2.resize(undistorted_rgb, (IMAGE_SIZE, IMAGE_SIZE))

        bboxes, classes, scores = self.model_wrapper.predict(resized_rgb)
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
        # TODO! Check if I have to copy the data not only the pointer (probably but I forgot how python works)
        _, im_buf_arr = cv2.imencode(".png", undistorted_rgb)
        shoe_bboxes.segimage.data = im_buf_arr.tobytes()
        shoe_bboxes.rects = []
        people_bboxes = []

        scale_x = w / IMAGE_SIZE
        scale_y = h / IMAGE_SIZE
        for clas, box in zip(classes, bboxes):
            # TODO! Check the signs of these values
            undistorted_box = [
                int(box[0] * scale_x),
                int(box[1] * scale_y),
                int(box[2] * scale_x),
                int(box[3] * scale_y),
            ]

            rect = Rect()
            rect.x = int(undistorted_box[0])
            rect.y = int(undistorted_box[1])
            rect.w = int(undistorted_box[2]) - rect.x
            rect.h = int(undistorted_box[3]) - rect.y

            if clas == 0:
                # Then it has detected people, so we populate the info for the people_bboxes
                people_bboxes.append(rect)
            elif clas == 1:
                # Then it has detected a shoe, so populetes the message for the classifier
                shoe_bboxes.rects.append(rect)

            if self._debug:
                colors = {0: (0, 255, 255), 1: (0, 165, 255)}

                pt1 = np.array([rect.x, rect.y])
                pt2 = np.array([int(undistorted_box[2]), int(undistorted_box[3])])
                pt1 = tuple(pt1)
                pt2 = tuple(pt2)
                color = tuple(reversed(colors[clas]))
                distance = 228.15 * 100 / rect.h
                # theta = np.arctan()
                # distance_y = 228.15 * 100 / rect.h
                # angle =
                name = f"{names[clas]}({distance} mm))"
                # draw bounding box
                undistorted_rgb = cv2.rectangle(undistorted_rgb, pt1, pt2, color, 2)
                # label location
                text_location = (pt1[0], min(pt2[1] + 30, h))
                # draw label underneath the bounding box
                undistorted_rgb = cv2.putText(
                    undistorted_rgb, name, text_location, font, 1, color, thickness=2
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
