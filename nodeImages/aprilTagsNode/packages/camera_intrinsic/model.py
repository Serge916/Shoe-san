import os
import cv2
import numpy as np
from dt_data_api import DataClient
import yaml
import rospy
from dt_apriltags import Detector
from .constants import ASSETS_DIR, TAG_SIZE

class Wrapper:
    def __init__(self):
        camera_intrinsic_parameters = "db3"
        parameter_dir = os.path.join(ASSETS_DIR, f"{camera_intrinsic_parameters}.yaml")

        with open(parameter_dir, 'r') as stream:
            parameters = yaml.safe_load(stream)

        self.distortionArray = np.array(
            parameters['distortion_coefficients']['data']).reshape((parameters['distortion_coefficients']['rows'],
                                                                       parameters['distortion_coefficients']['cols']))
        
        self.cameraMatrix = np.array(
            parameters['camera_matrix']['data']).reshape((parameters['camera_matrix']['rows'],
                                                                       parameters['camera_matrix']['cols']))

        self.camera_params = (self.cameraMatrix[0,0], self.cameraMatrix[1,1], self.cameraMatrix[0,2], self.cameraMatrix[1,2])

        self.at_detector = Detector(families='tag36h11',
                                    nthreads=1,
                                    quad_decimate=1.0,
                                    quad_sigma=0.0,
                                    refine_edges=1,
                                    decode_sharpening=0.25,
                                    debug=0)

    def predict(self, img):
        tags = self.at_detector.detect(img, True, self.camera_params, TAG_SIZE)
        return tags
    
    def rectify(self, img):
        
        # Distortion variables
        self.newK, self.regionOfInterest = cv2.getOptimalNewCameraMatrix(
            self.cameraMatrix,
            self.distortionArray,
            (img.shape[1], img.shape[0]),
            1,
            (img.shape[1], img.shape[0]),
        )
        # Undistort the image
        undistorted_img = cv2.undistort(img, self.cameraMatrix, self.distortionArray, None, self.newK)

        # Crop the image (if necessary)
        x, y, w, h = self.regionOfInterest

        undistorted_rgb = undistorted_img[y : y + h, x : x + w]
        # rospy.loginfo(undistorted_rgb.shape)
        resized_undistorted_rgb = cv2.resize(undistorted_rgb, (img.shape[1], img.shape[0]))
        # rospy.loginfo(resized_undistorted_rgb.shape)
        return resized_undistorted_rgb
