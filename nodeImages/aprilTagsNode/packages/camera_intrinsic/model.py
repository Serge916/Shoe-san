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
        detections = self.at_detector.detect(img)
        
        array_det = []
        for detection in detections:
            # Get the tag's corner points in the image
            image_points = np.array([detection.corners], dtype=np.float32)
            pt1 = (int(detection.corners[0][0]), int(detection.corners[0][1]))
            pt2 = (int(detection.corners[2][0]), int(detection.corners[2][1]))
            # Define the real-world coordinates of the tag's corners
            object_points = np.array([
                [-TAG_SIZE / 2, -TAG_SIZE / 2, 0],
                [ TAG_SIZE / 2, -TAG_SIZE / 2, 0],
                [ TAG_SIZE / 2,  TAG_SIZE / 2, 0],
                [-TAG_SIZE / 2,  TAG_SIZE / 2, 0]
            ], dtype=np.float32)

            # Solve the PnP problem to find the rotation and translation vectors
            retval, rvec, tvec = cv2.solvePnP(object_points, image_points, self.cameraMatrix, self.distortionArray)

            if retval:
                # Translation vector (tvec) gives the position of the tag in the camera coordinate system
                x, y, z = tvec[0][0], tvec[1][0], tvec[2][0]

                R, _ = cv2.Rodrigues(rvec)
                # Calculate the orientation of the tag in the camera frame
                # Assuming the Z-axis is forward and the X-axis is to the right
                angle_psi =np.pi/2 - np.arcsin(R[2, 0])

                # Calculate distance to the tag
                distance = np.sqrt(x**2+z**2)

                # Calculate angle to the tag (assuming the camera is facing along the Z-axis)
                angle_phi = np.arctan2(x, z)

                # Calculate the angle of the robot wrt the tag's orientation
                # pose = np.arcsin(detection.pose_R[2][0])*180/np.pi

                array_det.append([detection.tag_id, distance, angle_phi, angle_psi, pt1, pt2])
        return array_det
