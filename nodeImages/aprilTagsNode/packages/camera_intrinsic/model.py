import os
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

        cameraMatrix = np.array(
            parameters['camera_matrix']['data']).reshape((parameters['camera_matrix']['rows'],
                                                                       parameters['camera_matrix']['cols']))

        self.camera_params = (cameraMatrix[0,0], cameraMatrix[1,1], cameraMatrix[0,2], cameraMatrix[1,2])

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
