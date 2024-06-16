ASSETS_DIR = "/code/catkin_ws/src/object-detection/assets/"
IMAGE_SIZE = 416
DT_TOKEN = "dt1-3nT7FDbT7NLPrXykNJmqqh1v3fkywzQdti2VUSFrQa7HF4Q-43dzqWFnWd8KBa1yev1g3UKnzVxZkkTbfjLSzZjCy7j5Y6rJMe8GBP14axmwkrmn5f"
MODEL_NAME = "yolov5n"


import numpy as np

K = np.array(
    [
        [336.5568779901599, 0, 329.48689927534764],
        [0, 336.83379682532745, 247.82748160875434],
        [0, 0, 1],
    ]
)

D = np.array(
    [
        -0.276707022018512,
        0.04795023713853468,
        -0.0024864989237580143,
        -0.00033282476417615016,
        0,
    ]
)

OG_IMAGE_HEIGHT = 480
OG_IMAGE_WIDTH = 640
