import numpy as np

BASELINE = 0.1
RADIUS = 0.0318

TAG_ORIENTATIONS = {0: 0, 1: np.pi / 2, 2: np.pi, 3: -np.pi / 2}

# Process noise covariance matrix
Q = np.diag([0.001, 0.001, 0.0001])
# Measurement noise covariance
R = np.diag([0.00005, 0.000005])
