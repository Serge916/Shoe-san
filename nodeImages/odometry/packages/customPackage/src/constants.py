import numpy as np

BASELINE = 0.1
RADIUS = 0.0318

TAG_ORIENTATIONS = {0: 0, 1: np.pi / 2, 2: np.pi, 3: -np.pi / 2}

# Process noise covariance matrix
Q = np.diag([0.001, 0.001, 0.0001])
# Measurement noise covariance
R = np.diag([0.00005, 0.000005])

# Shoe colors
WHITE = 2.350988561514728583455766e-38
GREEN = 9.147676375112405838990107e-41
BLUE = 3.573311084028283530855510e-43
GREY = 9.133129495754249912976847e-39
YELLOW = 2.350952828403888300620457e-38

VALID_TIME = 10