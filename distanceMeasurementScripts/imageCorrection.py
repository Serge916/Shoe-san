import cv2
import numpy as np

image_height = 480
image_width = 640

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

img = cv2.imread("image8.png")

# Compute the optimal new camera matrix
newK, regionOfInterest = cv2.getOptimalNewCameraMatrix(
    K, D, (image_width, image_height), 1, (image_width, image_height)
)

print(newK)
print(regionOfInterest)

# Undistort the image
dst = cv2.undistort(img, K, D, None, newK)

# Crop the image (if necessary)
x, y, w, h = regionOfInterest
dst = dst[y : y + h, x : x + w]

# Save and display the undistorted image
cv2.imwrite("undistorted_image8.jpg", dst)
# cv2.imshow("Undistorted Image", dst)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
