import cv2 as cv
import numpy as np

path = "image/room4.JPG"

image = cv.imread(path , cv.IMREAD_GRAYSCALE)
print(image)
print(image.shape)
# Blur the image
gauss = cv.GaussianBlur(image, (7,7), 0)
# Apply Unsharp masking
unsharp_image = cv.addWeighted(image, 2, gauss, -1, 0)
# cv.imshow("gig" ,unsharp_image)
# cv.waitKey(0)

