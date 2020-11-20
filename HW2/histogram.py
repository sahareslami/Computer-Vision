import numpy as np
import cv2 as cv

def compute_histogram(img):
    width , height = img.shape
    histogram = np.zeros((256), int)
    for i in range(0 , width):
        for j in range(0, height):
            histogram[img[i][j]] += 1
    return histogram


