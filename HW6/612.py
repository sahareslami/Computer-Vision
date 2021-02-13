import cv2 as cv
import numpy as np


img = cv.imread("images/Lena.bmp" , cv.IMREAD_GRAYSCALE)


def apply_filter(img , ifilter, weight):
    w,l = img.shape
    wf, lf = ifilter.shape
    hwf , hlf = int(wf/2) , int(lf/2)
    result = np.zeros((w - wf + 1 , l - lf + 1) , int)
    maxi , mini = 0 , 255
    for i in range(hwf , w - hwf):
        for j in range(hlf , l - hlf):
            val = int(dots(ifilter , img[i - hwf: i + hwf + 1 , j - hlf : j + hlf + 1]) / weight)
            maxi , mini = max(maxi , val) , min(mini , val)
            result[i - hwf][j - hlf] = val


img = cv.imread("images/Lena.bmp" , cv.IMREAD_GRAYSCALE)
haar_filter = np.array([[1 , 1,  1 , 1],[1 , 1 , -1 , -1],[np.sqrt(2) , -1 * np.sqrt(2) , 0 , 0],[0 , 0 , np.sqrt(2) , -1 * sqrt(2)]])

level_1 = apply_filter(img , haar_filter , 2)
level_2 = 



