import cv2 as cv
import numpy as np


def dots(m1 , m2):
    if(m1.shape != m2.shape):
        print("WRONGE")
    w , l = m1.shape
    ans = 0
    for i in range(w):
        for j in range(l):
            ans += int(m2[i][j] * m1[i][j])
    return ans

def apply_filter(img , ifilter, weight):
    w,l = img.shape
    wf, lf = ifilter.shape
    hwf , hlf = int(wf/2) , int(lf/2)
    result = np.zeros((w - wf + 1 , l - lf + 1) , int)
    maxi , mini = 0 , 255
    for i in range(hwf , w - hwf):
        for j in range(hlf , l - hlf):
            val = int(dots(ifilter , img[i - hwf: i + hwf  , j - hlf : j + hlf]) / weight)
            maxi , mini = max(maxi , val) , min(mini , val)
            result[i - hwf][j - hlf] = val

    alpha = (maxi - mini) / 255
    scale = lambda x : (x - mini) / alpha
    applied = np.array(scale(result) , np.uint8)
    return applied


filter_num = int(input())
img = cv.imread("image/Lena.bmp" , cv.IMREAD_GRAYSCALE)
filters = []
filters.append(np.array([[1, 0],[0 , -1]] , int))
filters.append(np.array([[0 , 1],[-1 , 0]] , int))

result = apply_filter(img , filters[filter_num] , 1)
path = "result/3.4.2/img" + str(filter_num) + ".jpg"
cv.imwrite(path , result)