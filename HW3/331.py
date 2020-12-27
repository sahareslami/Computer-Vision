import cv2 as cv
import numpy as np
import eqlized 

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
            val = int(dots(ifilter , img[i - hwf: i + hwf + 1 , j - hlf : j + hlf + 1]) / weight)
            maxi , mini = max(maxi , val) , min(mini , val)
            result[i - hwf][j - hlf] = val

def apply_median_filter(img,window_size):
    w,l = img.shape
    result = np.zeros((w ,l) , np.uint8)
    hw = int(window_size / 2)

    for i in range(hw , w - hw):
        for j in range( hw,l - hw):
            result[i - hw][j - hw] = np.median(np.squeeze(np.asarray(img[i-hw:i+hw+1,j-hw:j+hw+1])))
    return result

path = "image/room6.jpg"

image = cv.imread(path , cv.IMREAD_GRAYSCALE)
img = eqlized.equalize(image)
cv.imwrite("result/3.3.1/eqlized.jpg" , img)
img = apply_median_filter(img,5)

#apply laplacian
filters = np.array([[0,1,0],[1,-4,1],[0,1,1]] , np.uint8)
laplacian = apply_filter(img , filters)
cv.imwrite("result/3.3.1/laplacian.jpg" , laplacian)
# Apply Unsharp masking
result = np.array(laplacian + img , np.uint8)
cv.imwrite("result/3.3.1/result.jpg" , result)
# cv.imshow("gig" ,gauss)
# cv.waitKey(0)

