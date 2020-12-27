import cv2 as cv
import numpy as np

def scale(img):
    maxi , mini = np.max(img) , np.min(img)
    a = (maxi - mini) / 255
    scl = lambda x : (x - mini) / a
    result = np.array(scl(img) , np.uint8)
    return result


filter_size = int(input("Enter input size: ")) 
alpha = float(input("Enter alpah: "))
img = cv.imread("image/Lena.bmp" , cv.IMREAD_GRAYSCALE)
blur_img = np.array(scale(cv.GaussianBlur(img,(filter_size,filter_size),0)) , np.uint8)
result = np.array(img + (alpha  * scale(blur_img - img))  , np.uint8)
neg = np.array(blur_img - img , np.uint8)
path = "result/3.5.1/" + str(filter_size) + "_" + str(alpha) + ".jpg"
cv.imwrite(path , result)
path = "result/3.5.1/blue" + str(filter_size) + "_" + str(alpha) + ".jpg"
cv.imwrite(path , neg)
cv.waitKey(0)
