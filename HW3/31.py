import cv2 as cv
import numpy as np

def apply_box_filter(img,size):
    hs = int(size / 2)
    w,l = img.shape
    blur_img = np.zeros((w-size + 1,l- size + 1 ) , np.uint8)
    for i in range(hs,w-hs):
        for j in range(hs,l-hs):
            blur_img[i - hs][j - hs] = np.uint8(np.sum(img[i-hs:i+hs+1,j-hs:j+hs+1]) / (size*size))
    return blur_img

def apply_median_filter(img,window_size):
    w,l = img.shape
    result = np.zeros((w ,l) , np.uint8)
    hw = int(window_size / 2)

    for i in range(hw , w - hw):
        for j in range( hw,l - hw):
            result[i - hw][j - hw] = np.median(np.squeeze(np.asarray(img[i-hw:i+hw+1,j-hw:j+hw+1])))
    return result
size = int(input("Enter the filter size: "))
iteration = int(input("Enter Iteration number:"))
path = "image/Lena.bmp"

img = cv.imread(path , cv.IMREAD_GRAYSCALE)
w,l = img.shape
pad_img = np.zeros((w+size-1,l+size-1) , np.uint8)
hs = int(size / 2)
pad_img[hs:w+hs,hs:l+hs] = img

path = "result/3.1/imag" + str(iteration) + ".jpg"
for i in range(0 , iteration):
    print(i)
    blur_img = apply_box_filter(pad_img , size)
    pad_img = np.zeros((w+size-1,l+size-1) , np.uint8)
    pad_img[hs:w+hs,hs:l+hs] = blur_img

pad_img[hs:w+hs,hs:l+hs] = img
cv.imwrite(path , blur_img)
cv.waitKey(0)


