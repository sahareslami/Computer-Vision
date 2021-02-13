import cv2 as cv
import numpy as np
import math

def quantizer(img , level):
    t_level = int(256 / level)
    result = np.uint8((np.floor(img / t_level)) * t_level)
    return result

def MSE(img , new_img):
    mse = np.mean(np.square(new_img - img))
    return mse

def PSNR(img , new_img):
    mse = MSE(img , new_img)
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

img = cv.imread("image/Pepper.bmp")

cv.imwrite("result/5.2.2/org_blue.jpg" , img[:,:,0])
cv.imwrite("result/5.2.2/org_green.jpg" , img[:,:,1])
cv.imwrite("result/5.2.2/org_red.jpg" , img[:,:,2])

result = np.zeros(img.shape , np.uint8)

result[:,:,0] = quantizer(img[:,:,0] , 4)
result[:,:,1] = quantizer(img[:,:,1] , 8)
result[:,:,2] = quantizer(img[:,:,2] , 8)

print("MSE" , MSE(result , img))
print("PSNR" , PSNR(result , img) )
cv.imwrite("result/5.2.2/result.jpg" , result)
cv.imwrite("result/5.2.2/blue.jpg" , result[:,:,0])
cv.imwrite("result/5.2.2/green.jpg" , result[:,:,1])
cv.imwrite("result/5.2.2/red.jpg" , result[:,:,2])


