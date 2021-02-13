import cv2 as cv
import numpy as np
import math

def MSE(img , new_img):
    mse = np.mean(np.square(new_img - img))
    return mse

def PSNR(img , new_img):
    mse = MSE(img , new_img)
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

def quantizer(img , level):
    t_level = int(256 / level)
    result = np.uint8((np.floor(img / t_level)) * t_level)
    return result

def quantizer_with_different(img , levels):
    result = np.zeros(img.shape , np.uint8)
    result[:,:,0] = quantizer(img[:,:,0], levels[0])
    result[:,:,1] = quantizer(img[:,:,1], levels[1])
    result[:,:,2] = quantizer(img[:,:,2], levels[2])
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


img = cv.imread('image/Girl.bmp')

cv.imwrite("result/5.2.3/blue.jpg" , img[:,:,0])
cv.imwrite("result/5.2.3/green.jpg" , img[:,:,1])
cv.imwrite("result/5.2.3/red.jpg" , img[:,:,2])


print("blue variance" , np.sqrt(img[:,:,0].var()))
print("green variance" , np.sqrt(img[:,:,1].var()))
print("red variance" , np.sqrt(img[:,:,2].var()))

#for 8 level
eight_color = quantizer_with_different(img , [2 , 2 , 2])
print("For 8 levels. MES" , MSE(eight_color , img))
print("For 8 levels. PSNR" , PSNR(eight_color , img))
cv.imwrite("result/5.2.3/8level.jpg" , eight_color)

# for 16 levels
levels_16 = np.array([[4 , 2 , 2], [2 , 4 , 2] , [2 , 2 , 4]])
index = 0
for level in levels_16:
    sixteen_color = quantizer_with_different(img , level)
    cv.imwrite("result/5.2.3/16level" + str(index) + ".jpg" , sixteen_color)
    print("For 8 levels. MES " + str(index), MSE(sixteen_color , img))
    print("For 8 levels. PSNR " + str(index) , PSNR(sixteen_color , img))
    index += 1


# for 32 levels

levels_32 = np.array([[4 , 4 , 2], [4 , 2 , 4] , [2 , 4 , 4] , [8 , 2 , 2] , [2 , 2 , 8] , [2 , 8 , 2]])
index = 0
for level in levels_32:
    sixteen_color = quantizer_with_different(img , level)
    # cv.imwrite("result/5.2.3/32level" + str(index) + ".jpg" , sixteen_color)
    print("For 16 levels. MES " + str(index), MSE(sixteen_color , img))
    print("For 16 levels. PSNR " + str(index) , PSNR(sixteen_color , img))    
    index += 1    






