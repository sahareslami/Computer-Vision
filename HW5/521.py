import cv2 as cv
import numpy as np
import pandas as pd
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


img = cv.imread('image/Pepper.bmp')

report = pd.DataFrame(index=['MSE' , 'PSNR'], columns = ['8' , '16' , '32' , '64'])

gray_level = np.array([8 , 16 , 32 , 64])

for level in gray_level:
    quantized_img = quantizer(img, level)
    report.set_value('MSE' , str(level) , MSE(quantized_img , img))
    report.set_value('PSNR' , str(level) , PSNR(quantized_img , img))
    cv.imwrite("result/5.2.1/" + str(level) +  ".jpg", quantized_img)


print(report)





