import cv2 as cv
import numpy as np
import pandas as pd
from skimage.util import *

def apply_median_filter(img,window_size):
    w,l = img.shape
    result = np.zeros((w ,l) , np.uint8)
    hw = int(window_size / 2)

    for i in range(hw , w - hw):
        for j in range( hw,l - hw):
            result[i - hw][j - hw] = np.median(np.squeeze(np.asarray(img[i-hw:i+hw+1,j-hw:j+hw+1])))
    return result

def apply_box_filter(img,size):
    hs = int(size / 2)
    w,l = img.shape
    blur_img = np.zeros((w,l) , np.uint8)
    for i in range(hs,w-hs):
        for j in range(hs,l-hs):
            blur_img[i - hs][j - hs] = np.uint8(np.sum(img[i-hs:i+hs+1,j-hs:j+hs+1]) / (size*size))
    return blur_img

def save_result(filter, size , variance , img , n_img):
    # path = "result/3.2.2/img_" + filter + "_var" + str(variance) + "_size" + str(size) + ".jpg"
    mse = np.square(n_img - img).mean()
    result.set_value(str(variance), filter , mse)
    # cv.imwrite(path , noised_img)

path = "image/Lena.bmp"
img = cv.imread(path , cv.IMREAD_GRAYSCALE)
noise_var = np.array([0.01 , 0.05 , 0.1])
window_sizes = np.array([3,5,7,9])

result = pd.DataFrame(index=['0.01' , '0.05' , '0.1'],columns=['nf' , 'median 3' , 'median 5' , 'median 7' , 'median 9' ,
 'box 3' , 'box 5' , 'box 7' , 'box 9'])

for noise in noise_var:

    noised_img = random_noise(img , mode="gaussian" , var = noise) 
    noised_img = np.array(noised_img * 255 , np.uint8)
    save_result('nf' , 0 , noise , img , noised_img)
    for ws in window_sizes:
        box_img = apply_box_filter(noised_img, ws)
        save_result('box ' + str(ws) , ws , noise ,img , box_img)
        median_img = apply_median_filter(noised_img , ws)
        save_result('median ' + str(ws) , ws , noise ,img , median_img)
print()
print(result)
result.to_csv("result.csv")
        


