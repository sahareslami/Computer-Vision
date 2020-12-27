import cv2 as cv
import numpy as np
import pandas
from skimage.util import *
import pandas as pd

def apply_median_filter(img,window_size):
    w,l = img.shape
    result = np.zeros((w ,l) , np.uint8)
    hw = int(window_size / 2)

    for i in range(hw , w - hw):
        for j in range( hw,l - hw):
            result[i - hw][j - hw] = np.median(np.squeeze(np.asarray(img[i-hw:i+hw+1,j-hw:j+hw+1])))
    print(result)
    return result

def save_result(size , variance , img , n_img):
    path = "result/3.2.1/var" + str(variance) + "_size" + str(size) + ".jpg"
    mse = np.square(n_img - img).mean()
    result.set_value(str(variance), str(size) , mse)
    cv.imwrite(path , n_img)


img = cv.imread("image/Lena.bmp",cv.IMREAD_GRAYSCALE)

noise_var = np.array([0.05 , 0.1 , 0.2])
window_sizes = np.array([3,5,7,9])

result = pd.DataFrame(index=['0.05' , '0.1' , '0.2'],columns=['nf' , '3' , '5' , '7' , '9'])

for noise in noise_var:
    noise_img = random_noise(img , mode="s&p" , salt_vs_pepper = 0.1)
    noise_img = np.array(noise_img * 255 , np.uint8)    
    save_result(0 , noise , img, noise_img)
    for win in window_sizes:
        n_img = apply_median_filter(noise_img,win)
        save_result(win , noise , img , n_img)



result.to_csv("result1.csv")
print(result)
