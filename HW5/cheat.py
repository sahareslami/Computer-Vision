from sklearn.cluster import MiniBatchKMeans
import numpy as np
import cv2
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

image = cv2.imread("image/Girl.bmp")
(h, w) = image.shape[:2]
image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
image = image.reshape((image.shape[0] * image.shape[1], 3))
clt = MiniBatchKMeans(8)
labels = clt.fit_predict(image)
quant = clt.cluster_centers_.astype("uint8")[labels]
quant = quant.reshape((h, w, 3))
image = image.reshape((h, w, 3))
quant = cv2.cvtColor(quant, cv2.COLOR_LAB2BGR)
image = cv2.cvtColor(image, cv2.COLOR_LAB2BGR)
print("mse" , MSE(image , quant))
print("psnr" , PSNR(image , quant))