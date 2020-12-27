import cv2 as cv
import numpy as np
from skimage.filters import unsharp_mask


a = int(input("filter: "))
b = int(input("alpha: "))
img = cv.imread("image/Lena.bmp" , cv.IMREAD_GRAYSCALE)

result = 255 * unsharp_mask(img, radius=a, amount=b)

path = "result/cheat/img" + str(a) + "_" + str(b) + ".jpg" 
path = "result/cheat/img" + ".jpg" 
cv.imwrite(path , result)

