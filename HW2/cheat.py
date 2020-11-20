import numpy as np
import cv2 as cv

w = int(input("window"))
nmbr = int(input("image"))
path = "image/HE" + str(nmbr) + ".jpg" 
img = cv.imread(path,cv.IMREAD_GRAYSCALE)

# create a CLAHE object (Arguments are optional).
clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(w,w))
cl1 = clahe.apply(img)

 
path = "result/2.2.1/CLAHE_HE" + str(nmbr) + "_" + str(w) + ".jpg"
cv.imwrite(path,cl1)
