import cv2 as cv
import numpy as np

def normalize(img):
    minimum = np.amin(img)
    return (img - minimum) * (255 / np.amax(img))

#compute Hue
def compute_H(pixel):
    width , height = img.shape[0:2]
    hue = np.zeros((width , height))
    for i in range (0,width):
        for j in range(0,height):
            pixel = img[i][j]
            den = ((pixel[0] - pixel[1]) ** 2 + (pixel[2] - pixel[1])*(pixel[0] - pixel[2])) ** (1/2)
            # den = ((pixel[2] - pixel[1]) ** 2 + ((pixel[0] - pixel[1]) * (pixel[2] - pixel[0]))) ** (1/2)
            exp = ((pixel[2] - pixel[1]) + (pixel[2] - pixel[0])) / 2
            if(den != 0):
                hue[i][j] = np.arccos(exp / den)
            if(pixel[0] > pixel[1]):
                hue[i][j] = 2 * np.pi - hue[i][j]
    print(hue)
    return hue


#compute Intensity
def compute_I(img):
    width , height = img.shape[0:2]
    intensity = np.zeros((width , height))
    for i in range (0,width):
        for j in range(0,height):
            pixel = img[i][j]
            intensity[i][j] = np.sum(pixel) / 3
    return intensity
    
#compute Saturation
def compute_S(img):
    width , height = img.shape[0:2]
    saturation = np.ones((width , height))
    for i in range (0,width):
        for j in range(0,height):
            pixel = img[i][j]
            saturation[i][j] = 1.00 - ((3 * np.amin(pixel)) / np.sum(pixel))
    # print(saturation)
    return saturation


img = cv.imread("image/Pepper.bmp")

img = img / 255

img_h = normalize(compute_H(img))
img_s = compute_S(img) * 255
img_i = normalize(compute_I(img))

cv.imwrite("result/5.1.1/hue3.jpg" , img_h)
cv.imwrite("result/5.1.1/saturation2.jpg" , img_s)
cv.imwrite("result/5.1.1/intensity2.jpg" , img_i)


