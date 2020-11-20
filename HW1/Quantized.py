import cv2 as cv
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

path = "image/Barbara.jpg"
img = cv.imread(path , cv.IMREAD_GRAYSCALE)
cv.imwrite("result/1.2.1/balck_white.jpg" , img)
# cv.imshow("barbara" , img)
# cv.waitKey(0)

width , height = img.shape
hist = cv.calcHist([img] , [0] , None , [256] , [0,256])
plt.plot(hist)
# plt.savefig("result/1.2.1/histogram.png")
# plt.hist(img.ravel() , 256 , [0,256])


eqlzd_img = cv.equalizeHist(img)
eqlzd_hist = cv.calcHist([eqlzd_img] , [0] , None , [256] , [0,256])

plt.plot(eqlzd_hist)
# plt.savefig("result/1.2.1/eqlzd_histogram.png")

name = "result/1.2.1/equalized_image.jpg"
cv.imwrite(name , eqlzd_img)

# cv.imshow("equal" , eqlzd_img)
# cv.waitKey(0)

gray_level = np.array([8 , 16 , 32 , 64 , 128])
result = [[],[]]


# for normal image
for level in gray_level:
    new_img = np.uint8((np.floor(img / level)) * level)
    name = "result/1.2.1/gray_level_" + str(level) + ".jpg"
    cv.imwrite(name , new_img)
    level_hist = ``cv.calcHist([new_img] , [0] , None , [256] , [0,256])``
    # plt.figure(str(level))
    # plt.plot(level_hist)
    # name = "result/1.2.1/gray_level_" + str(level) + "_hist.png"
    # plt.savefig(name)
    result[0].append(np.square(new_img - img).mean())


# quantized equlized image
for level in gray_level:
    new_img = np.uint8((np.floor(eqlzd_img / level)) * level)
    name = "result/1.2.1/gray_level_" + str(level) + "_histeq.jpg"
    cv.imwrite(name , new_img)
    level_hist = cv.calcHist([new_img] , [0] , None , [256] , [0,256])
    plt.figure(str(level))
    plt.plot(level_hist)
    name = "result/1.2.1/gray_level_" + str(level) + "_hist_histeq.png"
    plt.savefig(name)
    result[1].append(np.square(new_img - img).mean())
    print(np.square(new_img - img).mean())

print(result)