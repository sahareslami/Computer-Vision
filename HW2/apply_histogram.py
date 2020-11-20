import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import histogram as hst

img = cv.imread("image/Camera Man.bmp" , cv.IMREAD_GRAYSCALE)
hist = cv.calcHist([img] , [] , None , [256] , [0,256])
plt.figure("built_in fucntion")
plt.plot(hist)
# plt.savefig("result/2.1.1/Built_in_histogram.png")

histo = hst.compute_histogram(img)

fig = plt.figure()

#show original image
fig.add_subplot(221)
plt.title("Camera Man image")
plt.set_cmap('gray')
plt.imshow(img)

#show histogram
fig.add_subplot(222)
plt.title('histogram')
plt.stem(histo)# , use_line_collection=True)

plt.savefig("result/2.1.1/output.png")
print("Tejnr")