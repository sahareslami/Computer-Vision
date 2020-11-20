import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import histogram as hst

font = {'family': 'serif',
        'color':  'darkred',
        'weight': 'normal',
        'size': 8,
        }

img = cv.imread("image/Camera Man.bmp" , cv.IMREAD_GRAYSCALE)
width , height = img.shape
hstgrm = hst.compute_histogram(img)

#save histogram
plt.figure("built_in fucntion")
plt.plot(hstgrm , c = "C1")
plt.savefig("result/2.1.2/histogram.png")

#equlized process
pdf = hstgrm / (width*height)
cdf = np.cumsum(pdf) 
equlized = np.array(cdf * 255 , np.uint8)
map_to_equlized = lambda x: equlized[x]
equlized_img = map_to_equlized(img)

cv.imwrite("result/2.1.2/outputImage.jpg",equlized_img)

equalized_hist = hst.compute_histogram(equlized_img)
plt.figure("check")
plt.plot(equalized_hist , c = "C2")
plt.savefig("result/2.1.2/EQhistogram.png")


#output
fig = plt.figure("output")
#show original image
fig.add_subplot(221)
plt.title("Camera Man image" , fontdict= font)
plt.set_cmap('gray')
plt.imshow(img)
fig.tight_layout(pad=3.0)
#show original histogram
fig.add_subplot(222)
plt.title('histogram' , fontdict= font)
plt.plot(hstgrm , c = "C1")

#show equlized image
fig.add_subplot(223)
plt.title("Equlized Camera Man image" , fontdict= font)
plt.set_cmap('gray')
plt.imshow(equlized_img)

#show equlized histogram
fig.add_subplot(224)
plt.title('histogram' , fontdict= font)
plt.plot(equalized_hist , c = "C2")

plt.savefig("result/2.1.2/output.png")
