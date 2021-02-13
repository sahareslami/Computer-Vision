import cv2 as cv
import numpy as np

def normalize(img):
    minimum = np.amin(img)
    return (img - minimum) * (255 / np.amax(img))

def apply_box_filter(img):
    w,l = img.shape
    result = np.zeros((int(w / 2) , int(l / 2)) , np.uint8)
    for i in range(0,w,2):
        for j in range(0,l,2):
            index, jndex = int(i / 2) , int(j / 2)
            result[index][jndex] = np.uint8(np.sum(img[i:i+2,j:j+2]) / 4)
    return result

def make_size(img , width , length):
    result = np.zeros((width , length) , np.uint8)   
    mid_w , mid_l = int(width / 2) , int(length / 2)
    img_w , img_l = int(img.shape[0] / 2) , int(img.shape[0] / 2)
    result[mid_w - img_w : mid_w + img_w  , mid_l - img_l : mid_l + img_l] = img
    return result

img = cv.imread("images/Lena.bmp" , cv.IMREAD_GRAYSCALE)


pyramid = img
iteration = int(np.log2(img.shape[0]))
pre_img = img
all_pixels = int(img.shape[0] * img.shape[1])

for i in range(0,iteration):
    cur_img = apply_box_filter(pre_img)
    #calcute all pixels in pyramid
    all_pixels += int(cur_img.shape[0] * cur_img.shape[1])
    all_pixels = int(img.shape[0] * img.shape[1])
    # print(iteration)
    path = "result/6.1/Lena" + str(i) + ".jpg"
    # cv.imwrite(path, cur_img)
    #add margin to image 
    margin_img = make_size(cur_img , img.shape[0] , img.shape[1])
    #add to pyramid
    pyramid =  np.vstack((pyramid,margin_img))
    pre_img = cur_img
    


for i in range(0,iteration):
    cur_img = apply_box_filter(pre_img)
    #calcute all pixels in pyramid
    all_pixels += int(cur_img.shape[0] * cur_img.shape[1])
    all_pixels = int(img.shape[0] * img.shape[1])
    #calculate laplacian
    laplacian = normalize(img - cur_img)
    # print(iteration)
    path = "result/6.1/Lena" + str(i) + ".jpg"
    # cv.imwrite(path, cur_img)
    #add margin to image 
    margin_img = make_size(cur_img , img.shape[0] , img.shape[1])
    #add to pyramid
    pyramid =  np.vstack((pyramid,margin_img))
    pre_img = cur_img 

print(all_pixels)
# cv.imwrite("result/6.1/pyramid.jpg" , pyramid)
cv.waitKey(0)
