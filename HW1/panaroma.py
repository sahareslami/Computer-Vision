import cv2 as cv
import numpy as np

img1 = cv.imread("image/Car1.jpg")
img2 = cv.imread("image/Car2.jpg")
height , width , channel = img1.shape

roi_1 = np.array([[389, 545], [399, 802], [389, 941], [344, 456], [475, 940], [459, 560]]) 
roi_2 = np.array([[406, 122], [418, 383],[409, 513], [357, 25], [490, 511], [479, 136]])

A = np.empty((0,6), np.uint8)
# for x,y in roi_2[2:8]:
for x,y in roi_2:
    A = np.vstack((A , np.array([x ,y ,x*y,x*x,y*y,1])))

B = np.empty((0,6) , np.uint8)
# for x,y in roi_1[2:8]:
for x,y in roi_1:
    B = np.vstack((B , np.array([x , y , x*y ,x*x,y*y, 1])))

trans = np.linalg.solve(A , B)
inv_trans = np.linalg.inv(trans)

left = np.array(np.dot(np.array([0,0,0,0,0,1]) , trans),np.int64)
right = np.array(np.dot(np.array([height - 1,0,0,(height - 1) ** 2 ,0,1]) , trans),np.int64)
up = np.array(np.dot(np.array([0,width - 1 ,0,0,(width - 1) ** 2 ,1]) , trans),np.int64)
down = np.array(np.dot(np.array([height - 1,width - 1,(width - 1)*(height - 1),(height - 1)**2,(width - 1) ** 2,1]) , trans),np.int64)

min_h = min(left[0] , right[0] , up[0] , down[0])
max_h = max(left[0] , right[0] , up[0] , down[0])
min_w = min(left[1] , right[1] , up[1] , down[1])
max_w = max(left[1] , right[1] , up[1] , down[1])

reg_img_2 = np.zeros((1200 ,1540 , 3) , np.uint8)

for i in range(min_h,max_h):
    for j in range(min_w, max_w):
        pixel = np.array([i , j , i*j , i*i , j*j , 1], np.int64)
        target = np.dot(pixel, inv_trans)
        x , y = int(np.floor(target[0])) , int(np.floor(target[1]))
        if x >= height or x < 0 or y < 0 or  y >= width:
            continue    
        reg_img_2[i - min_h][j] = img2[x][y]

cv.imwrite("result/1.1.2/registered_2.jpg" , reg_img_2)
cv.imshow("registerred" , reg_img_2)

for i in range(0,height):
    for j in range(0,width):
        reg_img_2[i - min_h][j] = img1[i][j]

cv.imwrite("result/1.1.2/panaroma_2.jpg" , reg_img_2)
cv.imshow("reg" , reg_img_2)
cv.waitKey(0)
