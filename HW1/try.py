import cv2 as cv
import numpy as np
import math

theta = int(input("Enter the degree"))
theta = (np.pi / (180 / theta))
path = "image/Elaine.jpg"
img = cv.imread(path ,  cv.IMREAD_GRAYSCALE)
width , height = img.shape

# cv.imshow("image" , img)
# cv.waitKey(0)

trans = np.array([[np.cos(theta) , np.sin(theta)],[-np.sin(theta) , np.cos(theta)]])
inv_trans = np.linalg.inv(trans)

res = np.zeros((height, width) , np.uint8)
mode = input("Enter mode: \n 1 for nearest neighbour interpolation \n 2 for bilinear interpolation")
half = int(width / 2)
for r in range(int(-(width/2)) , int((width/2))):
        for c in range(-int(height / 2) , int(height/2)):

                x , y = np.dot(np.array([r,c]) , inv_trans)

                if(mode == "1"):

                        x , y = int(np.round(x) + (width / 2)) , int(np.round(y) + (height / 2))
                        if x < 0 or y < 0 or x >= width or y >= height:
                                continue
                        res[int(r + width / 2)][int(c + height / 2)] = img[x][y]

                if(mode == "2"):
                        fx , cx = int(np.floor(x)) , int(np.ceil(x))
                        fy , cy = int(np.floor(y)) , int(np.ceil(y))


                        up = np.dot(np.array([fx,fy]) ,trans) + half
                        right = np.dot(np.array([cx,fy]) ,trans )+ half
                        down =  np.dot(np.array([fx,cy]) , trans)+ half
                        left =  np.dot(np.array([cx,cy]) , trans)+ half

                        A = np.array([[up[0] , up[1] , up[1] * up[0]  , 1],
                                [down[0] , down[1] , down[1] * down[0] , 1] ,
                                [right[0] , right[1] , right[1] * right[0] , 1],
                                [left[0] , left[1] , left[1] * left[0] , 1]])
                        d = np.array(A , np.uint8)

                        if cx + half >= width or cy + half >= height or fx + half < 0 or fy + half < 0:
                                continue
                        b = np.array([[img[fx + half][fy + half]] , [img[fx + half][cy + half]] , [img[cx + half][fy + half]] , [img[cx + half][cy + half]]])
                        
                        if np.linalg.matrix_rank(A) != A.shape[0]:
                                value = img[fx + half][fy + half]
                        else:
                                coef = np.linalg.solve(A , b)
                                value = np.dot(np.array([r+half , c+ half , (r+half)*(c+half),1]) , coef) 
                        res[int(r + width / 2)][int(c + height / 2)] = int(value)
                        

name = "result/1.1.3/rotate_" + ("NNI_" if mode == "1" else "BLI_") + str((theta * 180) / np.pi) + ".jpg"
cv.imwrite(name , res)
cv.imshow("rotated" , res)
cv.waitKey(0)
                







