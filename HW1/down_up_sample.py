import cv2 as cv
import numpy as np


path = "image/Goldhill.bmp"
img = cv.imread(path , cv.IMREAD_GRAYSCALE)
width, height = img.shape

# cv.imshow("Godhill" , img)
# cv.waitKey(0)

# down sample using averging filter
shrngked_img = np.zeros((int(width / 2) , int(height / 2)) , np.uint8)
for x in range(0, int(width / 2)):
     for y in range(0, int(height / 2)):
        shrngked_img[x][y] = (int(img[2 * x][2 * y]) + int(img[2 * x + 1][2 * y]) + int(img[2 * x][2 * y + 1]) + int(img[2 * x + 1][2 * y + 1])) / 4

name = "result/1.2.2/downSample_Avg_Filer.jpg" 
cv.imwrite(name , shrngked_img)  
# cv.imshow("GodhillAVG" , shrngked_img)
# cv.waitKey(0)

# down sample without averaging filter
shrngked_1_img = np.zeros((int(width / 2) , int(height / 2)) , np.uint8)
for x in range(0, int(width / 2)):
     for y in range(0, int(height / 2)):
        shrngked_1_img[x][y] = (int)(img[2 * x][2 * y])

name = "result/1.2.2/DownSample.jpg"

cv.imwrite(name , shrngked_1_img)
# cv.imshow("GodhillADi" , shrngked_1_img)
# cv.waitKey(0)

# upsample image using pixel replication
pr_upsample_img = np.zeros((width , height) , np.uint8)
for x in range(0 , width):
    for y in range(0 , height):
        pr_upsample_img[x][y]  = shrngked_1_img[int(np.floor(x / 2))][int(np.floor(y / 2))]

name = "result/1.2.2/upSample_pxl_replc_avg.jpg"
cv.imwrite(name , pr_upsample_img)
cv.imshow("GodhillPR" , pr_upsample_img)
cv.waitKey(0)

# upsample image using bilinear interpolation

bi_upsample_img = np.zeros((width , height) , np.uint8)

for r in range(0,width):
    for c in range(0, height):
        fx = int(np.floor(r/2)) 
        cx = int(np.ceil(r/2)) 
        fy = int(np.floor(c/2))
        cy = int(np.ceil(c/2))
        # print(r,c,fx,cx,fy,cy)
        if r % 2 == 0:
            A = np.array([[2 * fx , 2 * fy , 4*fx * fy , 1],[2 * cx + 2, 2 * fy , 4 * (cx + 1) * fy , 1],[2 * fx ,2 * cy ,4*fx*cy , 1],[2* cx + 2  ,2 * cy , 4*(cx + 1)*cy, 1]])
        if c % 2 == 0:
            A = np.array([[2 * fx , 2 * fy , 4*fx * fy , 1],[2 * cx, 2 * fy , 4 * cx  * fy , 1],[2 * fx ,2 * cy + 2 ,4*fx*(cy + 1) , 1],[2* cx + 2  ,2 * cy , 4*cx*(cy + 1), 1]])
        else:
            A = np.array([[2 * fx , 2 * fy , 4*fx * fy , 1],[2 * cx, 2 * fy , 4 * cx * fy , 1],[2 * fx ,2 * cy ,4*fx*cy , 1],[2* cx ,2 * cy , 4*cx*cy, 1]])
        if cx >= width / 2 or cy >= height / 2:
            bi_upsample_img[r][c] = shrngked_1_img[fy][fx]   
            continue
        b = np.array([[shrngked_1_img[fx][fy]],[shrngked_1_img[cx][fy]],[shrngked_1_img[fx][cy]],[shrngked_1_img[cx][cy]]])
        # print(A , b)
        if np.linalg.matrix_rank(A) != A.shape[0]:
            bi_upsample_img[r][c] = shrngked_1_img[fx][fy]
            continue
        coef = np.linalg.solve(A , b)
        color = np.dot([[r,c,r*c,1]], coef)
        bi_upsample_img[r][c] = int(color)


name = "result/1.2.2/upSamle_BLI_avg.jpg"
cv.imwrite(name , bi_upsample_img)
cv.imshow("GodhillBLI" , bi_upsample_img)
cv.waitKey(0)

result = np.array([[np.square(pr_upsample_img - img).mean()] , 
                    [np.square(bi_upsample_img - img).mean()]])
print(result)
