import cv2 as cv
import numpy as np
import math
def normalize(img):
    minimum = np.amin(img)
    return (img - minimum) * (255 /np.amax(img))

def a(img_fft, t):
    img_copy = img_fft.copy()
    height, width = img_fft.shape
    minimum = (1 - t) * height
    maximum = t * height
    for i in range(height):
        for j in range(width):
            if (minimum > i > maximum) and (minimum > j > maximum):
                img_copy[i][j] = 0
    return img_copy

def b1(img_fft, t):
    img_copy = img_fft.copy()
    height, width = img_fft.shape
    maximum = t * height
    minimum = (1 - t) * height
    for i in range(height):
        for j in range(width):
            if (maximum >= i >= 0) and (maximum >= j >= 0):
                img_copy[i][j] = 0
    return img_copy

def b1_no_zero(img_fft, t):
    img_copy = img_fft.copy()
    height, width = img_fft.shape
    maximum = t * height
    for i in range(height):
        for j in range(width):
            if (maximum >= i >= 0) and (maximum >= j >= 0) and (i != 0 or j != 0):
                img_copy[i][j] = 0
    return img_copy

def b2(img_fft, t):
    img_copy = img_fft.copy()
    height, width = img_fft.shape
    maximum = t * height
    minimum = (1 - t) * height
    for i in range(height):
        for j in range(width):
            if (maximum >= i >= 0) and (height - 1 >= j >= minimum):
                img_copy[i][j] = 0
    return img_copy

def b3(img_fft, t):
    img_copy = img_fft.copy()
    height, width = img_fft.shape
    maximum = t * height
    minimum = (1 - t) * height
    for i in range(height):
        for j in range(width):
            if (height - 1 >= i >= minimum) and (maximum >= j >= 0):
                img_copy[i][j] = 0
    return img_copy


def b4(img_fft, t):
    img_copy = img_fft.copy()
    height, width = img_fft.shape
    minimum = (1 - t) * height
    for i in range(height):
        for j in range(width):
            if (height - 1 >= i >= minimum) and (height - 1 >= j >= minimum):
                img_copy[i][j] = 0
    return img_copy

def inverse_transform(img, phase, name):
    img_i = np.real(np.fft.ifft2(img * math.e ** (1j * phase)))
    cv.imwrite("result/4.2.2/" + name, normalize(img_i)) 
    img_final = cv.imread(name, 0) 
    return img_final

out_path = "result/4.2.2/"
lena_gray = cv.imread('image/Lena.bmp', 0)
cv.imwrite(out_path + 'lena_gray.jpg', lena_gray)
lena_fft = np.fft.fft2(lena_gray) 
magnitude = np.abs(lena_fft) 
phase = np.angle(lena_fft)
# cv.imshow('m', normalize(magnitude))

'''
# a , T = 1/4
lena_a_4 = a(magnitude, 0.25)
lena_i_a_4 = inverse_transform(lena_a_4, phase, 'lena_a_4.jpg')

# write magnitude after apply change
cv.imwrite(out_path + 'a_4_fourier.jpg', normalize(np.log(lena_a_4 + 1)))
cv.imwrite(out_path + 'a_4_fourier_shift.jpg', normalize(np.log(np.fft.ifftshift(lena_a_4) + 1)))
# a , T = 1/8
lena_a_8 = a(magnitude, 0.125) 
lena_i_a_8 = inverse_transform(lena_a_8, phase, 'lena_a_8.jpg')
cv.imwrite(out_path + 'a_8_fourier.jpg', normalize(np.log(lena_a_8 + 1)))

cv.imwrite(out_path + 'a_8_fourier_shift.jpg', normalize(np.log(np.fft.ifftshift(lena_a_8) + 1)))

# b , i , T = 1/4
lena_b1_4 = b1(magnitude, 0.25)
lena_i_b1_4 = inverse_transform(lena_b1_4, phase, 'lena_b1_4.jpg')
cv.imwrite(out_path + 'b1_4_fourier.jpg', normalize(np.log(lena_b1_4 + 1))) 
cv.imwrite(out_path + 'b1_4_fourier_shift.jpg', normalize(np.log(np.fft.ifftshift(lena_b1_4) + 1)))

# b , i , T = 1/8
lena_b1_8 = b1(magnitude, 0.125) 
lena_i_b1_8 = inverse_transform(lena_b1_8, phase, 'lena_b1_8.jpg')
cv.imwrite(out_path + 'b1_8_fourier.jpg', normalize(np.log(lena_b1_8 + 1))) 
cv.imwrite(out_path + 'b1_8_fourier_shift.jpg', normalize(np.log(np.fft.ifftshift(lena_b1_8) + 1)))

# b , i , not considering [0, 0]
lena_b1_4_no_zero = b1_no_zero(magnitude, 0.25) 
lena_i_b1_4_no_zero = inverse_transform(lena_b1_4_no_zero, phase, 'lena_b1_4_no_zero.jpg')

# b , i , not considering [0, 0]
lena_b1_8_no_zero = b1_no_zero(magnitude, 0.125) 
lena_i_b1_8_no_zero = inverse_transform(lena_b1_8_no_zero, phase, 'lena_b1_8_no_zero.jpg')

# b , ii , T = 1/4

lena_b2_4 = b2(magnitude, 0.25) 
lena_i_b2_4 = inverse_transform(lena_b2_4, phase, 'lena_b2_4.jpg')
cv.imwrite(out_path + 'b2_4_fourier.jpg', normalize(np.log(lena_b2_4 + 1))) 
cv.imwrite(out_path + 'b2_4_fourier_shift.jpg', normalize(np.log(np.fft.ifftshift(lena_b2_4) + 1)))

# b , ii , T = 1/8
lena_b2_8 = b2(magnitude, 0.125)
lena_i_b2_8 = inverse_transform(lena_b2_8, phase, 'lena_b2_8.jpg')
cv.imwrite(out_path + 'b2_8_fourier.jpg', normalize(np.log(lena_b2_8 + 1))) 
cv.imwrite(out_path + 'b2_8_fourier_shift.jpg', normalize(np.log(np.fft.ifftshift(lena_b2_8) + 1)))

# b , iii , T = 1/4
lena_b3_4 = b3(magnitude, 0.25) 
lena_i_b3_4 = inverse_transform(lena_b3_4, phase,'lena_b3_4.jpg')
cv.imwrite(out_path + 'b3_4_fourier.jpg', normalize(np.log(lena_b3_4 + 1)))
cv.imwrite(out_path + 'b3_4_fourier_shift.jpg', normalize(np.log(np.fft.ifftshift(lena_b3_4) + 1)))

# b , iii , T = 1/8
lena_b3_8 = b3(magnitude, 0.125) 
lena_i_b3_8 = inverse_transform(lena_b3_8, phase, 'lena_b3_8.jpg')
cv.imwrite(out_path + 'b3_8_fourier.jpg', normalize(np.log(lena_b3_8 + 1))) 
cv.imwrite(out_path + 'b3_8_fourier_shift.jpg', normalize(np.log(np.fft.ifftshift(lena_b3_8) + 1)))

# b , iv , T = 1/4
lena_b4_4 = b4(magnitude, 0.25) 
lena_i_b4_4 = inverse_transform(lena_b4_4, phase, 'lena_b4_4.jpg')
cv.imwrite(out_path + 'b4_4_fourier.jpg', normalize(np.log(lena_b4_4 + 1)))
cv.imwrite(out_path + 'b4_4_fourier_shift.jpg', normalize(np.log(np.fft.ifftshift(lena_b4_4) + 1)))

# b , iv , T = 1/8
lena_b4_8 = b4(magnitude, 0.125) 
lena_i_b4_8 = inverse_transform(lena_b4_8, phase, 'lena_b4_8.jpg')
cv.imwrite(out_path + 'b4_8_fourier.jpg', normalize(np.log(lena_b4_8 + 1)))
cv.imwrite(out_path + 'b4_8_fourier_shift.jpg', normalize(np.log(np.fft.ifftshift(lena_b4_8) + 1)))
'''

# b , all , T = 1/4
lena_all_4 = b4(b3(b2(b1(magnitude, 0.25) , 0.25) , 0.25) , 0.25)
lena_all_4_s = inverse_transform(lena_all_4, phase, 'lena_b_all_4.jpg')
cv.imwrite(out_path + 'b_all_4.jpg', normalize(np.log(lena_all_4 + 1)))
cv.imwrite(out_path + 'b_all_4_shift.jpg', normalize(np.log(np.fft.ifftshift(lena_all_4) + 1)))


# b , all , T = 1/8
lena_all_8 = b4(b3(b2(b1_no_zero(magnitude, 0.125) , 0.125) , 0.125) , 0.125)
lena_all_8_s = inverse_transform(lena_all_8, phase, 'lena_b_all_8.jpg')
cv.imwrite(out_path + 'b_all_8.jpg', normalize(np.log(lena_all_8 + 1)))
cv.imwrite(out_path + 'b_all_8_shift.jpg', normalize(np.log(np.fft.ifftshift(lena_all_8) + 1)))