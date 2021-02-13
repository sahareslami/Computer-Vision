import cv as cv
import numpy as np

def normalize(img):
    minimum = np.amin(img)
    return (img - minimum) * (255 / np.amax(img))

def filtering(this_img, this_filter):
    img = np.zeros(this_img.shape)
    for i in range(this_img.shape[0]):
        img[i] = this_img[i].copy()

    # compute p and q
    height, width = img.shape
    new_height, new_width = height * 2, width * 2

    # create fp
    new_img = np.zeros((new_height, new_width))
    for i in range(height):
        for j in range(width):
            new_img[i][j] = img[i][j]

    img_fft = np.fft.fft2(new_img)
    # DFT and shift
    img_fft = np.fft.fftshift(img_fft)
    # multiply filter
    img_filtering = img_fft * cv.resize(np.abs(this_filter), (new_height, new_width))
    # shift
    img_filtering = np.fft.ifftshift(img_filtering)
    # IDFT and real
    img_ifft = np.real(np.fft.ifft2(img_filtering))
    # resize image
    final_img = np.zeros((height, width))
    for i in range(height):
        for j in range(width):
            final_img[i][j] = img_ifft[i][j]
    return final_img

out_path = "result/4.1.1/"
barbara_gray = cv.imread('image/Barbara.bmp', cv.IMREAD_GRAYSCALE)
cv.imwrite(out_path + 'barbara_gray.jpg', barbara_gray)


# a filter
a = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]])
a = a / 16

# separate a
a_separate_c = np.array([1, 2, 1]) 
a_separate_c = a_separate_c / 4
a_separate_r = np.array([[1, 2, 1]])
a_separate_r = a_separate_r / 4

# b filter
b = np.array([[-1, -1, -1],[-1, 8, -1],[-1, -1, -1]])

# c filter
c = np.array([[0, -1, 0],[-1, 5, -1], [0, -1, 0]])

# compute FT of filters
a_f = np.fft.fft2(a)
b_f = np.fft.fft2(b)
c_f = np.fft.fft2(c)
a_separate_c_f = np.fft.fft(a_separate_c)
a_separate_r_f = np.fft.fft(a_separate_r)

# shift filters
a_shift = np.fft.fftshift(a_f)
b_shift = np.fft.fftshift(b_f)
c_shift = np.fft.fftshift(c_f)
a_separate_c_shift = np.abs(np.fft.fftshift(a_separate_c_f))
a_separate_r_shift = np.abs(np.fft.fftshift(a_separate_r_f))

# compute logarithm of filters
a_log = np.log(np.abs(a_shift) + 1)
b_log = np.log(np.abs(b_shift) + 1)
c_log = np.log(np.abs(c_shift) + 1)
a_separate_c_log = np.log(a_separate_c_shift + 1)
a_separate_r_log = np.log(a_separate_r_shift + 1)

# normalize filters
a_log_normalize = normalize(a_log)
b_log_normalize = normalize(b_log)
c_log_normalize = normalize(c_log)
a_separate_c_log_normalize = normalize(a_separate_c_log)
a_separate_r_log_normalize = normalize(a_separate_r_log)

# resize filters to show
a_separate_c_log_f = cv.resize(a_separate_c_log_normalize, (512, 512))
a_separate_r_log_f = cv.resize(a_separate_r_log_normalize, (512, 512))
a_log = cv.resize(a_log_normalize, (512, 512))
b_log = cv.resize(b_log_normalize, (512, 512))
c_log = cv.resize(c_log_normalize, (512, 512))

# write filters
cv.imwrite(out_path + 'a.jpg', a_log)
cv.imwrite(out_path + 'b.jpg', b_log) 
cv.imwrite(out_path + 'c.jpg', c_log)
cv.imwrite(out_path + 'a_separate_c.jpg', a_separate_c_log_f)
cv.imwrite(out_path + 'a_separate_r.jpg', a_separate_r_log_f)

'''
# read filters
a_fft = cv.imread('a.jpg', 0)
b_fft = cv.imread('b.jpg', 0) 
c_fft = cv.imread('c.jpg', 0) 
a_separate_c_fft = cv.imread('a_separate_c.jpg', 0)
a_separate_r_fft = cv.imread('a_separate_r.jpg', 0)

# show filters
cv.imshow('a', a_fft)
cv.imshow('b', b_fft)
cv.imshow('c', c_fft)
cv.imshow('a_separate_c',a_separate_c_fft)
cv.imshow('a_separate_r',a_separate_r_fft)
'''

# apply a filter
final_img_a = filtering(barbara_gray, a_shift)
cv.imwrite(out_path + 'barbara_a.jpg', final_img_a)

# apply b filter
final_img_b = filtering(barbara_gray, b_shift)
cv.imwrite(out_path + 'barbara_b.jpg', final_img_b)

# apply c filter
final_img_c = filtering(barbara_gray, c_shift)
cv.imwrite(out_path + 'barbara_c.jpg', final_img_c)



# apply a separate filters
img_a_separate_c = filtering(barbara_gray, a_separate_c_shift)
img_a_separate_r = filtering(barbara_gray, a_separate_r_shift)
cv.imwrite(out_path + 'barbara_a_separate_c.jpg', img_a_separate_c)
cv.imwrite(out_path + 'barbara_a_separate_r.jpg', img_a_separate_r)


final_img_for_seperate_filter = filtering(img_a_separate_c, a_separate_r_shift)
cv.imwrite(out_path + 'barbara_a_separate_final.jpg', final_img_for_seperate_filter) 
