import cv as cv
import numpy as np
def normalize(img):
    minimum = np.amin(img)
    return (img - minimum) * (255 / np.amax(img))


out_path = "result/4.1.2/"
baboon = cv.imread('image/Baboon.bmp', cv.IMREAD_GRAYSCALE)
f16 = cv.imread('image/F16.bmp', 0)
lena = cv.imread('image/Lena.bmp', 0)

cv.imwrite(out_path + 'baboon_gray.jpg', baboon)
cv.imwrite(out_path + 'f16_gray.jpg', f16)
cv.imwrite(out_path + 'lena_gray.jpg', lena)

# compute DFT of images
baboon_f = np.abs(np.fft.fft2(baboon)) 
f16_f = np.abs(np.fft.fft2(f16)) 
lena_f = np.abs(np.fft.fft2(lena))

# shift
baboon_f_shift = np.fft.fftshift(baboon_f)
f16_f_shift = np.fft.fftshift(f16_f) 
lena_f_shift = np.fft.fftshift(lena_f)

# log on no shifted image
baboon_f_log = np.log(baboon_f + 1) 
f16_f_log = np.log(f16_f + 1) 
lena_f_log = np.log(lena_f + 1)

# log on shifted image
baboon_f_shift_log = np.log(baboon_f_shift + 1)
f16_f_shift_log = np.log(f16_f_shift + 1) 
lena_f_shift_log = np.log(lena_f_shift + 1)

# write images
# no shift and no log

cv.imwrite(out_path + 'baboon_f.jpg', normalize(baboon_f))
cv.imwrite(out_path + 'f16_f.jpg', normalize(f16_f))
cv.imwrite(out_path + 'lena_f.jpg', normalize(lena_f))

# shift and no log
cv.imwrite(out_path + 'baboon_f_shift.jpg', normalize(baboon_f_shift)) 
cv.imwrite(out_path + 'f16_f_shift.jpg', normalize(f16_f_shift)) 
cv.imwrite(out_path + 'lena_f_shift.jpg', normalize(lena_f_shift))


# no shift and log
cv.imwrite(out_path + 'baboon_f_log.jpg', normalize(baboon_f_log))
cv.imwrite(out_path + 'f16_f_log.jpg', normalize(f16_f_log)) 
cv.imwrite(out_path + 'lena_f_log.jpg', normalize(lena_f_log))

# shift and log
cv.imwrite(out_path + 'baboon_f_shift_log.jpg', normalize(baboon_f_shift_log)) 
cv.imwrite(out_path + 'f16_f_shift_log.jpg', normalize(f16_f_shift_log)) 
cv.imwrite(out_path + 'lena_f_shift_log.jpg', normalize(lena_f_shift_log))

# read images
baboon_f = cv.imread(out_path + 'baboon_f.jpg', cv.IMREAD_GRAYSCALE)
f16_f = cv.imread(out_path + 'f16_f.jpg', cv.IMREAD_GRAYSCALE) 
lena_f = cv.imread(out_path + 'lena_f.jpg', cv.IMREAD_GRAYSCALE)

baboon_f_shift = cv.imread(out_path + 'baboon_f_shift.jpg', 0) 
f16_f_shift = cv.imread(out_path + 'f16_f_shift.jpg', 0) 
lena_f_shift = cv.imread(out_path + 'lena_f_shift.jpg', 0)
baboon_f_log = cv.imread(out_path + 'baboon_f_log.jpg', 0) 
f16_f_log = cv.imread(out_path + 'f16_f_log.jpg', 0)
lena_f_log = cv.imread(out_path + 'lena_f_log.jpg', 0)
baboon_f_shift_log = cv.imread(out_path + 'baboon_f_shift_log.jpg', 0) 
f16_f_shift_log = cv.imread(out_path + 'f16_f_shift_log.jpg', 0) 
lena_f_shift_log = cv.imread(out_path + 'lena_f_shift_log.jpg', 0)


 