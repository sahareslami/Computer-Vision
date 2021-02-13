import matplotlib.pyplot as plt

from skimage.restoration import (denoise_wavelet, estimate_sigma)
import cv2 as cv
from skimage.util import random_noise
from skimage.metrics import peak_signal_noise_ratio

def apply_median_filter(img,window_size):
    w,l = img.shape
    result = np.zeros((w ,l) , np.uint8)
    hw = int(window_size / 2)

    for i in range(hw , w - hw):
        for j in range( hw,l - hw):
            result[i - hw][j - hw] = np.median(np.squeeze(np.asarray(img[i-hw:i+hw+1,j-hw:j+hw+1])))
    print(result)
    return result

original = cv.imread("images/Lena.bmp" , cv.IMREAD_GRAYSCALE)

salf_paper = noise_img = random_noise(img , mode="s&p" , salt_vs_pepper = 0.1)
gussian_noise =  noised_img = random_noise(img , mode="gaussian" , var = 0.01) 
sigma = 0.12
noisy = random_noise(original, var=sigma**2)

fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(8, 5),
                       sharex=True, sharey=True)

plt.gray()


sigma_est = estimate_sigma(noisy, multichannel=True, average_sigmas=True)


im_bayes = denoise_wavelet(noisy, multichannel=True, convert2ycbcr=True,
                           method='BayesShrink', mode='soft',
                           rescale_sigma=True)
im_visushrink = denoise_wavelet(noisy, multichannel=True, convert2ycbcr=True,
                                method='VisuShrink', mode='soft',
                                sigma=sigma_est, rescale_sigma=True)


im_visushrink2 = denoise_wavelet(noisy, multichannel=True, convert2ycbcr=True,
                                 method='VisuShrink', mode='soft',
                                 sigma=sigma_est/2, rescale_sigma=True)
im_visushrink4 = denoise_wavelet(noisy, multichannel=True, convert2ycbcr=True,
                                 method='VisuShrink', mode='soft',
                                 sigma=sigma_est/4, rescale_sigma=True)


psnr_noisy = peak_signal_noise_ratio(original, noisy)
psnr_bayes = peak_signal_noise_ratio(original, im_bayes)
psnr_visushrink = peak_signal_noise_ratio(original, im_visushrink)
psnr_visushrink2 = peak_signal_noise_ratio(original, im_visushrink2)
psnr_visushrink4 = peak_signal_noise_ratio(original, im_visushrink4)

plt.show()