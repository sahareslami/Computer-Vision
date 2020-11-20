import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

def get_histogram(img):
    
    histogram = np.zeros((256) , int)
    for i in range(0 , img.shape[0]):
        for j in range(0 , img.shape[1]):
            histogram[img[i][j]] += 1
    return histogram

def equlization(histogram , img):
    size = img.shape[0] * img.shape[1]
    pdf = histogram / size
    cdf = np.cumsum(pdf)
    equlized = np.array(cdf * 255 , np.uint8)
    return lambda x: equlized[x]
    
def get_rank(img , intesity):
    rank = 0
    for i in range(0,img.shape[0]):
        for j in range(0,img.shape[1]):
            if img[i][j] <= intesity:
                rank = rank + 1
    return rank


def padding(img , w):
    x,y = img.shape
    pad_img = np.zeros((x + w, y + w) , np.uint8)

    h_w = int(w / 2)
    pad_img[h_w:x+h_w,h_w:y+h_w] = img
    for i in range(0, h_w):
        pad_img[i][h_w:y + h_w] = img[0]
        pad_img[x + i + h_w][h_w:y + h_w] = img[x - 1]


    for j in range(0 , h_w):
        pad_img[:, j] = pad_img[:,h_w]
        pad_img[:,j + y + h_w] = pad_img[:,y - 1 + h_w] 

    return pad_img


def AHE_v1(img , window):
    x,y = img.shape
    hw = int(window / 2)
    res = np.zeros((x - 2*hw , y - 2*hw) , np.uint8)

    for i in range(hw ,  x - hw ):
        for j in range(hw , y - hw ):
            rank  = get_rank(img[i-hw:i+hw,j-hw:j+hw], img[i][j])
            res[i - hw][j - hw] = np.uint8((rank * 255) / (window * window))
    return res


def AHE_v2(img , window):
    x,y = img.shape
    hw = int(window / 2)
    res = np.zeros((x  , y) , np.uint8)

    for i in range(hw ,  x - hw + 1 , hw):
        for j in range(hw , y - hw + 1, hw):
            histogram = get_histogram(img[i-hw:i+hw,j-hw:j+hw])
            map_to_eqlz = equlization(histogram , img[i-hw:i+hw,j-hw:j+hw])
            res[i-hw:i+hw,j-hw:j+hw] = map_to_eqlz(img[i-hw:i+hw,j-hw:j+hw])
    return res


def HE(img):
    res = np.zeros((img.shape) , np.uint8)
    hstgm = get_histogram(img)
    trans = equlization(hstgm , img)
    res = trans(img)
    return res
 
def clahe(img, win_shape):
    largest_dim = np.argmax(win_shape)
    img = np.swapaxes(img, 0, largest_dim)
    win_shape = list(win_shape)
    win_shape[0], win_shape[largest_dim] = win_shape[largest_dim], win_shape[0]
    img = np.pad(img, [((sz - 1) // 2, sz // 2) for sz in win_shape], "reflect")
    if _fast:
        if img.ndim == 2:
            res = _clahe_impl.clahe(
                img[..., None], *win_shape, 1, clip_limit)[..., 0]
        elif img.ndim == 3:
            res = _clahe_impl.clahe(img, *win_shape, clip_limit)

    else:
        res = clahe_nd(img, win_shape, clip_limit)
    return np.swapaxes(
        res[tuple(np.s_[(sz - 1) // 2 : -(sz // 2) or None]
                  for sz in win_shape)],0, largest_dim)
    


img_nmbr = int(input("Enter your input image: "))
window_size = int(input("Enter window size: "))
typee = input("Enter the method: ")


path = "image/HE" + str(img_nmbr) + ".jpg"
img = cv.imread(path , cv.IMREAD_GRAYSCALE)

pad_img = padding(img , window_size)


if(typee == "global"):
    path = "result/2.2.1/Global" + str(img_nmbr) + ".jpg"
    cv.imwrite(path, HE(img))
if(typee == "v1"):
    path = "result/2.2.1/original" + str(img_nmbr) + "-" + str(window_size) + ".jpg"
    cv.imwrite(path, AHE_v1(pad_img , window_size))
if(typee == "v2"):
    path = "result/2.2.1/fake" + str(img_nmbr) + "-" + str(window_size) + ".jpg"
    cv.imwrite(path, AHE_v2(img , window_size))

if(typee == "clahe"):
        path = "result/2.2.1/clahe" + str(img_nmbr) + "-" + str(window_size) + ".jpg"
    cv.imwrite(path, clahe(img , window_size))

