import numpy as np
import scipy.signal as signal
from PIL import Image
from scipy.misc import imread,imresize,imsave

def im2double(im):
    info = np.iinfo(im.dtype)
    return im.astype(np.double) / info.max


def readImg(filename):
    img=im2double(imread(filename))
    if len(img.shape) == 2:
        img=img[:, :, np.newaxis]
    return img

im1 = readImg("B.png").flatten()
im2 = readImg("BB.png").flatten()
print(im1)
print(im2)
print(np.linalg.norm(im1 - im2, 2))
