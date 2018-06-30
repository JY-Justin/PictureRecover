from PIL import Image
import numpy as np
from sklearn import linear_model
import cv2
import scipy.signal as signal
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

#im.show()
img = np.array(cv2.imread('E.png').copy())
img_initial = np.array(cv2.imread('E.png').copy())

[rows, cols, channels] = img.shape
img = img.astype('float64')
minX = img.min()
maxX = img.max()
img = (img - minX) / (maxX - minX)
noiseMask = np.array(img != 0, dtype='double')
print(noiseMask)
print(img)
print(img.dtype)

radius = 1

f = linear_model.LinearRegression()
for _ in range(5):
    for channel in range(channels):
        for row in range(rows):
            for col in range(cols):
                if row - radius < 0:
                    rowl = 0
                    rowr = rowl + 2 * radius
                elif row + radius >= rows:
                    rowr = rows - 1
                    rowl = rowr - 2 * radius
                else:
                    rowl = row - radius
                    rowr = row + radius

                if col - radius < 0:
                    colu = 0
                    cold = colu + 2 * radius
                elif col + radius >= cols:
                    cold = cols - 1
                    colu = cold - 2 * radius
                else:
                    colu = col - radius
                    cold = col + radius

                if noiseMask[row][col][channel] != 0.:
                    continue
                window = []
                index = []
                count = 0
                for i in range(rowl, rowr):
                    for j in range(colu, cold):
                        if noiseMask[i][j][channel] == 0. and (i == row and j == col):
                            continue
                        index.append([i, j])
                        window.append(img[i][j][channel])
                        #count = count + 1
                if len(index) != 0:
                    #f.fit(index, window)
                    #for i in range(rowl, rowr):
                        #for j in range(colu, cold):
                    if noiseMask[row][col][channel] != 0.:
                        pass
                    else:
                        img[row][col][channel] = sum(window) / len(window)
                else:
                    print("miss")
print(img)
#im_array = signal.medfilt2d(im_array, (3, 3))
print(img)
img = (img + minX) * (maxX - minX)
print(img)
im = Image.fromarray(img.astype('uint8')).convert('RGB')
im.show()
print(np.sum(img == 0))

im1 = np.array(cv2.imread('E_ori.png')).flatten()
im2 = img_initial.flatten()
im3 = img.flatten()
print(('{}({}):\n'
       'Distance between original and corrupted: {}\n'
        'Distance between original and reconstructed (regression): {}'
).format('E', 0.6, np.linalg.norm(im1-im2, 2), np.linalg.norm(im1-im3, 2)))