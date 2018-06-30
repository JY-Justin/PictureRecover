from PIL import Image
import numpy as np
from sklearn import linear_model
import cv2
import scipy.signal as signal
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

im = Image.open('A.png')
#im.show()
im_array = np.array(im)
[rows, cols] = im_array.shape
print(rows, cols)
im_array = im_array.astype('float64')
minX = im_array.min()
maxX = im_array.max()
im_array = (im_array - minX) / (maxX - minX)
im_initial = im_array.copy()
noiseMask = np.array(im_array != 0, dtype='double')
print(noiseMask)
print(im_array)
print(im_array.dtype)

radius = 1

f = linear_model.LinearRegression()
for _ in range(20):
    #for channel in range(channels):
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

                if noiseMask[row][col] != 0.:
                    continue
                window = []
                index = []
                count = 0
                for i in range(rowl, rowr):
                    for j in range(colu, cold):
                        if noiseMask[i][j] == 0. and (i == row and j == col):
                            continue
                        index.append([i, j])
                        window.append(im_array[i][j])
                        #count = count + 1
                if len(index) != 0:
                    #f.fit(index, window)
                    #for i in range(rowl, rowr):
                        #for j in range(colu, cold):
                    if noiseMask[row][col] != 0.:
                        pass
                    else:
                        im_array[row][col] = sum(window) / len(window)
                else:
                    print("miss")
print(im_array)
#im_array = signal.medfilt2d(im_array, (3, 3))
print(im_array)
im_array = (im_array + minX) * (maxX - minX)
print(im_array)
im = Image.fromarray(im_array.astype('uint8')).convert('RGB')
im.show()
print(np.sum(im_array == 0))
#im1 = np.array(Image.open('E_ori.png')).flatten()
#im2 = im_initial.flatten()
#im3 = im_array.flatten()
#print(('{}({}):\n'
#       'Distance between original and corrupted: {}\n'
#        'Distance between original and reconstructed (regression): {}'
#).format('E', 0.6, np.linalg.norm(im1-im2, 2), np.linalg.norm(im1-im3, 2)))