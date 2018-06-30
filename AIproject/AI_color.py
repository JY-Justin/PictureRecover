from PIL import Image
import numpy as np
import cv2
from sklearn import linear_model
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

im = Image.open('E.png')
#im.show()
im_array = np.array(im)
[rows, cols, channels] = im_array.shape
print(rows, cols, channels)
im_array = im_array.astype('float64')
minX = im_array.min()
maxX = im_array.max()
im_array = (im_array - minX) / (maxX - minX)
im_initial = im_array.copy()
noiseMask = np.array(im_array != 0, dtype='double')
print(noiseMask)
print(im_array)
print(im_array.dtype)
print(np.sum(noiseMask == 0))

radius = 4
#f = GaussianProcessRegressor()

f = linear_model.LinearRegression()
for _ in range(1):
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
                        window.append(im_array[i][j][channel])
                        count = count + 1
                if len(index) != 0:
                #if len(index) != 0 and count / (radius * radius) > 0.4:
                    f.fit(index, window)
                    for i in range(rowl, rowr):
                        for j in range(colu, cold):
                            if noiseMask[i][j][channel] != 0.:
                                pass
                            else:
                                im_array[i][j][channel] = f.predict([[i, j]])[0]
                else:
                    print("miss")
print(im_array)
im_array = (im_array + minX) * (maxX - minX)
print(im_array)
im = Image.fromarray(im_array.astype('uint8')).convert('RGB')
im.show()
print(np.sum(im_array == 0))
im1 = np.array(Image.open('E_ori.png')).flatten()
im2 = im_initial.flatten()
im3 = im_array.flatten()
print(('{}({}):\n'
        'Distance between original and corrupted: {}\n'
        'Distance between original and reconstructed (regression): {}'
).format('E', 0.6, np.linalg.norm(im1-im2, 2), np.linalg.norm(im1-im3, 2)))
