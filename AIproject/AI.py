from PIL import Image
import numpy as np
from sklearn import linear_model
import cv2
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

radius = 3
#f = GaussianProcessRegressor()

f = linear_model.LinearRegression()
for _ in range(5):
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
                    count = count + 1
            #if len(index) != 0:
            if len(index) != 0 and count / (radius * radius) > 0.4:
                f.fit(index, window)
                for i in range(rowl, rowr):
                    for j in range(colu, cold):
                        if noiseMask[i][j] != 0.:
                            pass
                        else:
                            im_array[i][j] = f.predict([[i, j]])[0]
            else:
                print("miss")

print(im_array)
im_array = im_array * (maxX - minX) + minX
im_array = np.minimum(im_array, 255)
im_array = np.maximum(im_array, 0)
print(im_array)
im = Image.fromarray(im_array)
im.show()
im1 = np.array(Image.open('AA.png')).flatten()
im2 = im_initial.flatten()
im3 = im_array.flatten()
print(('{}({}):\n'
       'Distance between original and corrupted: {}\n'
        'Distance between original and reconstructed (regression): {}'
).format('E', 0.6, np.linalg.norm(im1-im2, 2), np.linalg.norm(im1-im3, 2)))