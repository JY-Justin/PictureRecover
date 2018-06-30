from PIL import Image
import numpy as np
from sklearn import linear_model
import cv2
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

filename = "B.png"
img = np.array(cv2.imread(filename).copy())
img_initial = np.array(cv2.imread(filename).copy())
[rows, cols, channels] = img.shape
print(rows, cols, channels)
minX = img.min()
maxX = img.max()
img = (img - minX) / (maxX - minX)
noiseMask = np.array(img != 0, dtype='double')
print(noiseMask)

radius = 3
#f = GaussianProcessRegressor()

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
                        count = count + 1
                #if len(index) != 0:
                if len(index) != 0 and count / (radius * radius) > 0.4:
                    f.fit(index, window)
                    for i in range(rowl, rowr):
                        for j in range(colu, cold):
                            if noiseMask[i][j][channel] != 0.:
                                pass
                            else:
                                img[i][j][channel] = f.predict([[i, j]])[0]
                else:
                    print("miss")

print(img)
img = img * (maxX - minX) + minX
img = np.minimum(img, 255)
img = np.maximum(img, 0)
print(img)
im1 = np.array(cv2.imread("BB.png")).flatten()
im2 = img_initial.flatten()
im3 = img.flatten()
print(('{}({}):\n'
       'Distance between original and corrupted: {}\n'
        'Distance between original and reconstructed (regression): {}'
).format('E', 0.6, np.linalg.norm(im1-im2, 2), np.linalg.norm(im1-im3, 2)))