from PIL import Image
import cv2
import numpy as np
from sklearn import linear_model
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

im = Image.open('B.png')
#im.show()
im_array = np.array(im)
[rows, cols, channels] = im_array.shape
print(rows, cols, channels)
im_array = im_array.astype('float64')
minX = im_array.min()
maxX = im_array.max()
im_array = (im_array - minX) / (maxX - minX)
im_initial = im_array.copy()
noiseMask = np.zeros((rows, cols, channels))
for i in range(rows):
    for j in range(cols):
        for k in range(channels):
            if im_array[i][j][k] > 0:
                noiseMask[i][j][k] = 1
            else:
                noiseMask[i][j][k] = 0
print(noiseMask)
print(im_array)
print(im_array.dtype)
f = GaussianProcessRegressor()
for i in range(channels):
    for j in range(rows - 4):
        for k in range(cols - 4):
            window = []
            index = []
            for m in range(5):
                for n in range(5):
                    if im_array[j + m][k + n][i] > 0:
                        window.append(im_array[j + m][k + n][i])
                        index.append([m + 1, n + 1])
                    else:
                        pass
            if len(index) != 0:
                f.fit(index, window)
                for k in range(3):
                    for l in range(3):
                        if im_array[j + m][k + n][i] > 0 and \
                                im_array[j + m][k + n][i] == im_initial[j + m][k + n][i]:
                            pass
                        else:
                            im_array[j + m][k + n][i] = f.predict([[m + 1, n + 1]])[0]
            else:
                print("miss")

print(im_array)
im_array = (im_array + minX) * (maxX - minX)
print(im_array)
cv2.imwrite("C_fix.png", im_array)
#im = Image.fromarray(im_array)
#im.show()
