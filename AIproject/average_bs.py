import numpy as np
import cv2
import math
from sklearn import linear_model


def sigmoid(x):
    return 1 / (1 + math.pow(math.e, 5 * x))


def relu(x):
    return max(x, 0)


def tanh(x):
    return (math.exp(x) - math.exp(-x)) / (math.exp(x) + math.exp(-x))


def softmax(x_vector):
    sum_x = 0
    for x in x_vector:
        sum_x = sum_x + math.exp(x)
    x1 = [math.exp(x) / sum_x for x in x_vector]
    return x1


def mean_sigmoid(img, limit):
    [rows, cols, channels] = img.shape
    noiseMask = np.array(img != 0, dtype='double')
    minX = img.min()
    maxX = img.max()
    img = (img - minX) / (maxX - minX)
    resImg = img.copy()
    for channel in range(channels):
        for row in range(rows):
            for col in range(cols):
                if noiseMask[row, col, channel] < 1:
                    num_error = 0
                    num_right = 0
                    window = []
                    index = []
                    for i in range(-int(limit*15), int(limit*15+1)):
                        for j in range(-int(limit*15), int(limit*15+1)):
                            if row + i < 0 or row + i >= rows or col + j < 0 or col + j >= cols:
                                continue
                            if noiseMask[row + i, col + j, channel] < 1:
                                num_error = num_error + 1
                            else:
                                num_right = num_right + 1
                                index.append(sigmoid(math.sqrt(i * i+j * j)))
                                window.append(resImg[row + i, col + j, channel])
                    if num_right * 1.0 / (num_right + num_error) < limit:
                        continue
                    resImg[row, col, channel] = np.sum(np.array(index)*window)/np.sum(index)
    resImg = resImg * (maxX - minX) + minX
    resImg = np.minimum(resImg, 255)
    resImg = np.maximum(resImg, 0)
    return resImg, noiseMask


def mean_normal(img, noiseMask):
    [rows, cols, channels] = img.shape
    minX = img.min()
    maxX = img.max()
    img = (img - minX) / (maxX - minX)
    #noiseMask = np.array(img != 0, dtype='double')
    resImg = img.copy()
    radius = 1
    for _ in range(20):
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
                            window.append(resImg[i][j][channel])
                            # count = count + 1
                    if len(index) != 0:
                        # f.fit(index, window)
                        # for i in range(rowl, rowr):
                        # for j in range(colu, cold):
                        if resImg[row][col][channel] - sum(window) / len(window) < 0.0001:
                            pass
                        else:
                            resImg[row][col][channel] = sum(window) / len(window)
                    else:
                        print("miss")
    resImg = resImg * (maxX - minX) + minX
    resImg = np.minimum(resImg, 255)
    resImg = np.maximum(resImg, 0)
    return resImg

if __name__ == '__main__':
    files = {"A.png": 0.8, "B.png": 0.4, "C.png": 0.6}
    for filename in files.keys():
        print(filename)
        limit = 0.6
        img = np.array(cv2.imread(filename).copy())
        img_initial = np.array(cv2.imread(filename).copy())
        noiseRatio = files[filename]
        #img = mean_normal(img)

        for i in range(2400):
            if i < 1600:
                limit = 1 - noiseRatio
            else:
                limit = 0.2
            ans, noiseMask = mean_sigmoid(img, limit)
            img = ans
        img = mean_normal(img, noiseMask)
        if filename == "A.png":
            print('A')
            im1 = np.array(cv2.imread('AA.png')).flatten()
            #cv2.imwrite("fix_final_A.png", img)
        elif filename == "B.png":
            print('B')
            im1 = np.array(cv2.imread('BB.png')).flatten()
            #cv2.imwrite("fix_final_B.png", img)
        else:
            print('C')
            im1 = np.array(cv2.imread('CC.png')).flatten()
            #cv2.imwrite("fix_final_C.png", img)
        im2 = img_initial.flatten()
        im3 = img.flatten()
        print(np.linalg.norm(im1 - im2, 2))
        print(np.linalg.norm(im1 - im3, 2))
