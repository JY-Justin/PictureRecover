#
# This is my version of linear regression
#
import numpy as np
import cv2
import matplotlib.mlab as mlab
import math
from PIL import Image

def sigmoid(x):
    return 1/(1+math.pow(math.e, 6 * x))

def linearRegression(img,noiseLevel,filename,lr):
    [h, w, c] = img.shape

    noiseMask = ((img != 0)).astype(np.double)
    # normalize
    minX = np.min(np.min(np.min(img)))
    maxX = np.max(np.max(np.max(img)))
    img = (img - minX) / (maxX - minX)

    resImg = img.copy()
    # for each channel
    for k in range(c):
        # for each point
        for i in range(h):
            for j in range(w):
                # if it is to fix
                if noiseMask[i, j, k] < 1:
                    cntBlack = 0
                    cntWhite = 0
                    dataY = []
                    distance = []
                    # scan the point arround
                    # if the fine point 's ratio is smaller than limit, skip it in this iteration
                    for m in range(-2, 3, 1):
                        for n in range(-2, 3, 1):
                            if i+m < 0 or i+m >= h or j+n < 0 or j+n >= w:
                                continue
                            if noiseMask[i+m, j+n, k] < 1:
                                cntBlack += 1
                            else:
                                cntWhite += 1
                                distance.append(sigmoid(math.sqrt(m*m+n*n)/math.sqrt(8)))
                                dataY.append(resImg[i+m, j+n, k])
                    if cntWhite*1.0/(cntWhite+cntBlack) < lr:
                        continue
                    ans = 0
                    resImg[i, j, k] = np.sum(np.array(distance)*dataY)/np.sum(distance)
                    #resImg[i,j,k]=np.sum(dataY)*1.0/cntWhite

    resImg = np.minimum(resImg, 1)
    resImg = np.maximum(resImg, 0)
    resImg *= 255
    cv2.imwrite(filename, resImg)
    return resImg




if __name__=="__main__":
    files = {"E.png": 0.8}
    #files = {"A.png": 0.8, "B.png": 0.4, "C.png": 0.6}
    cnt = 0

    for filename in files.keys():
        print(filename)
        lr = 0.6
        img = np.array(cv2.imread(filename).copy())
        img_initial = np.array(cv2.imread(filename).copy())
        print(img)
        print(img_initial)
        noiseRatio = files[filename]

        for _ in range(2400):
            if _ < 1600:
                lr = 1 - noiseRatio
            else:
                lr = 0.2
            ans = linearRegression(img, noiseRatio, "../picout/linearBlockPool/"+filename, lr)
            img = ans
        #cv2.imwrite("fix_" + filename, img)
        im1 = np.array(cv2.imread('E_ori.png')).flatten()
        im2 = img_initial.flatten()
        im3 = img.flatten()
        print(np.linalg.norm(im1-im2, 2))
        print(np.linalg.norm(im1-im3, 2))
        #print(('{}({}):\n'
        #       'Distance between original and corrupted: {}\n'
        #        'Distance between original and reconstructed (regression): {}'
        #).format(filename, 0.6, np.linalg.norm(im1-im2, 2), np.linalg.norm(im1-im3, 2)))





