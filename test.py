import cv2
import numpy as np
from numpy import inf
import pandas as pd
import os
import matplotlib.pyplot as plt
from skimage import exposure, transform
from skimage import img_as_ubyte
import csv
from skimage.feature import hog
from sklearn.model_selection  import train_test_split
import classifiers as cs
import phog as PHOG
from os import walk, getcwd
import pywt
import mahotas as mt
import pickle
import keras.models
from sklearn.externals import joblib
from scipy import stats
from mahotas.features import surf
from skimage.filters import gaussian
from skimage.segmentation import active_contour

img_rows = 224
img_cols = 224

def manual_canny(img):
    edges = cv2.Canny(img,170,200)
    return edges

def auto_canny(image, sigma=0.33):
    blurred = cv2.GaussianBlur(image, (3, 3), 0)
    wide = cv2.Canny(blurred, 10, 200)
    tight = cv2.Canny(blurred, 225, 250)
    v = np.median(blurred)
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv2.Canny(image, lower, upper)
    return edged

def Dilated_Canny(image):
    gray = cv2.GaussianBlur(image, (7, 7), 0)
    edged = cv2.Canny(gray, 50, 100)
    edged = cv2.dilate(edged, None, iterations=1)
    edged = cv2.erode(edged, None, iterations=1)
    #cv2.imshow("edge", edged)
    #cv2.waitKey()
    #exit()
    return edged

def adjust_gamma(image, gamma=1.0):

   invGamma = 1.0 / gamma
   table = np.array([((i / 255.0) ** invGamma) * 255
      for i in np.arange(0, 256)]).astype("uint8")

   return cv2.LUT(image, table)

def preprocess(img):
    ##Resize
    img = cv2.resize(img, (img_rows, img_cols))

    ##GreyScale
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Adaptive Equalization
    #0.07
    img = exposure.equalize_adapthist(img, clip_limit=0.03) #0.57!! farrah(1), 0.75 norm/abn
    img = img_as_ubyte(img)


    #cv2.imshow("win", img)
    #cv2.waitKey()
    #cv2.destroyAllWindows()
    return img

def preprocess2(img):
    ##Resize
    img = cv2.resize(img, (img_rows, img_cols))

    ##GreyScale
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Contrast stretching
    p2, p98 = np.percentile(img, (2, 98))
    img = exposure.rescale_intensity(img, in_range=(p2, p98)) #0.64!! farrah(1), 0.65 norm/abn

    return img

def feature_hog(img):
    ##Call edge detection here if needed.. ##
    #img = Dilated_Canny(img)
    #img = auto_canny(img)
    #img = manual_canny(img)

    ##hog function
    fd, himg = hog(img, orientations=8, pixels_per_cell=(16, 16), cells_per_block=(1, 1), visualize=True,
                  multichannel=False)
    return himg

def feature_hog_desc(img):
    winSize = (img_rows, img_cols)
    blockSize = (112, 112)
    blockStride = (7, 7)
    cellSize = (56, 56)

    nbins = 9
    derivAperture = 1
    winSigma = -1.
    histogramNormType = 0
    L2HysThreshold = 0.2
    gammaCorrection = 0
    nlevels = 64
    useSignedGradients = True

    #img = Dilated_Canny(img)

    hog = cv2.HOGDescriptor(winSize, blockSize, blockStride, cellSize, nbins, derivAperture, winSigma,
                            histogramNormType, L2HysThreshold, gammaCorrection, nlevels, useSignedGradients)

    descriptor = hog.compute(img)
    print(descriptor)
    print(descriptor.shape)

    return descriptor

def feature_orb(img):
    orb = cv2.ORB_create()

    # find the keypoints with ORB
    kp = orb.detect(img, None)

    # compute the descrip0tors with ORB
    kp, des = orb.compute(img, kp)
    print(des.shape)
    print(des)
    return des
    #exit()

def feature_surf(img):

    spoints = surf.surf(img)
    print("Nr points: {}".format(len(spoints)))
    #print("points:", spoints)
    print("------------")
    #exit()

    return spoints[:95]

def feature_baseline(img):
    result = []
    pos = []

    pixels = 0
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if img[i, j] >= 200:
                pixels+=1
        result.append(pixels)
        pos.append(i)
        pixels=0


    imgarr = np.array([result])
    imgarr = imgarr.transpose()
    print(imgarr.shape)

    axes = plt.gca()
    axes.set_xlim([0, 224])
    plt.plot(result)
    plt.ylabel('Pixels')
    plt.show()

    #exit()
    return imgarr

def wave(img):
    titles = ['Approximation', ' Horizontal detail',
              'Vertical detail', 'Diagonal detail']
    coeffs2 = pywt.dwt2(img, 'bior1.3')
    LL, (LH, HL, HH) = coeffs2
    fig = plt.figure(figsize=(12, 3))
    for i, a in enumerate([LL, LH, HL, HH]):
        ax = fig.add_subplot(1, 4, i + 1)
        ax.imshow(a, interpolation="nearest", cmap=plt.cm.gray)
        ax.set_title(titles[i], fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])

    fig.tight_layout()
    plt.show()
    exit()
    pass

def morph(img):

    #img4 = cv2.fastNlMeansDenoising(img, None, 10, 10, 7)

    img = cv2.bilateralFilter(img, 9, 75, 75)

    kernel = np.ones((5, 5), np.uint8)
    img = cv2.morphologyEx(img, cv2.MORPH_GRADIENT, kernel)

    kernel2 = np.array([[-1, -1, -1], [-1, 11, -1], [-1, -1, -1]])
    img = cv2.filter2D(img, -1, kernel2)

    #img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    #img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)

    #to sharpen
    #denoise

    #cv2.imshow("c1",img)
    #cv2.waitKey()
    #cv2.destroyAllWindows()
    #exit()
    return img

def gabor(img):
    filters = []
    ksize = 31
    for theta in np.arange(0, np.pi, np.pi / 16):
        kern = cv2.getGaborKernel((ksize, ksize), 4.0, theta, 10.0, 0.5, 0, ktype=cv2.CV_32F)
    kern /= 1.5 * kern.sum()
    filters.append(kern)

    accum = np.zeros_like(img)
    for kern in filters:
        fimg = cv2.filter2D(img, cv2.CV_8UC3, kern)
    np.maximum(accum, fimg, accum)
    return accum

def haralick_features(image):
    #The haralick texture features are energy, entropy, homogeneity, correlation, contrast, dissimilarity and maximum probability.
    # calculate haralick texture features for 4 types of adjacency
    textures = mt.features.haralick(image)

    # take the mean of it and return it
    ht_mean = textures.mean(axis=0)
    print("haralick: ", ht_mean)
    print("haralick2: ", ht_mean.shape)
    print("------------------")

    return ht_mean

def binary_features(image):
    #mahotas.features.lbp.lbp_transform(image, radius, points, ignore_zeros=False, preserve_shape=True)
    textures = mt.features.lbp(image,3,12,False)
    #ht_mean = textures.mean(axis=0)
    print("Binary features: ", textures)
    print("---------------------------")
    return  textures

def stat_features(img):

    ls = []

    std = np.nanstd(img)
    #ls.append(std)

    var = np.nanvar(img)
    #ls.append(var)

    #mean = np.nanmean(img)
    #ls.append(mean)

    avg = np.average(img)
    #ls.append(avg)

    sum = np.nansum(img)
    #ls.append(sum)

    median = np.nanmedian(img)
    #ls.append(median)

    #max = np.max(img)
    #ls.append(max)

    #min = np.min(img)
    #ls.append(min)

    #ls = np.asarray(ls)
    '''
    grd = np.gradient(img)
    grd = np.asarray(grd[0])
    grd = grd.round(decimals=6)
    grd[grd == -inf] = 0
    ls.extend(grd)
    '''

    kurt = stats.kurtosis(img)
    kurt = np.asarray(kurt)
    kurt = kurt.round(decimals=6)
    kurt[kurt == -inf] = 0
    #ls.extend(kurt)

    entr = stats.entropy(img)
    entr = np.asarray(entr)
    entr = entr.round(decimals=1) # at 1, alone = 0.62
    entr[entr == -inf] = 0
    ls.extend(entr)

    skew = stats.skew(img)
    skew = np.asarray(skew)
    skew = skew.round(decimals=5)
    skew[skew == -inf] = 0
    #ls.extend(skew)

    print(ls)
    print("--------------------------")
    #exit()
    arr = np.asarray(ls)

    return arr

def loadimages():

    mypath = "hybrid2/"
    i = 0
    lbl = 0

    for (dirpath, dirnames, filenames) in walk(mypath):
        for subtype in dirnames:
            newpath = mypath + subtype +"/"

            for (dirpath2, dirnames2, images) in walk(newpath):

                for img in images:
                    label = lbl
                    imgname = img

                    img_path = newpath+img
                    img = cv2.imread(img_path)

                    print(imgname)

                    if imgname.find("radio") == -1:
                        print("doesnt contain radio (from mura)")
                        img = preprocess(img)
                    else:
                        print("contains radio")
                        img = preprocess2(img)

                    #cv2.imwrite(imgname,img)
                    img = gabor(img)
                    #img = wave(img)
                    #img = feature_hog_desc(img)
                    #img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

                    #temp=[]

                    #arr1 = haralick_features(img)
                    #temp.extend(arr1)

                    #arr2 = binary_features(img)
                    #temp.extend(arr2)

                    #arr3 = stat_features(img)
                    #temp.extend(arr3)

                    #img = np.asarray(temp)

                    #img = feature_surf(img)

                    img = img.flatten()
                    imgarr = np.array([img])
                    print(imgarr.shape)

                    y = [label]
                    if i != 0:
                        xtotal = np.concatenate((imgarr, xtotal), axis=0)
                        ytotal = np.concatenate((y, ytotal), axis=0)
                    else:
                        xtotal = imgarr
                        ytotal = y
                    i += 1
            lbl +=1
        return xtotal, ytotal

def main():

    xtotal, ytotal = loadimages()

    #xtotal = xtotal.astype('float32')
    #x_test = x_test.astype('float32')
    #xtotal /= 255
    #x_test /= 255

    x_train, x_test, y_train, y_test = train_test_split(xtotal, ytotal, test_size=0.3,random_state=42)

    concrete_strategy_a = cs.SvmAlg()
    context = cs.Context(concrete_strategy_a)
    context.context_interface(x_train, y_train, x_test, y_test)

if __name__ == "__main__":
    main()
