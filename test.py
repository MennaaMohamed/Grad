import cv2
import numpy as np

from numpy import inf
import pandas as pd
import os
import matplotlib.pyplot as plt
from skimage import exposure, transform
from skimage import img_as_ubyte
from skimage.feature import hog
from sklearn.model_selection  import train_test_split
import classifiers as cs
import math
#import phog as PHOG
from os import walk, getcwd
from PIL import Image
import pywt
#import mahotas as mt
import pickle
import keras.models
from sklearn.externals import joblib
import tensorflow as tf
import tensorflow.python.platform
from tensorflow.python.platform import gfile
import matplotlib.pyplot as plt
import csv
import pandas as pd
import os
from scipy import ndimage
from skimage import feature
from scipy import stats
#from mahotas.features import surf
from skimage.filters import gaussian
from skimage.segmentation import active_contour

from Codes.Histogram.Histogram import Count_XPixels, Count_YPixels

img_rows = 224
img_cols = 224

"""
def get_features(image_path, bins=8, angle=360., pyramid_levels=3):
    
    # Returns a feature vector containing a PHOG descriptor of a whole image.
    # :param image_path: Absolute path to an image
    # :param bins: Number of (orientation) bins on the histogram (optimal: 20)
    # :param angle: 180 or 360 (optimal: 360)
    # :param pyramid_levels: Number of pyramid levels (optimal: 3)
    # :return:
    

    feature_vec = phog(image_path, bins, angle, pyramid_levels)
    feature_vec = feature_vec.T[0]  # Transpose vector, take the first array
    return feature_vec

def phog(image_path, bin, angle, pyramid_levels):
    
    # Given and image I, phog computes the Pyramid Histogram of Oriented
    # Gradients over L pyramid levels and over a Region Of Interest.
    # 
    # :param image_path: Absolute path to an image of size MxN (Color or Gray)
    # :param bin: Number of (orientation) bins on the histogram
    # :param angle: 180 or 360
    # :param pyramid_levels: Number of pyramid levels
    # :return: Pyramid histogram of oriented gradients
    

    grayscale_img = image_path  # 0 converts it to grayscale

    bh = np.array([])
    bv = np.array([])
    if np.sum(np.sum(grayscale_img)) > 100.:
        # Matlab The default sigma is sqrt(2); the size of the filter is chosen automatically, based on sigma.
        # Threshold is applied automatically - the percentage is a bit different than in Matlab's implementation:
        # low_threshold: 10%
        # high_threshold: 20%
        edges_canny = feature.canny(grayscale_img, sigma=math.sqrt(2))
        [GradientY, GradientX] = np.gradient(np.double(grayscale_img))
        GradientYY = np.gradient(GradientY)[1]  # Take only the first matrix
        Gr = np.sqrt((GradientX * GradientX + GradientY * GradientY))

        index = GradientX == 0.
        index = index.astype(int)  # Convert boolean array to an int array
        GradientX[np.where(index > 0)] = np.power(10, 5)
        YX = GradientY / GradientX

        if angle == 180.:
            angle_values = np.divide((np.arctan(YX) + np.pi / 2.) * 180., np.pi)
        if angle == 360.:
            angle_values = np.divide((np.arctan2(GradientY, GradientX) + np.pi) * 180., np.pi)

        [bh, bv] = bin_matrix(angle_values, edges_canny, Gr, angle, bin)
    else:
        bh = np.zeros(image_path.shape[0], image_path.shape[1])
        bv = np.zeros(image_path.shape[0], image_path.shape[1])

    # Don't consider a roi, take the whole image instead
    bh_roi = bh
    bv_roi = bv
    p = phog_descriptor(bh_roi, bv_roi, pyramid_levels, bin)

    return p

def bin_matrix(angle_values, edge_image, gradient_values, angle, bin):
    
    # Computes a Matrix (bm) with the same size of the image where
    # (i,j) position contains the histogram value for the pixel at position (i,j)
    # and another matrix (bv) where the position (i,j) contains the gradient
    # value for the pixel at position (i,j)
    # 
    # :param angle_values: Matrix containing the angle values
    # :param edge_image: Edge Image
    # :param gradient_values: Matrix containing the gradient values
    # :param angle: 180 or 360
    # :param bin: Number of bins on the histogram
    # :return: bm - Matrix with the histogram values
    #         bv - Matrix with the gradient values (only for the pixels belonging to and edge)
    

    # 8-orientations/connectivity structure (Matlab's default is 8 for bwlabel)
    structure_8 = [[1, 1, 1],
                   [1, 1, 1],
                   [1, 1, 1]]

    [contorns, n] = ndimage.label(edge_image, structure_8)
    X = edge_image.shape[1]
    Y = edge_image.shape[0]
    bm = np.zeros((Y, X))
    bv = np.zeros((Y, X))
    nAngle = np.divide(angle, bin)
    for i in np.arange(1, n + 1):
        [posY, posX] = np.nonzero(contorns == i)
        posY = posY + 1
        posX = posX + 1
        for j in np.arange(1, (posY.shape[0]) + 1):
            pos_x = posX[int(j) - 1]
            pos_y = posY[int(j) - 1]
            b = np.ceil(np.divide(angle_values[int(pos_y) - 1, int(pos_x) - 1], nAngle))
            if b == 0.:
                bin = 1.
            if gradient_values[int(pos_y) - 1, int(pos_x) - 1] > 0:
                bm[int(pos_y) - 1, int(pos_x) - 1] = b
                bv[int(pos_y) - 1, int(pos_x) - 1] = gradient_values[int(pos_y) - 1, int(pos_x) - 1]

    return [bm, bv]

def phog_descriptor(bh, bv, pyramid_levels, bin):
    
    # Computes Pyramid Histogram of Oriented Gradient over an image.
    # 
    # :param bh: Matrix of bin histogram values
    # :param bv: Matrix of gradient values
    # :param pyramid_levels: Number of pyramid levels
    # :param bin: Number of bins
    # :return: Pyramid histogram of oriented gradients (phog descriptor)
    

    p = np.empty((0, 1), dtype=int)  # dtype=np.float64? # vertical size 0, horizontal 1

    for b in np.arange(1, bin + 1):
        ind = bh == b
        ind = ind.astype(int)  # convert boolean array to int array
        sum_ind = np.sum(bv[np.where(ind > 0)])
        p = np.append(p, np.array([[sum_ind]]), axis=0)  # append the sum horizontally to empty p array

    cella = 1.
    for l in np.arange(1, pyramid_levels + 1):  # defines a range (from, to, step)
        x = np.fix(np.divide(bh.shape[1], 2. ** l))
        y = np.fix(np.divide(bh.shape[0], 2. ** l))
        xx = 0.
        yy = 0.
        while xx + x <= bh.shape[1]:
            while yy + y <= bh.shape[0]:
                bh_cella = np.array([])
                bv_cella = np.array([])
                bh_cella = bh[int(yy + 1.) - 1:int(yy + y), int(xx + 1.) - 1:int(xx + x)]
                bv_cella = bv[int(yy + 1.) - 1:int(yy + y), int(xx + 1.) - 1:int(xx + x)]

                for b in np.arange(1, bin + 1):
                    ind = bh_cella == b
                    ind = ind.astype(int)  # convert boolean array to int array
                    sum_ind = np.sum(bv_cella[np.where(ind > 0)])
                    p = np.append(p, np.array([[sum_ind]]), axis=0)  # append the sum horizontally to p

                yy = yy + y

            cella = cella + 1.
            yy = 0.
            xx = xx + x

    if np.sum(p) != 0:
        p = np.divide(p, np.sum(p))

    return p
"""
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


# def call_matlab(img):
#
#     eng = matlab.engine.start_matlab()
#     img = Image.fromarray(img, 'RGB')
#     image_mat = matlab.uint8(list(img.getdata()))
#     image_mat.reshape((img.size[0], img.size[1], 3))
#     ret = eng.anna_phog_demo(image_mat)
#     return ret

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

# def feature_surf(img):
#
#     spoints = surf.surf(img)
#     print("Nr points: {}".format(len(spoints)))
#     #print("points:", spoints)
#     print("------------")
#     #exit()
#
#     return spoints[:95]

def feature_baseline(img):
    #Counting white pixels in each Row
    #number of counted pixels
    result = []
    #position of each counted pixel
    pos = []

    pixels = 0
    #shape[0] Height , shape[1] Width
    #White pixel: 200 or more
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if img[i, j] >= 200:
                pixels+=1
        result.append(pixels)
        #i = row number
        pos.append(i)
        pixels=0


    imgarr = np.array([result])
    imgarr = imgarr.transpose()
    print(imgarr.shape)

    axes = plt.gca()
    #set_xlim([min,max])
    axes.set_xlim([0, 224])
    #results = Counted pixels
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

    #reduces noise 
    #(src/dst img, d (diameter of each pixel), sigmaColor (Filter sigma in the color space), sigmaSpace (Filter sigma in the coordinate space))
    img = cv2.bilateralFilter(img, 9, 75, 75)

    kernel = np.ones((5, 5), np.uint8)
    #gives an outline of the object  (src img, op type of morphological trans, kernel Structuring element)
    img = cv2.morphologyEx(img, cv2.MORPH_GRADIENT, kernel)

    kernel2 = np.array([[-1, -1, -1], [-1, 11, -1], [-1, -1, -1]])
    #applies a linear filter to the image (src img, ddepth desired depth of img, kernel)
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
        #(ksize (size of filter returned), sigma (Standard deviation of the gaussian envelope),
        # theta(Orientation of the normal to the parallel stripes of a Gabor function), lambd (Wavelength of the sinusoidal factor),
        #gamma (Spatial aspect ratio), psi (Phase offset), ktype (Type of filter coefficients. It can be CV_32F or CV_64F))
        kern = cv2.getGaborKernel((ksize, ksize), 4.0, theta, 10.0, 0.5, 0, ktype=cv2.CV_32F)
    kern /= 1.5 * kern.sum()
    filters.append(kern)

    accum = np.zeros_like(img)
    for kern in filters:
        fimg = cv2.filter2D(img, cv2.CV_8UC3, kern)
    np.maximum(accum, fimg, accum)
    return accum

# def haralick_features(image):
#     #The haralick texture features are energy, entropy, homogeneity, correlation, contrast, dissimilarity and maximum probability.
#     # calculate haralick texture features for 4 types of adjacency
#     textures = mt.features.haralick(image)
#
#     # take the mean of it and return it
#     ht_mean = textures.mean(axis=0)
#     print("haralick: ", ht_mean)
#     print("haralick2: ", ht_mean.shape)
#     print("------------------")
#
#     return ht_mean

# def binary_features(image):
#     #mahotas.features.lbp.lbp_transform(image, radius, points, ignore_zeros=False, preserve_shape=True)
#     textures = mt.features.lbp(image,3,12,False)
#     #ht_mean = textures.mean(axis=0)
#     print("Binary features: ", textures)
#     print("---------------------------")
#     return  textures

def stat_features(img):

    ls = []
    #Compute the standard deviation along the specified axis, while ignoring NaNs
    std = np.nanstd(img)
    #ls.append(std)

    #Compute the variance along the specified axis, while ignoring NaNs.
    var = np.nanvar(img)
    #ls.append(var)

    #mean = np.nanmean(img)
    #ls.append(mean)

    #Compute the weighted average along the specified axis.
    avg = np.average(img)
    #ls.append(avg)

    #Return the sum of array elements over a given axis treating Not a Numbers (NaNs) as zero.
    sum = np.nansum(img)
    #ls.append(sum)

    #Compute the median along the specified axis, while ignoring NaNs.
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
    #flatness of the histogram
    kurt = stats.kurtosis(img)
    kurt = np.asarray(kurt)
    kurt = kurt.round(decimals=6)
    kurt[kurt == -inf] = 0
    #ls.extend(kurt)

    #Calculate the entropy of a distribution for given probability values.
    #randomness of the intensity of values
    entr = stats.entropy(img)
    entr = np.asarray(entr)
    entr = entr.round(decimals=1) # at 1, alone = 0.62
    entr[entr == -inf] = 0
    ls.extend(entr)

    #measures the asymmetry around the mean
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

                    ###Call Feature Extraction. ###

                    #print(img.shape)
                    #cv2.imwrite(imgname,img)
                    img = gabor(img)
                    #img = wave(img)
                    #img = feature_hog_desc(img)
                    #img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
                    temp=[]
                    arr1,arr2 = Count_XPixels(img)

                    # arr1,arr2 = Count_YPixels(img)

                    # temp.extend(arr1)
                    # temp.extend(arr2)


                    # img = np.asarray(temp)

                    #img = np.asarray(temp)

                    #img = feature_surf(img)

                    img = get_features(img)

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
