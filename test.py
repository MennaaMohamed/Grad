import cv2
import numpy as np
import matlab.engine
import math
from skimage import feature
from scipy import ndimage
from skimage import exposure, transform
from skimage import img_as_ubyte
from skimage.feature import hog
from sklearn.model_selection  import train_test_split
import classifiers as cs
from os import walk, getcwd
from PIL import Image

import pickle
from sklearn.externals import joblib
import tensorflow as tf
import tensorflow.python.platform
from tensorflow.python.platform import gfile
import matplotlib.pyplot as plt
import PHogFeatures as phogfeat
import csv
import pandas as pd
import os


img_rows = 224
img_cols = 224


def get_features(image_path, bins=8, angle=360., pyramid_levels=3):
    """
    Returns a feature vector containing a PHOG descriptor of a whole image.
    :param image_path: Absolute path to an image
    :param bins: Number of (orientation) bins on the histogram (optimal: 20)
    :param angle: 180 or 360 (optimal: 360)
    :param pyramid_levels: Number of pyramid levels (optimal: 3)
    :return:
    """

    feature_vec = phog(image_path, bins, angle, pyramid_levels)
    feature_vec = feature_vec.T[0]  # Transpose vector, take the first array
    return feature_vec

def phog(image_path, bin, angle, pyramid_levels):
    """
    Given and image I, phog computes the Pyramid Histogram of Oriented
    Gradients over L pyramid levels and over a Region Of Interest.

    :param image_path: Absolute path to an image of size MxN (Color or Gray)
    :param bin: Number of (orientation) bins on the histogram
    :param angle: 180 or 360
    :param pyramid_levels: Number of pyramid levels
    :return: Pyramid histogram of oriented gradients
    """

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
    """
    Computes a Matrix (bm) with the same size of the image where
    (i,j) position contains the histogram value for the pixel at position (i,j)
    and another matrix (bv) where the position (i,j) contains the gradient
    value for the pixel at position (i,j)

    :param angle_values: Matrix containing the angle values
    :param edge_image: Edge Image
    :param gradient_values: Matrix containing the gradient values
    :param angle: 180 or 360
    :param bin: Number of bins on the histogram
    :return: bm - Matrix with the histogram values
            bv - Matrix with the gradient values (only for the pixels belonging to and edge)
    """

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
    """
    Computes Pyramid Histogram of Oriented Gradient over an image.

    :param bh: Matrix of bin histogram values
    :param bv: Matrix of gradient values
    :param pyramid_levels: Number of pyramid levels
    :param bin: Number of bins
    :return: Pyramid histogram of oriented gradients (phog descriptor)
    """

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
    img = exposure.equalize_adapthist(img, clip_limit=0.03) #0.57!! farrah(1), 0.75 norm/abn
    img = img_as_ubyte(img)

    #mngheir khales == 0.607, 0.57
    #img = img_as_ubyte(img)

    return img

def preprocess2(img):
    ##Resize
    img = cv2.resize(img, (img_rows, img_cols))

    ##GreyScale
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    ##Exposure
    #img = exposure.equalize_adapthist(img, clip_limit=0.03)
    #img = img_as_ubyte(img)

    # Contrast stretching
    #p2, p98 = np.percentile(img, (2, 98))
    #img = exposure.rescale_intensity(img, in_range=(p2, p98)) #0.64!! farrah(1), 0.65 norm/abn

    # Histogram Equalization
    #img = exposure.equalize_hist(img) #vry bad 0.39!! farrah(1), 0.6 norm/abn

    # Adaptive Equalization
    #img = exposure.equalize_adapthist(img, clip_limit=0.03) #0.57!! farrah(1), 0.75 norm/abn
    #img = img_as_ubyte(img)

    #mngheir khales == 0.607, 0.57
    #img = img_as_ubyte(img)

    return img

def feature_hog(img):
    ##Call edge detection here if needed.. ##
    #Dilated_Canny(img)
    #auto_canny(img)
    #manual_canny(img)

    ##hog function
    fd, himg = hog(img, orientations=8, pixels_per_cell=(16, 16), cells_per_block=(1, 1), visualize=True,
                  multichannel=False)
    return himg

def call_matlab(img):

    eng = matlab.engine.start_matlab()
    img = Image.fromarray(img, 'RGB')
    image_mat = matlab.uint8(list(img.getdata()))
    image_mat.reshape((img.size[0], img.size[1], 3))
    ret = eng.anna_phog_demo(image_mat)
    return ret

def feature_orb(img):
    pass

def loadimages():

    mypath = "hybrid/"
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


                    img = get_features(img)
                    #print(img.shape)

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
    x_train, x_test, y_train, y_test = train_test_split(xtotal, ytotal, test_size=0.3, random_state=42)

    concrete_strategy_a = cs.RandomForestAlg()
    context = cs.Context(concrete_strategy_a)
    context.context_interface(x_train, y_train, x_test, y_test)

    '''
    print("accuracy= " ,acc)
    print("Confusion Matrix = \n", cm)
    print("Classification Report = \n", cr)
    print("Confusion Matrix = \n", cm)
    print("Classification Report = \n", cr)
    print("accuracy = ", acc)
      concrete_strategy_a = cs.CnnAlg()
    context = cs.Context(concrete_strategy_a)
    print(context.context_interface(x_train, y_train, x_test, y_test))
    '''

if __name__ == "__main__":
    main()
