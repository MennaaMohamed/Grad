import cv2
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from skimage import exposure, transform
from skimage import img_as_ubyte
import csv
from skimage.feature import hog
from sklearn.model_selection  import train_test_split
import classifiers as cs
from os import walk, getcwd
import pickle
from sklearn.externals import joblib

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
    return edged

def adjust_gamma(image, gamma=1.0):

   invGamma = 1.0 / gamma
   table = np.array([((i / 255.0) ** invGamma) * 255
      for i in np.arange(0, 256)]).astype("uint8")

   return cv2.LUT(image, table)

def preprocess(img):
    ##GreyScale
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    ##Resize
    img = cv2.resize(img, (img_rows, img_cols))

    ##Exposure
    #img = exposure.equalize_adapthist(img, clip_limit=0.03)
    #img = img_as_ubyte(img)

    # Contrast stretching
    p2, p98 = np.percentile(img, (2, 98))
    img = exposure.rescale_intensity(img, in_range=(p2, p98)) #0.64!! farrah(1), 0.65 norm/abn

    # Histogram Equalization
    #img = exposure.equalize_hist(img) #vry bad 0.39!! farrah(1), 0.6 norm/abn

    # Adaptive Equalization
    #img = exposure.equalize_adapthist(img, clip_limit=0.03) #0.57!! farrah(1), 0.75 norm/abn

    #mngheir khales == 0.607, 0.57
    #img = img_as_ubyte(img)

    return img

def feature_hog(img):
    ##Call edge detection here if needed.. ##
    Dilated_Canny(img)
    #auto_canny(img)
    #manual_canny(img)

    ##hog function
    fd, himg = hog(img, orientations=8, pixels_per_cell=(16, 16), cells_per_block=(1, 1), visualize=True,
                  multichannel=False)
    return himg

def feature_orb(img):
    pass

def loadimages():

    mypath = "datatemp/"
    i = 0
    lbl = 0

    for (dirpath, dirnames, filenames) in walk(mypath):
        for subtype in dirnames:
            newpath = mypath + subtype +"/"

            for (dirpath2, dirnames2, images) in walk(newpath):

                for img in images:
                    label = lbl
                    img_path = newpath+img
                    img = cv2.imread(img_path)

                    ####Call preprocess function here.####
                    img = preprocess(img)
                    #img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
                    ###Call Feature Extraction. ###
                    #img = feature_hog(img)

                    img = img.flatten()
                    imgarr = np.array([img])

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
    x_train, x_test, y_train, y_test = train_test_split(xtotal, ytotal, test_size=0.25, random_state=42)

    concrete_strategy_a = cs.CnnAlg()
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
