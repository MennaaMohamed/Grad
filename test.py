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

def preprocess(img):
    ##GreyScale
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    ##Resize
    img = cv2.resize(img, (img_rows, img_cols))

    ##Exposure
    img = exposure.equalize_adapthist(img, clip_limit=0.05)
    img = img_as_ubyte(img)

    return img

def loadimages():
    '''

        ###Feature Extraction###
        ##Hog
        fd, img = hog(img, orientations=8, pixels_per_cell=(16, 16),cells_per_block=(1, 1), visualize=True, multichannel=False)
    '''
    mypath = "data2/"
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

                    ###Call Feature Extraction. ###

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

    #links = loadpaths(links=[])

    xtotal, ytotal = loadimages()
    x_train, x_test, y_train, y_test = train_test_split(xtotal, ytotal, test_size=0.25, random_state=42)

    concrete_strategy_a = cs.SvmAlg()
    context = cs.Context(concrete_strategy_a)
    cm, acc, cr = context.context_interface(x_train, y_train, x_test, y_test)
    print("accuracy = ", acc)
    print("Confusion Matrix = \n", cm)
    print("Classification Report = \n", cr)

    '''
    concrete_strategy_a = cs.CnnAlg()
    context = cs.Context(concrete_strategy_a)
    print(context.context_interface(x_train, y_train, x_test, y_test))
    '''

if __name__ == "__main__":
    main()
