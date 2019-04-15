from ast import walk
import cv2
from keras.applications import ResNet50

import test

def extract_resnet():

    mypath = "hybrid2/"
    i = 0
    lbl = 0

    for (dirpath, dirnames, filenames) in walk(mypath):
        for subtype in dirnames:
            newpath = mypath + subtype + "/"

            for (dirpath2, dirnames2, images) in walk(newpath):

                for img in images:
                    label = lbl
                    imgname = img

                    img_path = newpath + img
                    img = cv2.imread(img_path)

                    print(img_path)

    # # X : images numpy array
    # resnet_model = ResNet50(input_shape=(xtotal,ytotal,3), weights='imagenet', include_top=False)
    # # Since top layer is the fc layer used for predictions
    # features_array = resnet_model.predict(X)
    # return features_array
extract_resnet()


