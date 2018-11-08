import abc
import cv2
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from skimage import exposure, transform
from skimage import img_as_ubyte
import csv
from skimage.feature import hog
from sklearn.model_selection import train_test_split
from keras import backend as K
import keras
from sklearn import svm
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report, confusion_matrix
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Activation
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D, BatchNormalization
from keras.optimizers import SGD
from keras.optimizers import Adam
import pickle
from sklearn.externals import joblib


class Context:
    """
    Define the interface of interest to clients.
    Maintain a reference to a Strategy object.
    """

    def __init__(self, strategy):
        self._strategy = strategy

    def context_interface(self, x, y, xt, yt):
        return self._strategy.algorithm_interface(x, y, xt, yt)

class Strategy(metaclass=abc.ABCMeta):
    
    @abc.abstractmethod
    def algorithm_interface(self, x, y, xt, yt):
        pass

class CnnAlg(Strategy):
    #Very Small Arch From Scratch
    # algorithm 1
    def __init__(self):

        self.num_classes = 2
        self.batch_size = 128
        self.img_rows = 240
        self.img_cols = 240
        self.epochs = 12

    def algorithm_interface(self, x_train, y_train, x_test, y_test):
        if K.image_data_format() == 'channels_first':
            x_train = x_train.reshape(x_train.shape[0], 1, self.img_rows, self.img_cols)
            x_test = x_test.reshape(x_test.shape[0], 1, self.img_rows, self.img_cols)
            input_shape = (1, self.img_rows, self.img_cols)
        else:
            x_train = x_train.reshape(x_train.shape[0], self.img_rows, self.img_cols, 1)
            x_test = x_test.reshape(x_test.shape[0], self.img_rows, self.img_cols, 1)
            input_shape = (self.img_rows, self.img_cols, 1)

            # more reshaping
        x_train = x_train.astype('float32')
        x_test = x_test.astype('float32')
        x_train /= 255
        x_test /= 255

        # convert class vectors
        y_train = keras.utils.to_categorical(y_train, self.num_classes)
        y_test = keras.utils.to_categorical(y_test, self.num_classes)

        model = Sequential()

        model.add(Conv2D(32, (3, 3), input_shape=input_shape))
        model.add(Activation('relu'))

        BatchNormalization(axis=-1)
        model.add(Conv2D(32, (3, 3)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        BatchNormalization(axis=-1)
        model.add(Conv2D(64, (3, 3)))
        model.add(Activation('relu'))
        BatchNormalization(axis=-1)
        model.add(Conv2D(64, (3, 3)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Flatten())

        BatchNormalization()
        model.add(Dense(512))
        model.add(Activation('relu'))
        BatchNormalization()
        model.add(Dropout(0.2))
        model.add(Dense(self.num_classes))

        model.add(Activation('softmax'))

        # Adam combines the good properties of Adadelta and RMSprop and hence tend to do better for most of the problems.
        model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

        model.fit(x_train, y_train,
                  batch_size=self.batch_size,
                  epochs=self.epochs,
                  verbose=1,
                  validation_data=(x_test, y_test))

        score = model.evaluate(x_test, y_test, verbose=0)
        print('Test loss:', score[0])
        print('Test accuracy:', score[1])

        model.save('mlp_model.h5')
        print("Saved model to disk")

        return model.summary()

class SvmAlg(Strategy):
    # algorithm 2
    def algorithm_interface(self, x_train, y_train, x_test, y_test):
        svclassifier = SVC(kernel='linear')
        svclassifier.fit(x_train, y_train)
        y_pred = svclassifier.predict(x_test)
        joblib.dump(svclassifier, 'models/svm2a.joblib')

        cm = confusion_matrix(y_test, y_pred)
        acc = accuracy_score(y_test, y_pred, normalize=True, sample_weight=None)
        cr = classification_report(y_test, y_pred)
        return cm, acc, cr

class RandomForest(Strategy):
    def algorithm_interface(self, x_train, y_train, x_test, y_test):
        pass

class RnnAlg(Strategy):
    def algorithm_interface(self, x_train, y_train, x_test, y_test):
        pass

class A_ALG(Strategy):
    def algorithm_interface(self, x, y, xt, yt):
        pass

class VGGALG(Strategy):
    def algorithm_interface(self, x, y, xt, yt):
        #fine tuned with layers frozen
        pass