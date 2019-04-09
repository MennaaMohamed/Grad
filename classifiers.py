import abc
import cv2
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
from skimage import exposure, transform
from skimage import img_as_ubyte
import csv
from skimage.feature import hog
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from keras import backend as K
from keras.models import Model
import keras
from sklearn.feature_selection import SelectFromModel
from sklearn.svm import LinearSVC
from sklearn import svm
from sklearn.feature_selection import RFE
from sklearn.svm import SVC
from keras.metrics import categorical_accuracy
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report, confusion_matrix
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Activation, Input
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D, BatchNormalization
from keras.optimizers import SGD
from keras.optimizers import Adam
import pickle
from sklearn.externals import joblib
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import RadiusNeighborsClassifier
from keras.callbacks import EarlyStopping,ModelCheckpoint

class Context:

    def __init__(self, strategy):
        self._strategy = strategy

    def context_interface(self, x, y, xt, yt):
        return self._strategy.algorithm_interface(x, y, xt, yt)

class Strategy(metaclass=abc.ABCMeta):
    
    @abc.abstractmethod
    def algorithm_interface(self, x, y, xt, yt):
        pass

class DecisionTreeAlg(Strategy):
    def algorithm_interface(self, x_train, y_train, x_test, y_test):
        clf = DecisionTreeClassifier(random_state=0)
        clf.fit(x_train, y_train)
        y_pred = clf.predict(x_test)

        cm = confusion_matrix(y_test, y_pred)
        acc = accuracy_score(y_test, y_pred, normalize=True, sample_weight=None)
        cr = classification_report(y_test, y_pred)
        print(cm)
        print(acc)
        print(cr)
        return cm, acc, cr

class NaiveAlg(Strategy):
    def algorithm_interface(self, x_train, y_train, x_test, y_test):

        xtotal = np.concatenate((x_train,x_test), axis=0)
        ytotal = np.concatenate((y_train, y_test), axis=0)

        print(x_train)
        print(ytotal.shape)
        print(xtotal.shape)

        lsvc = LinearSVC(C=0.01, penalty="l1", dual=False).fit(xtotal, ytotal)
        model = SelectFromModel(lsvc, prefit=True)
        X_new = model.transform(xtotal)
        #joblib.dump(model, 'models/svmtransform.joblib')
        print(X_new.shape)
        x_train, x_test, y_train, y_test = train_test_split(X_new, ytotal, test_size=0.3, random_state=42)


        clf = GaussianNB()
        clf.fit(x_train, y_train)
        y_pred = clf.predict(x_test)

        #joblib.dump(clf, 'models/naive.joblib')

        cm = confusion_matrix(y_test, y_pred)
        acc = accuracy_score(y_test, y_pred, normalize=True, sample_weight=None)
        cr = classification_report(y_test, y_pred)
        print (cm)
        print(acc)
        print(cr)
        return cm, acc, cr

class CnnAlg(Strategy):
    #Very Small Arch From Scratch
    # algorithm 1
    def __init__(self):

        self.num_classes = 3
        self.batch_size = 31
        self.img_rows = 224
        self.img_cols = 224
        self.epochs = 50

    def algorithm_interface(self, x_train, y_train, x_test, y_test):

        if K.image_data_format() == 'channels_first':
            x_train = x_train.reshape(x_train.shape[0], 1, self.img_rows, self.img_cols)
            x_test = x_test.reshape(x_test.shape[0], 1, self.img_rows, self.img_cols)
            #x_validate = x_validate.reshape(x_validate.shape[0], 1, self.img_rows, self.img_cols)
            input_shape = (1, self.img_rows, self.img_cols)
        else:
            x_train = x_train.reshape(x_train.shape[0], self.img_rows, self.img_cols, 1)
            x_test = x_test.reshape(x_test.shape[0], self.img_rows, self.img_cols, 1)
            #x_validate = x_validate.reshape(x_validate.shape[0], self.img_rows, self.img_cols, 1)
            input_shape = (self.img_rows, self.img_cols, 1)


        # normalize
        x_train = x_train.astype('float32')
        x_test = x_test.astype('float32')
        x_train /= 255
        x_test /= 255

        # convert class vectors
        y_train = keras.utils.to_categorical(y_train, self.num_classes)
        y_test = keras.utils.to_categorical(y_test, self.num_classes)
        #y_validate = keras.utils.to_categorical(y_validate, self.num_classes)

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


        datagen = ImageDataGenerator(
            featurewise_std_normalization=True,
            rotation_range=40,
            zoom_range = 0.2,
            rescale=1/255,
            vertical_flip=True,
            horizontal_flip=True,
            fill_mode = 'nearest')
        
        datagen.fit(x_train)


        #sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)

        model.compile(optimizer=Adam(0.00001), loss='categorical_crossentropy', metrics=['accuracy'])
        #model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

        ES = EarlyStopping(patience=5)

        # fits the model on batches with real-time data augmentation:
        model.fit_generator(datagen.flow(x_train, y_train, batch_size=self.batch_size, save_to_dir="cnnmodels/"),
                            epochs=self.epochs, validation_data=(x_test,y_test),callbacks=[ES])


        # Adam combines the good properties of Adadelta and RMSprop and hence tend to do better for most of the problems.

        #validation_data (x_test, y_test);

        #model.fit(x_train, y_train, validation_data=(x_test,y_test),batch_size=self.batch_size, epochs=self.epochs, callbacks=[ES])

        score = model.evaluate(x_test, y_test)
        print('Test loss:', score[0])
        print('Test accuracy:', score[1])

        model.save('cnnmodels/model5.h5')
        print("Saved model to disk")


        #model5??? 3 classes only.., mean squared error aw categorical same, validation data, 0.625
        #model5 sgd optimizer 0.3
        return model.summary()

class SvmAlg(Strategy):

    # algorithm 2
    def algorithm_interface(self, x_train, y_train, x_test, y_test):

        '''
        xtotal = np.concatenate((x_train,x_test), axis=0)
        ytotal = np.concatenate((y_train, y_test), axis=0)

        print(x_train)
        print(ytotal.shape)
        print(xtotal.shape)

        lsvc = LinearSVC(C=0.01, penalty="l1", dual=False).fit(xtotal, ytotal)
        model = SelectFromModel(lsvc, prefit=True)
        X_new = model.transform(xtotal)
        #joblib.dump(model, 'models/svmtransform.joblib')
        print(X_new.shape)
        x_train, x_test, y_train, y_test = train_test_split(X_new, ytotal, test_size=0.3, random_state=42)
        '''

        svclassifier = SVC(kernel='sigmoid')
        svclassifier.fit(x_train, y_train)
        y_pred = svclassifier.predict(x_test)

        cm = confusion_matrix(y_test, y_pred)
        acc = accuracy_score(y_test, y_pred, normalize=True, sample_weight=None)
        cr = classification_report(y_test, y_pred)

        #joblib.dump(svclassifier, 'models/svmfs.joblib')

        print (cm)
        print(acc)
        print(cr)
        #return cm, acc, cr

        pass

class KnnAlg(Strategy):
    def algorithm_interface(self, x_train, y_train, x_test, y_test):

        knnclassifier = KNeighborsClassifier(n_neighbors=5)
        knnclassifier.fit(x_train, y_train)
        y_pred = knnclassifier.predict(x_test)

        joblib.dump(knnclassifier, 'models/knn.joblib')

        cm = confusion_matrix(y_test, y_pred)
        acc = accuracy_score(y_test, y_pred, normalize=True, sample_weight=None)
        cr = classification_report(y_test, y_pred)
        print(cm)
        print(acc)
        print(cr)
        return cm, acc, cr

class RandomForestAlg(Strategy):
    def algorithm_interface(self, x_train, y_train, x_test, y_test):

        xtotal = np.concatenate((x_train, x_test), axis=0)
        ytotal = np.concatenate((y_train, y_test), axis=0)

        print(x_train)
        print(ytotal.shape)
        print(xtotal.shape)

        lsvc = LinearSVC(C=0.01, penalty="l1", dual=False).fit(xtotal, ytotal)
        model = SelectFromModel(lsvc, prefit=True)
        X_new = model.transform(xtotal)
        # joblib.dump(model, 'models/svmtransform.joblib')
        print(X_new.shape)
        x_train, x_test, y_train, y_test = train_test_split(X_new, ytotal, test_size=0.3, random_state=42)



        rfclassifier = RandomForestClassifier(n_estimators=100, max_depth=2,random_state = 0)
        rfclassifier.fit(x_train, y_train)
        y_pred = rfclassifier.predict(x_test)

        #joblib.dump(rfclassifier, 'models/randomforest.joblib')

        cm = confusion_matrix(y_test, y_pred)
        acc = accuracy_score(y_test, y_pred, normalize=True, sample_weight=None)
        cr = classification_report(y_test, y_pred)
        print(rfclassifier.feature_importances_)
        print (cm)
        print(acc)
        print(cr)
        return cm, acc, cr

class MlpAlg(Strategy):
    def algorithm_interface(self, x_train, y_train, x_test, y_test):
        mlclassifier = MLPClassifier(hidden_layer_sizes=(100,),activation='relu', solver ='adam', alpha = 0.0001,
        batch_size ='auto', learning_rate ='constant', learning_rate_init = 0.001, power_t = 0.5,
        max_iter = 200, shuffle = True, random_state = None, tol = 0.0001, verbose = False, warm_start = False,
        momentum = 0.9, nesterovs_momentum = True,
        early_stopping = False, validation_fraction = 0.1, beta_1 = 0.9, beta_2 = 0.999, epsilon = 1e-08)

        mlclassifier.fit(x_train, y_train)
        y_pred = mlclassifier.predict(x_test)

        joblib.dump(mlclassifier, 'models/mlpdatamr.joblib')

        cm = confusion_matrix(y_test, y_pred)
        acc = accuracy_score(y_test, y_pred, normalize=True, sample_weight=None)
        cr = classification_report(y_test, y_pred)
        return cm, acc, cr

class RnnAlg(Strategy):
    def algorithm_interface(self, x_train, y_train, x_test, y_test):
        pass

class A_ALG(Strategy):
    def algorithm_interface(self, x, y, xt, yt):
        pass

class VGGALG(Strategy):
    def __init__(self):

        self.num_classes = 3
        self.batch_size = 128
        self.img_rows = 224
        self.img_cols = 224
        self.epochs = 10

    def algorithm_interface(self, x_train, y_train, x_test, y_test):
        if K.image_data_format() == 'channels_first':
            x_train = x_train.reshape(x_train.shape[0], 3, self.img_rows, self.img_cols)
            x_test = x_test.reshape(x_test.shape[0], 3, self.img_rows, self.img_cols)
            input_shape = (3, self.img_rows, self.img_cols)
        else:
            x_train = x_train.reshape(x_train.shape[0], self.img_rows, self.img_cols, 3)
            x_test = x_test.reshape(x_test.shape[0], self.img_rows, self.img_cols, 3)
            input_shape = (self.img_rows, self.img_cols, 3)

        # more reshaping whats this
        x_train = x_train.astype('float32')
        x_test = x_test.astype('float32')
        x_train /= 255
        x_test /= 255

        # convert class vectors
        y_train = keras.utils.to_categorical(y_train, self.num_classes)
        y_test = keras.utils.to_categorical(y_test, self.num_classes)

        #vgg16
        model_vgg16_conv = keras.applications.vgg16.VGG16(weights='imagenet', include_top=False)
        model_vgg16_conv.summary()

        # Create your own input format (here 3x200x200)
        input = Input(shape=input_shape, name='image_input')

        # Use the generated model
        output_vgg16_conv = model_vgg16_conv(input)

        # Add the fully-connected layers
        x = Flatten(name='flatten')(output_vgg16_conv)
        x = Dense(4096, activation='relu', name='fc1')(x)
        x = Dense(4096, activation='relu', name='fc2')(x)
        x = Dense(self.num_classes, activation='softmax', name='predictions')(x)

        # Create your own model
        my_model = Model(input=input, output=x)
        ES = EarlyStopping(patience=6)

        # In the summary, weights and layers from VGG part will be hidden, but they will be fit during the training
        my_model.summary()

        sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
        my_model.compile(optimizer=sgd, loss='categorical_crossentropy',metrics=['accuracy'])

        Model_Fit = my_model.fit(x=x_train, y=y_train, validation_data=(x_test, y_test), batch_size=7,
                              epochs=self.epochs, verbose=1, callbacks=[ES])
        evaluation = my_model.evaluate(x_test, y_test)
        my_model.save('vggimagenet.h5')
        print(evaluation)

        '''
        vgg16 = keras.applications.vgg16.VGG16()
        vgg16.summary()

        model = Sequential()

        for layer in vgg16.layers:
            model.add(layer)

        model.layers.pop()
        #model.summary()

        for layer in model.layers:
            layer.trainable = False

        model.add(Dense(self.num_classes, activation='softmax'))
        #model.summary()
        model.compile(optimizer=Adam(lr=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

        model.fit(x_train, y_train,
                  batch_size=self.batch_size,
                  epochs=self.epochs,
                  verbose=1)

        score = model.evaluate(x_test, y_test, verbose=0)
        print('Test loss:', score[0])
        print('Test accuracy:', score[1])

        model.save('mlp_model.h5')
        print("Saved model to disk")
        
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

        #validation_data (x_test, y_test);
        model.fit(x_train, y_train,
                  batch_size=self.batch_size,
                  epochs=self.epochs,
                  verbose=1)

        score = model.evaluate(x_test, y_test, verbose=0)
        print('Test loss:', score[0])
        print('Test accuracy:', score[1])

        model.save('mlp_model.h5')
        print("Saved model to disk")

        return model.summary()
        '''
        return "hi"

class VGG_ALG_SCRATCH(Strategy):
    def __init__(self):
        self.num_classes = 4
        self.batch_size = 128
        self.img_rows = 224
        self.img_cols = 224
        self.epochs = 20

    def algorithm_interface(self, x_train, y_train, x_test, y_test):

        if K.image_data_format() == 'channels_first':
            x_train = x_train.reshape(x_train.shape[0], 1, self.img_rows, self.img_cols)
            x_test = x_test.reshape(x_test.shape[0], 1, self.img_rows, self.img_cols)
            input_shape = (1, self.img_rows, self.img_cols)
        else:
            x_train = x_train.reshape(x_train.shape[0], self.img_rows, self.img_cols, 1)
            x_test = x_test.reshape(x_test.shape[0], self.img_rows, self.img_cols, 1)
            input_shape = (self.img_rows, self.img_cols, 1)

        # more reshaping whats this
        x_train = x_train.astype('float32')
        x_test = x_test.astype('float32')
        x_train /= 255
        x_test /= 255

        # convert class vectors
        y_train = keras.utils.to_categorical(y_train, self.num_classes)
        y_test = keras.utils.to_categorical(y_test, self.num_classes)

        model = Sequential([
            Conv2D(64, (3, 3), input_shape=input_shape, padding='same',
                   activation='relu'),
            Conv2D(64, (3, 3), activation='relu', padding='same'),
            MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
            Conv2D(128, (3, 3), activation='relu', padding='same'),
            Conv2D(128, (3, 3), activation='relu', padding='same', ),
            MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
            Conv2D(256, (3, 3), activation='relu', padding='same', ),
            Conv2D(256, (3, 3), activation='relu', padding='same', ),
            Conv2D(256, (3, 3), activation='relu', padding='same', ),
            MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
            Conv2D(512, (3, 3), activation='relu', padding='same', ),
            Conv2D(512, (3, 3), activation='relu', padding='same', ),
            Conv2D(512, (3, 3), activation='relu', padding='same', ),
            MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
            Conv2D(512, (3, 3), activation='relu', padding='same', ),
            Conv2D(512, (3, 3), activation='relu', padding='same', ),
            Conv2D(512, (3, 3), activation='relu', padding='same', ),
            MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
            Flatten(),
            Dense(4096, activation='relu'),
            Dense(4096, activation='relu'),
            Dense(self.num_classes, activation='softmax')
        ])
        print("Our Model Summary: ")
        model.summary()

        # Adam combines the good properties of Adadelta and RMSprop and hence tend to do better for most of the problems.
        model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

        #validation_data (x_test, y_test);
        model.fit(x_train, y_train,
                  batch_size=self.batch_size,
                  epochs=self.epochs,
                  verbose=1)

        score = model.evaluate(x_test, y_test, verbose=0)
        print('Test loss:', score[0])
        print('Test accuracy:', score[1])

        model.save('mlp_model2.h5')
        print("Saved model to disk")
#        model.compile(optimizer=Adam(lr=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

        return model.summary()

class CNN2(Strategy):
    def __init__(self):

        self.num_classes = 4
        self.batch_size = 31
        self.img_rows = 224
        self.img_cols = 224
        self.epochs = 40

    def algorithm_interface(self, x_train, y_train, x_test, y_test):
        ES = EarlyStopping(patience=6)
        if K.image_data_format() == 'channels_first':
            x_train = x_train.reshape(x_train.shape[0], 1, self.img_rows, self.img_cols)
            x_test = x_test.reshape(x_test.shape[0], 1, self.img_rows, self.img_cols)
            #x_validate = x_validate.reshape(x_validate.shape[0], 1, self.img_rows, self.img_cols)
            input_shape = (1, self.img_rows, self.img_cols)
        else:
            x_train = x_train.reshape(x_train.shape[0], self.img_rows, self.img_cols, 1)
            x_test = x_test.reshape(x_test.shape[0], self.img_rows, self.img_cols, 1)
            #x_validate = x_validate.reshape(x_validate.shape[0], self.img_rows, self.img_cols, 1)
            input_shape = (self.img_rows, self.img_cols, 1)


        # normalize
        x_train = x_train.astype('float32')
        x_train /= 255
        x_test = x_test.astype('float32')
        x_test /= 255


        # convert class vectors
        y_train = keras.utils.to_categorical(y_train, self.num_classes)
        y_test = keras.utils.to_categorical(y_test, self.num_classes)

        print(x_train.shape[1:])

        model = Sequential()
        model.add(ZeroPadding2D((1, 1), input_shape=input_shape))
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(ZeroPadding2D((1, 1)))
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(MaxPooling2D((2, 2), strides=(2, 2)))
        model.add(ZeroPadding2D((1, 1)))
        model.add(Conv2D(128, (3, 3), activation='relu'))
        model.add(ZeroPadding2D((1, 1)))
        model.add(Conv2D(128, (3, 3), activation='relu'))
        model.add(MaxPooling2D((2, 2), strides=(2, 2)))

        print('000')
        model.add(ZeroPadding2D((1, 1)))
        model.add(Conv2D(256, (3, 3), activation='relu'))
        model.add(ZeroPadding2D((1, 1)))
        model.add(Conv2D(256, (3, 3), activation='relu'))
        model.add(ZeroPadding2D((1, 1)))
        model.add(Conv2D(256, (3, 3), activation='relu'))
        model.add(MaxPooling2D((2, 2), strides=(2, 2)))
        print('000')
        #
        model.add(ZeroPadding2D((1, 1)))
        model.add(Conv2D(512, (3, 3), activation='relu'))
        model.add(ZeroPadding2D((1, 1)))
        model.add(Conv2D(512, (3, 3), activation='relu'))
        model.add(ZeroPadding2D((1, 1)))
        model.add(Conv2D(512, (3, 3), activation='relu'))
        model.add(MaxPooling2D((2, 2), strides=(2, 2)))
        print('000')
        #
        model.add(ZeroPadding2D((1, 1)))
        model.add(Conv2D(512, (3, 3), activation='relu'))
        model.add(ZeroPadding2D((1, 1)))
        model.add(Conv2D(512, (3, 3), activation='relu'))
        model.add(ZeroPadding2D((1, 1)))
        model.add(Conv2D(512, (3, 3), activation='relu'))
        model.add(MaxPooling2D((2, 2), strides=(2, 2)))
        model.add(Flatten())
        model.add(Dense(4096, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(4096, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(self.num_classes, activation='softmax'))

        #opt = SGD(lr=0.01)
        opt = SGD(lr=0.0001, decay=1e-6, momentum=0.9, nesterov=True)
        #model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])
        #model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

        model.compile(loss='binary_crossentropy', optimizer='adadelta', metrics=[categorical_accuracy])
        model.summary()

        Model_Fit = model.fit(x=x_train, y=y_train, validation_data=(x_test, y_test), batch_size=7,
                              epochs=self.epochs, verbose=1, callbacks=[ES])
        evaluation = model.evaluate(x_test, y_test)

        print(evaluation)