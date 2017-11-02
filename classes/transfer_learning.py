import random
import numpy as np
import keras
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten, Lambda, Input
from keras.initializers import RandomNormal
from keras.optimizers import SGD, RMSprop, Adam
from keras import backend as K
from itertools import product
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras.callbacks import LearningRateScheduler
from sklearn.model_selection import train_test_split
import math
from classes.nn_network import NNNetwork
from classes.base_model import BaseModel
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


class TransferLearning(BaseModel):

    def define_model_structure(self, base_model):
        '''Given the base model, defines which layers will be trainable and set FC layers.'''

        for layer in base_model.layers:
            layer.trainable = False  # turn of all layers for training
        #base_model.layers[1].trainable = True # letting the first conv layer train

        # adding a dense layer behind flatten_1 layers (i.e., layer[-4])
        x = Dense(128, activation='relu', name="dense_new1")(base_model.layers[-4].output)
        x = Dropout(0.5, name='dropout_new')(x)
        predictions = Dense(self.num_classes_to_predict, activation='softmax', name='softmax_new1')(x)  # new softmax layer
        self.model = Model(input=base_model.input, output=predictions)

    def train_model(self, nb_epoch=5, model_name='transfer_learning.h5', batch_size = 128, load_model=False):
        '''Trains transfer learning model.'''

        NNNetwork.train_model(self, x = self.x_train, model = self.model, load_model = load_model, model_name = model_name)

        if not load_model:
            print("Training on {} isntances of data.".format(len(self.x_train)))
            self.model.compile(loss=keras.losses.categorical_crossentropy, optimizer='rmsprop', metrics=['accuracy'])
            # converting class inds (e.g. 04567) labels into e.g. 01234
            y_train_inds = [np.where(self.class_inds == y) for y in self.y_train]
            # getting 01234 classes into one hot
            y_train_one_hot = keras.utils.to_categorical(y_train_inds, self.num_classes_to_predict)
            self.model.fit(self.x_train, y_train_one_hot, batch_size=batch_size, epochs=nb_epoch, shuffle=True)
            self.model.save(model_name)



