import numpy as np
import keras
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from classes.nn_network import NNNetwork
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from hyperopt import Trials, STATUS_OK, tpe, hp, fmin
from hyperas.distributions import choice, uniform, conditional
from hyperas import optim


class BaseModel(NNNetwork):

    def __init__(self, data_handler, c_04567):
        NNNetwork.__init__(self, data_handler)
        self.y_test_pred_proba = np.array([])
        self.y_pred_class = np.array([])
        self.x_train, self.y_train = self.dh.get_data(return_train=True, c_04567=c_04567)
        self.x_test, self.y_test = self.dh.get_data(return_train=False, c_04567=c_04567)

        if c_04567:
            self.class_inds = self.dh.ind_04567
            self.class_names = self.dh.class_names04567
        else:
            self.class_inds = self.dh.ind_12389
            self.class_names = self.dh.class_names12389

        self.num_classes_to_predict = len(self.class_inds)


    def train_model_for_tuning(self, params):
        print('Tuning')
        model = Sequential()
        model.add(Conv2D(32, (3, 3), activation='relu', input_shape=self.dh.input_shape))
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(params['dropout1']))
        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(self.num_classes_to_predict, activation='softmax'))

        model.compile(loss=keras.losses.categorical_crossentropy, optimizer='rmsprop', metrics=['accuracy'])
        # converting 04567 labels into 01234
        y_train_inds = [np.where(self.class_inds == y) for y in self.y_train]
        # getting 01234 classes into one hot
        y_train_one_hot = keras.utils.to_categorical(y_train_inds, self.num_classes_to_predict)
        model.fit(self.x_train, y_train_one_hot, batch_size=128, epochs=1)

        y_test_inds = [np.where(self.class_inds == y) for y in self.y_test]
        y_test_one_hot = keras.utils.to_categorical(y_test_inds, self.num_classes_to_predict)
        score, acc = model.evaluate(self.x_test,y_test_one_hot, verbose=0)
        print('Test accuracy:', acc)
        return {'loss': -acc, 'status': STATUS_OK, 'model': model}

    def hyper_param_tuning(self):
        print('Hyper param uning')

        space = {'dropout1': hp.uniform('dropout1', .25, .75)}

        trials = Trials()
        best = fmin(self.train_model_for_tuning, space, algo=tpe.suggest, max_evals=5, trials=trials)

        return best


    def define_model_structure(self):
        model = Sequential()
        model.add(Conv2D(32, (3, 3), activation='relu', input_shape=self.dh.input_shape))
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        model.add(Conv2D(128, (3, 3), activation='relu')) #new
        model.add(MaxPooling2D(pool_size=(2, 2))) #new
        model.add(Dropout(0.25))#new
        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(self.num_classes_to_predict, activation='softmax'))
        self.model = model

    def train_model(self, nb_epoch=5, model_name='base_model.h5', batch_size = 128, load_model=False):
        NNNetwork.train_model(self, x = self.x_train, model = self.model, load_model = load_model, model_name = model_name)
        if not load_model:
            self.model.compile(loss=keras.losses.categorical_crossentropy, optimizer='rmsprop', metrics=['accuracy'])
            # converting 04567 labels into 01234
            y_train_inds = [np.where(self.class_inds == y) for y in self.y_train]
            # getting 01234 classes into one hot
            y_train_one_hot = keras.utils.to_categorical(y_train_inds,self.num_classes_to_predict)
            self.model.fit(self.x_train,  y_train_one_hot, batch_size=batch_size, epochs=nb_epoch)
            self.model.save(model_name)

    def __predict_class(self, x, max_acc_level):
        '''Return class label given input X.'''

        y_pred_proba = self.model.predict(x)
        y_pred_class_inds = y_pred_proba.argsort(axis=1)[:, -max_acc_level:]
        y_test_preds = self.dh.get_class_index(y_pred_class_inds, max_acc_level, self.class_inds)
        return y_test_preds

    def compute_accuracy(self, max_acc_level = 3, verbose = True):
        '''Computes accuracy on clasifying classes 04567 on the test set.'''

        self.y_pred_class = self.__predict_class(self.x_test, max_acc_level)
        accs_at_level = NNNetwork._compute_accuracy_at_level(self, max_acc_level=max_acc_level,
                                             y_pred_classes=self.y_pred_class, y_true=self.y_test.reshape(-1, 1),
                                             verbose=verbose)

        return accs_at_level

    def plot_confusion_matrix(self, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
        '''Computes and plot confusion matrix.'''

        if len(self.y_pred_class) == 0:
            # this computes accs_at_level
            print("Getting class prediction first.")
            # max level 1 because we want to have a confusion matrix with normal accuracy
            self.y_pred_class = self.__predict_class(self.x_test, max_acc_level = 1)
        cm = confusion_matrix(self.y_test, self.y_pred_class[:, -1])
        NNNetwork.plot_confusion_matrix(self, cm, self.class_names, normalize, title, cmap)


