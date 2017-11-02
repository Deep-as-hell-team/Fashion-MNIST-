import numpy as np
import keras
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from classes.nn_network import NNNetwork
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from hyperopt import Trials, STATUS_OK, tpe, hp, fmin
from keras.optimizers import Adam
from keras.callbacks import LearningRateScheduler
import classes

class BaseModel(NNNetwork):

    def __init__(self, data_handler, c_04567):
        super().__init__(data_handler, c_04567=c_04567)
        self.y_test_pred_proba = np.array([])
        self.y_pred_class = np.array([])


    def _get_hyper_params(self):
        h_params = {'num_layers': hp.choice('num_layers',
                                         [{'layers': 'two', },
                                          {'layers': 'three',
                                           'filter_size2': hp.choice('filter_size2', [32, 64, 128]),
                                           'kernel_size2': hp.choice('kernel_size2', [3, 4]),
                                           'dropout2': hp.uniform('dropout2', .25, .75)}
                                          ]),

                 'filter_size0': hp.choice('filter_size0', [16, 32, 64]),
                 'filter_size1': hp.choice('filter_size1', [16, 32, 64]),

                 'kernel_size0': hp.choice('kernel_size0', [3, 4, 5]),
                 'kernel_size1': hp.choice('kernel_size1', [3, 4]),

                 'dropout0': hp.uniform('dropout0', .25, .75),
                 'dropout1': hp.uniform('dropout1', .25, .75),

                 'fc_size0': hp.choice('fc_size0', [64, 128, 256]),

                 'nb_epochs': 40,
                 'lr': hp.choice('lr', [0.01, 0.001, 0.0001]),
                 'optimizer': Adam,
                 'optimizer': 'adam',
                 'activation': 'relu'
                 }
        return h_params

    def _define_model_structure(self, params):
        '''Defines arbitrary structure of a model.'''

        model = Sequential()
        model.add(Conv2D(params['filter_size0'], (params['kernel_size0'], params['kernel_size0']),
                         activation=params['activation'], input_shape=self.dh.input_shape))
        model.add(Conv2D(params['filter_size1'], (params['kernel_size1'], params['kernel_size1']),
                         activation=params['activation']))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(params['dropout0']))

        if params['num_layers']['layers'] == 'three':
            model.add(Conv2D(params['num_layers']['filter_size2'],
                             (params['num_layers']['kernel_size2'], params['num_layers']['kernel_size2']),
                             activation=params['activation']))
            model.add(Dropout(params['num_layers']['dropout2']))

        model.add(Flatten())
        model.add(Dense(params['fc_size0'], activation=params['activation']))
        model.add(Dropout(params['dropout1']))
        model.add(Dense(self.num_classes_to_predict, activation='softmax'))
        return model


    def _model_fit(self, h_params, model):

        model.compile(loss=keras.losses.categorical_crossentropy, optimizer=h_params['optimizer'](h_params['lr']),
                      metrics=['accuracy'])

        lrate = LearningRateScheduler(classes.step_decay)
        earlyStopping = keras.callbacks.EarlyStopping(monitor='acc', min_delta=0.01, patience=3, verbose=0, mode='auto')
        callbacks_list = [lrate, earlyStopping]

        # training
        y_train_inds = self.dh.get_base_index(self.y_train)  # converting 04567 labels into 01234
        y_train_one_hot = keras.utils.to_categorical(y_train_inds,
                                                     self.num_classes_to_predict)  # getting 01234 classes into one hot
        model.fit(self.x_train, y_train_one_hot, batch_size=128, epochs=h_params['nb_epochs'], callbacks=callbacks_list,
                  verbose=0)
        return model

    def _predict_class(self, model, x, max_acc_level):
        '''Return class label given input X.'''

        y_pred_proba = model.predict(x)
        y_pred_class_inds = y_pred_proba.argsort(axis=1)[:, -max_acc_level:]
        y_test_preds = self.dh.get_class_index(y_pred_class_inds, max_acc_level, self.class_inds)
        return y_test_preds

    def compute_accuracy(self, model, max_acc_level = 3, verbose = True):
        '''Computes accuracy on clasifying classes 04567 on the test set.'''

        self.y_pred_class = self._predict_class(model, self.x_test, max_acc_level)
        accs_at_level = NNNetwork._compute_accuracy_at_level(self, max_acc_level=max_acc_level,
                                             y_pred_classes=self.y_pred_class, y_true=self.y_test.reshape(-1, 1),
                                             verbose=verbose)

        return accs_at_level

    def plot_confusion_matrix(self, model, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
        '''Computes and plot confusion matrix.'''

        if len(self.y_pred_class) == 0:
            # this computes accs_at_level
            print("Getting class prediction first.")
            # max level 1 because we want to have a confusion matrix with normal accuracy
            self.y_pred_class = self._predict_class(model, self.x_test, max_acc_level = 1)
        cm = confusion_matrix(self.y_test, self.y_pred_class[:, -1])
        NNNetwork.plot_confusion_matrix(self, cm, self.class_names, normalize, title, cmap)


