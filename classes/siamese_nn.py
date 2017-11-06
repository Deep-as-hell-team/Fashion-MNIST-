import random
import numpy as np
import keras
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten, Lambda, Input
from keras.initializers import RandomNormal
from keras.optimizers import Adam
from keras import backend as K
from itertools import product
from keras.layers import Conv2D, MaxPooling2D
from keras.callbacks import LearningRateScheduler
from sklearn.model_selection import train_test_split
from classes.nn_network import NNNetwork
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from hyperopt import hp
import classes

class SiameseNN(NNNetwork):
    '''A class running a Siamese network inspired by Koch et al.(2015)
    https://www.cs.cmu.edu/~rsalakhu/papers/oneshot1.pdf
    '''

    def __init__(self, data_handler):

        # c_04567=True because we always work with 04567 when working with Siamese
        super().__init__(data_handler, c_04567=True)

        self.y_train_orig = self.dh.y_train_04567_orig
        self.x_train_orig = self.dh.x_train_04567_orig

        self.accs_at_level = []
        self.__create_pairs()

    def _get_hyper_params(self):
        h_params = {'num_layers': hp.choice('num_layers',
                                            [{'layers': 'two', },
                                             {'layers': 'three',
                                              'filter_size2': hp.choice('filter_size2', [32, 64, 128, 256, 512]),
                                              'kernel_size2': hp.choice('kernel_size2', [3, 4]),
                                              'dropout2': hp.uniform('dropout2', 0, .75)
                                             }
                                             ]),

                    'filter_size0': hp.choice('filter_size0', [16, 32, 64, 128, 256, 512]),
                    'filter_size1': hp.choice('filter_size1', [16, 32, 64, 128, 256, 512]),

                    'kernel_size0': hp.choice('kernel_size0', [3, 4, 5]),
                    'kernel_size1': hp.choice('kernel_size1', [3, 4]),

                    'dropout0': hp.uniform('dropout0', .25, .75),
                    'dropout1': hp.uniform('dropout1', .25, .75),

                    'fc_size0': hp.choice('fc_size0', [64, 128, 256, 512]),

                    'nb_epochs': 80,
                    'lr': hp.choice('lr', [0.01, 0.001, 0.0001]),
                    'optimizer': Adam,
                    'activation': 'relu'
                    }
        return h_params

    def __create_pairs(self):
        ''' Create positive and negative pairs between training data of 04567.
        Alternates between positive and negative pairs.

        Maximally creates (min(class_size) - 1) * num_classes pairs.

        Taken from https://gist.github.com/mmmikael/0a3d4fae965bdbec1f9d.
        '''

        x, y = self.x_train, self.y_train
        num_classes = len(self.class_inds)
        data_ind_per_class = [np.where(y == i)[0] for i in self.class_inds]

        x_pairs, y_pairs = [], []
        min_data_per_class = min([len(data_ind_per_class[d]) for d in range(num_classes)])
        for d in range(num_classes):
            for i in range(min_data_per_class - 1):
                z1, z2 = data_ind_per_class[d][i], data_ind_per_class[d][i + 1]
                x_pairs += [[x[z1], x[z2]]]
                inc = random.randrange(1, num_classes)
                dn = (d + inc) % num_classes  # this guarantees that the same class will not be selected
                z1, z2 = data_ind_per_class[d][i], data_ind_per_class[dn][i]
                x_pairs += [[x[z1], x[z2]]]
                y_pairs += [1, 0]

        x_pairs, y_pairs = np.array(x_pairs), np.array(y_pairs)
        self.x_train_pairs, self.x_val_pairs, self.y_train_pairs, self.y_val_pairs = train_test_split(x_pairs,
                                                                                                      y_pairs,
                                                                                                      test_size=0.3,
                                                                                                      random_state=42)
        print("Number of training pairs is %d " % self.x_train_pairs.shape[0])
        print("Number of val pairs is %d " % self.x_val_pairs.shape[0])

    def _define_model_structure(self, params):
        '''Defines model structure similar to Koch et al. (2015).
        Kernel size is smaller, due to our input.
        We also used dropout, because the network was overfitting.
        '''

        conv_init_w = RandomNormal(mean=0.0, stddev=0.01)
        init_b = RandomNormal(mean=0.5, stddev=0.01)
        fc_init_w = RandomNormal(mean=0.0, stddev=2 * 0.1)

        model = Sequential()
        model.add(Conv2D(params['filter_size0'], (params['kernel_size0'], params['kernel_size0']),
                         activation=params['activation'], input_shape=self.dh.input_shape,
                         kernel_initializer=conv_init_w, bias_initializer=init_b))
        #model.add(Dropout(params['dropout0']))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=2))

        model.add(Conv2D(params['filter_size1'], (params['kernel_size1'], params['kernel_size1']),
                         activation=params['activation'], kernel_initializer=conv_init_w, bias_initializer=init_b))
        model.add(Dropout(params['dropout1']))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=2))


        if params['num_layers']['layers'] == 'threee':
            model.add(Conv2D(params['num_layers']['filter_size2'], (params['num_layers']['kernel_size2'],
                                                                    params['num_layers']['kernel_size2']),
                             activation=params['activation'], kernel_initializer=conv_init_w, bias_initializer=init_b))
            model.add(Dropout(params['num_layers']['dropout2']))

        model.add(Flatten())

        # because we re-use the same instance `base_network`,
        # the weights of the network will be shared across the two branches
        input_a = Input(shape=self.dh.input_shape)
        input_b = Input(shape=self.dh.input_shape)

        processed_a = model(input_a)
        processed_b = model(input_b)

        abs_diff = Lambda(get_abs_diff, output_shape=abs_diff_output_shape)([processed_a, processed_b])

        flattened_weighted_distance = Dense(1,
                                            activation='sigmoid',
                                            kernel_initializer=fc_init_w,
                                            bias_initializer=init_b)(abs_diff)

        return Model(input=[input_a, input_b], output=flattened_weighted_distance)

    def _model_fit(self, h_params, model):

        model.compile(loss='binary_crossentropy', optimizer=h_params['optimizer'](h_params['lr']), metrics=['accuracy'])

        lrate = LearningRateScheduler(classes.step_decay)
        earlyStopping = keras.callbacks.EarlyStopping(monitor='acc', min_delta=0.01, patience=3, verbose=0, mode='auto')
        callbacks_list = [lrate, earlyStopping]

        model.fit([self.x_train_pairs[:, 0], self.x_train_pairs[:, 1]],  # pairs
                   self.y_train_pairs,  # labels of the pairs
                   callbacks=callbacks_list,
                   batch_size=128,
                   nb_epoch=h_params['nb_epochs'],
                   verbose = 0
                  )

        return model


    def _predict_pair(self, model, x_pair1, x_pair2):
        '''Return true/false whether a pair of x is a pair or not.'''

        y_pred_proba = model.predict([x_pair1, x_pair2])
        y_pred_class = y_pred_proba > 0.5
        return y_pred_class.reshape(-1)

    def get_accuracy(self, model, max_acc_level = -1, verbose = True):
        '''Computes the verification (binary) accuracy of classifying pairs of images'''

        y_pred_class = self._predict_pair(model, self.x_val_pairs[:, 0], self.x_val_pairs[:, 1])

        accur = np.sum(y_pred_class == self.y_val_pairs) / len(self.y_val_pairs)
        if verbose:
            print('* Accuracy of classifying the val set: {:.2%}'.format(accur))
        return [{'acc':accur, 'y_pred': y_pred_class}]


    def _predict_class(self, model, x_test, x_train, y_train, max_acc_level):
        '''One shot class prediction.'''

        x_test_pairs = np.array(list(product(x_test, x_train)))

        # predict the probability of a pair being similar.
        y_test_pred_prob = model.predict([x_test_pairs[:, 0], x_test_pairs[:, 1]])

        # reshaping such that each row contains k*number of class probabilities
        y_test_pred_prob = y_test_pred_prob.reshape(len(x_test), len(x_train))

        # getting top k prediction for a class
        y_test_pred_class_inds = y_test_pred_prob.argsort(axis=1)[:, -max_acc_level:]

        y_test_preds = np.array([y_train[y_test_pred_class_inds[i]]
                                 for i in range(len(y_test_pred_class_inds))])
        return y_test_preds

    def compute_one_shot_accuracy(self, model, max_acc_level = 1, verbose = True):
        '''Compute one-shot classificaiton accuracy.'''

        y_test_preds = self._predict_class(model, self.x_test, self.x_train_orig,  self.y_train_orig, max_acc_level)
        self.accs_at_level = NNNetwork._compute_accuracy(self, max_acc_level=max_acc_level,
                                                         y_pred_classes=y_test_preds, y_true = self.y_test.reshape(-1, 1),
                                                         verbose = verbose)

        self.accs_at_level


    def plot_confusion_matrix(self, model, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
        '''Computes and plot confusion matrix of one-shot classification.'''

        if len(self.accs_at_level) == 0:
            # this computes accs_at_level
            print("Computing one shot accuracy first.")
            self.compute_one_shot_accuracy(max_acc_level = 1, verbose = False)

        y_pred_level_1 = self.accs_at_level[0]['y_pred']
        cm = confusion_matrix(self.dh.y_test_04567, y_pred_level_1)
        NNNetwork.plot_confusion_matrix(self, cm, self.dh.class_names04567, normalize, title, cmap)

# additional functions that are used for NN

def abs_diff_output_shape(shapes):
    shape1, shape2 = shapes
    return shape1

def get_abs_diff(vects):
    x, y = vects
    return K.abs(x - y)

# def contrastive_loss(y, d):
#     ''' Contrastive loss from Hadsell-et-al.'06
#         http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
#     '''
#     margin = 1
#     return K.mean((1 - y) * K.square(d) + y * K.square(K.maximum(margin - d, 0)))

# def get_l2_norm(vects):
#     x, y = vects
#     return K.sqrt(K.mean(K.square(x - y)))

