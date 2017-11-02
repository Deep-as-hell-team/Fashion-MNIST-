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
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


class SiameseNN(NNNetwork):

    def __init__(self, data_handler):
        NNNetwork.__init__(self, data_handler)
        self.accs_at_level = []

    def create_pairs(self):
        ''' Create positive and negative pairs between training data of 04567.
        Alternates between positive and negative pairs.

        Maximally creates (min(class_size) - 1) * num_classes pairs.
        '''

        x, y = self.dh.x_train_04567, self.dh.y_train_04567

        inds = self.dh.ind_04567
        class_indices = [np.where(y == i)[0] for i in inds]
        num_classes = len(inds)

        x_pairs = []
        y_pairs = []
        min_data_per_class = min([len(class_indices[d]) for d in range(num_classes)])
        for d in range(num_classes):
            for i in range(min_data_per_class - 1):
                z1, z2 = class_indices[d][i], class_indices[d][i + 1]
                x_pairs += [[x[z1], x[z2]]]
                inc = random.randrange(1, num_classes)
                dn = (d + inc) % num_classes  # this guarantees that the same class will not be selected
                z1, z2 = class_indices[d][i], class_indices[dn][i]
                x_pairs += [[x[z1], x[z2]]]
                y_pairs += [1, 0]

        x_pairs, y_pairs = np.array(x_pairs), np.array(y_pairs)
        self.x_train_pairs, self.x_val_pairs, self.y_train_pairs, self.y_val_pairs = train_test_split(x_pairs,
                                                                                                      y_pairs,
                                                                                                      test_size=0.3,
                                                                                                      random_state=42)
        print("Number of training pairs is %d " % self.x_train_pairs.shape[0])
        print("Number of val pairs is %d " % self.x_val_pairs.shape[0])

    def define_model_structure(self):
        '''Define network structure by Koch.
        10, 10 --> 7,7, --> 4,4 --> 4,4

        '''
        conv_init_w = RandomNormal(mean=0.0, stddev=0.01)
        init_b = RandomNormal(mean=0.5, stddev=0.01)
        fc_init_w = RandomNormal(mean=0.0, stddev=2 * 0.1)

        model = Sequential()
        model.add(Conv2D(64, kernel_size=(7, 7), activation='relu', input_shape=self.dh.input_shape,
                         kernel_initializer=conv_init_w, bias_initializer=init_b))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=2))
        model.add(Conv2D(128, (4, 4), activation='relu', kernel_initializer=conv_init_w, bias_initializer=init_b))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=2))
        model.add(Conv2D(128, (4, 4), activation='relu', kernel_initializer=conv_init_w, bias_initializer=init_b))
        model.add(Flatten())

        input_a = Input(shape=self.dh.input_shape)
        input_b = Input(shape=self.dh.input_shape)

        # because we re-use the same instance `base_network`,
        # the weights of the network will be shared across the two branches

        processed_a = model(input_a)
        processed_b = model(input_b)

        # abs_diff = Lambda(get_l2_norm, output_shape = abs_diff_output_shape)([processed_a, processed_b])

        abs_diff = Lambda(get_abs_diff, output_shape=abs_diff_output_shape)([processed_a, processed_b])

        flattened_weighted_distance = Dense(1,
                                            activation='sigmoid',
                                            kernel_initializer=fc_init_w,
                                            bias_initializer=init_b)(abs_diff)

        self.model = Model(input=[input_a, input_b], output=flattened_weighted_distance)

    def train_model(self, nb_epoch=5, model_name='transfer_learning.h5', load_model=False):
        '''Train model.'''

        NNNetwork.train_model(self, x=self.x_train_pairs, model=self.model, load_model=load_model, model_name=model_name)

        if not load_model:
            optimizer = Adam()
            self.model.compile(loss='binary_crossentropy', optimizer=optimizer)

            lrate = LearningRateScheduler(step_decay)
            callbacks_list = [lrate]

            self.model.fit([self.x_train_pairs[:, 0], self.x_train_pairs[:, 1]],  # pairs
                      self.y_train_pairs,  # labels of the pairs
                      callbacks=callbacks_list,
                      batch_size=128,
                      nb_epoch=nb_epoch)

            self.model.save(model_name)

    def compute_accuracy(self, verbose = True):
        '''Computes the verification (binary) accuracy of classifying pairs of images.'''

        if self.model is None:
            raise ValueError('Self.model is not defined. Run define_model_structure() and train_model() first. ')

        # getting predictions
        self.y_test_pred_proba = self.model.predict([self.x_val_pairs[:, 0], self.x_val_pairs[:, 1]])

        # getting labels
        y_pred_class = self.y_test_pred_proba > 0.5
        y_pred_class = y_pred_class.reshape(-1)

        # compute accuracy
        accur = np.sum(y_pred_class == self.y_val_pairs) / len(self.y_val_pairs)
        if verbose:
            print('* Accuracy of classifying the val set: {:.2%}'.format(accur))
        return accur, y_pred_class


    def compute_one_shot_accuracy(self, max_acc_level = 1, verbose = True, k=1):
        '''Compute test accuracy of classification on test data 04567.


        @:param k: number of augmented pictures. NotImplemented.
        @:param max_acc_level: if max_acc_level = 2, the class will look accuracy at level 1 and 2
        '''

        # create pairs
        x_test_pairs = np.array(list(product(self.dh.x_test_04567, self.dh.x_train_04567_orig)))

        # predict the probability of a pair being similar.
        y_test_pred_prob = self.model.predict([x_test_pairs[:, 0], x_test_pairs[:, 1]])

        # reshaping such that each row contains k*number of class probabilities
        y_test_pred_prob = y_test_pred_prob.reshape(len(self.dh.x_test_04567), len(self.dh.x_train_04567_orig))

        # getting top k prediction for a class
        y_test_pred_class_inds = y_test_pred_prob.argsort(axis=1)[:, -max_acc_level:]

        y_test_preds = np.array([self.dh.y_train_04567_orig[y_test_pred_class_inds[i]]
                                 for i in range(len(y_test_pred_class_inds))])

        self.accs_at_level = NNNetwork._compute_accuracy_at_level(self, max_acc_level=max_acc_level,
                                                                  y_pred_classes=y_test_preds, y_true = self.dh.y_test_04567.reshape(-1, 1),
                                                                  verbose = verbose)

        self.accs_at_level

    def plot_confusion_matrix(self, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
        '''Computes and plot confusion matrix of one-shot classification.'''

        if len(self.accs_at_level) == 0:
            # this computes accs_at_level
            print("Computing one shot accuracy first.")
            self.compute_one_shot_accuracy(max_acc_level = 1, verbose = False)

        y_pred_level_1 = self.accs_at_level[0]['y_pred']
        cm = confusion_matrix(self.dh.y_test_04567, y_pred_level_1)
        class_names_04567 = self.dh.class_names[np.unique(self.dh.y_test_04567)]
        NNNetwork.plot_confusion_matrix(self, cm, class_names_04567, normalize, title, cmap)

def step_decay(epoch):
    '''Learning rate step decay following the original paper.'''
    initial_lrate = 0.001
    drop = 0.99
    epochs_drop = 1
    lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))
    return lrate

def contrastive_loss(y, d):
    ''' Contrastive loss from Hadsell-et-al.'06
        http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    '''
    margin = 1
    return K.mean((1 - y) * K.square(d) + y * K.square(K.maximum(margin - d, 0)))

def get_abs_diff(vects):
    x, y = vects
    return K.abs(x - y)

def get_l2_norm(vects):
    x, y = vects
    return K.sqrt(K.mean(K.square(x - y)))

def abs_diff_output_shape(shapes):
    shape1, shape2 = shapes
    return shape1