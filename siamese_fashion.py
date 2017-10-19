# -*- coding: utf-8 -*-
"""
Created on Wed Oct 18 12:45:14 2017

@author: sibrahim
"""

import random
import numpy as np


from keras.datasets import mnist
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Input, Lambda
from keras.optimizers import SGD, RMSprop
from keras import backend as K
from sklearn.metrics import classification_report

from sklearn.metrics import accuracy_score as accuracy

#
data=np.load("FashionData/FashionPDEngDM.npz")

##Labeled training set for classes 1,2,3,8,9 (30000 samples)
x_train_12389_labeled=data["x_train_12389_labeled"]
y_train_12389_labeled=data["y_train_12389_labeled"]

##Labeled training set for classes 0,4,5,6,7 (just 5 samples)
x_train_04567_labeled=data["x_train_04567_labeled"]
y_train_04567_labeled=data["y_train_04567_labeled"]

##Unlabeled training set for classes 0,4,5,6,7 (29992 samples)
x_train_04567_unlabeled=data["x_train_04567_unlabeled"]

##Labeled test set for classes 1,2,3,8,9
x_test_12389=data["x_test_12389"]
y_test_12389=data["y_test_12389"]

##Labeled test set for classes 0,4,5,6,7 (this is where we are interested to obtain the highest accuracy possible - project goal)
x_test_04567=data["x_test_04567"]
y_test_04567=data["y_test_04567"]
    
def get_abs_diff(vects):
    x, y = vects
    return K.abs(x - y)  

def abs_diff_output_shape(shapes):
    shape1, shape2 = shapes
    return shape1  

def create_pairs(x, digit_indices):
    '''Positive and negative pair creation.
    Alternates between positive and negative pairs.
    '''
    pairs = []
    labels = []
    n = min([len(digit_indices[d]) for d in range(10)]) - 1
    for d in range(10):
        for i in range(n):
            z1, z2 = digit_indices[d][i], digit_indices[d][i+1]
            pairs += [[x[z1], x[z2]]]
            inc = random.randrange(1, 10)
            dn = (d + inc) % 10
            z1, z2 = digit_indices[d][i], digit_indices[dn][i]
            pairs += [[x[z1], x[z2]]]
            labels += [1, 0]
    return np.array(pairs), np.array(labels)


def create_base_network(input_dim):
    '''Base network to be shared (eq. to feature extraction).
    '''
    seq = Sequential()
    seq.add(Dense(128, input_shape=(input_dim,), activation='relu'))
    seq.add(Dropout(0.1))
    seq.add(Dense(128, activation='relu'))
    seq.add(Dropout(0.1))
    seq.add(Dense(128, activation='relu'))
    return seq

#

np.random.seed(1337)  # for reproducibility

x_train = np.vstack((x_train_12389_labeled, x_train_04567_labeled, x_train_04567_labeled))
y_train = np.append(y_train_12389_labeled, y_train_04567_labeled)
y_train = np.append(y_train, y_train_04567_labeled)
x_test = np.vstack((x_test_12389, x_test_04567))
y_test = np.append(y_test_12389, y_test_04567)
print('shape X_train', x_train.shape)
print('shape y_train', y_train.shape)
print('shape X_test', x_test.shape)
print('shape y_test', y_test.shape)
X_train = x_train.astype('float32')
X_test = x_test.astype('float32')
X_train /= 255
X_test /= 255
input_dim = 784
nb_epoch = 20

# create training+test positive and negative pairs
digit_indices = [np.where(y_train == i)[0] for i in range(10)]
tr_pairs, tr_y = create_pairs(X_train, digit_indices)
print('digit_indices1', digit_indices)
print('X_train1', X_train)
print('shape digit_indices1', len(digit_indices))
print('shape X_train1', len(X_train))
print('tr_pairs1', tr_pairs)
print('tr_y 1', tr_y)


digit_indices = [np.where(y_test == i)[0] for i in range(10)]
te_pairs, te_y = create_pairs(X_test, digit_indices)

# network definition
base_network = create_base_network(input_dim)

input_a = Input(shape=(input_dim,))
input_b = Input(shape=(input_dim,))

# because we re-use the same instance `base_network`,
# the weights of the network
# will be shared across the two branches
processed_a = base_network(input_a)
processed_b = base_network(input_b)

abs_diff = Lambda(get_abs_diff, output_shape = abs_diff_output_shape)([processed_a, processed_b])

flattened_weighted_distance = Dense(1, activation = 'sigmoid')(abs_diff)

model = Model(input=[input_a, input_b], output = flattened_weighted_distance)

# train

rms = RMSprop()
model.compile(loss = 'binary_crossentropy', optimizer=rms, metrics = ['accuracy'])

model.fit([tr_pairs[:, 0], tr_pairs[:, 1]], tr_y,
          #validation_data=([te_pairs[:, 0], te_pairs[:, 1]], te_y),
          batch_size=128, nb_epoch=nb_epoch)

# compute final accuracy on training and test sets
tr_pred = model.predict([tr_pairs[:, 0], tr_pairs[:, 1]])
tr_acc = accuracy(tr_y, tr_pred.round())

te_pred = model.predict([te_pairs[:, 0], te_pairs[:, 1]])
te_acc = accuracy(te_y, te_pred.round())

print('* Accuracy on the training set: {:.2%}'.format(tr_acc))
print('* Accuracy on the test set: {:.2%}'.format(te_acc))

Y_test = np.argmax(te_y, axis=1) # Convert one-hot to index
y_pred = model.predict_classes(te_pred)
print(classification_report(te_y, te_pred))
