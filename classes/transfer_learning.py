import numpy as np
import keras
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten, Lambda, Input
from hyperopt import Trials, STATUS_OK, tpe, hp, fmin
from keras.callbacks import LearningRateScheduler
from classes.base_model import BaseModel
import classes


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

    def __train_model_tuning(self, params):
        '''Train a model during hyper params tuning.

        @:base_model: pretrained model
        @:param params: space of hyper params
        @:return dict(loss, status, model)
        '''

        # to make sure that the first layer wont be trained while the second not
        self.base_model.layers[0].trainable = params['train_conv0'] and params['train_conv1']
        self.base_model.layers[1].trainable = params['train_conv1']

        # adding a dense layer behind flatten_1 layers (i.e., layer[-4])
        x = Dense(params['fc_size0'], activation=params['activation'], name="dense_new0")(self.base_model.layers[-4].output)
        x = Dropout(params['dropout0'], name='dropout_new0')(x)
        predictions = Dense(self.num_classes_to_predict, activation='softmax', name='softmax_new0')(x)
        model = Model(input=self.base_model.input, output=predictions)

        model.compile(loss=keras.losses.categorical_crossentropy, optimizer=params['optimizer'],
                      metrics=['accuracy'])

        # converting 04567 labels into 01234
        y_train_inds = self.dh.get_base_index(self.y_train)
        # getting 01234 classes into one hot
        y_train_one_hot = keras.utils.to_categorical(y_train_inds, self.num_classes_to_predict)

        # setting callback
        lrate = LearningRateScheduler(classes.step_decay)
        earlyStopping = keras.callbacks.EarlyStopping(monitor='acc', min_delta = 0.01, patience=3, verbose=0, mode='auto')
        callbacks_list = [lrate, earlyStopping]

        model.fit(self.x_train, y_train_one_hot, batch_size=128, epochs=params['nb_epochs'], callbacks = callbacks_list, verbose = 0)

        # model evaluation
        y_test_inds = self.dh.get_base_index(self.y_test)
        y_test_one_hot = keras.utils.to_categorical(y_test_inds, self.num_classes_to_predict)
        score, acc = model.evaluate(self.x_test, y_test_one_hot, verbose=0)

        print('Test accuracy of model {}/{} is {:.2%}:'.format(self.tuning_iter, self.max_evals,  acc))
        self.tuning_iter+=1
        return {'loss': -acc, 'status': STATUS_OK, 'model': model}

    def get_hyper_param_space(self):
        '''Return the space of hyperparameters to tune the model.'''

        space = {'train_conv0': hp.choice('train_conv0', [True, False]),
                 'train_conv1': hp.choice('train_conv1', [True, False]),

                 'dropout0': hp.uniform('dropout0', .25, .75),
                 'fc_size0': hp.choice('fc_size0', [64, 128, 256]),

                 'nb_epochs': 40,
                 'optimizer': 'adam',
                 'activation': 'relu'
                 }
        return space

    def tune_hyper_params(self, base_model, max_evals):
        '''Tune hyper parameters. Returns the best model'''

        # nice exmaple at https://github.com/fchollet/keras/issues/1591

        self.base_model = base_model
        return super(BaseModel, self).tune_hyper_params()



