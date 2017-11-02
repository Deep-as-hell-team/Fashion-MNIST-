import numpy as np
import keras
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten, Lambda, Input
from hyperopt import Trials, STATUS_OK, tpe, hp, fmin
from keras.callbacks import LearningRateScheduler
from classes.base_model import BaseModel
from keras.optimizers import Adam
import classes


class TransferLearning(BaseModel):

    def __init__(self, data_handler, c_04567):
        super().__init__(data_handler, c_04567)
        self.base_model = None

    def _get_hyper_params(self):
        '''Return the space of hyperparameters to tune the model.'''

        h_params = {
                 'train_conv0': hp.choice('train_conv0', [True, False]),
                 'train_conv1': hp.choice('train_conv1', [True, False]),

                 'dropout0': hp.uniform('dropout0', .25, .75),
                 'fc_size0': hp.choice('fc_size0', [64, 128, 256]),

                 'nb_epochs': 40,
                  'lr': hp.choice('lr', [0.001, 0.0001, 0.00001]),
                 'optimizer': Adam,
                 'activation': 'relu'
                 }
        return h_params

    def _define_model_structure(self, params):
        '''Given the base model, defines which layers will be trainable and set FC layers.'''

        if self.base_model is None:
            raise ValueError("Base model has not been defined. Call set_base_model(...) first.")

        self.base_model.layers[0].trainable = params['train_conv0'] and params['train_conv1']
        self.base_model.layers[1].trainable = params['train_conv1']

        # adding a dense layer behind flatten_1 layers (i.e., layer[-4])
        x = Dense(params['fc_size0'], activation=params['activation'], name="dense_new0")(
            self.base_model.layers[-4].output)
        x = Dropout(params['dropout0'], name='dropout_new0')(x)
        predictions = Dense(self.num_classes_to_predict, activation='softmax', name='softmax_new0')(x)
        model = Model(input=self.base_model.input, output=predictions)
        return model

    def _model_fit(self, h_params, model):
        model.compile(loss=keras.losses.categorical_crossentropy, optimizer=h_params['optimizer'](h_params['lr']),
                      metrics=['accuracy'])

        # converting 04567 labels into 01234
        y_train_inds = self.dh.get_base_index(self.y_train)
        # getting 01234 classes into one hot
        y_train_one_hot = keras.utils.to_categorical(y_train_inds, self.num_classes_to_predict)

        # setting callback
        lrate = LearningRateScheduler(classes.step_decay)
        earlyStopping = keras.callbacks.EarlyStopping(monitor='acc', min_delta=0.01, patience=3, verbose=0, mode='auto')
        callbacks_list = [lrate, earlyStopping]

        model.fit(self.x_train, y_train_one_hot, batch_size=128, epochs=h_params['nb_epochs'], callbacks=callbacks_list,
                  verbose=0)
        return model

    def set_base_model(self, base_model):
        self.base_model = base_model






