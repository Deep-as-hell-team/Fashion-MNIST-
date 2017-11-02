import abc
import numpy as np
import matplotlib.pyplot as plt
import itertools
import keras
from hyperopt import Trials, STATUS_OK, tpe, hp, fmin

class NNNetwork():

    def __init__(self, data_handler, c_04567):
        self.dh = data_handler
        self.tuning_iter = 0

        self.x_train, self.y_train = self.dh.get_data(return_train=True, c_04567=c_04567)
        self.x_test, self.y_test = self.dh.get_data(return_train=False, c_04567=c_04567)

        if c_04567:
            self.class_inds = self.dh.ind_04567
            self.class_names = self.dh.class_names04567
        else:
            self.class_inds = self.dh.ind_12389
            self.class_names = self.dh.class_names12389

        self.num_classes_to_predict = len(self.class_inds)

    @abc.abstractmethod
    def _define_model_structure(self):
        '''Define an arbitary structure model structure and saves it in self.model.'''
        raise NotImplemented("Not implemented.")

    @abc.abstractmethod
    def _model_fit(self, h_params, model):
        raise NotImplemented("Not implemented.")

    def _get_hyper_params(self):
        raise NotImplemented("Not implemented.")

    def train_model(self, h_params, hyper_param_tuning = True):
        '''Train a model given hyper params tuning.

        @:param h_params: space of hyper params
        @:return dict(loss, status, model)
        '''

        model = self._define_model_structure(h_params)
        model = self._model_fit(h_params, model)

        accs = self.compute_accuracy(model, max_acc_level=1, verbose=False)
        acc = accs[0]['acc']
        if hyper_param_tuning:
            print('Test accuracy of model {}/{} is {:.2%}:'.format(self.tuning_iter, self.max_evals, acc))
        else:
            print('Test accuracy of model {:.2%}:'.format(acc))
        self.tuning_iter += 1
        return {'loss': -acc, 'status': STATUS_OK, 'model': model}

    def tune_hyper_params(self, max_evals):
        '''Tune hyper parameters. Returns the best model'''
        # nice exmaple at https://github.com/fchollet/keras/issues/1591

        trials = Trials()
        self.tuning_iter = 0
        self.max_evals = max_evals
        # tpe = Tree of Parzen Estimators
        best = fmin(self.train_model, self._get_hyper_params(), algo=tpe.suggest, max_evals=max_evals, trials=trials)
        best_model_acc = np.max([-result['loss'] for result in trials.results])  # it is interpreted as loss
        return best, best_model_acc, trials

    @abc.abstractmethod
    def compute_accuracy(model, max_acc_level, verbose):
        '''Compute accuracy on the test set.'''
        raise NotImplemented("Not implemented.")

    def _compute_accuracy_at_level(self, max_acc_level, y_pred_classes, y_true, verbose = True):

        accs_at_level = []
        for acc_level in range(1, max_acc_level + 1):
            # y_test_pred_k and dh.y_test_04567 needs to be 2D
            y_test_pred_at_level = y_pred_classes[:, -acc_level:]
            test_accur = np.sum(np.any(y_test_pred_at_level == y_true, axis=1)) / len(y_true)
            accs_at_level.append({'acc': test_accur, 'y_pred': y_test_pred_at_level.reshape(-1)})
            if verbose:
                print('*Accuracy of classifying the test set at level {}: {:.2%}'.format(acc_level, test_accur))
        return accs_at_level

    def plot_confusion_matrix(self, cm, classes_names, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
        ''' Plots the confusion matrix.

        @:param cm: confusion matrix computed by sklearn
        @:param classes_names: names of the classes
        @:param normalize: whether to print normlized matrix
        @:param title: title of the confusion matrix
        @:param cmap: colors of the plot

        '''
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            print("Normalized confusion matrix")
        else:
            print('Confusion matrix, without normalization')

        print(cm)

        plt.figure()
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(classes_names))
        plt.xticks(tick_marks, classes_names, rotation=45)
        plt.yticks(tick_marks, classes_names)

        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], fmt),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.show()
