import abc
import numpy as np
import matplotlib.pyplot as plt
import itertools
import keras

class NNNetwork():

    def __init__(self, data_handler):
        self.dh = data_handler

    @abc.abstractmethod
    def define_model_structure(self):
        '''Define a model structure and saves it in self.model.'''
        pass

    @abc.abstractmethod
    def train_model(self, x, model, model_name, load_model):
        '''Trains the define model'''

        if load_model:
            self.model = keras.models.load_model(model_name)
        else:
            if x.shape[0] == 0:
                raise InputError('You need to create / have training data first. ')
            if model is None:
                raise InputError('You need to first define model structure. Call define_model_structure.')

    def hyper_param_tuning(self, **kwargs):
        # self.train_model(self.model, )
        pass

    @abc.abstractmethod
    def compute_accuracy(self, verbose):
        '''Compute accuracy on the test set.'''

    def _compute_accuracy_at_level(self, max_acc_level, y_pred_classes, y_true, verbose = True):

        accs_at_level = []
        for acc_level in range(1, max_acc_level + 1):
            # y_test_pred_k and dh.y_test_04567 needs to be 2D
            y_test_pred_at_level = y_pred_classes[:, -acc_level:]
            test_accur = np.sum(np.any(y_test_pred_at_level == y_true, axis=1)) / len(y_true)
            accs_at_level.append({'accuracy': test_accur, 'y_pred': y_test_pred_at_level.reshape(-1)})
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
