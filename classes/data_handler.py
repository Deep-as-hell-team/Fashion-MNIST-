import numpy as np
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import keras

class DataHandler():
    def __is_channel_first(self):
        return K.image_data_format() == 'channels_first'

    def __preprocess_data(self, X, y):
        '''Preprocess data such that it convert 1D X into 3D array (h, w, c) and normalized the data.'''

        if self.__is_channel_first():
            X = X.reshape(X.shape[0], 1, self.img_rows, self.img_cols)
        else:
            X = X.reshape(X.shape[0], self.img_rows, self.img_cols, 1)

        X = X.astype('float32') / 255
        # convert class vectors to binary class matrices
        # y = keras.utils.to_categorical(y, num_classes)
        return X, y

    def load_data(self, path=None):
        if path is None:
            self.__load_fashion_data()

    def get_class_index(self, y_pred_inds, max_acc_level, class_inds):
        '''Convert an array of e.g. 0-4 to class indexes e.g. 04567

        @:param y_pred_inds: 2D array of (num_data, max_acc_level)
        @:param max_acc_level: number of best guesses
        '''
        y_pred_inds = y_pred_inds.reshape(-1)
        y_test_preds = np.array([class_inds[y_ind] for y_ind in y_pred_inds])
        y_test_preds = y_test_preds.reshape(-1, max_acc_level)

        return y_test_preds

    def get_base_index(self, y_class_inds):
        '''Convert an arrray of e.g. 04567 into 01234

        @:param y_class_inds: class indexes (04567)
        '''
        class_inds = np.unique(y_class_inds)
        return [np.where(class_inds == y) for y in y_class_inds]

    def __load_fashion_data(self, path="FashionData/FashionPDEngDM.npz"):
        '''Load Fashion MNIST data set.'''

        self.img_rows, self.img_cols = 28, 28
        self.ind_04567 = [0, 4, 5, 6, 7]
        self.ind_12389 = [1, 2, 3, 8, 9]
        data = np.load(path)  # load data

        self.class_names = np.array(['T-shirt/top', 'Trouser', 'Pullover', 'Dress',
                                    'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'])
        self.class_names04567 = self.class_names[self.ind_04567]
        self.class_names12389 = self.class_names[self.ind_12389]

        #self.num_classes_04567 = 5 # classes to classify (04567)

        # Extracting data
        # - Classes 04567 have one data observation per class.
        # - Classes 12389 have 6000 observation for class.

        x_train_12389, y_train_12389 = data["x_train_12389_labeled"], data["y_train_12389_labeled"]
        x_test_12389, y_test_12389 = data["x_test_12389"], data["y_test_12389"]

        x_train_04567, y_train_04567 = data["x_train_04567_labeled"], data["y_train_04567_labeled"]
        x_train_04567_unlabeled = data["x_train_04567_unlabeled"]
        x_test_04567, y_test_04567 = data["x_test_04567"], data["y_test_04567"]

        self.input_shape = (1, self.img_rows, self.img_cols) if self.__is_channel_first() else (
                            self.img_rows, self.img_cols, 1)


        # preprocessing & saving data
        self.x_train_12389, self.y_train_12389 = self.__preprocess_data(x_train_12389, y_train_12389)
        self.x_train_04567, self.y_train_04567 = self.__preprocess_data(x_train_04567, y_train_04567)

        self.x_test_12389, self.y_test_12389 = self.__preprocess_data(x_test_12389, y_test_12389)
        self.x_test_12389, self.y_test_12389 = self.__preprocess_data(x_test_12389, y_test_12389)
        self.x_test_04567, self.y_test_04567 = self.__preprocess_data(x_test_04567, y_test_04567)

        # set the origianlly loaded data
        self.__set_original_data(self.x_train_04567, self.y_train_04567,
                                 self.x_train_12389, self.y_train_12389,
                                 self.x_test_04567, self.y_test_04567,
                                 self.x_test_12389, self.y_test_12389
                                 )

    def __visualize_augmented_data(self, x, y, x_batch, y_batch, num_columns=2):
        '''Visualize a batch of augmented data.'''

        f, axarr = plt.subplots(x.shape[0], num_columns, figsize=(5, 5))
        for j in range(len(x_batch)):
            ground_ind = np.where(y == y_batch[j])
            axarr[j, 0].imshow(x[ground_ind].reshape((self.img_rows, self.img_rows)), cmap='gray')
            axarr[j, 1].imshow(x_batch[j].reshape((self.img_rows, self.img_rows)), cmap='gray')
            axarr[j, 0].axis('off')
            axarr[j, 1].axis('off')
        plt.show()

    def augment_data(self, use_train, c_04567, N, visualize=False, **kwargs):
        ''' Augment currectly loaded data.

        @:param use_train: BOOL wheather to augment train data
        @:param class_04567: BOOL wheather to augment 04567 or 12389
        @:param N: number of images to generates per class
        @:param visualize: whether to visualize the augmented images
        @:param **kwaks: parameters to the Keras.ImageGenerator

        Return: (x, y) augmneted data
        '''

        num_columns = 2  # for priting
        if kwargs is None:
            print("Parameters for the image generator are empty. Running the generator with the defautl params.")

        x, y = self.__get_original_data(return_train=use_train, c_04567=c_04567)

        img_gen = ImageDataGenerator(**kwargs)
        # print(img_gen)
        img_gen.fit(x)

        data_gen = img_gen.flow(x, y, batch_size=x.shape[0])  # we generate as much as the size of the data

        x_aug = []
        y_aug = []

        for i, data_batch in enumerate(data_gen):
            if N <= i:  # in every iteration, one instance per class is generated
                break

            x_batch, y_batch = data_batch
            x_aug.append(x_batch)
            y_aug.append(y_batch)

            if visualize:
                self.__visualize_augmented_data(x, y, x_batch, y_batch)

        # reshaping such that the array has the shape of (# data, 28, 28, 1)
        x_aug = np.array(x_aug).reshape((-1, self.img_rows, self.img_rows, 1))

        # reshaping to shape of (# data)
        y_aug = np.array(y_aug).reshape((-1))

        self.__set_data(x_aug, y_aug, set_train=use_train, c_04567=c_04567)

    def get_data(self, return_train, c_04567):
        '''Getter for the loaded data (can be alrady augmented).'''

        if return_train and c_04567:
            return self.x_train_04567, self.y_train_04567

        if return_train and not c_04567:
            return self.x_train_12389, self.y_train_12389

        if not return_train and c_04567:
            return self.x_test_04567, self.y_test_04567

        if not return_train and not c_04567:
            return self.x_test_12389, self.y_test_12389

    def __set_data(self, x, y, set_train, c_04567):
        '''Setter for the loaded data.'''
        if set_train and c_04567:
            self.x_train_04567, self.y_train_04567 = x, y

        if set_train and not c_04567:
            self.x_train_12389, self.y_train_12389 = x, y

        if not set_train and c_04567:
            self.x_test_04567, self.y_test_04567 = x, y

        if not set_train and not c_04567:
            self.x_test_12389, self.y_test_12389 = x, y

    def __get_original_data(self, return_train, c_04567):
        '''Getter for the initially loaded data (they are untouched).'''

        if return_train and c_04567:
            return self.x_train_04567_orig, self.y_train_04567_orig

        if return_train and not c_04567:
            return self.x_train_12389_orig, self.y_train_12389_orig

        if not return_train and c_04567:
            return self.x_test_04567_orig, self.y_test_04567_orig

        if not return_train and not c_04567:
            return self.x_test_12389_orig, self.y_test_12389_orig

    def __set_original_data(self, x_train_04567, y_train_04567, x_train_12389, y_train_12389, x_test_04567,
                            y_test_04567,
                            x_test_12389, y_test_12389):
        '''Setter for the initially loaded data (they are untouched) - is called only once.'''

        self.x_train_04567_orig, self.y_train_04567_orig = np.copy(x_train_04567), np.copy(y_train_04567)
        self.x_train_12389_orig, self.y_train_12389_orig = np.copy(x_train_12389), np.copy(y_train_12389)
        self.x_test_04567_orig, self.y_test_04567_orig = np.copy(x_test_04567), np.copy(y_test_04567)
        self.x_test_12389_orig, self.y_test_12389_orig = np.copy(x_test_12389), np.copy(y_test_12389)

    def __get_val_data(self):
        raise (NotImplementedError())

    def train_model(self):
        raise (NotImplementedError())

    def validate_model(self):
        raise (NotImplementedError())

    def test_model(self):
        raise (NotImplementedError())