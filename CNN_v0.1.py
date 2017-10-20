##This code is a CNN example for the PDEng DM module, 2017-2018, TU/e

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report
from keras.metrics import categorical_accuracy, binary_accuracy
import numpy as np


# read training and test sets
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



## Example of trainining a CNN on classes 1,2,3,8,9
# input image dimensions
img_rows, img_cols = 28, 28
#set CNN parameters
batch_size = 100
num_classes = 10
epochs = 10

#create tensor variant of 2D images

def preprocess_data(X, y):
    #create tensor variant of 2D images
    if K.image_data_format() == 'channels_first':
        X = X.reshape(X.shape[0], 1, img_rows, img_cols)
        input_shape = (1, img_rows, img_cols)
        print('inputshape1', input_shape)
    else:
        X = X.reshape(X.shape[0], img_rows, img_cols, 1)
        input_shape = (img_rows, img_cols, 1)
        print('inputshape2', input_shape)
    X = X.astype('float32') / 255.

    # convert class vectors to binary class matrices
    y = keras.utils.to_categorical(y, num_classes)
    return X, y, input_shape
x_train_12389_labeled, y_train_12389_labeled, input_shape = preprocess_data(x_train_12389_labeled, y_train_12389_labeled)
x_test_12389, y_test_12389, _ = preprocess_data(x_test_12389, y_test_12389)

x_train_04567_labeled, y_train_04567_labeled, input_shape = preprocess_data(x_train_04567_labeled, y_train_04567_labeled)
x_test_04567, y_test_04567, _ = preprocess_data(x_test_04567, y_test_04567)
print('len x_train_04567_labeled', len(x_train_04567_labeled))
print('shape x_train_04567_labeled', x_train_04567_labeled.shape)

x_train = np.vstack((x_train_12389_labeled, x_train_04567_labeled))
y_train = np.vstack((y_train_12389_labeled, y_train_04567_labeled))
print('len xtrain', len(x_train))
print('shape xtrain', x_train.shape)
print('shape ytrain', y_train.shape)
x_test = np.vstack((x_test_12389, x_test_04567))
y_test = np.vstack((y_test_12389, y_test_04567))
print('len xtest', len(x_test))

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer='rmsprop',
              metrics=['accuracy'])

datagen = ImageDataGenerator(
    featurewise_center=True,
    featurewise_std_normalization=True,
    rotation_range=90,
    horizontal_flip=True,
    vertical_flip = True)

# compute quantities required for featurewise normalization
# (std, mean, and principal components if ZCA whitening is applied)
datagen.fit(x_train)

# fits the model on batches with real-time data augmentation:
model.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size),
                    steps_per_epoch=len(x_train) / batch_size, epochs=epochs)

#model.fit(x_train,y_train, #changed
#          batch_size=batch_size,
#          epochs=epochs,
#          shuffle=True,
#          verbose=1,
#          validation_data=(x_test, y_test))

score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

Y_test = np.argmax(y_test, axis=1) # Convert one-hot to index
y_pred = model.predict_classes(x_test)
print(classification_report(Y_test, y_pred))

