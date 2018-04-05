
'''
    Trains a simple deep NN on the MNIST dataset.
    Gets to 98.40% test accuracy after 20 epochs
    (there is *a lot* of margin for parameter tuning).
    2 seconds per epoch on a K520 GPU.

    adapted from: https://github.com/keras-team/keras/blob/master/examples/mnist_mlp.py

    updated by Tuka Alhanai CSAIL MIT, 5th April 2018

    script updated to show how to introduce and evaluate features into different layers of neural network.
'''

# from __future__ import print_function

import keras
from keras.datasets import mnist
from keras.models import Model
from keras.layers import Dense, Dropout, Input, concatenate
from keras.optimizers import RMSprop
import numpy as np

# generating all layer combinations of features
# raw, even, ge5
# 0 0 0
# 0 0 1
# 0 0 2
# 0 1 0
# 0 1 1
# 0 1 2
# 0 2 0
# 0 2 1
# 0 2 2

# raw pixel features remain in the first layer only
Nfeats = 3
layers_feats = []
for i_raw in [0]:

    # odd and greater-equal than 5 can be in any of the three layes (input, middle, output)
    for i_odd in [0,1,2]:
        for i_ge5 in [0,1,2]:
            print(i_raw, i_odd, i_ge5)
            layers_feats.append([i_raw, i_odd, i_ge5])



# define hyperparameters
batch_size = 128
num_classes = 10
epochs = 20

# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(60000, 784)
x_test = x_test.reshape(10000, 784)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

# generatingg features that are 1 if number is even
x_train_even = np.zeros(x_train.shape[0])
index_even = np.where(y_train[:,[0,2,4,6,8]] == 1)[0]
x_train_even[index_even] = 1

x_test_even = np.zeros(x_test.shape[0])
index_even = np.where(y_test[:,[0,2,4,6,8]] == 1)[0]
x_test_even[index_even] = 1

# generating feature that is 1 if number greater than and equal to 5
x_train_ge5 = np.zeros(x_train.shape[0])
index_ge5 = np.where(y_train[:,5:9] == 1)[0]
x_train_ge5[index_ge5] = 1

x_test_ge5 = np.zeros(x_test.shape[0])
index_ge5 = np.where(y_test[:,5:9] == 1)[0]
x_test_ge5[index_ge5] = 1

# looping through each layer combination of feature inputs
for exp in range(len(layers_feats)):
    print('===')
    print('... EVALUATING EXPERIMENT ...', exp)
    print('... FEATURES ARE INPUT (RAW, EVEN, GE5)', layers_feats[exp])

    # build model

    # define inputs
    input_raw = Input(shape=(x_train.shape[1],))
    input_even = Input(shape=(1,))
    input_ge5 = Input(shape=(1,))

    # defining features of input layar
    if layers_feats[exp][1] == 0 and layers_feats[exp][2] == 0:
        input0 = concatenate([input_raw, input_even, input_ge5])
    elif layers_feats[exp][1] == 0 and layers_feats[exp][2] != 0:
        input0 = concatenate([input_raw, input_even])
    elif layers_feats[exp][1] != 0 and layers_feats[exp][2] == 0:
        input0 = concatenate([input_raw, input_ge5])
    else:
        input0 = input_raw

    # build input layer
    x1 = Dense(16, activation='relu')(input0)
    x1 = Dropout(0.2)(x1)

    # check if features need to be input into middle layer
    if layers_feats[exp][1] == 1 and layers_feats[exp][2] == 1:
        input1 = concatenate([x1, input_even, input_ge5])
    elif layers_feats[exp][1] == 1 and layers_feats[exp][2] != 1:
        input1 = concatenate([x1, input_even])
    elif layers_feats[exp][1] != 1 and layers_feats[exp][2] == 1:
        input1 = concatenate([x1, input_ge5])
    else:
        input1 = x1

    # build middle layer
    x1 = Dense(16, activation='relu')(input1)
    x1 = Dropout(0.2)(x1)

    # check if features need to be input into final output
    if layers_feats[exp][1] == 2 and layers_feats[exp][2] == 2:
        input2 = concatenate([x1, input_even, input_ge5])
    elif layers_feats[exp][1] == 2 and layers_feats[exp][2] != 2:
        input2 = concatenate([x1, input_even])
    elif layers_feats[exp][1] != 2 and layers_feats[exp][2] == 2:
        input2 = concatenate([x1, input_ge5])
    else:
        input2 = x1

    # build output layer
    out = Dense(num_classes, activation='softmax')(input2)

    model = Model(inputs=[input_raw, input_even, input_ge5], outputs=out)

    # show model topology
    # model.summary()

    # compile model
    model.compile(loss='categorical_crossentropy',
                  optimizer=RMSprop(),
                  metrics=['accuracy'])

    # train model
    print('... TRAINING MODEL ...')
    history = model.fit([x_train, x_train_even, x_train_ge5], y_train,
                        batch_size=batch_size,
                        epochs=epochs,
                        verbose=0,
                        validation_data=([x_test, x_test_even, x_test_ge5], y_test))

    # evalaute model
    score = model.evaluate([x_test, x_test_even, x_test_ge5], y_test, verbose=0)

    # print performance
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

    del model

# eof