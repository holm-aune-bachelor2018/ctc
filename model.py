import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Masking
from keras.layers import BatchNormalization
from keras.utils import np_utils

np.random.seed(7)
loss = 'categorical_crossentropy'
optimizer = 'adam'
metrics = ['accuracy']


def simpleLSTM(units, input_shape, X, y):
    """
    :param units:
    :param input_shape: (batch_size(number of seq), time_steps(sequence length), input_dim(mfcc_features))
    :param X:
    :param Y:
    :return:
    """

    # Model compiling and fit
    model = Sequential()
    model.add(Masking(mask_value=0., input_shape=input_shape))
    #model.add(BatchNormalization())
    model.add(LSTM(units))
    model.add(Dense(y.shape[1], activation='softmax'))




'''
    model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
    model.fit(X,y, epochs=100, batch_size=1, verbose=0)

    #Results
    scores = model.evaluate(X, y, verbose=0)
    return scores
'''
